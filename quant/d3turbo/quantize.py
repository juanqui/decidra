#!/usr/bin/env python3
"""Quantize d3-turbo to W4A8 (INT4 group-64 + INT8 per-token) for Decidra.

The shipped recipe is GPTQ-only. AWQ is available for experiments but is not
used for the published artifact: both AWQ+GPTQ test artifacts were far from
BF16, while GPTQ-only stayed within 0.04 on the same row.

    python3 quant/d3turbo/quantize.py --no-awq --group 64 --samples 1024 \\
        --calib /srv/models/d3-calibration-block-v1.jsonl \\
        --out /srv/models/d3-turbo-w4a8-g64-v4

The previous artifact was quantized from a randomly initialized model because
`oneshot(model=<path>)` loads `AutoModelForCausalLM`; this checkpoint is a
`Qwen3_5Model` whose state dict lives under `language_model.*`. This script
loads the correct class first and passes the instance to `oneshot`.

Calibration uses the checkpoint's real `prefix_bidir` mask: a dense
`[B,1,L,L]` visibility tensor where state queries see only the state and
question queries see the whole row. The GDN layers get the 2-D padding mask.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier

MODEL = Path("/srv/models/d3-turbo")
OUT = Path("/srv/models/d3-turbo-w4a8-g64")
CALIB = Path("/srv/models/d3-calibration-block-v1.jsonl")
IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*mtp.*",
    "readout",
]
AUX = [
    "chat_template.jinja",
    "config.json",
    "decision_config.json",
    "d3_compose.py",
    "d3_engine.py",
    "d3_fast.py",
    "d3_format.py",
    "d3_kernels.py",
    "d3_runtime.py",
    "d3_server.py",
    "modeling_d3.py",
    "pipeline_d3.py",
    "preprocessor_config.json",
    "requirements.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "readout.safetensors",
]


def mappings() -> list[AWQMapping]:
    config = json.loads((MODEL / "config.json").read_text())
    layer_types = config["text_config"]["layer_types"]
    full_attn = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    linear_attn = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]

    def alt(indices: list[int]) -> str:
        return "(?:" + "|".join(str(i) for i in indices) + ")"

    return [
        AWQMapping(
            f"re:.*layers\\.{alt(full_attn)}\\.input_layernorm",
            [
                f"re:.*layers\\.{alt(full_attn)}\\.self_attn\\.{p}"
                for p in ("q_proj", "k_proj", "v_proj")
            ],
        ),
        AWQMapping(
            f"re:.*layers\\.{alt(linear_attn)}\\.input_layernorm",
            [
                f"re:.*layers\\.{alt(linear_attn)}\\.linear_attn\\.{p}"
                for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            ],
        ),
        AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]),
        AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
    ]


class BlockCalibration(torch.utils.data.Dataset):
    """Full prompts with the d3-turbo block visibility mask, one row per item."""

    def __init__(self, path: Path, tokenizer, limit: int, max_length: int = 2048):
        self.rows = []
        self.max_length = max_length
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            if len(ids) > self.max_length:
                continue
            self.rows.append((ids, int(row["state_len"])))
            if len(self.rows) >= limit:
                break
        if not self.rows:
            raise SystemExit(f"no calibration rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ids, split = self.rows[i]
        length = len(ids)
        mask = torch.zeros(1, length, length, dtype=torch.bool)
        for q in range(length):
            mask[0, q, : split if q < split else length] = True
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "state_len": split,
            "full_mask": mask,
            "linear_mask": torch.ones(length, dtype=torch.long),
        }


def collate(batch: list[dict], pad: int) -> dict:
    if len(batch) != 1:
        raise ValueError("block calibration uses batch size 1")
    row = batch[0]
    ids = row["input_ids"][None, :]
    length = ids.shape[1]
    positions = torch.arange(length, dtype=torch.long)[None, None, :].expand(4, 1, -1)
    return {
        "input_ids": ids,
        "attention_mask": {
            "full_attention": row["full_mask"][None],
            "linear_attention": row["linear_mask"][None],
        },
        "position_ids": positions,
    }


def post(out: Path) -> None:
    # The decision checkpoint has no lm_head; a CausalLM wrapper adds one.
    # Drop it so HF/vLLM-SR loaders do not see a shape that conflicts with
    # vocab_size. Then merge the source vision tower in BF16: the language
    # model is quantized, but the 27-block image encoder is not.
    from safetensors.torch import load_file, save_file

    weights = out / "model.safetensors"
    state = load_file(str(weights))
    stripped = state.pop("lm_head.weight", None) is not None
    visual = {}
    for shard in sorted(MODEL.glob("model-*.safetensors")):
        with safe_open(str(shard), "pt") as handle:
            for key in handle.keys():
                if key.startswith("visual."):
                    visual[key] = handle.get_tensor(key)
    if not visual:
        raise SystemExit("source checkpoint has no visual.* tensors")
    state.update(visual)
    save_file(state, str(weights), metadata={"format": "pt"})
    print(
        f"{'stripped lm_head.weight; ' if stripped else ''}merged {len(visual)} BF16 visual tensors",
        flush=True,
    )
    shutil.copy(MODEL / "readout.safetensors", out / "readout.safetensors")
    readout = json.loads((MODEL / "decision_config.json").read_text())
    ids = readout["token_ids"]
    codes = readout["codes"]
    temperature = float(readout.get("temperature", 1.0))
    head = {
        "bias": [0.0] * len(ids),
        "verbalizer_ids": ids,
        "slots": {
            "num_slots": len(ids),
            "ranges": {"noul": [0, 2], "score": [0, 6], "choice": [0, len(ids)]},
            "verbalizers": codes,
            "template_version": "d3-code-readout-v1",
        },
        "note": "d3-turbo code-readout; separate F32 readout; zero bias",
    }
    directory = out / "decision_head"
    directory.mkdir(exist_ok=True)
    (directory / "decision_head.json").write_text(json.dumps(head, indent=1))
    (directory / "calibration.json").write_text(
        json.dumps(
            {"version": 1, "per_kind": {"noul": temperature, "score": temperature, "choice": temperature}},
            indent=1,
        )
    )
    copied = 0
    for name in AUX:
        source = MODEL / name
        if source.exists() and not (out / name).exists():
            shutil.copy(source, out / name)
            copied += 1
    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    config["attention_mode"] = "prefix_bidir"
    config_path.write_text(json.dumps(config, indent=1))
    print(f"copied {copied} aux files; head {len(ids)} codes T={temperature}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--max-memory", default="20GiB")
    parser.add_argument("--pipeline", default="basic")
    parser.add_argument("--causal-calib", action="store_true")
    parser.add_argument("--sequential-targets", default=None)
    parser.add_argument(
        "--no-awq",
        action="store_true",
        help="GPTQ-only; this is the shipped recipe",
    )
    parser.add_argument("--calib", type=Path, default=CALIB)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    if args.causal_calib:
        rows = []
        for line in args.calib.read_text().splitlines():
            if line.strip():
                rows.append({"text": json.loads(line)["text"]})
                if len(rows) >= args.samples:
                    break
        dataset = Dataset.from_list(rows)
        loader = None
        print(f"causal calibration rows: {len(dataset)}", flush=True)
    else:
        dataset = BlockCalibration(args.calib, tokenizer, args.samples, args.max_seq)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, pad),
        )
        print(f"calibration rows: {len(dataset)}", flush=True)

    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        str(MODEL),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={0: args.max_memory, "cpu": "90GiB"},
        key_mapping={r"^language_model\.": "model."},
        output_loading_info=True,
    )
    missing = [k for k in loading.get("missing_keys", []) if k != "lm_head.weight"]
    if missing:
        raise SystemExit(f"unexpected missing weights: {missing[:8]}")
    # A random lm_head wastes 2.5 GiB and computes full-vocab logits we never
    # use. A one-row head keeps the forward valid and the save small.
    model.lm_head = torch.nn.Linear(model.config.hidden_size, 1, bias=False)
    with safe_open(str(MODEL / "model-00001-of-00011.safetensors"), "pt") as handle:
        probe = handle.get_tensor("language_model.layers.0.mlp.gate_proj.weight")
    loaded = model.model.layers[0].mlp.gate_proj.weight
    if loaded.device.type != "meta" and not torch.equal(
        loaded.float().cpu(), probe.float()
    ):
        raise SystemExit("layer-0 gate_proj does not match the source tensor")
    print(
        "loaded Qwen3_5ForCausalLM (language_model.* -> model.*); layer0 gate device "
        f"{model.model.layers[0].mlp.gate_proj.weight.device}",
        flush=True,
    )

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            group_size=args.group,
            strategy="group",
            dynamic=False,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="token",
            dynamic=True,
        ),
        format=CompressionFormat.pack_quantized,
    )
    recipe = [] if args.no_awq else [
        AWQModifier(mappings=mappings(), duo_scaling="both", n_grid=args.n_grid),
    ]
    recipe.append(
        GPTQModifier(
            config_groups={"group_0": scheme},
            targets="Linear",
            ignore=IGNORE,
            actorder="weight",
        )
    )
    oneshot(
        model=model,
        recipe=recipe,
        dataset=loader if loader is not None else dataset,
        tokenizer=tokenizer,
        output_dir=str(args.out),
        max_seq_length=args.max_seq,
        num_calibration_samples=len(dataset),
        pipeline=args.pipeline,
        sequential_targets=(
            [args.sequential_targets] if args.sequential_targets else None
        ),
    )
    print(f"quantized -> {args.out}", flush=True)
    post(args.out)
    print("post-processing done", flush=True)


if __name__ == "__main__":
    main()
