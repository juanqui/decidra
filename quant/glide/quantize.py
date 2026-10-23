#!/usr/bin/env python3
"""Quantize GLiDE to W4A8 (INT4 group-64 weights, INT8 per-token activations).

Recipe: AWQ smoothing followed by GPTQ, calibrated on the decision training
corpus rendered with GLiDE's own prompt template. Vision, the 255-way readout,
and the LM head are not quantized.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier

IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*", "readout"]


def mappings(model_dir: Path) -> list[AWQMapping]:
    config = json.loads((model_dir / "config.json").read_text())
    layer_types = config["text_config"]["layer_types"]
    full = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    linear = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]

    def alt(indices: list[int]) -> str:
        return "(?:" + "|".join(str(i) for i in indices) + ")"

    return [
        AWQMapping(
            f"re:.*layers\\.{alt(full)}\\.input_layernorm",
            [f"re:.*layers\\.{alt(full)}\\.self_attn\\.{p}" for p in ("q_proj", "k_proj", "v_proj")],
        ),
        AWQMapping(
            f"re:.*layers\\.{alt(linear)}\\.input_layernorm",
            [f"re:.*layers\\.{alt(linear)}\\.linear_attn\\.{p}" for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")],
        ),
        AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]),
        AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
    ]


class TokenCalibration(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise SystemExit(f"no calibration rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        ids = self.rows[index]["input_ids"]
        length = len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "full_mask": torch.ones(1, length, length, dtype=torch.bool),
            "linear_mask": torch.ones(length, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("GLiDE calibration uses batch size 1")
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


def eval_prompts(model_dir: Path, path: Path, tokenizer, limit: int) -> list[dict]:
    sys.path.insert(0, str(model_dir))
    from glide.prompt import Prompter, options_for

    prompter = Prompter(tokenizer, json.loads((model_dir / "decision_config.json").read_text())["codes"])
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") not in ("choice", "noul", "score"):
            continue
        opts = row.get("options")
        if isinstance(opts, str):
            opts = ast.literal_eval(opts)
        opts = [str(o) for o in opts]
        kind = row["kind"]
        if kind == "choice":
            q = {"type": "choice", "instructions": row["question"], "criteria": {o: o for o in opts}}
        elif kind == "score":
            q = {"type": "score", "instructions": row["question"], "criteria": opts}
        else:
            q = {"type": "noul", "instructions": row["question"]}
        text = prompter.text(row["state"], q, options_for(q))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not 0 < len(ids) <= 2048:
            continue
        rows.append({"id": row.get("id"), "kind": kind, "input_ids": ids})
        if len(rows) >= limit:
            break
    return rows


def eval_logits(model, rows: list[dict], device) -> dict:
    out = {}
    for row in rows:
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)[None, :]
        length = ids.shape[1]
        mask = {
            "full_attention": torch.ones(1, 1, length, length, dtype=torch.bool, device=device),
            "linear_attention": torch.ones(1, length, dtype=torch.long, device=device),
        }
        pos = torch.arange(length, dtype=torch.long, device=device)[None, None, :].expand(4, 1, -1)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[0, -1]
        out[row["id"]] = {
            "kind": row["kind"],
            "logits": logits.float().cpu().tolist(),
        }
    return out


def compare_eval(before: dict, after: dict) -> dict:
    agree = total = 0
    maes, deltas = [], []
    for key, a in before.items():
        if key not in after:
            continue
        b = after[key]
        n = min(len(a["logits"]), len(b["logits"]))
        if n < 2:
            continue
        pa = torch.softmax(torch.tensor(a["logits"][:n]), dim=0).tolist()
        pb = torch.softmax(torch.tensor(b["logits"][:n]), dim=0).tolist()
        total += 1
        agree += int(max(range(n), key=pa.__getitem__) == max(range(n), key=pb.__getitem__))
        maes.append(sum(abs(x - y) for x, y in zip(pa, pb)) / n)
        deltas.append(max(abs(x - y) for x, y in zip(pa, pb)))
    return {
        "rows": total,
        "top1": agree,
        "mean_mae": sum(maes) / max(1, len(maes)),
        "max_delta": max(deltas, default=0.0),
        "max_mae": max(maes, default=0.0),
    }


def postprocess(source: Path, out: Path) -> None:
    """Drop the unused LM head and copy the decision-model sidecars."""
    from safetensors.torch import load_file, save_file

    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [p.name for p in out.glob("model*.safetensors")]
    for name in files:
        path = out / name
        state = load_file(str(path))
        if state.pop("lm_head.weight", None) is not None:
            save_file(state, str(path), metadata={"format": "pt"})
            if index_path.exists():
                index["weight_map"].pop("lm_head.weight", None)
                index_path.write_text(json.dumps(index, indent=2))
            print(f"stripped lm_head from {name}", flush=True)
    for name in ("decision_config.json", "readout.safetensors", "chat_template.jinja",
                 "generation_config.json", "LICENSE", "NOTICE", "README.md", "requirements.txt"):
        src = source / name
        if src.exists():
            shutil.copy2(src, out / name)
    for name in ("glide", "licenses"):
        src = source / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
    print(f"postprocessed {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/GLiDE"))
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--max-memory", default="20GiB")
    parser.add_argument("--pipeline", choices=["sequential", "basic"], default="sequential")
    parser.add_argument("--no-awq", action="store_true")
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument("--eval-baseline", type=Path, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--eval-rows", type=int, default=32)
    args = parser.parse_args()

    dataset = TokenCalibration(args.calib)
    count = min(args.samples, len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    print(f"calibration rows: {count}", flush=True)

    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={0: args.max_memory, "cpu": "90GiB"},
        output_loading_info=True,
    )
    missing = [k for k in loading.get("missing_keys", []) if k != "lm_head.weight"]
    unexpected = [k for k in loading.get("unexpected_keys", []) if "readout" not in k]
    if missing:
        raise SystemExit(f"missing weights: {missing[:8]}")
    if unexpected:
        print(f"unexpected weights: {unexpected[:8]}", flush=True)
    # GLiDE has no LM head; use the 255-way decision readout for the
    # before/after accuracy check, then swap in a one-row dummy while the
    # quantizer runs so the full vocab head does not waste memory.
    from safetensors.torch import load_file
    device = next(model.parameters()).device
    readout = torch.nn.Linear(model.config.text_config.hidden_size, 255, bias=False).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        readout.weight.copy_(load_file(str(args.model / "readout.safetensors"))["weight"].to(torch.bfloat16))
    model.lm_head = readout
    eval_rows = eval_prompts(args.model, args.eval, tokenizer, args.eval_rows) if args.eval else []
    if args.eval_baseline:
        before = json.loads(args.eval_baseline.read_text())
        print(f"loaded eval baseline {args.eval_baseline}", flush=True)
    else:
        before = eval_logits(model, eval_rows, device) if eval_rows else None
    print(f"eval rows: {len(eval_rows)}", flush=True)
    if args.baseline_only:
        if args.eval_out and before is not None:
            args.eval_out.write_text(json.dumps(before, indent=1))
            print(f"wrote baseline {args.eval_out}", flush=True)
        return
    model.lm_head = torch.nn.Linear(model.config.text_config.hidden_size, 1, bias=False)

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
        AWQModifier(mappings=mappings(args.model), duo_scaling="both", n_grid=args.n_grid),
    ]
    recipe.append(GPTQModifier(
        config_groups={"group_0": scheme},
        targets="Linear",
        ignore=IGNORE,
        actorder="weight",
    ))
    oneshot(
        model=model,
        recipe=recipe,
        dataset=loader,
        tokenizer=tokenizer,
        output_dir=str(args.out),
        max_seq_length=2048,
        num_calibration_samples=count,
        pipeline=args.pipeline,
        sequential_targets=["Qwen3_5DecoderLayer"] if args.pipeline == "sequential" else None,
    )
    if before is not None:
        model.lm_head = readout.to(device)
        after = eval_logits(model, eval_rows, device)
        metrics = compare_eval(before, after)
        print(f"eval metrics {metrics}", flush=True)
        if args.eval_out:
            args.eval_out.write_text(json.dumps({"before": before, "after": after, "metrics": metrics}, indent=1))
        if metrics["top1"] < metrics["rows"] * 0.90:
            print("WARNING: quantized top-1 retention below 90%", flush=True)
    postprocess(args.model, args.out)
    print(f"done {args.out}", flush=True)


if __name__ == "__main__":
    main()
