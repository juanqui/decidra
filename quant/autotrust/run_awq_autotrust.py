#!/usr/bin/env python3
"""AutoTrust-JEV-27B-v1.1 AWQ + GPTQ quantization for the Decidra native engine.

Target format: compressed-tensors W4A16, asymmetric INT4 group 64 -- the native
engine's highest-quality supported weight layout (Marlin W4A16 + runtime INT8
activation path). Calibration: 1024 kind-balanced prompts rendered with the
model's shipped jev_format (system prompt, chat template, 255 answer codes),
matching the serving distribution.

Post-processing:
- splice the untouched BF16 vision tower back in (llm-compressor drops it),
- keep lm_head (the folded 255-code readout) BF16 and indexed,
- copy the tokenizer/chat/readout/format files,
- emit the engine's decision_head/decision_head.json + calibration.json with the
  255 readout token ids, zero bias, and the model's readout temperature.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from safetensors import safe_open
from safetensors.torch import save_file

MODEL = Path("/srv/models/autotrust-jev-27b-v1.1")
OUT = Path("/srv/models/autotrust-jev-27b-v1.1-awq-g64")
CALIB = Path("/srv/models/jev-awq/corpus/autotrust-calibration-v1.jsonl")
IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*"]
AUX = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "decision_config.json",
    "vllm_readout.json",
    "jev_format.py",
    "serve_vllm.sh",
]


def load_calibration() -> Dataset:
    rows = [json.loads(line) for line in CALIB.read_text().splitlines() if line.strip()]
    print(f"calibration samples: {len(rows)}", flush=True)
    return Dataset.from_list(rows)


def mappings() -> list[AWQMapping]:
    config = json.loads((MODEL / "config.json").read_text())
    layer_types = config["text_config"]["layer_types"]
    full_attn = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    linear_attn = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    print(f"layer mix: {len(full_attn)} full-attention, {len(linear_attn)} linear-attention", flush=True)

    def alt(indices: list[int]) -> str:
        return "(?:" + "|".join(str(i) for i in indices) + ")"

    return [
        AWQMapping(
            f"re:.*layers\\.{alt(full_attn)}\\.input_layernorm",
            [f"re:.*layers\\.{alt(full_attn)}\\.self_attn\\.{proj}" for proj in ("q_proj", "k_proj", "v_proj")],
        ),
        AWQMapping(
            f"re:.*layers\\.{alt(linear_attn)}\\.input_layernorm",
            [
                f"re:.*layers\\.{alt(linear_attn)}\\.linear_attn\\.{proj}"
                for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            ],
        ),
        AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]),
        AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
    ]


def splice_vision(out: Path) -> None:
    main = out / "model.safetensors"
    assert main.exists(), main
    with safe_open(str(main), "pt") as handle:
        main_keys = list(handle.keys())
    vision_prefixes = ("model.visual.", "visual.")
    if any(key.startswith(vision_prefixes) for key in main_keys):
        print("vision tensors already present", flush=True)
        return
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    by_file: dict[str, list[str]] = {}
    for key, filename in index["weight_map"].items():
        if key.startswith(vision_prefixes):
            by_file.setdefault(filename, []).append(key)
    tensors = {}
    for filename, keys in by_file.items():
        with safe_open(str(MODEL / filename), "pt") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    target = out / "model-00002-of-00002.safetensors"
    save_file(tensors, str(target))
    weight_map = {key: main.name for key in main_keys}
    weight_map.update({key: target.name for key in tensors})
    total_size = main.stat().st_size + target.stat().st_size
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=1)
    )
    print(f"spliced {len(tensors)} vision tensors -> {target.name}", flush=True)


def keep_lm_head(out: Path) -> None:
    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if "lm_head.weight" in index["weight_map"]:
        print("lm_head already indexed", flush=True)
        return
    source = MODEL / "lm_head.safetensors"
    assert source.exists(), source
    target = out / "lm_head.safetensors"
    shutil.copy(source, target)
    index["weight_map"]["lm_head.weight"] = target.name
    total = sum((out / name).stat().st_size for name in set(index["weight_map"].values()))
    index.setdefault("metadata", {})["total_size"] = total
    index_path.write_text(json.dumps(index, indent=1))
    print("spliced lm_head.safetensors", flush=True)


def patch_config(out: Path) -> None:
    path = out / "config.json"
    config = json.loads(path.read_text())
    quant = config.get("quantization_config")
    if quant is not None:
        ignore = list(quant.get("ignore") or [])
        for pattern in ("re:.*visual.*", "lm_head"):
            if pattern not in ignore:
                ignore.append(pattern)
        quant["ignore"] = ignore
        path.write_text(json.dumps(config, indent=2))
        print("patched ignore:", ignore, flush=True)


def write_head(out: Path) -> None:
    readout = json.loads((MODEL / "vllm_readout.json").read_text())
    ids = readout["readout_token_ids"]
    codes = readout["codes"]
    temperature = float(readout.get("temperature", 1.0))
    head = {
        "bias": [0.0] * len(ids),
        "verbalizer_ids": ids,
        "slots": {
            "num_slots": len(ids),
            "ranges": {"noul": [0, 2], "score": [0, 6], "choice": [0, len(ids)]},
            "verbalizers": codes,
            "template_version": "autotrust-code-v1",
        },
        "note": "AutoTrust-JEV-27B-v1.1 jev-code-readout-v1; 255 code rows of lm_head; zero bias",
    }
    directory = out / "decision_head"
    directory.mkdir(exist_ok=True)
    (directory / "decision_head.json").write_text(json.dumps(head, indent=1))
    (directory / "calibration.json").write_text(
        json.dumps({"version": 1, "per_kind": {"noul": temperature, "score": temperature, "choice": temperature}}, indent=1)
    )
    print(f"wrote engine head: {len(ids)} codes, T={temperature}", flush=True)


def post(out: Path) -> None:
    splice_vision(out)
    keep_lm_head(out)
    patch_config(out)
    copied = 0
    for name in AUX:
        source = MODEL / name
        if source.exists() and not (out / name).exists():
            shutil.copy(source, out / name)
            copied += 1
    print(f"copied {copied} aux files", flush=True)
    write_head(out)


def main() -> None:
    dataset = load_calibration()
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=False, group_size=64,
            strategy="group", dynamic=False, zp_dtype="torch.int8",
        ),
    )
    recipe = [
        AWQModifier(mappings=mappings(), duo_scaling="both", n_grid=40),
        GPTQModifier(
            config_groups={"group_0": scheme},
            targets="Linear",
            ignore=IGNORE,
            actorder="weight",
        ),
    ]
    print("recipe: AWQ smoothing + GPTQ W4A16 g64 actorder=weight", flush=True)
    oneshot(
        model=str(MODEL),
        recipe=recipe,
        dataset=dataset,
        output_dir=str(OUT),
        max_seq_length=1024,
        num_calibration_samples=len(dataset),
    )
    print(f"quantized -> {OUT}", flush=True)
    post(OUT)
    print("post-processing done", flush=True)


if __name__ == "__main__":
    main()
