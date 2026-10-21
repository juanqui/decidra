#!/usr/bin/env python3
"""Verify a compressed-tensors W4A8 artifact against its BF16 source.

The previous d3-turbo artifact was quantized from a randomly initialized
model. This gate dequantizes every packed projection and requires a cosine
similarity of at least `--min-cos` with the source tensor, so a silent
wrong-class load cannot ship again.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

SOURCE = Path("/srv/models/d3-turbo")


def dequant(packed: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    nibbles = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    q = nibbles.reshape(packed.shape[0], -1).float()
    out = packed.shape[0]
    cols = q.shape[1]
    # Symmetric int4 stores q + 8; recover the signed value.
    q = q - 8.0
    q = q.reshape(out, cols // group, group)
    return (q * scale.float()[:, :, None]).reshape(out, cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    # 4-bit group-128 quantization legitimately lands near 0.98-0.99 on small
    # projections; a wrong-class/random load lands near 0. The gate is a
    # random-init detector, not a bit-exactness claim.
    parser.add_argument("--min-cos", type=float, default=0.95)
    parser.add_argument("--group", type=int, default=0)
    args = parser.parse_args()

    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    source_map = index["weight_map"]
    shards: dict[str, object] = {}
    failures = []
    checked = 0
    with safe_open(str(args.quant / "model.safetensors"), "pt") as quant:
        keys = list(quant.keys())
        packed_keys = [k for k in keys if k.endswith(".weight_packed")]
        print(f"packed modules: {len(packed_keys)}", flush=True)
        for i, key in enumerate(packed_keys):
            module = key[: -len(".weight_packed")]
            src_key = module + ".weight"
            if src_key not in source_map:
                failures.append((module, "missing source"))
                continue
            shard = source_map[src_key]
            handle = shards.get(shard)
            if handle is None:
                handle = safe_open(str(args.source / shard), "pt")
                shards[shard] = handle
            src = handle.get_tensor(src_key).float()
            packed = quant.get_tensor(key)
            scale = quant.get_tensor(module + ".weight_scale")
            group = args.group or (src.shape[-1] // scale.shape[-1])
            w = dequant(packed, scale, group)
            if w.shape != src.shape:
                failures.append((module, f"shape {tuple(w.shape)} vs {tuple(src.shape)}"))
                continue
            cos = torch.nn.functional.cosine_similarity(
                w.flatten(), src.flatten(), dim=0
            ).item()
            checked += 1
            if not math.isfinite(cos) or cos < args.min_cos:
                failures.append((module, f"cos {cos:.6f}"))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(packed_keys)}", flush=True)
    # Unpacked tensors (norms, A_log, dt_bias, conv1d, embeddings) must be
    # bit-identical to the source.
    with safe_open(str(args.quant / "model.safetensors"), "pt") as quant:
        for key in quant.keys():
            if key.endswith(".weight_packed"):
                continue
            if key.endswith(".weight_scale") or key.endswith(".weight_shape"):
                continue
            if key not in source_map:
                continue
            shard = source_map[key]
            handle = shards.get(shard)
            if handle is None:
                handle = safe_open(str(args.source / shard), "pt")
                shards[shard] = handle
            a = quant.get_tensor(key)
            b = handle.get_tensor(key)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                failures.append((key, "unpacked tensor differs"))
    print(f"checked {checked} modules; failures {len(failures)}")
    for module, why in failures[:20]:
        print(f"  FAIL {module}: {why}")
    if failures:
        raise SystemExit(1)
    print("quantization weight gate passed")


if __name__ == "__main__":
    main()
