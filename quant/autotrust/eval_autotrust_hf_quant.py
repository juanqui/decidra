#!/usr/bin/env python3
"""HF transformers forward of the quantized AutoTrust checkpoint (CPU).

Same 16 held-out prompts/code ids as the BF16 reference and the native probe.
Comparing this to BF16 isolates the quantization from the engine's forward;
comparing it to the engine isolates the engine's forward/format handling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

MODEL = Path("/srv/models/autotrust-jev-27b-v1.1-awq-g64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/tmp/autotrust_sample.jsonl")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--out", default="/tmp/autotrust_hf_quant.json")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = [json.loads(line) for line in Path(args.sample).read_text().splitlines() if line.strip()]
    rows = rows[: args.rows]
    print(f"loading quantized {MODEL} ...", flush=True)
    started = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"model ready in {time.time() - started:.1f}s", flush=True)
    results = []
    for row in rows:
        started = time.time()
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([row["token_ids"]], dtype=torch.long), use_cache=False
            ).logits[0, -1].float()
        values = torch.tensor([float(logits[i]) for i in row["code_ids"]]) / row["temperature"]
        probabilities = torch.softmax(values, dim=-1).tolist()
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        results.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "choice": row["options"][best],
                "probabilities": probabilities,
                "seconds": round(time.time() - started, 2),
            }
        )
        print(
            row["id"], "tokens", len(row["token_ids"]), "choice", row["options"][best],
            "p", [round(p, 6) for p in probabilities], f"{time.time() - started:.1f}s", flush=True,
        )
    Path(args.out).write_text(json.dumps({"rows": results}, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
