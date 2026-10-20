#!/usr/bin/env python3
"""BF16 AutoTrust readout on the held-out probe rows (CPU, causal forward).

Mirrors the model's own readout: the softmax over the n answer-code token logits
at the last prompt position, divided by the readout temperature. The forward is
the HF causal path, the same one the native engine implements, so the BF16-vs-
engine comparison isolates the INT4/INT8 quantization error.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

MODEL = Path("/srv/models/autotrust-jev-27b-v1.1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/tmp/autotrust_sample.jsonl")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default="/tmp/autotrust_bf16.json")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    rows = [json.loads(line) for line in Path(args.sample).read_text().splitlines() if line.strip()]
    rows = rows[args.offset : args.offset + args.rows]
    print(f"loading {MODEL} (bf16, cpu) ...", flush=True)
    started = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"model ready in {time.time() - started:.1f}s", flush=True)

    results = []
    for row in rows:
        started = time.time()
        ids = row["token_ids"]
        code_ids = row["code_ids"]
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False
            ).logits[0, -1].float()
        values = torch.tensor([float(logits[i]) for i in code_ids]) / row["temperature"]
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
            row["id"], row["kind"], "tokens", len(ids),
            "choice", row["options"][best],
            "p", [round(p, 6) for p in probabilities],
            f"{time.time() - started:.1f}s", flush=True,
        )
    Path(args.out).write_text(json.dumps({"rows": results}, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
