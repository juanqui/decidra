#!/usr/bin/env python3
"""Prepare AutoTrust readout probes: token ids + answer-code ids per held-out row.

Uses the model's own tokenizer on the rendered jev_format text (chat template),
the readout token ids from vllm_readout.json, and writes one JSON for the native
engine probe plus an annotated sample for the BF16 and comparison scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = Path("/srv/models/autotrust-jev-27b-v1.1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="/srv/models/jev-awq/corpus/autotrust-eval-v1.jsonl")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--probe", default="/tmp/autotrust_probe.json")
    parser.add_argument("--sample", default="/tmp/autotrust_sample.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.eval).read_text().splitlines() if line.strip()]
    # the eval file groups rows by kind; take a kind-balanced sample (60/40)
    shares = {"choice": 0.60, "noul": 0.40}
    chosen = []
    for kind, share in shares.items():
        pool = [r for r in rows if r["kind"] == kind]
        chosen.extend(pool[: round(args.rows * share)])
    rows = chosen
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    readout = json.loads((MODEL / "vllm_readout.json").read_text())
    codes = readout["codes"]
    code_ids = readout["readout_token_ids"]
    temperature = float(readout.get("temperature", 1.0))

    probe = []
    with open(args.sample, "w") as handle:
        for row in rows:
            n = len(row["options"])
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            row_ids = code_ids[:n]
            probe.append({"id": row["id"], "tokens": ids, "codes": row_ids})
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "kind": row["kind"],
                        "options": row["options"],
                        "code_names": codes[:n],
                        "token_ids": ids,
                        "code_ids": row_ids,
                        "temperature": temperature,
                        "tokens": len(ids),
                    }
                )
                + "\n"
            )
    Path(args.probe).write_text(json.dumps({"rows": probe}))
    print(f"probe rows: {len(probe)} (choice/noul) | tokens {min(len(r['tokens']) for r in probe)}..{max(len(r['tokens']) for r in probe)}")
    print(f"wrote {args.probe} and {args.sample}")


if __name__ == "__main__":
    main()
