#!/usr/bin/env python3
"""d3-turbo calibration prompts with the exact prefix_bidir state split.

Reads the shared decision corpus, renders each prompt with the checkpoint's
own `d3_format.render`, tokenizes it once, and records the state block's token
count from the tokenizer offsets. Quantization calibration builds the same
dense block mask the engine uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL = Path("/srv/models/d3-turbo")
sys.path.insert(0, str(MODEL))
import d3_format  # noqa: E402


def question_of(row: dict) -> dict:
    if row["kind"] == "choice":
        return {
            "type": "choice",
            "criteria": {str(k): None for k in row["options"]},
            "instructions": row["question"],
        }
    return {
        "type": "noul",
        "criteria": {"false": None, "true": None},
        "instructions": row["question"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/srv/models/d3-calibration.jsonl")
    parser.add_argument("--out", default="/srv/models/d3-calibration-block-v1.jsonl")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    codes, _ = d3_format.answer_codes(tokenizer)
    rows = []
    with open(args.input) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") not in ("choice", "noul"):
                continue
            if len(row.get("state") or "") > args.max_chars:
                continue
            text = d3_format.render(tokenizer, row["state"], question_of(row), codes)
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            n = len(enc["input_ids"])
            if n > args.max_tokens:
                continue
            split = d3_format.state_token_count(
                enc["offset_mapping"], d3_format.state_end_char(text, row["state"])
            )
            if not 0 < split < n:
                continue
            rows.append({"text": text, "tokens": n, "state_len": split, "kind": row["kind"]})
    # Stratify by kind and length, deterministic.
    choice = sorted((r for r in rows if r["kind"] == "choice"), key=lambda r: r["tokens"])
    noul = sorted((r for r in rows if r["kind"] == "noul"), key=lambda r: r["tokens"])
    want_c = min(args.samples * 3 // 5, len(choice))
    want_n = min(args.samples - want_c, len(noul))
    picked = []
    for pool, want in ((choice, want_c), (noul, want_n)):
        if want:
            step = max(1, len(pool) // want)
            picked.extend(pool[::step][:want])
    with open(args.out, "w") as handle:
        for row in picked:
            handle.write(json.dumps(row) + "\n")
    print(
        f"wrote {len(picked)} rows to {args.out}: "
        f"tokens {min(r['tokens'] for r in picked)}..{max(r['tokens'] for r in picked)}, "
        f"state {min(r['state_len'] for r in picked)}..{max(r['state_len'] for r in picked)}"
    )


if __name__ == "__main__":
    main()
