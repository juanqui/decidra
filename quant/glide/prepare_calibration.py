#!/usr/bin/env python3
"""Render GLiDE calibration prompts from the decision training corpus.

The source corpus is the raw decision corpus used for d3 training
(`calibration.jsonl`), not generic web text. Rows are sampled by question kind
with a fixed seed, rendered with the checkpoint's own Prompter, and tokenized
with the checkpoint tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def question(row: dict) -> dict:
    kind = row["kind"]
    options = list(row.get("options") or [])
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {str(o): str(o) for o in options},
        }
    if kind == "score":
        return {
            "type": "score",
            "instructions": row["question"],
            "criteria": [str(o) for o in options],
        }
    if kind == "noul":
        return {"type": "noul", "instructions": row["question"]}
    raise ValueError(f"unsupported kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/GLiDE"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model))
    from glide.prompt import Prompter, options_for

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    decision = json.loads((args.model / "decision_config.json").read_text())
    prompter = Prompter(tokenizer, decision["codes"])
    rows = defaultdict(list)
    with args.input.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") in ("choice", "noul", "score"):
                rows[row["kind"]].append(row)
    if not rows:
        raise SystemExit(f"no usable rows in {args.input}")

    total = sum(len(v) for v in rows.values())
    wanted = {kind: max(1, args.samples * len(v) // total) for kind, v in rows.items()}
    # Correct rounding so the output has exactly the requested number.
    while sum(wanted.values()) < args.samples:
        kind = max(rows, key=lambda k: len(rows[k]) / wanted[k])
        wanted[kind] += 1
    rng = random.Random(args.seed)
    picked = []
    for kind, pool in rows.items():
        rng.shuffle(pool)
        taken = 0
        for row in pool:
            q = question(row)
            text = prompter.text(row["state"], q, options_for(q))
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not (0 < len(ids) <= args.max_tokens):
                continue
            picked.append({
                "id": row.get("id"),
                "kind": kind,
                "domain": row.get("domain"),
                "tokens": len(ids),
                "input_ids": ids,
            })
            taken += 1
            if taken >= wanted[kind]:
                break
        print(f"{kind}: {taken}/{wanted[kind]}", flush=True)
    if len(picked) < args.samples:
        print(f"warning: wrote {len(picked)} rows (requested {args.samples})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in picked:
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {len(picked)} rows to {args.out} sha256={digest}", flush=True)
    print("token range", min(r["tokens"] for r in picked), max(r["tokens"] for r in picked), flush=True)


if __name__ == "__main__":
    main()
