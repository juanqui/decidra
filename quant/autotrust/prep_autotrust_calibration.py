#!/usr/bin/env python3
"""AutoTrust-JEV-27B-v1.1 calibration and held-out prompts, in the model's own format.

Renders raw decision rows with the model's shipped ``jev_format`` (system prompt,
chat template, 255 answer codes) exactly as the serving engine does. The
calibration set is kind-balanced and length-stratified; the held-out set is
disjoint. Both are written as JSONL for llm-compressor (``text``) and for the
BF16-vs-quant readout evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

MODEL = Path("/srv/models/autotrust-jev-27b-v1.1")
sys.path.insert(0, str(MODEL))
import jev_format  # noqa: E402


def question_of(row: dict) -> dict | None:
    kind = row["kind"]
    if kind == "choice":
        keys = list(row["options"])
        return {
            "type": "choice",
            "criteria": {str(key): None for key in keys},
            "instructions": row["question"],
        }
    if kind == "noul":
        return {
            "type": "noul",
            "criteria": {"false": None, "true": None},
            "instructions": row["question"],
        }
    return None  # the model's format has no score readout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/srv/models/jev-awq/corpus/calibration.jsonl")
    parser.add_argument("--calib-out", default="/srv/models/jev-awq/corpus/autotrust-calibration-v1.jsonl")
    parser.add_argument("--eval-out", default="/srv/models/jev-awq/corpus/autotrust-eval-v1.jsonl")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--mix", default="choice=0.60,noul=0.40")
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    codes, code_ids = jev_format.answer_codes(tokenizer)
    readout = json.loads((MODEL / "vllm_readout.json").read_text())
    assert readout["codes"] == codes, "readout codes differ from jev_format.answer_codes"
    assert readout["readout_token_ids"] == code_ids, "readout token ids differ"
    print(f"answer codes: {len(codes)}; temperature {readout.get('temperature', 1.0)}", flush=True)

    weights = {}
    for part in args.mix.split(","):
        kind, share = part.split("=")
        weights[kind.strip()] = float(share)

    rows: list[dict] = []
    with open(args.input) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["kind"] not in weights:
                continue
            if len(row.get("state") or "") > args.max_chars:
                continue
            question = question_of(row)
            if question is None:
                continue
            prompt = jev_format.render(tokenizer, row["state"], question, codes)
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens > args.max_tokens:
                continue
            rows.append({"row": row, "question": question, "text": prompt, "tokens": tokens})
    print(f"usable rows: {len(rows)}", flush=True)

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        by_kind[item["row"]["kind"]].append(item)

    rng = random.Random(args.seed)
    calibration: list[dict] = []
    for kind, share in weights.items():
        pool = sorted(by_kind[kind], key=lambda item: item["tokens"])
        want = min(round(args.samples * share), len(pool))
        picks = [pool[i] for i in range(0, len(pool), max(1, len(pool) // want))][:want]
        calibration.extend(picks)
    chosen = {id(item) for item in calibration}
    rng.shuffle(calibration)
    print(
        f"calibration: {len(calibration)} rows, tokens {min(i['tokens'] for i in calibration)}"
        f"..{max(i['tokens'] for i in calibration)}",
        flush=True,
    )

    evaluable = [item for item in rows if id(item) not in chosen]
    eval_rows: list[dict] = []
    for kind, share in weights.items():
        pool = sorted((i for i in evaluable if i["row"]["kind"] == kind), key=lambda item: item["tokens"])
        want = min(round(args.eval_samples * share), len(pool))
        picks = [pool[i] for i in range(0, len(pool), max(1, len(pool) // want))][:want]
        eval_rows.extend(picks)
    print(f"held-out: {len(eval_rows)} rows", flush=True)

    with open(args.calib_out, "w") as handle:
        for item in calibration:
            handle.write(json.dumps({"text": item["text"]}) + "\n")
    with open(args.eval_out, "w") as handle:
        for item in eval_rows:
            row = item["row"]
            handle.write(
                json.dumps(
                    {
                        "id": f"{row['kind']}-{abs(hash(item['text'])) % (1 << 32):08x}",
                        "kind": row["kind"],
                        "state": row["state"],
                        "question": row["question"],
                        "options": row["options"],
                        "truth": row.get("truth"),
                        "text": item["text"],
                        "tokens": item["tokens"],
                    }
                )
                + "\n"
            )
    print(f"wrote {args.calib_out} and {args.eval_out}", flush=True)


if __name__ == "__main__":
    main()
