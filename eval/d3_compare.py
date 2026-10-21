#!/usr/bin/env python3
"""Compare d3-turbo engine logits with the checkpoint's BF16 reference.

The probe rows carry `tokens`, `split` and the option count; the engine output
carries `logits` for each row. Probabilities are softmax over the question's
own options at the checkpoint temperature (1.0 for d3-turbo).
"""
from __future__ import annotations

import argparse
import json
import math


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    e = [math.exp(v - m) for v in values]
    s = sum(e)
    return [v / s for v in e]


def kl(p: list[float], q: list[float]) -> float:
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0 and b > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="/tmp/d3_probe_split32.json")
    parser.add_argument("--engine", default="/tmp/d3_engine_eval32.json")
    parser.add_argument("--hf", default="/tmp/d3_hf_eval.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-mae", type=float, default=0.02)
    parser.add_argument("--max-delta", type=float, default=0.10)
    parser.add_argument("--min-agree", type=float, default=0.95)
    parser.add_argument("--near-tie", type=float, default=0.05)
    args = parser.parse_args()

    probe = {
        row["id"]: row
        for row in json.load(open(args.probe))["rows"]
    }
    engine = {
        row["id"]: row for row in json.load(open(args.engine))["rows"]
    }
    hf = {row["id"]: row for row in json.load(open(args.hf))["rows"]}
    ids = [i for i in probe if i in engine and i in hf]
    if not ids:
        raise SystemExit("no overlapping rows")
    agree = 0
    maes = []
    deltas = []
    safe_deltas = []
    kls = []
    unsafe_flips = []
    for pid in ids:
        row = probe[pid]
        n = len(row["options"])
        ref = hf[pid]["probabilities"]
        logits = [v / args.temperature for v in engine[pid]["logits"][:n]]
        got = softmax(logits)
        delta = max(abs(a - b) for a, b in zip(got, ref))
        mae = sum(abs(a - b) for a, b in zip(got, ref)) / n
        got_top = max(range(n), key=got.__getitem__)
        ref_top = max(range(n), key=ref.__getitem__)
        if got_top == ref_top:
            agree += 1
        else:
            ranked = sorted(ref, reverse=True)
            margin = ranked[0] - ranked[1]
            if margin > args.near_tie:
                unsafe_flips.append((pid, margin))
        maes.append(mae)
        deltas.append(delta)
        ranked = sorted(ref, reverse=True)
        if ranked[0] - ranked[1] > args.near_tie:
            safe_deltas.append(delta)
        kls.append((kl(got, ref) + kl(ref, got)) / 2)
        print(
            f"{pid} {row['kind']:6} top1 {max(range(n), key=got.__getitem__)}"
            f"/{max(range(n), key=ref.__getitem__)} max {delta:.4f} mae {mae:.4f}"
        )
    print(
        f"rows {len(ids)} top1 {agree}/{len(ids)} "
        f"mean mae {sum(maes)/len(maes):.4f} max mae {max(maes):.4f} "
        f"max delta {max(deltas):.4f} "
        f"max delta (non-near-tie) {max(safe_deltas, default=0.0):.4f} "
        f"mean kl {sum(kls)/len(kls):.4f}"
    )
    mean_mae = sum(maes) / len(maes)
    if (
        agree / len(ids) < args.min_agree
        or mean_mae > args.max_mae
        or max(safe_deltas, default=0.0) > args.max_delta
        or unsafe_flips
    ):
        print(f"unsafe flips: {unsafe_flips}")
        raise SystemExit("accuracy gate failed")
    print("accuracy gate passed")


if __name__ == "__main__":
    main()
