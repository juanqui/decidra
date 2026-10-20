#!/usr/bin/env python3
"""Compare the engine's INT4/INT8 AutoTrust readout against the BF16 readout."""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, "/home/juanqui/Git/jev-engine/eval")
from client import kl_divergence, max_abs_delta, mean_abs_error  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/tmp/autotrust_sample.jsonl")
    parser.add_argument("--bf16", default="/tmp/autotrust_bf16.json")
    parser.add_argument("--engine", default="/tmp/autotrust_engine.json")
    args = parser.parse_args()

    sample = {r["id"]: r for r in map(json.loads, open(args.sample))}
    bf16 = {r["id"]: r for r in json.load(open(args.bf16))["rows"]}
    engine = {r["id"]: r for r in json.load(open(args.engine))["rows"]}
    ids = [pid for pid in sample if pid in bf16 and pid in engine]
    print(f"compared rows: {len(ids)}")

    by_kind: dict[str, list[tuple[list[float], list[float]]]] = {}
    deltas = []
    for pid in ids:
        row = sample[pid]
        values = torch.tensor(engine[pid]["logits"]) / row["temperature"]
        probabilities = torch.softmax(values, dim=-1).tolist()
        reference = bf16[pid]["probabilities"]
        delta = max_abs_delta(probabilities, reference)
        deltas.append((delta, pid, row["kind"], probabilities, reference))
        by_kind.setdefault(row["kind"], []).append((probabilities, reference))
        top = max(range(len(probabilities)), key=probabilities.__getitem__)
        ref_top = max(range(len(reference)), key=reference.__getitem__)
        print(
            pid, row["kind"], "top1",
            row["options"][top], "|", row["options"][ref_top],
            f"max {delta:.3e}",
            "p", [round(p, 5) for p in probabilities],
            "ref", [round(p, 5) for p in reference],
        )

    agreement = sum(
        1
        for pid in ids
        if max(range(len(engine[pid]["logits"])), key=lambda i: engine[pid]["logits"][i])
        == max(range(len(bf16[pid]["probabilities"])), key=lambda i: bf16[pid]["probabilities"][i])
    )
    all_pairs = [(p, q) for pairs in by_kind.values() for p, q in pairs]
    mae = sum(mean_abs_error(p, q) for p, q in all_pairs) / len(all_pairs)
    kl = sum(
        (kl_divergence(p, q) + kl_divergence(q, p)) / 2 for p, q in all_pairs
    ) / len(all_pairs)
    print(
        f"top1 agreement {agreement}/{len(ids)} | max {max(d[0] for d in deltas):.3e}"
        f" | mae {mae:.3e} | KL {kl:.3e}"
    )
    for kind, pairs in by_kind.items():
        kmae = sum(mean_abs_error(p, q) for p, q in pairs) / len(pairs)
        kkl = sum((kl_divergence(p, q) + kl_divergence(q, p)) / 2 for p, q in pairs) / len(pairs)
        kmax = max(max_abs_delta(p, q) for p, q in pairs)
        print(f"  {kind}: n={len(pairs)} max {kmax:.3e} mae {kmae:.3e} KL {kkl:.3e}")


if __name__ == "__main__":
    main()
