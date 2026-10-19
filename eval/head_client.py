#!/usr/bin/env python3
"""Direct vLLM read-out of the 28-row suite, bypassing the Rust front.

For each suite row this builds the exact text prompt, asks vLLM's
``/v1/completions`` for the decision-head token log probabilities, applies the
head bias and per-kind temperature locally, and (optionally) compares the
result with the Rust ``/v1/decide`` answer on the same prompt.

    python3 head_client.py --vllm-url http://127.0.0.1:8002 \
        --rust-url http://127.0.0.1:8501 --out out/suite_head_vs_rust.json

Only the first 16 choice labels use the trained verbalizer slots; the suite
never exceeds four options, so this path needs no tokenizer.  Wide choices
would require the tokenizer's label token ids (see client.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from client import (
    DecideClient,
    DecideError,
    HeadCalibration,
    build_decision_text,
    canonical_options,
    top1_index,
)
from suite import (
    DEFAULT_SUITE,
    compare,
    evaluate_thresholds,
    load_suite,
    row_request,
    run_live,
    truth_accuracy,
)


def parse_logprob_entry(entry: Any) -> dict[int, float]:
    """vLLM may return either a mapping or a list of token/logprob objects."""
    out: dict[int, float] = {}
    if isinstance(entry, dict):
        for key, value in entry.items():
            token_id = int(str(key).split(":")[-1])
            out[token_id] = float(value)
        return out
    if isinstance(entry, list):
        for item in entry:
            if not isinstance(item, dict):
                continue
            token = item.get("token", item.get("token_id"))
            token_id = int(str(token).split(":")[-1])
            out[token_id] = float(item["logprob"])
        return out
    raise ValueError(f"unrecognized top_logprobs entry: {type(entry).__name__}")


class DirectVLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def decision_logprobs(self, text: str, token_ids: Sequence[int]) -> tuple[dict[int, float], dict[str, Any]]:
        body = {
            "model": self.model,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "add_special_tokens": False,
            "return_tokens_as_token_ids": True,
            "logprobs": len(token_ids),
            "logprob_token_ids": list(token_ids),
        }
        response = self.session.post(
            self.base_url + "/v1/completions", json=body, timeout=self.timeout
        )
        if response.status_code >= 400:
            raise DecideError(
                f"vLLM /v1/completions -> {response.status_code}: {response.text[:800]}"
            )
        data = response.json()
        entry = data["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs = parse_logprob_entry(entry)
        return logprobs, data.get("usage") or {}


def direct_rows(
    vllm: DirectVLLM,
    head: HeadCalibration,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for row in rows:
        kind, state, question, options = row_request(row)
        opts = canonical_options(kind, options)
        target = list(opts)
        entry: dict[str, Any] = {
            "id": row["id"],
            "kind": kind,
            "truth": row.get("truth"),
            "options": target,
        }
        try:
            text = build_decision_text(kind, state, question, options)
            if kind == "choice":
                lo, hi = head.ranges["choice"]
                if len(opts) > hi - lo:
                    raise ValueError("direct suite path supports <=16 choices")
                token_ids = head.verbalizer_ids[lo : lo + len(opts)]
                logprobs, usage = vllm.decision_logprobs(text, token_ids)
                probabilities = head.probabilities(
                    kind, logprobs, len(opts), label_token_ids=token_ids
                )
            else:
                lo, hi = head.ranges[kind]
                token_ids = head.verbalizer_ids[lo:hi]
                logprobs, usage = vllm.decision_logprobs(text, token_ids)
                probabilities = head.probabilities(kind, logprobs)
            best = top1_index(probabilities)
            entry.update(
                {
                    "ok": True,
                    "choice": target[best],
                    "choice_index": best,
                    "probabilities": probabilities,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    },
                }
            )
        except (DecideError, ValueError, KeyError, IndexError) as error:
            entry.update({"ok": False, "error": str(error)})
        results.append(entry)
    return {
        "label": f"direct-vllm:{vllm.base_url}",
        "url": vllm.base_url,
        "model": vllm.model,
        "rows": results,
        "errors": sum(not row["ok"] for row in results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="jev-awq")
    parser.add_argument("--rust-url", default=None, help="optional Rust backend to compare")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--head-dir", default=None)
    parser.add_argument("--out", default="out/suite_head_vs_rust.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-top1", type=float, default=1.0)
    parser.add_argument("--max-kl", type=float, default=0.001)
    parser.add_argument("--max-mae", type=float, default=0.0002)
    parser.add_argument("--max-delta", type=float, default=0.001)
    args = parser.parse_args(argv)

    try:
        rows = load_suite(args.suite)
        base = Path(args.head_dir) if args.head_dir else Path(__file__).resolve().parent / "data"
        head = HeadCalibration.from_paths(
            base / "decision_head.json", base / "calibration.json"
        )
        vllm = DirectVLLM(args.vllm_url, args.model, timeout=args.timeout)
        direct = direct_rows(vllm, head, rows)
        if direct["errors"]:
            print(
                f"warning: {direct['errors']}/{len(rows)} direct vLLM rows failed",
                file=sys.stderr,
            )
        if args.rust_url:
            rust = run_live(
                args.rust_url,
                rows,
                label=f"live:{args.rust_url}",
                model=None,
                strategy="auto",
                timeout=args.timeout,
                retries=3,
            )
        else:
            rust = {
                "label": "none",
                "rows": [
                    {"id": row["id"], "ok": False, "error": "no --rust-url"}
                    for row in rows
                ],
                "errors": len(rows),
            }
        comparison = compare(direct, rust, rows)
        verdict = evaluate_thresholds(
            comparison["summary"],
            min_top1=args.min_top1,
            max_kl=args.max_kl,
            max_mae=args.max_mae,
            max_delta=args.max_delta,
            max_errors=0,
        )
        report = {
            "tool": "jev-eval/suite",
            "suite": {"path": str(Path(args.suite).resolve()), "rows": len(rows)},
            "backends": {"a": direct, "b": rust},
            "rows": comparison["rows"],
            "summary": comparison["summary"],
            "truth_accuracy": {"a": truth_accuracy(direct, rows), "b": truth_accuracy(rust, rows)},
            **verdict,
            "direct_vllm": {
                "url": args.vllm_url,
                "model": args.model,
                "head": str(base),
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        s = comparison["summary"]
        print(
            f"direct vLLM vs {args.rust_url or 'none'}: compared={s['compared']}/{len(rows)} "
            f"top1={s['top1_agreement']} mean_KL={s['mean_kl']} "
            f"mean_MAE={s['mean_mae']} max_delta={s['max_delta']} "
            f"pass={verdict['pass']} -> {out}"
        )
        return 0 if verdict["pass"] else 1
    except (OSError, ValueError, KeyError, DecideError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
