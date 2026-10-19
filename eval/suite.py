#!/usr/bin/env python3
"""Run the deterministic 28-row decision suite against one or two backends.

Suite inputs
------------
``awq-v2-suite.json`` (and every other saved ``*-suite.json``) contains only
results: id/truth/kind/choice/probabilities.  It has **no prompts**.  The
prompts are reconstructed from the canonical suite the saved results were
produced with: ``/opt/decider-stack/deploy/suite.jsonl`` (the default of the
repo's ``run_suite.py``).  A copy lives at ``data/suite.jsonl`` so this harness
is self-contained.  ``data/awq-v2-suite.json`` is the saved production-v2
result used as the default reference backend; ``data/awq-v4-suite.json`` is
kept for historical cross-quantization comparisons.

Two modes
---------
* ``--a-url URL --b-url URL``: live A vs live B on identical request bodies.
* ``--a-url URL``: live A vs the saved v2 result file (``--baseline``).

Each row sends exactly the body ``run_suite.py`` sends, except that ``score``
requests omit ``options`` (the server ignores them and always uses 0..5; the
saved results have six probabilities even where the suite's ``criteria`` list
is shorter).  Choice descriptions in ``criteria`` are not part of the prompt;
only the keys are, matching ``run_suite.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from client import (
    DecideClient,
    DecideError,
    align_probabilities,
    canonical_options,
    kl_divergence,
    max_abs_delta,
    mean_abs_error,
    top1_index,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = ROOT / "data" / "suite.jsonl"
DEFAULT_BASELINE = ROOT / "data" / "awq-v2-suite.json"
DEFAULT_OUT = ROOT / "out" / "suite.json"


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in seen:
            raise ValueError(f"duplicate suite id {row['id']!r} at line {line_number}")
        seen.add(row["id"])
        question = row["question"]
        if question["type"] not in ("noul", "score", "choice"):
            raise ValueError(f"unsupported kind {question['type']!r} at line {line_number}")
        if not row.get("state"):
            raise ValueError(f"empty state for {row['id']!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty suite: {path}")
    return rows


def row_request(row: Mapping[str, Any]) -> tuple[str, str, str, list[str] | None]:
    question = row["question"]
    kind = question["type"]
    if kind == "choice":
        criteria = question.get("criteria") or {}
        options: list[str] | None = [str(key) for key in criteria]
        if not 2 <= len(options) <= 256:
            raise ValueError(f"{row['id']}: choice has {len(options)} options")
    else:
        options = None
    return kind, row["state"], question.get("instructions") or "", options


def target_options(kind: str, options: Sequence[str] | None) -> list[str]:
    return canonical_options(kind, options)


def _canonical_from_question(row: Mapping[str, Any]) -> list[str]:
    kind, _, _, options = row_request(row)
    return target_options(kind, options)


def run_live(
    url: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    model: str | None,
    strategy: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    client = DecideClient(url, model=model, timeout=timeout, retries=retries)
    results: list[dict[str, Any]] = []
    for row in rows:
        kind, state, question, options = row_request(row)
        result: dict[str, Any] = {
            "id": row["id"],
            "kind": kind,
            "truth": row.get("truth"),
            "options": target_options(kind, options),
        }
        try:
            answer = client.decide(
                kind,
                state,
                question,
                options,
                strategy=strategy,
                model=model,
            )
            result.update(
                {
                    "ok": True,
                    "choice": answer.get("choice"),
                    "choice_index": answer.get("choice_index"),
                    "probabilities": align_probabilities(
                        answer, target_options(kind, options)
                    ),
                    "usage": answer.get("usage"),
                    "num_model_requests": answer.get("num_model_requests"),
                }
            )
        except (DecideError, ValueError, KeyError) as error:
            result.update({"ok": False, "error": str(error)})
        results.append(result)
    return {
        "label": label,
        "url": url,
        "model": model,
        "strategy": strategy,
        "rows": results,
        "errors": sum(not row["ok"] for row in results),
    }


def load_baseline(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    saved = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(saved, list):
        raise ValueError(f"baseline {path} has no rows list")
    by_id = {row["id"]: row for row in saved}
    results: list[dict[str, Any]] = []
    for row in rows:
        expected_options = _canonical_from_question(row)
        saved_row = by_id.get(row["id"])
        result: dict[str, Any] = {
            "id": row["id"],
            "kind": row["question"]["type"],
            "truth": row.get("truth"),
            "options": expected_options,
        }
        if saved_row is None:
            result.update({"ok": False, "error": "missing from baseline"})
        else:
            probabilities = [float(value) for value in saved_row["probabilities"]]
            if len(probabilities) != len(expected_options):
                result.update(
                    {
                        "ok": False,
                        "error": (
                            f"baseline has {len(probabilities)} probabilities, "
                            f"expected {len(expected_options)}"
                        ),
                    }
                )
            else:
                result.update(
                    {
                        "ok": True,
                        "choice": saved_row.get("choice"),
                        "choice_index": top1_index(probabilities),
                        "probabilities": probabilities,
                    }
                )
        results.append(result)
    return {
        "label": label or f"saved:{Path(path).name}",
        "path": str(path),
        "rows": results,
        "errors": sum(not row["ok"] for row in results),
    }


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    a_by_id = {row["id"]: row for row in a["rows"]}
    b_by_id = {row["id"]: row for row in b["rows"]}
    per_row: list[dict[str, Any]] = []
    per_kind: dict[str, dict[str, Any]] = {}
    for suite_row in rows:
        row_id = suite_row["id"]
        kind = suite_row["question"]["type"]
        left = a_by_id[row_id]
        right = b_by_id[row_id]
        entry: dict[str, Any] = {
            "id": row_id,
            "kind": kind,
            "truth": suite_row.get("truth"),
            "a": {
                "ok": left["ok"],
                "choice": left.get("choice"),
                "probabilities": left.get("probabilities"),
                "error": left.get("error"),
            },
            "b": {
                "ok": right["ok"],
                "choice": right.get("choice"),
                "probabilities": right.get("probabilities"),
                "error": right.get("error"),
            },
        }
        if left["ok"] and right["ok"] and len(left["probabilities"]) == len(right["probabilities"]):
            p = left["probabilities"]
            q = right["probabilities"]
            delta = max_abs_delta(p, q)
            mae = mean_abs_error(p, q)
            kl_ab = kl_divergence(p, q)
            kl_ba = kl_divergence(q, p)
            top1 = top1_index(p) == top1_index(q)
            entry.update(
                {
                    "comparable": True,
                    "delta_max": delta,
                    "mae": mae,
                    "kl_ab": kl_ab,
                    "kl_ba": kl_ba,
                    "top1_agree": top1,
                }
            )
            bucket = per_kind.setdefault(
                kind,
                {"compared": 0, "top1_agree": 0, "kl_ab": 0.0, "mae": 0.0, "delta_max": 0.0},
            )
            bucket["compared"] += 1
            bucket["top1_agree"] += int(top1)
            bucket["kl_ab"] += kl_ab
            bucket["mae"] += mae
            bucket["delta_max"] = max(bucket["delta_max"], delta)
        else:
            entry["comparable"] = False
        per_row.append(entry)

    comparable = [row for row in per_row if row["comparable"]]
    count = len(comparable)
    summary = {
        "compared": count,
        "skipped": len(per_row) - count,
        "errors_a": a["errors"],
        "errors_b": b["errors"],
        "top1_agreement": (
            sum(row["top1_agree"] for row in comparable) / count if count else None
        ),
        "mean_kl_ab": sum(row["kl_ab"] for row in comparable) / count if count else None,
        "mean_kl_ba": sum(row["kl_ba"] for row in comparable) / count if count else None,
        "mean_mae": sum(row["mae"] for row in comparable) / count if count else None,
        "max_delta": max((row["delta_max"] for row in comparable), default=None),
        "max_delta_id": (
            max(comparable, key=lambda row: row["delta_max"])["id"] if comparable else None
        ),
    }
    summary["mean_kl"] = (
        (summary["mean_kl_ab"] + summary["mean_kl_ba"]) / 2
        if summary["mean_kl_ab"] is not None
        else None
    )
    for bucket in per_kind.values():
        n = bucket["compared"]
        bucket["top1_agreement"] = bucket.pop("top1_agree") / n
        bucket["mean_kl_ab"] = bucket.pop("kl_ab") / n
        bucket["mean_mae"] = bucket.pop("mae") / n
    summary["per_kind"] = per_kind
    return {"rows": per_row, "summary": summary}


def truth_accuracy(
    backend: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in backend["rows"]}
    correct = 0
    total = 0
    per_kind: dict[str, dict[str, int]] = {}
    for suite_row in rows:
        result = by_id[suite_row["id"]]
        if not result["ok"] or "truth" not in suite_row:
            continue
        kind = suite_row["question"]["type"]
        truth = suite_row["truth"]
        probabilities = result["probabilities"]
        if kind == "choice":
            predicted: Any = str(result.get("choice"))
            target = str(truth)
            hit = predicted == target
        elif kind == "noul":
            predicted = str(result.get("choice")).lower()
            target = "true" if bool(truth) else "false"
            hit = predicted == target
        else:  # score: rounded expectation, as validate_awq.py does
            expected = sum(i * p for i, p in enumerate(probabilities))
            predicted = int(round(expected))
            target = int(truth)
            hit = predicted == target
        total += 1
        correct += int(hit)
        bucket = per_kind.setdefault(kind, {"correct": 0, "total": 0})
        bucket["correct"] += int(hit)
        bucket["total"] += 1
    overall = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_kind": per_kind,
    }
    return overall


def evaluate_thresholds(
    summary: Mapping[str, Any],
    *,
    min_top1: float,
    max_kl: float,
    max_mae: float,
    max_delta: float,
    max_errors: int,
) -> dict[str, Any]:
    checks = {
        "compared_positive": (summary["compared"] or 0) > 0,
        "top1_agreement": summary["top1_agreement"] is not None
        and summary["top1_agreement"] >= min_top1,
        "mean_kl": summary["mean_kl"] is not None and summary["mean_kl"] <= max_kl,
        "mean_mae": summary["mean_mae"] is not None and summary["mean_mae"] <= max_mae,
        "max_delta": summary["max_delta"] is not None and summary["max_delta"] <= max_delta,
        "errors": summary["errors_a"] + summary["errors_b"] <= max_errors,
    }
    return {
        "thresholds": {
            "min_top1": min_top1,
            "max_mean_kl": max_kl,
            "max_mean_mae": max_mae,
            "max_delta": max_delta,
            "max_errors": max_errors,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-url", required=True, help="backend A base URL")
    parser.add_argument("--b-url", help="backend B base URL; omit to use --baseline")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=None, help="optional model field for both live backends")
    parser.add_argument("--strategy", default="auto", choices=["auto", "single", "tournament", "permute"])
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-top1", type=float, default=0.96)
    parser.add_argument("--max-kl", "--max-mean-kl", dest="max_kl", type=float, default=0.01)
    parser.add_argument("--max-mae", type=float, default=0.005)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--no-strict", action="store_true", help="always exit 0")
    args = parser.parse_args(argv)

    try:
        rows = load_suite(args.suite)
        a = run_live(
            args.a_url,
            rows,
            label=f"live:{args.a_url}",
            model=args.model,
            strategy=args.strategy,
            timeout=args.timeout,
            retries=args.retries,
        )
        if args.b_url:
            b = run_live(
                args.b_url,
                rows,
                label=f"live:{args.b_url}",
                model=args.model,
                strategy=args.strategy,
                timeout=args.timeout,
                retries=args.retries,
            )
        else:
            b = load_baseline(args.baseline, rows)
        comparison = compare(a, b, rows)
        verdict = evaluate_thresholds(
            comparison["summary"],
            min_top1=args.min_top1,
            max_kl=args.max_kl,
            max_mae=args.max_mae,
            max_delta=args.max_delta,
            max_errors=args.max_errors,
        )
        report = {
            "tool": "jev-eval/suite",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suite": {
                "path": str(Path(args.suite).resolve()),
                "rows": len(rows),
                "source": (
                    "prompts from /opt/decider-stack/deploy/suite.jsonl "
                    "(copied to data/suite.jsonl); saved result file lacks inputs"
                ),
            },
            "backends": {"a": a, "b": b},
            "rows": comparison["rows"],
            "summary": comparison["summary"],
            "truth_accuracy": {"a": truth_accuracy(a, rows), "b": truth_accuracy(b, rows)},
            **verdict,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

        summary = comparison["summary"]
        print(f"A = {a['label']}  ({a['errors']} errors)")
        print(f"B = {b['label']}  ({b['errors']} errors)")
        print(
            f"compared {summary['compared']}/{len(rows)}  "
            f"top1={_fmt(summary['top1_agreement'], 4)}  "
            f"mean_KL={_fmt(summary['mean_kl'], 4)}  "
            f"mean_MAE={_fmt(summary['mean_mae'], 4)}  "
            f"max_delta={_fmt(summary['max_delta'], 4)} ({summary['max_delta_id']})"
        )
        print(f"thresholds: {'PASS' if verdict['pass'] else 'FAIL'}  -> {out}")
        return 0 if verdict["pass"] or args.no_strict else 1
    except (OSError, ValueError, KeyError, DecideError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
