#!/usr/bin/env python3
"""Render a compact Markdown + JSON report from suite/bench outputs.

The source JSON files are the untouched outputs of ``suite.py`` and
``bench.py``; this tool only summarizes them, so it can be re-run as results
arrive.  Example:

    python3 report.py --suite out/suite.json --bench out/bench.json \
        --out-prefix out/report
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_PREFIX = ROOT / "out" / "report"


def _num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        magnitude = abs(value)
        if magnitude != 0.0 and magnitude < 10 ** (-digits):
            return f"{value:.{digits}e}"
        if 1e-4 <= magnitude < 1e6:
            text = f"{value:.{digits}f}"
            return text.rstrip("0").rstrip(".") if "." in text else text
        return f"{value:.{digits}g}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_suite(section: Mapping[str, Any], *, per_row: bool) -> list[str]:
    a = section["backends"]["a"]
    b = section["backends"]["b"]
    summary = section["summary"]
    checks = section["checks"]
    thresholds = section["thresholds"]
    a_label = a["label"].removeprefix("live:")
    b_label = b["label"].removeprefix("live:")
    lines = [
        "## Suite comparison",
        "",
        f"- A: `{a_label}`" + (f" (`{a['url']}`)" if a.get("url") else ""),
        f"- B: `{b_label}`"
        + (f" (`{b['url']}`)" if b.get("url") else ""),
        f"- Rows compared: **{summary['compared']}**"
        + (f" (skipped {summary['skipped']})" if summary["skipped"] else ""),
        f"- Request errors: A={summary['errors_a']}, B={summary['errors_b']}",
        "",
        "| Metric | Value | Threshold | Status |",
        "|---|---:|---:|:---:|",
        f"| Top-1 agreement | {_pct(summary['top1_agreement'])} | "
        f">= {_pct(thresholds['min_top1'], 0)} | {_status(checks['top1_agreement'])} |",
        f"| Mean KL (symmetric) | {_num(summary['mean_kl'])} | "
        f"<= {_num(thresholds['max_mean_kl'])} | {_status(checks['mean_kl'])} |",
        f"| Mean MAE | {_num(summary['mean_mae'])} | "
        f"<= {_num(thresholds['max_mean_mae'])} | {_status(checks['mean_mae'])} |",
        f"| Max abs delta | {_num(summary['max_delta'])}"
        + (f" (`{summary['max_delta_id']}`)" if summary.get("max_delta_id") else "")
        + f" | <= {_num(thresholds['max_delta'])} | {_status(checks['max_delta'])} |",
        f"| Request errors | {summary['errors_a'] + summary['errors_b']} | "
        f"<= {thresholds['max_errors']} | {_status(checks['errors'])} |",
        "",
        f"**Suite verdict: {_status(section['pass'])}**",
        "",
    ]

    per_kind = summary.get("per_kind") or {}
    if per_kind:
        lines += [
            "| Kind | Compared | Top-1 agree | Mean KL | Mean MAE | Max delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for kind in ("noul", "score", "choice"):
            bucket = per_kind.get(kind)
            if not bucket:
                continue
            lines.append(
                f"| {kind} | {bucket['compared']} | {_pct(bucket['top1_agreement'])} | "
                f"{_num(bucket['mean_kl_ab'])} | {_num(bucket['mean_mae'])} | "
                f"{_num(bucket['delta_max'])} |"
            )
        lines.append("")

    truth = section.get("truth_accuracy") or {}
    if truth:
        lines += [
            "| Backend | Truth correct | Accuracy | noul | score | choice |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ("a", "b"):
            result = truth.get(name)
            if not result:
                continue
            kinds = result.get("per_kind") or {}

            def ratio(kind: str) -> str:
                item = kinds.get(kind)
                if not item:
                    return "-"
                return f"{item['correct']}/{item['total']}"

            lines.append(
                f"| {name.upper()} | {result['correct']}/{result['total']} | "
                f"{_pct(result['accuracy'])} | {ratio('noul')} | {ratio('score')} | "
                f"{ratio('choice')} |"
            )
        lines.append("")

    if per_row and section.get("rows"):
        lines += [
            "| Row | Kind | A choice | B choice | Top-1 | Max delta | KL A→B | MAE |",
            "|---|---|---|---|:---:|---:|---:|---:|",
        ]
        for row in section["rows"]:
            if not row.get("comparable"):
                lines.append(
                    f"| {row['id']} | {row['kind']} | - | - | - | - | - | - |"
                )
                continue
            lines.append(
                f"| {row['id']} | {row['kind']} | {row['a'].get('choice')} | "
                f"{row['b'].get('choice')} | "
                f"{'yes' if row['top1_agree'] else 'no'} | "
                f"{_num(row['delta_max'], 4)} | {_num(row['kl_ab'], 4)} | "
                f"{_num(row['mae'], 4)} |"
            )
        lines.append("")
    return lines


def render_bench(section: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Benchmark",
        "",
        f"- URL: `{section['url']}`"
        + (f" (model `{section['model']}`)" if section.get("model") else ""),
        f"- Mode: `{section['mode']}`; concurrency: "
        f"{section['parameters']['concurrency']}; seed: {section['seed']}",
        "",
        "| Mode | Target tokens | Conc | Requests | OK | Err | req/s | p50 ms | p95 ms | p99 ms | Prompt tok/s | Tokens/req |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in section.get("levels", []):
        latency = level.get("latency_ms") or {}
        lines.append(
            f"| {level.get('mode', '-')} | "
            f"{level.get('target_tokens') or '-'} | {level['concurrency']} | "
            f"{level['requests']} | {level['ok']} | {level['errors']} | "
            f"{_num(level['req_per_s'], 3)} | {_num(latency.get('p50'), 1)} | "
            f"{_num(latency.get('p95'), 1)} | {_num(latency.get('p99'), 1)} | "
            f"{_num(level['prompt_tokens'].get('per_s'), 0)} | "
            f"{_num(level['prompt_tokens'].get('mean'), 0)} |"
        )
    lines.append("")

    warmups = section.get("warmups") or []
    if warmups:
        pieces = [
            f"{item.get('mode', '?')}: {item['requests']} req, "
            f"{item['errors']} errors, {item.get('latency_s', 0):.2f}s"
            for item in warmups
        ]
        lines += ["Warmup: " + "; ".join(pieces) + ".", ""]

    estimated = [
        level
        for level in section.get("levels", [])
        if (level.get("prompt_tokens") or {}).get("estimated_rows")
    ]
    if estimated:
        lines.append(
            "> Some prompt-token counts are estimates (`usage.prompt_tokens` was "
            "missing on those responses)."
        )
        lines.append("")
    errors = sum(level.get("errors", 0) for level in section.get("levels", []))
    lines += [f"**Benchmark errors: {errors}**", ""]
    return lines


def build_compact(
    title: str,
    suite: Mapping[str, Any] | None,
    bench: Mapping[str, Any] | None,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "tool": "jev-eval/report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
    }
    suite_pass: bool | None = None
    bench_pass: bool | None = None
    if suite is not None:
        suite_pass = bool(suite.get("pass"))
        compact["suite"] = {
            "suite": suite.get("suite"),
            "backends": {
                name: {
                    key: backend.get(key)
                    for key in ("label", "url", "path", "model", "strategy", "errors")
                    if key in backend
                }
                for name, backend in suite.get("backends", {}).items()
            },
            "summary": suite.get("summary"),
            "truth_accuracy": suite.get("truth_accuracy"),
            "thresholds": suite.get("thresholds"),
            "checks": suite.get("checks"),
            "pass": suite_pass,
        }
    if bench is not None:
        levels = bench.get("levels", [])
        bench_pass = all(level.get("errors", 0) == 0 for level in levels) if levels else False
        compact["bench"] = {
            "url": bench.get("url"),
            "model": bench.get("model"),
            "mode": bench.get("mode"),
            "seed": bench.get("seed"),
            "parameters": bench.get("parameters"),
            "warmups": bench.get("warmups"),
            "levels": [
                {
                    "mode": level.get("mode"),
                    "target_tokens": level.get("target_tokens"),
                    "concurrency": level.get("concurrency"),
                    "requests": level.get("requests"),
                    "ok": level.get("ok"),
                    "errors": level.get("errors"),
                    "error_rate": level.get("error_rate"),
                    "wall_s": level.get("wall_s"),
                    "req_per_s": level.get("req_per_s"),
                    "latency_ms": level.get("latency_ms"),
                    "prompt_tokens": level.get("prompt_tokens"),
                    "completion_tokens": level.get("completion_tokens"),
                }
                for level in levels
            ],
            "errors_total": sum(level.get("errors", 0) for level in levels),
            "pass": bench_pass,
        }
    compact["pass"] = bool(
        (suite_pass if suite_pass is not None else True)
        and (bench_pass if bench_pass is not None else True)
    )
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", help="suite.py output JSON")
    parser.add_argument("--bench", help="bench.py output JSON")
    parser.add_argument("--out-prefix", default=str(DEFAULT_PREFIX))
    parser.add_argument("--title", default="JEV-27B-VL decision backend evaluation")
    parser.add_argument(
        "--per-row", action="store_true", help="include the 28-row table in Markdown"
    )
    parser.add_argument(
        "--note", action="append", default=[], help="extra note line (repeatable)"
    )
    args = parser.parse_args(argv)
    if not args.suite and not args.bench:
        parser.error("at least one of --suite or --bench is required")

    try:
        suite = json.loads(Path(args.suite).read_text()) if args.suite else None
        bench = json.loads(Path(args.bench).read_text()) if args.bench else None
        if suite is not None and suite.get("tool") != "jev-eval/suite":
            raise ValueError(f"{args.suite} is not a jev-eval suite output")
        if bench is not None and bench.get("tool") != "jev-eval/bench":
            raise ValueError(f"{args.bench} is not a jev-eval bench output")

        compact = build_compact(args.title, suite, bench)
        lines = [f"# {args.title}", ""]
        lines += [
            f"Generated: {compact['generated_at']}",
            "",
        ]
        if suite is not None:
            lines += render_suite(suite, per_row=args.per_row)
        if bench is not None:
            lines += render_bench(bench)
        lines += [
            "## Notes",
            "",
            "- KL is symmetric mean of P||Q and Q||P; MAE and max abs delta are over",
            "  option probabilities on identical prompts.",
            "- Live backends of the same weights should agree almost exactly; batching",
            "  and quantization kernels can move logits slightly. See README.md.",
            "- The suite's 28 prompts come from the canonical suite JSONL, not from the",
            "  saved result file (which contains no inputs).",
            "- Truth accuracy is informational only. The saved v2 baseline scores",
            "  0/5 on the suite's `score` truths, so those targets do not line up with",
            "  the six-way score head; the pass/fail gate is backend equivalence.",
        ]
        for note in args.note:
            lines.append(f"- {note}")
        lines += [
            "",
            f"**Overall: {_status(compact['pass'])}**",
            "",
        ]

        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = prefix.with_suffix(".md")
        json_path = prefix.with_suffix(".json")
        markdown_path.write_text("\n".join(lines))
        json_path.write_text(json.dumps(compact, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {markdown_path}")
        print(f"wrote {json_path}")
        print(f"overall: {_status(compact['pass'])}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
