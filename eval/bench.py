#!/usr/bin/env python3
"""Concurrency and long-context benchmark for a JEV ``/v1/decide`` backend.

Short mode replays the 28-row decision suite (with a per-request nonce so no
two requests share a prompt prefix).  Long mode builds synthetic, *unique*
documents at approximately 3k and 9k tokens; uniqueness is deliberate so a
server-side prefix cache cannot turn the measurement into a cache test.

Every concurrency level gets freshly generated prompts (the tag includes the
level).  Requests are therefore unique across the whole run, at the cost of a
cold prefix cache for every level.  Use ``--reuse-short-prompts`` only when you
explicitly want to measure a warm/shared-prefix case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from client import (
    DecideClient,
    DecideError,
    build_decision_text,
    build_payload,
)
from suite import DEFAULT_SUITE, load_suite, row_request

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "out" / "bench.json"

_WORDS = (
    "amber basis candor delta ember fable gauge harbor ivory juniper kernel lumen "
    "meadow nectar orbit pepper quartz ripple sable tundra umber vapor willow xenon "
    "yonder zephyr alloy bitumen cinder drift elk fathom glacier hollow inlet jasper "
    "kestrel lattice marrow nimbus onyx prairie quill raven sparrow thistle upland "
    "vellum wattle yarrow zinnia"
).split()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile; deterministic for a given sample."""
    if not values:
        raise ValueError("percentile of empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def unique_document(nonce: str, target_tokens: int, chars_per_token: float, seed: int) -> str:
    """Deterministic ~target-token document, unique at its first byte."""
    target_chars = max(256, int(target_tokens * chars_per_token))
    digest = hashlib.sha256(f"{seed}:{nonce}".encode("utf-8")).digest()
    chunks = [f"Unique benchmark document {nonce}. "]
    size = len(chunks[0])
    counter = 0
    while size < target_chars:
        digest = hashlib.sha256(digest).digest()
        counter += 1
        picked = [
            _WORDS[digest[index] % len(_WORDS)]
            for index in range(0, 16, 2)
        ]
        sentence = f"Record {counter}: {' '.join(picked)}. "
        chunks.append(sentence)
        size += len(sentence)
    return "".join(chunks)[:target_chars]


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(math.ceil(len(text) / chars_per_token)))


def short_spec(
    row: Mapping[str, Any],
    *,
    tag: str,
    model: str | None,
    unique: bool,
    chars_per_token: float,
) -> dict[str, Any]:
    kind, state, question, options = row_request(row)
    if unique:
        state = f"[bench instance {tag}]\n{state}"
    body = build_payload(kind, state, question, options, model=model)
    text = build_decision_text(kind, state, question, options)
    return {
        "kind": kind,
        "state": state,
        "question": question,
        "options": options,
        "payload": body,
        "estimated_prompt_tokens": estimate_tokens(text, chars_per_token),
    }


def long_spec(
    *,
    tag: str,
    target_tokens: int,
    model: str | None,
    seed: int,
    chars_per_token: float,
) -> dict[str, Any]:
    document = unique_document(tag, target_tokens, chars_per_token, seed)
    kind = "noul"
    question = "Is this document about a billing incident?"
    state = document
    options = None
    body = build_payload(kind, state, question, options, model=model)
    text = build_decision_text(kind, state, question, options)
    return {
        "kind": kind,
        "state": state,
        "question": question,
        "options": options,
        "payload": body,
        "estimated_prompt_tokens": estimate_tokens(text, chars_per_token),
        "target_tokens": target_tokens,
    }


def run_warmup(
    url: str,
    specs: Sequence[Mapping[str, Any]],
    *,
    count: int,
    model: str | None,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    if count <= 0 or not specs:
        return {"requests": 0, "errors": 0, "error_rate": 0.0, "latency_s": 0.0}
    client = DecideClient(url, model=model, timeout=timeout, retries=retries)
    errors = 0
    started = time.perf_counter()
    for index in range(count):
        spec = specs[index % len(specs)]
        try:
            client.decide(
                spec["kind"],
                spec["state"],
                spec["question"],
                spec["options"],
                model=model,
            )
        except (DecideError, ValueError):
            errors += 1
    return {
        "requests": count,
        "errors": errors,
        "error_rate": errors / count,
        "latency_s": time.perf_counter() - started,
    }


def run_level(
    url: str,
    specs: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    model: str | None,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    local = threading.local()

    def init_local() -> None:
        local.client = DecideClient(
            url, model=model, timeout=timeout, retries=retries
        )

    def client() -> DecideClient:
        instance = getattr(local, "client", None)
        if instance is None:
            init_local()
            instance = local.client
        return instance

    def one(spec: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            answer = client().decide(
                spec["kind"],
                spec["state"],
                spec["question"],
                spec["options"],
                model=model,
            )
            elapsed = time.perf_counter() - started
            usage = answer.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            return {
                "ok": True,
                "latency_s": elapsed,
                "prompt_tokens": int(prompt_tokens)
                if prompt_tokens is not None
                else int(spec["estimated_prompt_tokens"]),
                "prompt_tokens_estimated": prompt_tokens is None,
                "completion_tokens": int(completion_tokens or 0),
            }
        except (DecideError, ValueError, KeyError) as error:
            return {
                "ok": False,
                "latency_s": time.perf_counter() - started,
                "error": str(error)[:500],
            }

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=concurrency, initializer=init_local
    ) as executor:
        futures = [executor.submit(one, spec) for spec in specs]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_started

    successes = [result for result in results if result["ok"]]
    errors = [result for result in results if not result["ok"]]
    latencies = [result["latency_s"] * 1000.0 for result in successes]
    prompt_tokens = sum(result["prompt_tokens"] for result in successes)
    completion_tokens = sum(result["completion_tokens"] for result in successes)
    estimated = sum(result["prompt_tokens_estimated"] for result in successes)
    return {
        "concurrency": concurrency,
        "requests": len(specs),
        "ok": len(successes),
        "errors": len(errors),
        "error_rate": len(errors) / len(specs) if specs else None,
        "wall_s": wall_s,
        "req_per_s": len(successes) / wall_s if wall_s > 0 else None,
        "attempted_per_s": len(specs) / wall_s if wall_s > 0 else None,
        "latency_ms": {
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "p99": percentile(latencies, 0.99) if latencies else None,
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "prompt_tokens": {
            "total": prompt_tokens,
            "mean": prompt_tokens / len(successes) if successes else None,
            "per_s": prompt_tokens / wall_s if wall_s > 0 else None,
            "estimated_rows": estimated,
        },
        "completion_tokens": {
            "total": completion_tokens,
            "per_s": completion_tokens / wall_s if wall_s > 0 else None,
        },
        "error_samples": [result.get("error") for result in errors[:5]],
    }


def parse_ints(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("JEV_DECIDE_URL", "http://127.0.0.1:8501"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["short", "long", "both"], default="short")
    parser.add_argument("--concurrency", default="1,4,8,12,16,32")
    parser.add_argument("--requests", type=int, default=64, help="requests per short level")
    parser.add_argument("--long-requests", type=int, default=12, help="requests per long level")
    parser.add_argument("--long-sizes", default="3000,9000", help="target prompt tokens")
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument(
        "--reuse-short-prompts",
        action="store_true",
        help="disable per-request nonces (measures a shared prefix cache; not the default)",
    )
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    try:
        levels = parse_ints(args.concurrency)
        if any(level <= 0 for level in levels):
            raise ValueError("concurrency values must be positive")
        long_sizes = parse_ints(args.long_sizes)
        if any(size <= 0 for size in long_sizes):
            raise ValueError("long sizes must be positive")
        if args.requests <= 0 or args.long_requests <= 0:
            raise ValueError("request counts must be positive")
        rows = load_suite(args.suite)
        health = None
        if not args.skip_health:
            health = DecideClient(
                args.url, model=args.model, timeout=args.timeout, retries=args.retries
            ).health()

        report: dict[str, Any] = {
            "tool": "jev-eval/bench",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "model": args.model,
            "mode": args.mode,
            "seed": args.seed,
            "parameters": {
                "concurrency": levels,
                "requests_per_short_level": args.requests,
                "requests_per_long_level": args.long_requests,
                "long_sizes": long_sizes,
                "chars_per_token": args.chars_per_token,
                "warmup": args.warmup,
                "unique_short_prompts": not args.reuse_short_prompts,
                "health": health,
            },
            "warmups": [],
            "levels": [],
        }

        if args.mode in ("short", "both"):
            warmup_specs = [
                short_spec(
                    rows[index % len(rows)],
                    tag=f"warmup-short-{index}-{args.seed}",
                    model=args.model,
                    unique=not args.reuse_short_prompts,
                    chars_per_token=args.chars_per_token,
                )
                for index in range(max(1, min(args.requests, len(rows))))
            ]
            warmup = run_warmup(
                args.url,
                warmup_specs,
                count=args.warmup,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
            )
            warmup.update({"mode": "short"})
            report["warmups"].append(warmup)
            for concurrency in levels:
                specs = [
                    short_spec(
                        rows[index % len(rows)],
                        tag=f"short-{args.seed}-{concurrency}-{index}",
                        model=args.model,
                        unique=not args.reuse_short_prompts,
                        chars_per_token=args.chars_per_token,
                    )
                    for index in range(args.requests)
                ]
                result = run_level(
                    args.url,
                    specs,
                    concurrency=concurrency,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                result.update({"mode": "short", "target_tokens": None})
                report["levels"].append(result)
                print(
                    f"short c={concurrency:<3} ok={result['ok']}/{result['requests']} "
                    f"req/s={_fmt(result['req_per_s'])} p50={_fmt(result['latency_ms']['p50'], 1)}ms "
                    f"p95={_fmt(result['latency_ms']['p95'], 1)}ms "
                    f"p99={_fmt(result['latency_ms']['p99'], 1)}ms "
                    f"tok/s={_fmt(result['prompt_tokens']['per_s'], 0)}",
                    flush=True,
                )

        if args.mode in ("long", "both"):
            warmup_specs = []
            for index in range(max(1, min(args.long_requests, 4))):
                size = long_sizes[index % len(long_sizes)]
                warmup_specs.append(
                    long_spec(
                        tag=f"warmup-long-{args.seed}-{index}",
                        target_tokens=size,
                        model=args.model,
                        seed=args.seed,
                        chars_per_token=args.chars_per_token,
                    )
                )
            warmup = run_warmup(
                args.url,
                warmup_specs,
                count=args.warmup,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
            )
            warmup.update({"mode": "long"})
            report["warmups"].append(warmup)
            for target in long_sizes:
                for concurrency in levels:
                    specs = [
                        long_spec(
                            tag=f"long-{args.seed}-{target}-{concurrency}-{index}",
                            target_tokens=target,
                            model=args.model,
                            seed=args.seed,
                            chars_per_token=args.chars_per_token,
                        )
                        for index in range(args.long_requests)
                    ]
                    result = run_level(
                        args.url,
                        specs,
                        concurrency=concurrency,
                        model=args.model,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    result.update(
                        {
                            "mode": "long",
                            "target_tokens": target,
                            "estimated_prompt_tokens_mean": (
                                sum(spec["estimated_prompt_tokens"] for spec in specs)
                                / len(specs)
                            ),
                        }
                    )
                    report["levels"].append(result)
                    print(
                        f"long  ~{target} c={concurrency:<3} ok={result['ok']}/{result['requests']} "
                        f"req/s={_fmt(result['req_per_s'])} "
                        f"p50={_fmt(result['latency_ms']['p50'], 1)}ms "
                        f"p95={_fmt(result['latency_ms']['p95'], 1)}ms "
                        f"p99={_fmt(result['latency_ms']['p99'], 1)}ms "
                        f"tok/s={_fmt(result['prompt_tokens']['per_s'], 0)} "
                        f"tokens={_fmt(result['prompt_tokens']['mean'], 0)}",
                        flush=True,
                    )

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out}")
        return 0
    except (OSError, ValueError, KeyError, DecideError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
