#!/usr/bin/env python3
"""Persistent-connection saturation benchmark for the Decidra ``/v1/decide`` API.

Unlike ``bench.py``, every worker opens its HTTP connection before the timed
window and all workers release together from a barrier. That removes client
connect/setup skew from the measurement, which matters on slower hosts and at
high concurrency. Server-side batch metrics are sampled before and after each
level and reported as a delta.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from bench import (
    batch_metrics_snapshot,
    batch_spec,
    percentile,
    scrape_metrics,
    select_batch_rows,
)
from client import DecideClient
from suite import DEFAULT_SUITE, load_suite

ROOT = Path(__file__).resolve().parent


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def metric_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    if not before or not after:
        return None
    b = before.get("batch_size") or {}
    a = after.get("batch_size") or {}
    count = float(a.get("count", 0)) - float(b.get("count", 0))
    total = float(a.get("sum", 0)) - float(b.get("sum", 0))
    return {
        "count": count,
        "mean": (total / count) if count > 0 else None,
        "queue_depth_max": after.get("queue_depth"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--requests", type=int, default=128, help="total timed requests per level")
    parser.add_argument("--tokens", type=int, default=69)
    parser.add_argument("--chars-per-token", type=float, default=3.38)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=str(ROOT / "out" / "saturate.json"))
    args = parser.parse_args(argv)

    rows = load_suite(args.suite)
    pool = select_batch_rows(rows, target_tokens=args.tokens, chars_per_token=args.chars_per_token) or list(rows)
    run_id = uuid.uuid4().hex[:8]
    report: dict[str, Any] = {
        "url": args.url,
        "requests_per_level": args.requests,
        "seed": args.seed,
        "levels": [],
    }

    for concurrency in parse_ints(args.concurrency):
        per_worker = max(1, args.requests // concurrency)
        total = per_worker * concurrency
        clients = [DecideClient(args.url, timeout=args.timeout) for _ in range(concurrency)]
        for client in clients:
            client.health()
        # One untimed decision warms the engine; health() above already opened
        # every worker's HTTP connection.
        warm = clients[0]
        warm_spec = batch_spec(
            pool[0],
            tag=f"sat-warm-{run_id}-{concurrency}",
            target_tokens=args.tokens,
            model=None,
            seed=args.seed,
            chars_per_token=args.chars_per_token,
        )
        for _ in range(2):
            warm.decide(warm_spec["kind"], warm_spec["state"], warm_spec["question"], warm_spec["options"])

        before = batch_metrics_snapshot(scrape_metrics(args.url))
        barrier = threading.Barrier(concurrency + 1)
        latencies: list[float] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            client = clients[index]
            barrier.wait()
            for j in range(per_worker):
                spec = batch_spec(
                    pool[(index + j) % len(pool)],
                    tag=f"sat-{run_id}-{concurrency}-{index}-{j}",
                    target_tokens=args.tokens,
                    model=None,
                    seed=args.seed,
                    chars_per_token=args.chars_per_token,
                )
                started = time.perf_counter()
                try:
                    client.decide(spec["kind"], spec["state"], spec["question"], spec["options"])
                    elapsed = time.perf_counter() - started
                    with lock:
                        latencies.append(elapsed)
                except Exception as error:  # noqa: BLE001 - report, do not abort the level
                    with lock:
                        errors.append(str(error)[:200])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
        wall_started = time.perf_counter()
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - wall_started
        after = batch_metrics_snapshot(scrape_metrics(args.url))
        ms = [value * 1000.0 for value in latencies]
        level = {
            "concurrency": concurrency,
            "requests": total,
            "ok": len(latencies),
            "errors": len(errors),
            "wall_s": wall,
            "req_per_s": len(latencies) / wall if wall > 0 else None,
            "latency_ms": {
                "p50": percentile(ms, 0.50) if ms else None,
                "p95": percentile(ms, 0.95) if ms else None,
                "p99": percentile(ms, 0.99) if ms else None,
                "mean": statistics.mean(ms) if ms else None,
            },
            "server_batch": metric_delta(before, after),
            "error_samples": errors[:3],
        }
        report["levels"].append(level)
        print(
            f"c={concurrency:<3} ok={level['ok']}/{total} rps={level['req_per_s']:.2f} "
            f"p50={level['latency_ms']['p50']:.1f}ms p95={level['latency_ms']['p95']:.1f}ms "
            f"batch_mean={(level['server_batch'] or {}).get('mean')}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
