#!/usr/bin/env python3
"""Benchmark a d3-turbo `/v1/systemone` server (Decidra or vLLM-SR engine).

Sends the same typed decision request to a URL, with optional question
subsets, warmups, and concurrent copies. Latency is wall time per request;
throughput is completed requests and questions per second. The server's
`usage.input_tokens` is reported when present.

Example:
    python3 eval/d3_bench.py --url http://127.0.0.1:8899 \
        --request eval/data/d3_ticket_2k_64q.json --questions 1,8,64 \
        --warmup 2 --runs 10 --concurrency 1,8
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent


def post(url: str, body: bytes, timeout: float) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return (time.perf_counter() - started) * 1000.0, payload


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def payload_for(
    request: dict[str, Any],
    question_count: int | None,
    state_repeat: int = 1,
    nonce: str | None = None,
) -> tuple[bytes, int]:
    body = dict(request)
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SystemExit("request JSON needs a non-empty questions object")
    if question_count is not None:
        if question_count < 1 or question_count > len(questions):
            raise SystemExit(f"--questions {question_count} outside 1..{len(questions)}")
        questions = dict(list(questions.items())[:question_count])
        body["questions"] = questions
    if state_repeat > 1:
        state = body.get("state")
        if not isinstance(state, str):
            raise SystemExit("--state-repeat needs a string state")
        body["state"] = "\n\n".join([state] * state_repeat)
    if nonce is not None:
        state = body.get("state")
        if not isinstance(state, str):
            raise SystemExit("--cold needs a string state")
        body["state"] = f"{state}\n\n[bench nonce {nonce}]"
    return json.dumps(body, separators=(",", ":")).encode(), len(questions)


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": percentile(values, 0.50),
        "p80_ms": percentile(values, 0.80),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.mean(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8899")
    parser.add_argument("--request", default=str(ROOT / "data" / "d3_ticket_2k_64q.json"))
    parser.add_argument("--questions", default="1,8,64", help="question subsets to test")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--concurrency", default="1,8", help="concurrent copies")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cold", action="store_true", help="change the state each request (cache miss)")
    parser.add_argument("--state-repeat", type=int, default=1, help="repeat the state text N times")
    parser.add_argument("--out", default=str(ROOT / "out" / "d3_bench.json"))
    args = parser.parse_args(argv)

    request = json.loads(Path(args.request).read_text())
    counts = [int(part) for part in args.questions.split(",") if part.strip()]
    levels = [int(part) for part in args.concurrency.split(",") if part.strip()]
    report: dict[str, Any] = {
        "url": args.url,
        "request": str(Path(args.request).resolve()),
        "runs": args.runs,
        "levels": [],
    }

    for count in counts:
        body, questions = payload_for(request, count, args.state_repeat)
        for warm in range(max(0, args.warmup)):
            warm_body = (
                payload_for(request, count, args.state_repeat, f"warm-{count}-{warm}")[0]
                if args.cold
                else body
            )
            post(args.url, warm_body, args.timeout)
        times: list[float] = []
        tokens = 0
        for run in range(args.runs):
            request_body = (
                payload_for(request, count, args.state_repeat, f"run-{count}-{run}")[0]
                if args.cold
                else body
            )
            ms, payload = post(args.url, request_body, args.timeout)
            times.append(ms)
            tokens += int((payload.get("usage") or {}).get("input_tokens") or 0)
        latency = summarize(times)
        level = {
            "questions": questions,
            "concurrency": 1,
            "runs": args.runs,
            "ok": len(times),
            "latency": latency,
            "req_per_s": 1000.0 / latency["mean_ms"] if latency["mean_ms"] else None,
            "questions_per_s": (
                1000.0 * questions / latency["mean_ms"] if latency["mean_ms"] else None
            ),
            "input_tokens": tokens,
            "input_tokens_per_s": (
                tokens / (sum(times) / 1000.0) if times else None
            ),
        }
        report["levels"].append(level)
        print(
            f"{questions:3d}q c=1  p50={latency['p50_ms']:.1f}ms "
            f"p95={latency['p95_ms']:.1f}ms req/s={level['req_per_s']:.2f} "
            f"q/s={level['questions_per_s']:.0f}",
            flush=True,
        )

        if len(levels) > 1 or levels[0] != 1:
            for concurrency in levels:
                if concurrency == 1:
                    continue
                latencies: list[float] = []
                errors: list[str] = []
                lock = threading.Lock()
                barrier = threading.Barrier(concurrency + 1)

                def worker() -> None:
                    barrier.wait()
                    try:
                        request_body = (
                            payload_for(
                                request,
                                count,
                                args.state_repeat,
                                f"conc-{count}-{concurrency}-{uuid.uuid4().hex}",
                            )[0]
                            if args.cold
                            else body
                        )
                        ms, _ = post(args.url, request_body, args.timeout)
                    except Exception as error:  # noqa: BLE001 - report the level
                        with lock:
                            errors.append(str(error)[:200])
                        return
                    with lock:
                        latencies.append(ms)

                threads = [threading.Thread(target=worker) for _ in range(concurrency)]
                for thread in threads:
                    thread.start()
                wall_started = time.perf_counter()
                barrier.wait()
                for thread in threads:
                    thread.join()
                wall = time.perf_counter() - wall_started
                level = {
                    "questions": questions,
                    "concurrency": concurrency,
                    "ok": len(latencies),
                    "errors": len(errors),
                    "error_samples": errors[:3],
                    "latency": summarize(latencies),
                    "wall_s": wall,
                    "req_per_s": len(latencies) / wall if wall else None,
                    "questions_per_s": (len(latencies) * questions / wall) if wall else None,
                }
                report["levels"].append(level)
                print(
                    f"{questions:3d}q c={concurrency:<2d} "
                    f"p50={level['latency']['p50_ms'] or 0:.1f}ms "
                    f"req/s={level['req_per_s'] or 0:.2f} "
                    f"q/s={level['questions_per_s'] or 0:.0f} "
                    f"errors={len(errors)}",
                    flush=True,
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
