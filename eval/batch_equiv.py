#!/usr/bin/env python3
"""Compare per-request decision-head logits from individual vs batched calls.

Inputs are two JSON files of per-request head logits: the first produced by
individual calls, the second by batched calls (for example the native worker's
``batch`` CLI).  The harness deliberately accepts the common container shapes
rather than requiring one exact wrapper:

* a list of records;
* ``{"requests": [...]}``, ``{"results": [...]}``, ``{"rows": [...]}`` or
  ``{"items": [...]}``;
* a mapping of request id -> record.

A record is either a bare list of numbers or an object with ``logits`` (or
``head_logits``/``head``) plus an optional ``id``/``request_id``.  When both
files carry explicit ids the records are aligned by id; otherwise file order is
used.  The report asserts equal request counts, per-request top-1 agreement,
and a bounded maximum absolute logit delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ID_KEYS = ("id", "request_id", "prompt_id", "name")
LOGIT_KEYS = ("logits", "head_logits", "head")
CONTAINER_KEYS = ("requests", "results", "rows", "items")


class EquivError(ValueError):
    """Malformed input or an equivalence assertion failure."""


def _vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise EquivError(f"logits must be a list of numbers, got {type(value).__name__}")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise EquivError(f"logits contain a non-numeric value: {error}") from error


def _normalize(data: Any) -> tuple[list[tuple[str, list[float]]], bool]:
    """Return ``[(request_id, logits)]`` and whether ids came from the file."""
    while isinstance(data, dict):
        for key in CONTAINER_KEYS:
            if key in data:
                data = data[key]
                break
        else:
            if any(key in data for key in ID_KEYS + LOGIT_KEYS):
                data = [data]
                break
            return _records(list(data.items()), explicit_ids=True)

    if isinstance(data, list):
        return _records(list(enumerate(data)), explicit_ids=False)
    raise EquivError(f"expected a JSON object or array, got {type(data).__name__}")


def _records(
    records: Sequence[tuple[Any, Any]], *, explicit_ids: bool
) -> tuple[list[tuple[str, list[float]]], bool]:
    normalized: list[tuple[str, list[float]]] = []
    all_ids_explicit = bool(records)
    for index, (key, record) in enumerate(records):
        if isinstance(record, (list, tuple)):
            request_id = str(key)
            vector = _vector(record)
            if not explicit_ids:
                all_ids_explicit = False
        elif isinstance(record, dict):
            request_id = None
            for id_key in ID_KEYS:
                if id_key in record:
                    request_id = str(record[id_key])
                    break
            else:
                request_id = str(key) if explicit_ids else str(index)
                if not explicit_ids:
                    all_ids_explicit = False
            raw = None
            for logit_key in LOGIT_KEYS:
                if logit_key in record:
                    raw = record[logit_key]
                    break
            if raw is None:
                raise EquivError(
                    f"record {request_id!r} has no logits field "
                    f"({', '.join(LOGIT_KEYS)})"
                )
            vector = _vector(raw)
        else:
            raise EquivError(
                f"record {key!r} must be a logits list or object, "
                f"got {type(record).__name__}"
            )
        normalized.append((request_id, vector))
    if not normalized:
        raise EquivError("no requests found")
    return normalized, all_ids_explicit


def load_logits(path: str | Path) -> tuple[list[tuple[str, list[float]]], bool]:
    try:
        data = json.loads(Path(path).read_text())
    except OSError as error:
        raise EquivError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise EquivError(f"invalid JSON in {path}: {error}") from error
    return _normalize(data)


def _top1(vector: Sequence[float]) -> int:
    if not vector:
        raise EquivError("empty logits vector")
    return max(range(len(vector)), key=vector.__getitem__)


def compare(
    individual: list[tuple[str, list[float]]],
    batched: list[tuple[str, list[float]]],
    *,
    align_ids: bool,
) -> dict[str, Any]:
    if len(individual) != len(batched):
        raise EquivError(
            f"request count differs: individual={len(individual)} batched={len(batched)}"
        )
    if align_ids:
        batched_by_id = {request_id: vector for request_id, vector in batched}
        if len(batched_by_id) != len(batched):
            raise EquivError("batched file contains duplicate request ids")
        missing = [request_id for request_id, _ in individual if request_id not in batched_by_id]
        if missing:
            raise EquivError(f"batched file is missing ids: {', '.join(missing[:5])}")
        pairs = [(request_id, vector, batched_by_id[request_id])
                 for request_id, vector in individual]
    else:
        pairs = [
            (request_id, vector, batched[index][1])
            for index, (request_id, vector) in enumerate(individual)
        ]

    max_delta = 0.0
    delta_total = 0.0
    delta_count = 0
    top1_matches = 0
    mismatches: list[str] = []
    worst: tuple[float, str] | None = None
    for request_id, left, right in pairs:
        if len(left) != len(right):
            raise EquivError(
                f"request {request_id!r}: logits length differs "
                f"({len(left)} vs {len(right)})"
            )
        if _top1(left) == _top1(right):
            top1_matches += 1
        else:
            mismatches.append(
                f"{request_id}: individual={_top1(left)} batched={_top1(right)}"
            )
        request_max = 0.0
        for a, b in zip(left, right):
            delta = abs(a - b)
            max_delta = max(max_delta, delta)
            request_max = max(request_max, delta)
            delta_total += delta
            delta_count += 1
        if worst is None or request_max > worst[0]:
            worst = (request_max, request_id)

    return {
        "requests": len(pairs),
        "top1_matches": top1_matches,
        "top1_mismatches": mismatches,
        "max_delta": max_delta,
        "mean_delta": delta_total / delta_count if delta_count else 0.0,
        "worst_request": worst[1] if worst else None,
        "worst_request_delta": worst[0] if worst else 0.0,
        "aligned_by": "id" if align_ids else "order",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("individual", help="JSON logits from individual calls")
    parser.add_argument("batched", help="JSON logits from batched calls")
    parser.add_argument(
        "--max-delta",
        type=float,
        default=1e-4,
        help="maximum allowed absolute logit delta (default: 1e-4)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    try:
        individual, individual_ids = load_logits(args.individual)
        batched, batched_ids = load_logits(args.batched)
        align_ids = bool(individual_ids and batched_ids)
        if align_ids:
            individual_set = {request_id for request_id, _ in individual}
            batched_set = {request_id for request_id, _ in batched}
            if len(individual_set) != len(individual):
                raise EquivError("individual file contains duplicate request ids")
            if individual_set != batched_set:
                missing = sorted(individual_set - batched_set)
                extra = sorted(batched_set - individual_set)
                raise EquivError(
                    "request ids differ"
                    + (f"; missing from batched: {', '.join(missing[:5])}" if missing else "")
                    + (f"; extra in batched: {', '.join(extra[:5])}" if extra else "")
                )
        report = compare(individual, batched, align_ids=align_ids)
        failures: list[str] = []
        if report["top1_matches"] != report["requests"]:
            failures.append(
                f"top-1 agreement {report['top1_matches']}/{report['requests']}"
            )
        if report["max_delta"] > args.max_delta:
            failures.append(
                f"max |delta| {report['max_delta']:.6g} > {args.max_delta:.6g}"
            )
        report["max_delta_limit"] = args.max_delta
        report["ok"] = not failures
        report["failures"] = failures
    except EquivError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(
            f"requests: {report['requests']} (aligned by {report['aligned_by']})\n"
            f"top-1:    {report['top1_matches']}/{report['requests']}\n"
            f"max |d|:  {report['max_delta']:.6g} (limit {args.max_delta:.6g})\n"
            f"mean |d|: {report['mean_delta']:.6g}\n"
            f"worst:    {report['worst_request']} ({report['worst_request_delta']:.6g})"
        )
        for mismatch in report["top1_mismatches"][:10]:
            print(f"  mismatch {mismatch}")

    if not report["ok"]:
        print("FAIL: " + "; ".join(report["failures"]), file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
