#!/usr/bin/env python3
"""Functional live checks for the Rust decision server.

1. 256-option choice through /v1/decide (default wide-choice strategy).
2. OpenAI-style /v1/decisions with predicate + choice + score questions.
3. Image state must be rejected with 400 (native vision is not implemented).

Writes a JSON record and the PNG used. Does not benchmark.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def post(session: requests.Session, url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    started = time.perf_counter()
    response = session.post(url, json=body, timeout=timeout)
    elapsed = time.perf_counter() - started
    try:
        payload = response.json()
    except ValueError:
        payload = {"_text": response.text[:1000]}
    return response.status_code, payload, elapsed


def check_256(url: str, timeout: float) -> dict[str, Any]:
    options = [f"option-{index:03d}" for index in range(256)]
    body = {
        "kind": "choice",
        "state": "A synthetic routing request that must pick exactly one of the listed labels.",
        "question": "Which option has the highest decision probability?",
        "options": options,
    }
    with requests.Session() as session:
        status, payload, elapsed = post(session, url + "/v1/decide", body, timeout)
    probabilities = payload.get("probabilities") if isinstance(payload, dict) else None
    ok = (
        status == 200
        and isinstance(probabilities, list)
        and len(probabilities) == 256
        and abs(sum(probabilities) - 1.0) <= 1e-6
        and payload.get("choice") in options
        and payload.get("choice_index") in range(256)
        and (payload.get("usage") or {}).get("prompt_tokens", 0) > 0
    )
    return {
        "name": "choice_256",
        "ok": bool(ok),
        "status": status,
        "elapsed_s": elapsed,
        "probability_count": len(probabilities) if isinstance(probabilities, list) else None,
        "probability_sum": sum(probabilities) if isinstance(probabilities, list) else None,
        "choice": payload.get("choice") if isinstance(payload, dict) else None,
        "choice_index": payload.get("choice_index") if isinstance(payload, dict) else None,
        "adaptation": payload.get("adaptation") if isinstance(payload, dict) else None,
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
        "num_model_requests": payload.get("num_model_requests") if isinstance(payload, dict) else None,
        "error": None if ok else payload,
    }


def check_decisions(url: str, timeout: float) -> dict[str, Any]:
    body = {
        "model": "jev-awq",
        "input": "Customer: my card was charged twice for one coffee. Please refund the duplicate charge.",
        "questions": [
            {
                "type": "predicate",
                "name": "followup",
                "instructions": "Does this message require a billing follow-up?",
            },
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team should handle this request?",
                "choices": [
                    {"value": "billing", "description": "Charges, refunds, invoices"},
                    {"value": "support", "description": "API errors, integrations, bugs"},
                    {"value": "sales", "description": "Pricing, quotes, new purchases"},
                ],
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent is this request?",
                "levels": [
                    {"label": "Routine"},
                    {"label": "Elevated"},
                    {"label": "Urgent"},
                    {"label": "Emergency"},
                ],
            },
        ],
    }
    with requests.Session() as session:
        status, payload, elapsed = post(session, url + "/v1/decisions", body, timeout)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    ok = status == 200 and isinstance(answers, list) and len(answers) == 3
    if ok:
        kinds = [answer.get("type") for answer in answers]
        predicate = answers[0]
        choice = answers[1]
        score = answers[2]
        ok = (
            kinds == ["predicate", "choice", "score"]
            and 0.0 <= float(predicate.get("probability", -1)) <= 1.0
            and choice.get("choice") in ("billing", "support", "sales")
            and abs(sum(item["probability"] for item in choice.get("probabilities", [])) - 1.0) <= 1e-6
            and abs(sum(item["probability"] for item in score.get("probabilities", [])) - 1.0) <= 1e-6
            and (payload.get("usage") or {}).get("input_tokens", 0) > 0
        )
    return {
        "name": "decisions",
        "ok": bool(ok),
        "status": status,
        "elapsed_s": elapsed,
        "answers": answers,
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
        "error": None if ok else payload,
    }


def check_image(url: str, png_path: Path, timeout: float) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        return {"name": "image", "ok": False, "skipped": True, "reason": f"PIL unavailable: {error}"}

    image = Image.new("RGB", (512, 512), (220, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((128, 128, 384, 384), fill=(30, 60, 220))
    draw.rectangle((208, 208, 304, 304), fill=(245, 245, 245))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(raw)
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    body = {
        "kind": "choice",
        "state": [
            {"type": "text", "text": "An image with a large blue square inside a red background. "},
            {"image": data_url},
        ],
        "question": "Which color covers the largest area in this image?",
        "options": ["red", "blue", "green"],
    }
    with requests.Session() as session:
        status, payload, elapsed = post(session, url + "/v1/decide", body, timeout)
    message = ""
    if isinstance(payload, dict):
        message = str(
            (payload.get("error") or {}).get("message")
            or payload.get("detail")
            or ""
        )
    ok = status == 400 and "native vision not implemented" in message
    return {
        "name": "image_rejected",
        "ok": bool(ok),
        "status": status,
        "elapsed_s": elapsed,
        "png": str(png_path),
        "png_bytes": len(raw),
        "png_size": list(image.size),
        "message": message,
        "error": None if ok else payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", default="out/functional_checks.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--skip-image", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checks = [check_256(args.url, args.timeout), check_decisions(args.url, args.timeout)]
    if not args.skip_image:
        checks.append(check_image(args.url, out.with_name("check-image.png"), args.timeout))
    else:
        checks.append({"name": "image", "ok": False, "skipped": True, "reason": "--skip-image"})
    result = {"url": args.url, "checks": checks, "pass": all(c["ok"] for c in checks)}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    for check in checks:
        if check.get("skipped"):
            print(f"{check['name']}: SKIPPED ({check.get('reason')})")
        else:
            print(
                f"{check['name']}: {'PASS' if check['ok'] else 'FAIL'} "
                f"status={check['status']} elapsed={check['elapsed_s']:.2f}s "
                f"choice={check.get('choice')} message={check.get('message')} "
            )
    print(f"wrote {out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
