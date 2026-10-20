#!/usr/bin/env python3
"""Verify Decidra response signing (HMAC-SHA256 over the exact body).

Run against a server started with DECIDRA_SIGNING_KEY=test-key. Checks that
/v1/decide carries `x-jev-signature: sha256=<hex>` matching the raw bytes and
that /health is not signed.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any

import requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--key", default="test-key")
    parser.add_argument("--out", default="out/signing_check.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    key = args.key.encode("utf-8")
    body = {
        "kind": "noul",
        "state": "Customer: the duplicate charge was refunded, thank you.",
        "question": "Does this message require a billing follow-up?",
    }
    result: dict[str, Any] = {}
    with requests.Session() as session:
        response = session.post(args.url + "/v1/decide", json=body, timeout=args.timeout)
        raw = response.content
        header = response.headers.get("x-jev-signature")
        expected = "sha256=" + hmac.new(key, raw, hashlib.sha256).hexdigest()
        result["decide"] = {
            "status": response.status_code,
            "header": header,
            "expected": expected,
            "body_bytes": len(raw),
            "signature_match": header == expected,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }
        health = session.get(args.url + "/health", timeout=args.timeout)
        result["health"] = {
            "status": health.status_code,
            "header": health.headers.get("x-jev-signature"),
            "signed": "x-jev-signature" in health.headers,
        }
        result["pass"] = bool(
            response.status_code == 200
            and result["decide"]["signature_match"]
            and health.status_code == 200
            and not result["health"]["signed"]
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
