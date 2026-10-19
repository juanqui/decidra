#!/usr/bin/env python3
"""Tiny deterministic mock of the JEV decision API (no GPU, stdlib only).

It implements the same contract as ``decide_shim.py``:

* ``GET /health``
* ``GET /v1/models``
* ``GET /v1/decide/info``
* ``POST /v1/decide`` for ``noul``, ``score``, and ``choice`` (2..256 options)

Probabilities come from the *real* head bias/temperature math in ``client.py``
applied to synthetic next-token log probabilities derived from a SHA-256 of the
request.  That makes the mock useful for exercising the client, suite, bench,
and report end-to-end, but it is not a model and its numbers are meaningless.

Run:
    python3 mock_server.py --port 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from client import (
    DATA_DIR,
    HeadCalibration,
    build_decision_text,
    canonical_options,
    top1_index,
)

MAX_BODY = 2 * 1024 * 1024


def synthetic_logprobs(token_ids: Sequence[int], material: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for index, token_id in enumerate(token_ids):
        digest = hashlib.sha256(f"{material}|{token_id}".encode("utf-8")).digest()
        uniform = int.from_bytes(digest[:8], "big") / float(1 << 64)
        result[int(token_id)] = -3.0 * uniform - 0.02 * index
    return result


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "jev-mock/1.0"

    # ---------------------------------------------------------------- helpers
    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"detail": message})

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_BODY:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ready", "model": self.server.model, "head": "mock"})
        elif self.path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "jev-eval",
                        }
                    ],
                },
            )
        elif self.path == "/v1/decide/info":
            head: HeadCalibration = self.server.head
            self._send(
                200,
                {
                    "protocol": "jev27-bare-v1",
                    "model": self.server.model,
                    "max_options": 256,
                    "native_choice_options": 16,
                    "temperatures": head.temperatures,
                    "defaults": {"strategy": "tournament"},
                    "mock": True,
                },
            )
        else:
            self._error(404, f"unknown path {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/decide":
            self._error(404, f"unknown path {self.path}")
            return
        try:
            body = self._read_json()
            result = self._decide(body)
        except (ValueError, KeyError, TypeError) as error:
            self._error(400, str(error))
            return
        delay = float(getattr(self.server, "delay_ms", 0.0)) / 1000.0
        if delay > 0:
            time.sleep(delay)
        self._send(200, result)

    # ------------------------------------------------------------------- logic
    def _decide(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        kind = body.get("kind")
        if kind not in ("noul", "score", "choice"):
            raise ValueError("kind must be noul|score|choice")
        question = body.get("question")
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        state = body.get("state", "")
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be a string, object, or list")

        if kind == "choice":
            options = canonical_options(kind, body.get("options"))
            label_ids = [
                (
                    self.server.head.verbalizer_ids[8 + index]
                    if index < 16
                    else 200000 + index
                )
                for index in range(len(options))
            ]
        else:
            options = canonical_options(kind)
            label_ids = None

        text = build_decision_text(kind, state, question, options)
        material = json.dumps(
            {"kind": kind, "state": state, "question": question, "options": options},
            ensure_ascii=False,
            sort_keys=True,
        )
        head: HeadCalibration = self.server.head
        if label_ids is None:
            token_ids = head.verbalizer_ids[
                head.ranges[kind][0] : head.ranges[kind][1]
            ]
            logprobs = synthetic_logprobs(token_ids, material)
            probabilities = head.probabilities(kind, logprobs)
        else:
            logprobs = synthetic_logprobs(label_ids, material)
            probabilities = head.probabilities(
                kind, logprobs, len(options), label_token_ids=label_ids
            )

        best = top1_index(probabilities)
        prompt_tokens = max(1, int(math.ceil(len(text) / 4.0)))
        return {
            "kind": kind,
            "effective_kind": kind,
            "options": options,
            "probabilities": probabilities,
            "choice_index": best,
            "choice": options[best],
            "adaptation": "native"
            if kind != "choice" or len(options) <= 16
            else body.get("strategy", "auto"),
            "protocol": "jev27-bare-v1",
            "model": self.server.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
            "num_model_requests": 1,
            "mock": True,
        }


def make_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    model: str = "jev-eval-mock",
    delay_ms: float = 0.0,
    verbose: bool = False,
    head_dir: str | None = None,
) -> ThreadingHTTPServer:
    from pathlib import Path

    base = Path(head_dir) if head_dir else DATA_DIR
    head = HeadCalibration.from_paths(
        base / "decision_head.json", base / "calibration.json"
    )
    server = ThreadingHTTPServer((host, port), MockHandler)
    server.daemon_threads = True
    server.model = model  # type: ignore[attr-defined]
    server.head = head  # type: ignore[attr-defined]
    server.delay_ms = delay_ms  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic decision mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="jev-eval-mock")
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--head-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    server = make_server(
        args.host,
        args.port,
        model=args.model,
        delay_ms=args.delay_ms,
        verbose=args.verbose,
        head_dir=args.head_dir,
    )
    host, port = server.server_address[:2]
    print(f"MOCK_URL=http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
