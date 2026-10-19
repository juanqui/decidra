#!/usr/bin/env python3
"""Exact client for the JEV-27B-VL ``/v1/decide`` protocol.

The protocol is the one implemented by ``/srv/models/jev-awq/decide_shim.py``
and mirrored by the Rust server:

* request:  ``POST /v1/decide`` with
  ``{kind, state, question, options?, strategy?, model?}``
* response: ``{kind, effective_kind, options, probabilities, choice_index, choice,
  adaptation, protocol, model, usage, num_model_requests}``

``noul`` is always false/true, ``score`` is always 0..5, and ``choice`` accepts
2..256 options.  Choice options above the first 16 letter labels are read from
the label token itself (head bias 0), exactly as the Python front does.

This module also contains the *offline* head math: given next-token log
probabilities for the verbalizer token ids, it reproduces the server's
``probabilities`` (head bias + per-kind temperature + optional fitted
calibration).  ``mock_server.py`` and the smoke test use that path; normal
callers only need ``DecideClient``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MAX_OPTIONS = 256
LABELS: list[str] = list(string.ascii_uppercase) + [
    "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
]
DATA_DIR = Path(__file__).resolve().parent / "data"
NOUL_OPTIONS = ["false", "true"]
SCORE_OPTIONS = [str(i) for i in range(6)]


class DecideError(RuntimeError):
    """Transport, HTTP, or protocol-contract failure."""


def choice_labels(count: int) -> list[str]:
    """A..Z, AA..IV: the exact label order used by the Python front."""
    if not 1 <= count <= MAX_OPTIONS:
        raise ValueError(f"choice label count must be 1..{MAX_OPTIONS}, got {count}")
    return LABELS[:count]


def canonical_options(kind: str, options: Sequence[str] | None = None) -> list[str]:
    if kind == "noul":
        return list(NOUL_OPTIONS)
    if kind == "score":
        return list(SCORE_OPTIONS)
    if kind == "choice":
        opts = list(options or [])
        if not 2 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_OPTIONS} options, got {len(opts)}")
        return opts
    raise ValueError(f"kind must be noul|score|choice, got {kind!r}")


def _state_text(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False)
    if isinstance(state, list):
        parts: list[str] = []
        for part in state:
            if isinstance(part, str):
                parts.append(part)
            else:
                raise ValueError("image/list states are outside this text-only harness")
        return "".join(parts)
    raise ValueError(f"unsupported state type: {type(state).__name__}")


def build_decision_text(
    kind: str,
    state: Any,
    question: str,
    options: Sequence[str] | None = None,
) -> str:
    """The exact text prompt a text-only request becomes, for token estimates.

    The server does the real templating; this mirrors ``build_content`` in
    ``decide_shim.py`` so benchmarks can report an independent token estimate
    when a backend omits ``usage``.
    """
    opts = canonical_options(kind, options)
    lines = (
        [f"{LABELS[i]}) {option}" for i, option in enumerate(opts)]
        if kind == "choice"
        else list(opts)
    )
    return (
        f"[kind] {kind}\n[state] {_state_text(state)}"
        f"\n[question] {question}\n[options]\n"
        + "\n".join(lines)
        + "\n[decision]:"
    )


def build_payload(
    kind: str,
    state: Any,
    question: str,
    options: Sequence[str] | None = None,
    strategy: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    """Build the request body.  ``noul``/``score`` omit ``options`` on purpose:
    the server fixes them to false/true and 0..5 respectively."""
    body: dict[str, Any] = {"kind": kind, "state": state, "question": question}
    if kind == "choice":
        body["options"] = canonical_options(kind, options)
    elif kind not in ("noul", "score"):
        raise ValueError(f"kind must be noul|score|choice, got {kind!r}")
    if strategy and strategy != "auto":
        body["strategy"] = strategy
    if model:
        body["model"] = model
    return body


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(value - top) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass(frozen=True)
class HeadCalibration:
    """The decision-head slots, bias, temperatures, and optional fitted remap."""

    bias: list[float]
    verbalizer_ids: list[int]
    ranges: dict[str, tuple[int, int]]
    temperatures: dict[str, float]
    fitted: dict[str, dict[str, Any]] | None = None
    template_version: str = "bare-v1"

    @classmethod
    def from_paths(
        cls,
        head_path: str | Path = DATA_DIR / "decision_head.json",
        calibration_path: str | Path = DATA_DIR / "calibration.json",
        fitted_path: str | Path | None = None,
    ) -> "HeadCalibration":
        head = json.loads(Path(head_path).read_text())
        calibration = json.loads(Path(calibration_path).read_text())
        fitted = None
        if fitted_path:
            path = Path(fitted_path)
            if not path.exists():
                raise FileNotFoundError(path)
            fitted = json.loads(path.read_text())
        ranges = {
            kind: (int(bounds[0]), int(bounds[1]))
            for kind, bounds in head["slots"]["ranges"].items()
        }
        head_cal = cls(
            bias=[float(value) for value in head["bias"]],
            verbalizer_ids=[int(value) for value in head["verbalizer_ids"]],
            ranges=ranges,
            temperatures={
                kind: float(value)
                for kind, value in calibration["per_kind"].items()
            },
            fitted=fitted,
            template_version=head["slots"].get("template_version", "bare-v1"),
        )
        head_cal.validate()
        return head_cal

    def validate(self) -> None:
        if len(self.bias) != len(self.verbalizer_ids):
            raise ValueError("head bias and verbalizer_ids lengths differ")
        for kind in ("noul", "score", "choice"):
            if kind not in self.ranges:
                raise ValueError(f"missing slot range for {kind}")
            if kind not in self.temperatures:
                raise ValueError(f"missing temperature for {kind}")
            lo, hi = self.ranges[kind]
            if not 0 <= lo < hi <= len(self.bias):
                raise ValueError(f"bad slot range for {kind}: {lo}..{hi}")
        if self.ranges["noul"] != (0, 2):
            raise ValueError(f"unexpected noul slots {self.ranges['noul']}")
        if self.ranges["score"] != (2, 8):
            raise ValueError(f"unexpected score slots {self.ranges['score']}")
        if self.ranges["choice"][0] != 8:
            raise ValueError(f"unexpected choice slots {self.ranges['choice']}")

    def probabilities(
        self,
        kind: str,
        logprobs: Mapping[int, float],
        n_options: int | None = None,
        label_token_ids: Sequence[int] | None = None,
    ) -> list[float]:
        """Reproduce ``_s1_pass`` from next-token log probabilities.

        ``logprobs`` maps token id -> log probability.  Missing ids become
        -1e9, as in the Python front.  For choices above 16 labels the caller
        must supply the tokenizer's single-token label ids (the Python front
        reads them from the tokenizer; this harness deliberately has no
        transformers dependency).
        """
        if kind == "choice":
            n = int(n_options or 0)
            if not 2 <= n <= MAX_OPTIONS:
                raise ValueError(f"choice n_options must be 2..{MAX_OPTIONS}")
            lo, hi = self.ranges["choice"]
            if label_token_ids is None:
                if n > hi - lo:
                    raise ValueError(
                        "choice with >16 options needs label_token_ids "
                        "(single-token labels from the model tokenizer)"
                    )
                token_ids = self.verbalizer_ids[lo : lo + n]
            else:
                token_ids = [int(value) for value in label_token_ids]
                if len(token_ids) != n:
                    raise ValueError("label_token_ids length must equal n_options")
            bias = [self.bias[lo + i] if lo + i < hi else 0.0 for i in range(n)]
        elif kind in ("noul", "score"):
            lo, hi = self.ranges[kind]
            expected = hi - lo
            if n_options is not None and int(n_options) != expected:
                raise ValueError(f"{kind} has fixed {expected} options")
            token_ids = self.verbalizer_ids[lo:hi]
            bias = self.bias[lo:hi]
        else:
            raise ValueError(f"unknown kind {kind!r}")

        raw = [
            max(float(logprobs.get(token_id, -1e9)), -1e9) + b
            for token_id, b in zip(token_ids, bias)
        ]
        if self.fitted and kind in self.fitted:
            fit = self.fitted[kind]
            scale = float(fit["scale"])
            fit_bias = [float(value) for value in fit["bias"]]
            if len(fit_bias) != len(raw):
                raise ValueError(
                    f"fitted {kind} bias has {len(fit_bias)} slots, need {len(raw)}"
                )
            scaled = [value * scale + fit_bias[i] for i, value in enumerate(raw)]
        else:
            scaled = [value / self.temperatures[kind] for value in raw]
        return softmax(scaled)


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    if len(p) != len(q):
        raise ValueError("KL inputs have different lengths")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            total += pi * math.log((pi + eps) / (qi + eps))
    return total


def mean_abs_error(p: Sequence[float], q: Sequence[float]) -> float:
    if not p:
        return 0.0
    return sum(abs(a - b) for a, b in zip(p, q)) / len(p)


def max_abs_delta(p: Sequence[float], q: Sequence[float]) -> float:
    return max((abs(a - b) for a, b in zip(p, q)), default=0.0)


def top1_index(probabilities: Sequence[float]) -> int:
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def align_probabilities(result: Mapping[str, Any], target_options: Sequence[str]) -> list[float]:
    """Return probabilities in ``target_options`` order.

    Servers echo ``options``; if a backend reorders them we re-map by option
    string.  Identical prompts are still required for a meaningful comparison.
    """
    probs = [float(value) for value in result["probabilities"]]
    options = result.get("options")
    if not options or list(options) == list(target_options) or len(options) != len(probs):
        return probs
    if sorted(options) != sorted(target_options):
        return probs
    by_name: dict[Any, float] = {}
    for name, probability in zip(options, probs):
        by_name.setdefault(name, probability)
    return [by_name.get(name, 0.0) for name in target_options]


class DecideClient:
    """Small retrying HTTP client for one decision backend."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        timeout: float = 600.0,
        retries: int = 3,
        backoff: float = 0.5,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        if session is None:
            import requests  # local import keeps offline head math import-light

            session = requests.Session()
        self.session = session

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import requests

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(
                    method,
                    self.base_url + path,
                    timeout=self.timeout,
                    **kwargs,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff * (2**attempt))
        raise DecideError(f"{method} {path} failed: {last_error}") from last_error

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health")
        if response.status_code >= 400:
            raise DecideError(f"GET /health -> {response.status_code}: {response.text[:400]}")
        return response.json()

    def info(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/decide/info")
        if response.status_code >= 400:
            raise DecideError(f"GET /v1/decide/info -> {response.status_code}: {response.text[:400]}")
        return response.json()

    def decide(
        self,
        kind: str,
        state: Any,
        question: str,
        options: Sequence[str] | None = None,
        *,
        strategy: str = "auto",
        model: str | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        payload = build_payload(
            kind,
            state,
            question,
            options,
            strategy=strategy,
            model=model or self.model,
        )
        response = self._request("POST", "/v1/decide", json=payload)
        if response.status_code >= 400:
            raise DecideError(
                f"POST /v1/decide -> {response.status_code}: {response.text[:1000]}"
            )
        try:
            result = response.json()
        except ValueError as error:
            raise DecideError(f"non-JSON response: {response.text[:400]}") from error
        if validate:
            self.validate_result(result, kind, options)
        return result

    @staticmethod
    def validate_result(result: Mapping[str, Any], kind: str, options: Sequence[str] | None) -> None:
        if not isinstance(result, Mapping) or "probabilities" not in result:
            raise DecideError("response has no probabilities")
        probs = [float(value) for value in result["probabilities"]]
        expected = 2 if kind == "noul" else 6 if kind == "score" else len(options or [])
        if len(probs) != expected:
            raise DecideError(f"expected {expected} probabilities for {kind}, got {len(probs)}")
        if any(not math.isfinite(value) or value < 0.0 for value in probs):
            raise DecideError("probabilities are non-finite or negative")
        total = sum(probs)
        if abs(total - 1.0) > 1e-4:
            raise DecideError(f"probabilities sum to {total:.6f}, expected 1")
        if "choice_index" in result:
            index = int(result["choice_index"])
            if not 0 <= index < len(probs):
                raise DecideError(f"choice_index {index} out of range")
            if "choice" in result:
                expected_choice = canonical_options(kind, options)[index]
                if str(result["choice"]) != str(expected_choice):
                    raise DecideError(
                        f"choice {result['choice']!r} does not match option {expected_choice!r}"
                    )


def _parse_options(args: argparse.Namespace) -> list[str] | None:
    if args.option:
        return list(args.option)
    if args.options is None:
        return None
    text = args.options.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise SystemExit("--options JSON must be a list")
        return [str(item) for item in parsed]
    return [part for part in text.split(",") if part != ""]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call a JEV /v1/decide backend")
    parser.add_argument(
        "--url",
        default=os.environ.get("JEV_DECIDE_URL", "http://127.0.0.1:8501"),
        help="backend base URL (default: %(default)s)",
    )
    parser.add_argument("--kind", choices=["noul", "score", "choice"], default="noul")
    parser.add_argument("--state", default="")
    parser.add_argument("--state-file", help="read state text from a file")
    parser.add_argument("--question", default="")
    parser.add_argument("--options", help="JSON list or comma-separated choice options")
    parser.add_argument("--option", action="append", help="repeatable choice option")
    parser.add_argument("--strategy", default="auto", choices=["auto", "single", "tournament", "permute"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--health", action="store_true", help="call GET /health and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the request body only")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    client = DecideClient(
        args.url, model=args.model, timeout=args.timeout, retries=args.retries
    )
    try:
        if args.health:
            print(json.dumps(client.health(), indent=2 if args.pretty else None))
            return 0
        state = Path(args.state_file).read_text() if args.state_file else args.state
        options = _parse_options(args)
        if args.kind in ("noul", "score"):
            options = None
        payload = build_payload(
            args.kind, state, args.question, options, args.strategy, args.model
        )
        if args.dry_run:
            print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
            return 0
        result = client.decide(
            args.kind,
            state,
            args.question,
            options,
            strategy=args.strategy,
            model=args.model,
        )
        print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
        return 0
    except (DecideError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
