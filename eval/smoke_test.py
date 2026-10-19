#!/usr/bin/env python3
"""End-to-end smoke test against the local mock server (no GPU, no model).

    python3 smoke_test.py -v

Starts ``mock_server.py`` on an ephemeral port, exercises the client for all
three decision kinds (including 256 choices), runs ``suite.py`` (mock A vs mock
B), ``bench.py`` (short + both long sizes), and ``report.py`` as subprocesses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Any

from client import (
    DATA_DIR,
    DecideClient,
    HeadCalibration,
    build_decision_text,
    choice_labels,
    softmax,
)
from mock_server import make_server

ROOT = Path(__file__).resolve().parent
SERVER = None
THREAD = None
BASE_URL = ""


def setUpModule() -> None:
    global SERVER, THREAD, BASE_URL
    SERVER = make_server(port=0)
    host, port = SERVER.server_address[:2]
    BASE_URL = f"http://{host}:{port}"
    THREAD = threading.Thread(target=SERVER.serve_forever, daemon=True)
    THREAD.start()


def tearDownModule() -> None:
    if SERVER is not None:
        SERVER.shutdown()
        SERVER.server_close()
    if THREAD is not None:
        THREAD.join(timeout=5)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=300,
    )


@lru_cache(maxsize=1)
def suite_result() -> tuple[Path, dict[str, Any]]:
    out = ROOT / "out" / "smoke-suite.json"
    process = _run(
        "suite.py",
        "--a-url",
        BASE_URL,
        "--b-url",
        BASE_URL,
        "--out",
        str(out),
    )
    if process.returncode != 0:
        raise AssertionError(
            f"suite.py failed ({process.returncode})\nstdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return out, json.loads(out.read_text())


@lru_cache(maxsize=1)
def bench_result() -> tuple[Path, dict[str, Any]]:
    out = ROOT / "out" / "smoke-bench.json"
    process = _run(
        "bench.py",
        "--url",
        BASE_URL,
        "--mode",
        "both",
        "--concurrency",
        "1,2",
        "--requests",
        "4",
        "--long-requests",
        "2",
        "--long-sizes",
        "3000,9000",
        "--warmup",
        "1",
        "--out",
        str(out),
    )
    if process.returncode != 0:
        raise AssertionError(
            f"bench.py failed ({process.returncode})\nstdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return out, json.loads(out.read_text())


class HarnessSmokeTest(unittest.TestCase):
    def test_labels_and_prompt(self) -> None:
        labels = choice_labels(256)
        self.assertEqual(len(labels), 256)
        self.assertEqual(labels[:3], ["A", "B", "C"])
        self.assertEqual(labels[25], "Z")
        self.assertEqual(labels[26], "AA")
        self.assertEqual(labels[-1], "IV")
        text = build_decision_text("choice", "state", "q", ["x", "y"])
        self.assertIn("[kind] choice\n[state] state", text)
        self.assertIn("A) x\nB) y", text)
        self.assertTrue(text.endswith("\n[decision]:"))

    def test_head_math(self) -> None:
        head = HeadCalibration.from_paths(
            DATA_DIR / "decision_head.json", DATA_DIR / "calibration.json"
        )
        ids = head.verbalizer_ids[head.ranges["noul"][0] : head.ranges["noul"][1]]
        logprobs = {ids[0]: -0.3, ids[1]: -1.2}
        probs = head.probabilities("noul", logprobs)
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        raw = [
            (logprobs[ids[i]] + head.bias[i]) / head.temperatures["noul"]
            for i in range(2)
        ]
        expected = softmax(raw)
        for got, want in zip(probs, expected):
            self.assertAlmostEqual(got, want, places=12)
        # Missing token ids read as -1e9 rather than crashing.
        self.assertEqual(len(head.probabilities("score", {})), 6)

        label_ids = [1000 + index for index in range(256)]
        logprobs = {token: -0.01 * index for index, token in enumerate(label_ids)}
        large = head.probabilities(
            "choice", logprobs, 256, label_token_ids=label_ids
        )
        self.assertEqual(len(large), 256)
        self.assertAlmostEqual(sum(large), 1.0, places=12)

    def test_client_all_kinds(self) -> None:
        client = DecideClient(BASE_URL, retries=0)
        noul = client.decide("noul", "The charge was duplicated.", "Needs follow-up?")
        self.assertEqual(len(noul["probabilities"]), 2)
        self.assertIn(noul["choice"], ("false", "true"))

        score = client.decide("score", "Everything is on fire.", "How urgent?")
        self.assertEqual(len(score["probabilities"]), 6)
        self.assertIn(score["choice"], [str(i) for i in range(6)])

        options = [f"option-{index}" for index in range(256)]
        wide = client.decide("choice", "Pick one.", "Which option?", options)
        self.assertEqual(len(wide["probabilities"]), 256)
        self.assertEqual(wide["options"], options)
        self.assertAlmostEqual(sum(wide["probabilities"]), 1.0, places=9)

    def test_suite_mock_vs_mock(self) -> None:
        _, result = suite_result()
        self.assertEqual(result["tool"], "jev-eval/suite")
        self.assertEqual(result["summary"]["compared"], 28)
        self.assertEqual(result["summary"]["errors_a"], 0)
        self.assertEqual(result["summary"]["errors_b"], 0)
        self.assertTrue(result["pass"])
        self.assertEqual(result["summary"]["top1_agreement"], 1.0)
        self.assertLess(result["summary"]["max_delta"], 1e-12)
        self.assertEqual(result["summary"]["mean_kl"], 0.0)
        # Deterministic mock: every probability vector is bit-identical.
        for row in result["rows"]:
            self.assertTrue(row["comparable"])
            self.assertEqual(row["a"]["probabilities"], row["b"]["probabilities"])

    def test_bench_sweep(self) -> None:
        _, result = bench_result()
        self.assertEqual(result["tool"], "jev-eval/bench")
        levels = result["levels"]
        self.assertEqual(len(levels), 6)  # 2 short + 2 sizes x 2 concurrency
        self.assertEqual({level["mode"] for level in levels}, {"short", "long"})
        self.assertEqual(
            {level["target_tokens"] for level in levels if level["mode"] == "long"},
            {3000, 9000},
        )
        for level in levels:
            self.assertEqual(level["errors"], 0)
            self.assertEqual(level["ok"], level["requests"])
            self.assertGreater(level["req_per_s"], 0)
            self.assertIsNotNone(level["latency_ms"]["p50"])
            self.assertIsNotNone(level["latency_ms"]["p95"])
            self.assertIsNotNone(level["latency_ms"]["p99"])
            self.assertGreater(level["prompt_tokens"]["total"], 0)
        long_levels = [level for level in levels if level["mode"] == "long"]
        for level in long_levels:
            # The synthetic document is sized at 4 chars/token; allow tokenizer drift.
            self.assertGreater(level["prompt_tokens"]["mean"], level["target_tokens"] * 0.5)

    def test_report(self) -> None:
        suite_path, suite = suite_result()
        bench_path, bench = bench_result()
        prefix = ROOT / "out" / "smoke-report"
        process = _run(
            "report.py",
            "--suite",
            str(suite_path),
            "--bench",
            str(bench_path),
            "--out-prefix",
            str(prefix),
            "--per-row",
        )
        self.assertEqual(
            process.returncode,
            0,
            f"report.py failed\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}",
        )
        markdown = prefix.with_suffix(".md").read_text()
        compact = json.loads(prefix.with_suffix(".json").read_text())
        self.assertIn("Suite verdict: PASS", markdown)
        self.assertIn("Overall: PASS", markdown)
        self.assertIn("| short |", markdown.replace("| short  |", "| short |"))
        self.assertTrue(compact["suite"]["pass"])
        self.assertTrue(compact["bench"]["pass"])
        self.assertTrue(compact["pass"])
        self.assertEqual(len(compact["bench"]["levels"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
