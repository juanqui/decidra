# Independent evaluation harness

Self-contained Python 3 harness for the `/v1/decide` protocol. It exercises the Rust
server, the Python shim, or both over HTTP; it never imports the model, starts/stops vLLM,
or touches a GPU. The only local server it starts is a deterministic stdlib mock.

See [`../docs/RESULTS.md`](../docs/RESULTS.md) for the measured live-eval results and
[`../docs/KERNELS.md`](../docs/KERNELS.md) for the serving-kernel analysis.

## Files

| File | Purpose |
|---|---|
| `client.py` | Exact `/v1/decide` client (noul/score/choice, 2..256 options), offline head math, CLI. |
| `suite.py` | 28-row deterministic suite; live-vs-live or live-vs-saved comparison with KL/MAE/delta/top-1 thresholds. |
| `head_client.py` | Direct vLLM `/v1/completions` logprob read-out plus local head math; compares against Rust `/v1/decide`. |
| `bench.py` | Concurrency sweep on short prompts and unique ~3k/~9k-token documents; latency, req/s, error rate, token throughput. |
| `report.py` | Markdown + JSON summary of suite and bench outputs. |
| `functional_checks.py` | 256-option choice, `/v1/decisions`, and a generated 512x512 image decision. |
| `sign_check.py` | HMAC-SHA256 signature verification for `/v1/decide`; confirms `/health` is unsigned. |
| `mock_server.py` | Deterministic stdlib mock of the decision API (no GPU, no model). |
| `smoke_test.py` | Offline end-to-end test: client → mock → suite → bench → report. |
| `data/suite.jsonl` | The 28 suite prompts, reconstructed from the canonical suite JSONL. |
| `data/awq-v2-suite.json` | Saved production-v2 results (default reference). |
| `data/awq-v4-suite.json` | Saved AWQ-v4 results, for historical cross-quantization comparisons. |
| `data/decision_head.json`, `data/calibration.json` | Decision-head bias/verbalizers and per-kind temperatures. |
| `data/fitted-calibration.json` | Optional serving-side fit; not enabled by default. |

Requirements: Python 3.10+, `requests`, and `Pillow` for the image check.

```bash
cd eval
python3 -m pip install -r requirements.txt
```

## Data provenance

Every saved `*-suite.json` contains results only (`id`, `truth`, `kind`, `choice`,
`probabilities`), not prompts. The prompts in `data/suite.jsonl` were reconstructed from
the canonical suite used by the original evaluation (`suite.jsonl` with ids `r01`..`r28`)
and match the saved rows id-for-id.

Two details preserve the historical request shapes:

- For `choice`, options are the **keys** of `question.criteria` in file order.
  Descriptions are not part of the decision prompt.
- For `score`, requests omit `options`; the server always uses `0..5`.

The saved v2 baseline's `score` truths disagree with the six-way head (0/5), so truth
accuracy is informational only. The pass/fail gate measures backend equivalence.

## Offline smoke test

No server, model, or GPU required:

```bash
cd eval
python3 smoke_test.py -v
```

This starts `mock_server.py` on an ephemeral port and runs the client, suite, bench, and
report as subprocesses. Expected: 6 tests pass, suite deltas exactly 0, bench 0 errors.

## Live commands

The live environment used for `docs/RESULTS.md`:

- Rust server: `http://127.0.0.1:8501`
- vLLM 0.31.0: `http://127.0.0.1:8002`, model `jev-awq`
- Python shim (optional live-vs-live): `http://127.0.0.1:8500`

All commands below run from `eval/`. `scripts/run-eval.sh` automates this sequence and
writes to `eval/out/`.

### 1. Primary exactness: Rust vs direct vLLM read-out

```bash
python3 head_client.py \
  --vllm-url http://127.0.0.1:8002 \
  --rust-url http://127.0.0.1:8501 \
  --out out/suite_head_vs_rust.json
```

For every row, the same prompt is sent directly to vLLM `/v1/completions` for the head
token logprobs, the bias/temperature/softmax math is applied locally, and the result is
compared with Rust `/v1/decide`. This isolates the Rust prompt/label/head implementation
from front-end concurrency. Default gates are strict: top-1 1.0, mean KL `1e-3`,
mean MAE `2e-4`, max delta `1e-3`, zero errors.

Measured: **28/28**, max delta **4.44e-16**, mean KL **3.53e-18**, 0 errors.

### 2. Saved production-v2 baseline: strict and tolerance variants

```bash
# Strict gates. This intentionally FAILS on the documented cross-build drift.
python3 suite.py \
  --a-url http://127.0.0.1:8501 \
  --baseline data/awq-v2-suite.json \
  --min-top1 1.0 --max-kl 0.001 --max-mae 0.0005 --max-delta 0.001 \
  --no-strict \
  --out out/suite_rust_vs_v2.json

# Cross-build tolerance. This is the acceptance variant.
python3 suite.py \
  --a-url http://127.0.0.1:8501 \
  --baseline data/awq-v2-suite.json \
  --min-top1 1.0 --max-kl 0.001 --max-mae 0.001 --max-delta 0.01 \
  --out out/suite_rust_vs_v2_tolerance.json
```

Live Rust vs live shim on identical bodies:

```bash
python3 suite.py \
  --a-url http://127.0.0.1:8500 \
  --b-url http://127.0.0.1:8501 \
  --out out/suite_py_vs_rust.json
```

Measured v2 comparison: top-1 **28/28**, mean KL **1.505e-5**, mean MAE **6.83e-4**,
max delta **4.933e-3** (`r22`), 0 errors. The strict `1e-3` max-delta gate fails by design;
see the tolerance rationale below.

### 3. Benchmark

```bash
python3 bench.py \
  --url http://127.0.0.1:8501 \
  --mode both \
  --concurrency 1,4,8,12 \
  --requests 128 \
  --long-requests 8 \
  --long-sizes 3000,9000 \
  --chars-per-token 3.38 \
  --warmup 5 \
  --out out/bench_combined_rust.json
```

Separate short/long runs used the same parameters but `--mode short` / `--mode long`.
Each short level sends 128 requests; each long size/level sends 8. Long documents are
unique per request and use a nonce at the start so vLLM prefix caching does not inflate
throughput. Measured: 512 short + 32 long requests, **0 errors**.

### 4. Functional checks

```bash
python3 functional_checks.py \
  --url http://127.0.0.1:8501 \
  --out out/functional_checks.json
```

Checks a 256-option choice (`tournament`, 17 upstream passes), `/v1/decisions` with
predicate + choice + score, and one 512x512 image choice. The image check requires Pillow.
Use `--skip-image` only when intentionally testing without Pillow.

### 5. Signing

Start the Rust server with `JEV_SIGNING_KEY=test-key`, then:

```bash
python3 sign_check.py \
  --url http://127.0.0.1:8501 \
  --key test-key \
  --out out/signing_check.json
```

It verifies HMAC-SHA256 over the exact response bytes and that `/health` is not signed.

### 6. Report

```bash
python3 report.py \
  --suite out/suite_rust_vs_v2_tolerance.json \
  --bench out/bench_combined_rust.json \
  --out-prefix out/live-report \
  --per-row
```

Writes `out/live-report.md` and `out/live-report.json`. The source JSONs remain
authoritative.

## Tolerance rationale

Two different comparisons need two different gates:

1. **Rust vs direct vLLM (same process, same prompt): strict.** The Rust server and the
   local head math perform the same operations in float64. Measured disagreement is
   `4.44e-16` max, mean KL `3.53e-18`. Gates: top-1 1.0, mean KL `1e-3`, MAE `2e-4`,
   max delta `1e-3`.

2. **Rust vs the saved production-v2 baseline: documented cross-build tolerance.**
   The saved baseline was produced on 2026-10-09 by an older serving configuration. The
   current server uses vLLM 0.31.0 with FP8 KV cache, chunked prefill, and SM89 kernels
   (Marlin W4A16, FlashInfer, Triton/FLA GDN), which move low-order logits. The gap is
   deterministic (repeated rows vary by exactly 0.0) and top-1/truth accuracy is
   unchanged at 28/28. Measured: mean KL `1.505e-5`, mean MAE `6.83e-4`, max delta
   `4.933e-3` (`r22`). Gates: top-1 1.0, mean KL `1e-3`, MAE `1e-3`, max delta `0.01`.

The strict saved-baseline variant (`--max-delta 0.001 --max-mae 0.0005`) is retained for
inspection but is not the acceptance criterion. The harness defaults (`max-delta 0.05`,
`max-mae 0.005`) are for cross-quantization comparisons such as AWQ v3 vs v4; they are too
loose for same-model serving verification.

## Outputs

- `out/suite*.json`: per-row probabilities for both backends, comparison metrics,
  per-kind breakdown, truth accuracy, thresholds/checks, `pass`.
- `out/bench*.json`: run parameters, warmups, and one entry per concurrency level.
- `out/functional_checks.json`, `out/signing_check.json`: functional contract checks.
- `out/live-report.{md,json}`: compact report and machine summary.
- `out/` and `__pycache__/` are ignored by git; results are reproducible from the commands
  above and are not committed.

## Pitfalls

1. **Prefix caching.** Do not build long prompts by repeating a filler prefix. The
   generator puts a unique nonce first in every long document and short state.
2. **Batching numerics.** Real backends are not bit-identical across batch shapes. The
   direct-vLLM comparison shows the Rust math is exact; the saved-v2 comparison is subject
   to serving-build drift.
3. **Wide choices.** More than 16 options uses `tournament` by default and becomes
   `ceil(n/16) + 1` model requests; `permute` uses four passes. This changes latency and
   `usage`.
4. **Score is fixed.** Options are always `0..5`; supplied options are ignored.
5. **Model name.** The harness sends `model` only when asked. Public names include
   `autotrust/JEV-27B-VL`, `autotrust/JEV-27B`, and `jev-awq`.
6. **Text by default.** `client.py` rejects image states; `functional_checks.py` exercises
   the image path separately.
7. **No secrets.** Signing tests use an explicit local test key; no key is read from a
   file or committed.
