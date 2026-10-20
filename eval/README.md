# Independent evaluation harness

Self-contained Python 3 harness for the `/v1/decide` protocol. It exercises a running
`decidra` over HTTP; it never imports the model, starts/stops a serving process, or
touches a GPU. The only local server it starts is a deterministic stdlib mock.

The runtime under test is the in-process native CUDA backend. There is no direct
oracle read-out; native-engine numerics are validated against CPU golden tensors in
[`../native/README.md`](../native/README.md) and summarized in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

## Files

| File | Purpose |
|---|---|
| `client.py` | Exact `/v1/decide` client (noul/score/choice, 2..256 options), offline head math, CLI. |
| `suite.py` | 28-row deterministic suite; live-vs-live or live-vs-saved comparison with KL/MAE/delta/top-1 thresholds. |
| `bench.py` | Concurrency sweep on short prompts and unique ~3k/~9k-token documents; latency, req/s, error rate, token throughput. `--mode batch` is the 69-token batching/scheduler sweep. |
| `batch_equiv.py` | Compare per-request head logits from individual vs batched calls: request count, top-1 agreement, max absolute delta. |
| `report.py` | Markdown + JSON summary of suite and bench outputs. |
| `functional_checks.py` | 256-option choice, `/v1/decisions`, and image-state rejection (400). |
| `sign_check.py` | HMAC-SHA256 signature verification for `/v1/decide`; confirms `/health` is unsigned. |
| `mock_server.py` | Deterministic stdlib mock of the decision API (no GPU, no model). |
| `smoke_test.py` | Offline end-to-end test: client → mock → suite → bench → report. |
| `data/suite.jsonl` | The 28 suite prompts, reconstructed from the canonical suite JSONL. |
| `data/awq-cpu-suite28.json` | CPU AWQ golden results; default reference and acceptance baseline. |
| `data/awq-v2-suite.json` | Historical production-v2 (previous serving stack) results; drift context only. |
| `data/awq-v4-suite.json` | Saved AWQ-v4 results, for historical cross-quantization comparisons. |
| `data/decision_head.json`, `data/calibration.json` | Decision-head bias/verbalizers and per-kind temperatures. |
| `data/fitted-calibration.json` | Optional serving-side fit; not enabled by default. |

Requirements: Python 3.10+, `requests`, and `Pillow` for the image-rejection check.

```bash
cd eval
python3 -m pip install -r requirements.txt
```

## Data provenance

Every saved `*-suite.json` contains results only (`id`, `truth`, `kind`, `choice`,
`probabilities`), not prompts. The prompts in `data/suite.jsonl` were reconstructed from
the canonical suite used by the original evaluation (`suite.jsonl`, ids `r01`..`r28`).
The saved baselines contain results only. `awq-cpu-suite28.json` is the CPU AWQ golden
result (the correct acceptance reference for the native engine). `awq-v2-suite.json` was
collected on the previous serving stack and is kept as historical drift context; it
differs from the CPU golden by max **0.01377** / mean MAE **0.00156** on its own.

## Offline smoke test

```bash
cd eval
python3 smoke_test.py -v
```

This starts `mock_server.py`, runs the client, suite, benchmark, and report generation.
No server or GPU is required.

## Against a running server

```bash
# From the repository root; requires decidra on 127.0.0.1:8501.
./scripts/run-eval.sh
```

The script runs:

1. The CPU AWQ golden comparison with strict gates (`--no-strict`, so failures are
   recorded rather than fatal).
2. The same CPU golden comparison with the acceptance tolerance
   (`top-1 1.0`, `max_kl 0.001`, `max_mae 0.001`, `max_delta 0.011`).
3. The historical production-v2 comparison, labeled as context only.
4. The combined short/long benchmark.
5. Functional checks (256 choices, `/v1/decisions`, image rejection).
6. Optional signing when `DECIDRA_EVAL_SIGNING_KEY` is set.

Reports land in `eval/out/`. Override the target with `DECIDRA_EVAL_RUST_URL`.

## Batching sweep

`bench.py --mode batch` sends 16 unique ~69-token decision requests at each
concurrency level (default `1,2,4,8,16`). Prompts are unique per level, so the
sweep does not measure a warm prefix cache. Each level reports requests/s,
p50/p95/p99 latency, prompt tokens/s, and—when the server exposes `/metrics`—a
per-level delta of the `decidra_native_batch_size` histogram (count, mean, buckets)
plus the peak observed `decidra_native_batch_queue_depth` gauge.

```bash
# From the repository root; server already running on 127.0.0.1:8501.
./scripts/bench-batch.sh
# Writes eval/out/bench_batch.json
```

Override the target and sweep with `DECIDRA_SMOKE_URL`, `DECIDRA_BATCH_OUT`,
`DECIDRA_BATCH_CONCURRENCY`, `DECIDRA_BATCH_REQUESTS`, `DECIDRA_BATCH_TOKENS`, and
`DECIDRA_BATCH_WARMUP`. Equivalent direct invocation:

```bash
cd eval
python3 bench.py --url http://127.0.0.1:8501 --mode batch \
  --batch-concurrency 1,2,4,8,16 --batch-requests 16 --batch-tokens 69 \
  --out out/bench_batch.json
```

Against the mock server the same command works, but `/metrics` is absent, so
`server_metrics.available` is `false`.

## Batched vs individual logits

`batch_equiv.py` compares two JSON files of per-request head logits (one from
individual calls, one from batched calls). It asserts equal request counts,
per-request top-1 agreement, and a bounded maximum absolute delta. Accepted
shapes are a list of records, `{"requests"|"results"|"rows"|"items": [...]}`,
or a mapping of request id to logits; a record is a bare vector or an object
with `logits`/`head_logits` (plus optional `id`/`request_id`). With explicit ids
in both files, records are aligned by id.

```bash
cd eval
python3 batch_equiv.py individual.json batched.json --max-delta 1e-4
```

The native worker's `batch` CLI is the intended producer of these JSON files;
this harness does not call an inference API itself.

## Tolerances

`suite.py` compares probability distributions row by row. The acceptance baseline is the
CPU AWQ golden (dequantized with `compressed_tensors` offline); native BF16 kernels differ
from it by a small amount. Measure the actual deltas on each deployment. The historical
v2 baseline is useful for context but is not an acceptance gate: that baseline itself
differs from CPU AWQ by max 0.01377 / mean MAE 0.00156. A change in top-1 agreement, or a
CPU-golden tolerance failure, is a real regression. Native-engine golden checks
(layer/final-norm cosine and logits) are the primary numerical evidence.
