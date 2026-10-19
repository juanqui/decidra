# Independent evaluation harness

Self-contained Python 3 harness for the `/v1/decide` protocol. It exercises a running
`jev-engine` over HTTP; it never imports the model, starts/stops a serving process, or
touches a GPU. The only local server it starts is a deterministic stdlib mock.

The runtime under test is the in-process native CUDA backend. There is no direct-vLLM
comparison anymore; native-engine numerics are validated against CPU golden tensors in
[`../native/README.md`](../native/README.md) and summarized in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

## Files

| File | Purpose |
|---|---|
| `client.py` | Exact `/v1/decide` client (noul/score/choice, 2..256 options), offline head math, CLI. |
| `suite.py` | 28-row deterministic suite; live-vs-live or live-vs-saved comparison with KL/MAE/delta/top-1 thresholds. |
| `bench.py` | Concurrency sweep on short prompts and unique ~3k/~9k-token documents; latency, req/s, error rate, token throughput. |
| `report.py` | Markdown + JSON summary of suite and bench outputs. |
| `functional_checks.py` | 256-option choice, `/v1/decisions`, and image-state rejection (400). |
| `sign_check.py` | HMAC-SHA256 signature verification for `/v1/decide`; confirms `/health` is unsigned. |
| `mock_server.py` | Deterministic stdlib mock of the decision API (no GPU, no model). |
| `smoke_test.py` | Offline end-to-end test: client → mock → suite → bench → report. |
| `data/suite.jsonl` | The 28 suite prompts, reconstructed from the canonical suite JSONL. |
| `data/awq-v2-suite.json` | Saved production-v2 results, used as the regression baseline. |
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
The saved baselines were collected on the previous serving stack; they are regression
references, not a numerical oracle for the native engine.

## Offline smoke test

```bash
cd eval
python3 smoke_test.py -v
```

This starts `mock_server.py`, runs the client, suite, benchmark, and report generation.
No server or GPU is required.

## Against a running server

```bash
# From the repository root; requires jev-engine on 127.0.0.1:8501.
./scripts/run-eval.sh
```

The script runs:

1. The saved-v2 baseline comparison with strict gates (`--no-strict`, so failures are
   recorded rather than fatal).
2. The same comparison with the documented cross-engine tolerance
   (`top-1 1.0`, `max_kl 0.001`, `max_mae 0.0025`, `max_delta 0.01`).
3. The combined short/long benchmark.
4. Functional checks (256 choices, `/v1/decisions`, image rejection).
5. Optional signing when `JEV_EVAL_SIGNING_KEY` is set.

Reports land in `eval/out/`. Override the target with `JEV_EVAL_RUST_URL`.

## Tolerances

`suite.py` compares probability distributions row by row. The tolerance run exists
because the saved baseline came from a different engine build and serving stack; the
native BF16 kernels legitimately differ by ~1e-3 per probability. Measured on the AWQ v2
suite: top-1 **28/28**, mean KL **4.50e-5**, mean MAE **1.143e-3**, max delta **9.734e-3**
(`r22`). The acceptance tolerance is therefore `max_mae 0.0025`, `max_delta 0.01`. A change
in top-1 agreement, or a tolerance failure, is a real regression. Native-engine golden
checks (layer/final-norm cosine and logits) are the primary numerical evidence.
