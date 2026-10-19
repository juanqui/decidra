# Live evaluation results

Measured on 2026-10-19 with the release Rust server against vLLM 0.31.0 on **GPU 1 only**
(RTX 4090, 24 GB, SM89). GPU 0, the production stack, ports 8001/8500, and model weights
were not touched. The Rust server was stopped after the run; the test vLLM instance was
left healthy on `127.0.0.1:8002`.

Environment:

- Rust `jev-engine` release build, `127.0.0.1:8501`, `max_options=256`,
  `native_choice_options=16`, `max_concurrency=8`, request timeout 600 s.
- vLLM `0.31.0`, model `jev-awq`, artifact `/srv/models/jev-awq/awq-w4a16-v2`,
  `max_model_len=16384`, `max_num_seqs=8`, `kv_cache_dtype=fp8`, prefix caching on,
  chunked prefill on, `logprobs_mode=processed_logprobs`,
  `gpu_memory_utilization=0.95`.
- Temperatures: `noul=1.0143134751`, `choice=1.0161280671`, `score=1.0036213516`.
- External tuning traffic was quiescent for the final suite and both benches
  (0 external completions in the 5 s before each run).

The harness and exact commands are in [`../eval/README.md`](../eval/README.md). Raw
outputs are written to `eval/out/` and are not committed.

## 1. Primary exactness: Rust vs direct vLLM read-out

For each of the 28 suite rows, the same prompt was sent directly to vLLM
`/v1/completions` for the decision-head token logprobs. The reference bias, per-kind
temperature, and softmax were applied locally, then compared with Rust `/v1/decide`.

| Metric | Result | Gate |
|---|---:|---:|
| Rows compared | **28/28** | 28 |
| Top-1 agreement | **100%** | 100% |
| Mean KL (symmetric) | **3.53e-18** | <= 1e-3 |
| Mean MAE | **3.95e-17** | <= 2e-4 |
| Max abs delta | **4.44e-16** (`r22`) | <= 1e-3 |
| Request errors | **0** | 0 |

**PASS.** This is float64 round-off. Rust prompt construction, label selection, head bias,
per-kind temperatures, and softmax reproduce the reference read-out.

## 2. Saved production-v2 baseline: cross-build drift

Baseline: the saved v2 result file
`/srv/models/jev-awq/awq-w4a16-v2/eval/suite-28rows.json`, copied to
[`../eval/data/awq-v2-suite.json`](../eval/data/awq-v2-suite.json).

| Metric | Result | Tolerance gate |
|---|---:|---:|
| Rows compared | **28/28** | 28 |
| Top-1 agreement | **28/28 (100%)** | 100% |
| Mean KL (symmetric) | **1.505e-5** | <= 1e-3 |
| Mean MAE | **6.83e-4** | <= 1e-3 |
| Max abs delta | **4.933e-3** (`r22`) | <= 0.01 |
| Request errors | **0** | 0 |

**PASS with the documented 0.01 tolerance.** The strict `1e-3` max-delta / `5e-4` MAE
variant fails on the same data; it is retained as `eval/out/suite_rust_vs_v2.json` but is
not the acceptance criterion.

Why the gap is serving drift, not Rust math:

- Rust vs direct vLLM on the **same process and prompt** is exact to `4.44e-16`.
- The saved-v2 gap is deterministic: repeated `r01`, `r02`, and `r22` requests varied by
  exactly 0.0, with zero external traffic.
- Top-1 and truth accuracy are identical (21/28; the five `score` truth targets are known
  not to line up with the six-way head).
- The saved baseline was produced on 2026-10-09 by an older serving build. The current
  stack uses vLLM 0.31.0 with FP8 KV cache, chunked prefill, and SM89 kernel choices
  (Marlin W4A16, FlashInfer attention, Triton/FLA GDN). Those change low-order logits.
  See [`KERNELS.md`](KERNELS.md).

Cross-build comparisons therefore use `max_delta=0.01`, `max_mae=1e-3`, `max_kl=1e-3`,
top-1 100%, and zero errors. Same-model exactness keeps the strict `1e-3` gate.

## 3. Benchmarks

All requests succeeded: **512/512 short + 32/32 long, 0 errors**.

### Short decision prompts (76-77 prompt tokens/request)

| Concurrency | Requests | OK | Errors | req/s | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 0 | 19.204 | 50.3 | 66.9 | 81.0 |
| 4 | 128 | 128 | 0 | 28.901 | 129.9 | 193.2 | 193.8 |
| 8 | 128 | 128 | 0 | **34.200** | **226.2** | **280.6** | 282.3 |
| 12 | 128 | 128 | 0 | **34.257** | **344.8** | 494.4 | 501.9 |

Throughput plateaus at the vLLM `max_num_seqs=8` limit; concurrency above 8 adds queueing
latency without throughput.

### Long-context unique documents

| Target tokens | Conc | Requests | OK | Errors | p50 ms | Aggregate prompt tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 3k | 1 | 8 | 8 | 0 | **947** | **3,150** |
| 3k | 4 | 8 | 8 | 0 | **3,591** | **3,188** |
| 9k | 1 | 8 | 8 | 0 | **2,934** | **3,077** |
| 9k | 4 | 8 | 8 | 0 | **11,540** | **3,088** |

Actual prompt counts were 2,973-2,978 tokens for the 3k target and 9,014-9,022 for the
9k target. At concurrency 4 latency scales roughly 4x while token throughput stays flat:
long prompts are prefill-compute bound and the scheduler serializes them (`running=1`).
Long documents are synthetic and unique per request to defeat prefix caching.

## 4. GPU 1 saturation

Sampled with `nvidia-smi -i 1 -l 1` during the benches:

- Utilization: **100% max** (short mean 74.3%, long mean 93.1%; the windows include idle
  tails).
- Power: **410-422 W under load** (short max 388 W in the sampler window; long max
  420.83 W, mean 389.6 W).
- SM clock: **~2,910 MHz under load** as reported by the tuning run; the sampler CSV
  recorded utilization, memory, and power only. Idle SM clock was ~210 MHz; the card
  reports 3,300 MHz max.
- Memory: **~22.1 GiB used** (22,069-22,129 MiB of 24,564 MiB).

Saturation starts at concurrency 4-8. The 100% utilization and 410-422 W draw, together
with flat long-context throughput from concurrency 1 to 4, show the GPU was saturated by
model prefill rather than by the HTTP front end.

## 5. Functional checks

| Check | Result |
|---|---|
| 256-option choice | 256 probabilities, sum 1.0; choice `option-000`; `adaptation=tournament`; **17 model requests**; 2,873 prompt tokens; **1.079 s** |
| `/v1/decisions` | predicate 0.946285; choice `billing` 0.999891; score 2.104860 over 4 levels; usage 180 input / 3 output; 0.149 s |
| 512x512 image choice | choice `red` 0.996926; three probabilities summing to 1.0; 312 prompt tokens; 0.176 s |
| HMAC signing (`test-key`) | `/v1/decide` `x-jev-signature` matches HMAC-SHA256 over the exact 361-byte body; `/health` unsigned |
| `/health`, `/v1/models`, `/v1/decide/info` | 200; `max_options=256`, `native_choice_options=16`, protocol `jev27-bare-v1` |

The image check is a contract/path check with one synthetic image, not a vision-quality
benchmark.

## 6. Caveats

- **Hybrid GDN is not batch-invariant.** The model mixes full attention with GDN
  linear-attention layers; batch composition and prefill chunking shift low-order logits
  at the ~5e-4 level, and up to ~5e-3 across concurrent prefill. Top-1 decisions remain
  stable and the exactness check above (same process, same prompt) is float64-exact.
- Maximum tested context is **16,384 tokens** (`max_model_len`).
- Synthetic long documents are not representative of a specific real corpus.
- Benchmark latency includes HTTP, the Rust front, and vLLM. p99 at 12 concurrent short
  requests reflects queueing at `max_num_seqs=8`.
- SM89/Ada kernel choices and FP8 KV cache can change low-order logits versus other
  builds, GPUs, or serving configurations. See [`KERNELS.md`](KERNELS.md).

## 7. Reproduce

With vLLM on 8002 and the Rust server on 8501 already running:

```bash
./scripts/run-eval.sh
```

This runs the direct-vLLM exactness check, the v2 baseline comparison (strict and
tolerance variants), the combined benchmark, functional checks, an optional signing check,
and a Markdown/JSON report under `eval/out/`. It does not start or stop GPU services.
Individual commands and gate explanations are in [`../eval/README.md`](../eval/README.md).
