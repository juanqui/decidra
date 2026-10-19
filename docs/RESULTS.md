# Correctness and performance results

All results below are for the native backend. The old vLLM-based measurements are
superseded and are not part of this deployment.

## Native engine numerics

From `native/README.md`, validated against CPU golden data (PyTorch
`compressed_tensors` dequantization is used only offline to produce the golden tensors):

### AWQ 27B, 36-token prompt

| tensor | max_abs | mean_abs | rel_rms | cosine |
|---|---|---|---|---|
| layer 0 | 0.125 | 1.5e-4 | 0.35% | 0.9999942 |
| layer 1 (isolated) | 0.125 | 1.4e-4 | 0.18% | 0.9999983 |
| layer 3 (full stack) | 0.25 | 6.2e-4 | 0.38% | 0.9999928 |
| layer 4 (isolated) | 0.125 | 1.8e-4 | 0.11% | 0.9999994 |
| layer 63 (full stack) | 32 | 6.4e-2 | 0.96% | 0.9999539 |
| final norm | 0.375 | 9.6e-3 | 1.25% | 0.9999221 |
| head logits (3721/1802) | 0.0977 | 0.0177 | 1.18% | 0.9999306 |
| full logits (last) | 0.0625 | 3.0e-3 | 0.26% | 0.9999969 |

Top-1 token and the last logit are identical to golden (`867`, `3.578125`). Two-way
softmax is **0.8568** vs golden **0.8559** (abs diff 9.6e-4); full-vocab log-sum-exp
differs by 9.4e-4. Large-activation `max_abs` is exactly one BF16 ulp.

### Tiny reference model

| tensor | max_abs | mean_abs | rel_rms |
|---|---|---|---|
| embedding | 0 | 0 | 0 |
| layer 0 (GDN) | 9.8e-4 | 7.4e-5 | 0.42% |
| layer 1 (GDN) | 9.8e-4 | 7.6e-5 | 0.37% |
| layer 2 (GDN) | 9.8e-4 | 4.9e-5 | 0.26% |
| layer 3 (full attn) | 1.5e-3 | 1.1e-4 | 0.33% |
| final norm | 4.7e-2 | 1.0e-2 | 1.3% |
| last-token logits | 1.2e-2 | 3.0e-3 | 1.2% |

## Server/API evidence

- `cargo test`: 31 unit tests and 7 integration tests pass against `MockBackend`. These
  cover exact prompt bytes, tournament/permute pass counts, `/v1/decisions`,
  `/v1/systemone`, HMAC signing, and 400/501/502/504 semantics.
- `cargo check --all-targets --features native` compiles the server against the public
  `jev_native::Engine`, `preload_all`, `set_cache_weights`, `set_fused_w4a16`, and
  `decision_logits` API.
- `--check-config` validates the production tokenizer/head labels without loading the GPU.
- `scripts/native-smoke.sh` runs one real forward and asserts two finite head logits.

### Live server vs saved AWQ v2 baseline

Measured with the release binary on GPU 1 (resident weights, `JEV_NATIVE_FUSED_W4A16=false`)
against `eval/data/awq-v2-suite.json`, 28 rows:

| metric | value |
|---|---|
| rows compared | 28/28, 0 errors |
| top-1 agreement | **1.0 (28/28)** |
| mean KL | **4.496e-5** |
| mean MAE | **1.143e-3** |
| max delta | **9.734e-3** (`r22`) |

Per kind:

| kind | rows | top-1 | mean KL | mean MAE | max delta |
|---|---|---|---|---|---|
| choice | 15 | 1.0 | 1.437e-5 | 3.876e-4 | 4.434e-3 |
| noul | 8 | 1.0 | 4.030e-5 | 2.081e-3 | 9.734e-3 |
| score | 5 | 1.0 | 1.436e-4 | 1.908e-3 | 6.767e-3 |

The baseline was collected on the previous serving stack, so these deltas are cross-engine
numerical drift (consistent with the ~1% BF16 rel_rms above), not API differences. The
acceptance run uses `max_mae 0.0025`, `max_delta 0.01`; the strict `max_mae 0.0005` gate is
recorded but expected to fail.

Live endpoint verification (same release binary):

- `/v1/decide` noul/score/choice: 200 with normalized probabilities.
- `/v1/decisions`: predicate + choice + score, 200.
- `/v1/systemone`: noul + choice, 200.
- 256-option choice: 200 in 4.56 s.
- Image state: 400 `native vision not implemented`.
- Generation routes: 501 `not_implemented_error`.
- HMAC signing: `x-jev-signature` matches the exact body; `/health` unsigned.
- `/metrics`: HTTP and backend counters/histograms exposed.
- Preload: 5.94 s, 10.86 GiB device free after loading 64 layers.

## Performance

Resident native engine on GPU 1 (`--preload`, 12.66 GiB used / 10.86 GiB free), measured
in `native/README.md`:

| prompt | reference path (dequant + cuBLAS) | fused `--fused-w4a16` |
|---|---|---|
| 36 tokens | 176.6 ms (first 267 ms) | 405 ms |
| 76 tokens | 202.9 ms | 460 ms |
| 1024 tokens | 1277.6 ms | 2689 ms |

Preload 5.8 s, init 0.2 s. GPU sample during the 36-token reference run: 100%
utilization, ~231 W peak; the fused kernel is latency-bound at ~156 W. Resident outputs are
bit-identical to the streaming (`--no-cache`) run.

The fused W4A16 kernel is numerically correct but slower than the reference path on this
workload (kernel-only averages: gate_proj 0.67 vs 0.63 ms, in_proj_qkv 1.20 vs 0.80 ms,
out_proj 0.55 vs 0.19 ms, down_proj 1.54 vs 0.63 ms, in_proj_a/b 0.40 vs 0.014 ms). It
stays off by default.

API-level throughput and latency under concurrency are **not yet measured**. Do not reuse
the previous vLLM-front numbers; the backend and scheduling model changed. Record real
numbers with:

```bash
./scripts/run-eval.sh
```

The benchmark writes `eval/out/bench_combined_rust.json`; reports land in `eval/out/`.
Wide strategies perform multiple serialized forwards per request (16 group + 1 final for a
256-option tournament, 4 for permute), so their API latency scales with pass count.
