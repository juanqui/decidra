# Correctness and performance results

All results below are for the native backend. Previous serving-stack measurements are
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

### Live server vs CPU AWQ golden (primary)

Measured with the release binary on GPU 1 (resident weights, hybrid fused policy
`JEV_NATIVE_FUSED_W4A16=true`, the server default) against
`eval/data/awq-cpu-suite28.json`, 28 rows:

| metric | value |
|---|---|
| rows compared | 28/28, 0 errors |
| top-1 agreement | **1.0 (28/28)** |
| mean KL | **3.055e-5** |
| mean MAE | **8.452e-4** |
| max delta | **1.0345e-2** (`r20`, choice) |

Per kind:

| kind | rows | top-1 | mean KL | mean MAE | max delta |
|---|---|---|---|---|---|
| choice | 15 | 1.0 | 2.590e-5 | 5.924e-4 | 1.0345e-2 |
| noul | 8 | 1.0 | 8.925e-6 | 8.831e-4 | 3.194e-3 |
| score | 5 | 1.0 | 7.885e-5 | 1.543e-3 | 8.081e-3 |

Acceptance gates: `top1 1.0`, `max_kl 0.001`, `mean_mae 0.001`, `max_delta 0.011`.
**PASS.** The result is at BF16 noise; the largest row is 1.0345e-2 against a 1.1e-2
delta gate. The strict `mean_mae 0.0005` / `max_delta 0.001` run is recorded separately
and fails as expected.

### Historical production-v2 baseline (context only)

Against `eval/data/awq-v2-suite.json`: top-1 **28/28**, mean KL **8.070e-5**, mean MAE
**1.640e-3**, max delta **1.4533e-2** (`r22`). This baseline is not an acceptance gate:
it was collected on a vLLM-based stack and itself differs from CPU AWQ golden by max
**0.01377** / mean MAE **0.00156**. The CPU golden is the correct reference.

### Live endpoint verification

- `/v1/decide` noul/score/choice: 200 with normalized probabilities.
- `/v1/decisions`: predicate + choice + score, 200.
- `/v1/systemone`: noul + choice, 200.
- 256-option choice: 200 in 3.48 s.
- Image state: 400 `native vision not implemented`.
- Generation routes: 501 `not_implemented_error`.
- HMAC signing: `x-jev-signature` matches the exact body; `/health` unsigned.
- `/metrics`: HTTP and backend counters/histograms exposed.
- Preload: 5.88 s, 10.86 GiB device free after loading 64 layers.
- Short benchmark (16 unique requests/level, hybrid default): c=1 **8.440 req/s**, p50
  **123.7 ms**; c=4 **8.484 req/s**, p50 **459.3 ms**, ~**587 tok/s**. Throughput does
  not scale with concurrency because the native engine serializes forwards; c=4 latency
  is queueing, not batching.

## Performance

Resident native engine on GPU 1 (`--preload`, 12.66 GiB used / 10.86 GiB free), measured
in `native/README.md`:

| prompt | reference path (dequant + cuBLAS) | hybrid fused (`--fused-w4a16`) |
|---|---|---|
| 36 tokens | 174.7 ms (first 259 ms) | **84.8 ms** (2.06x) |
| 76 tokens | 202.2 ms | **126.4 ms** (1.60x) |
| 1024 tokens | 1261.2 ms | **1261.2 ms** (falls back, 1.00x) |

Preload 5.8 s, init 0.2 s. GPU sample during the 36-token hybrid run: 100% utilization,
~184 W mean / 213 W peak. Resident outputs are bit-identical to the streaming
(`--no-cache`) run. Golden after residency (hybrid): final norm rel_rms 1.07% (reference
1.25%), full logits rel_rms 0.24%, top-1 identical.

The fused policy is `N >= 5120 && M <= 256`; otherwise the engine falls back to
dequant + cuBLAS. This is why hybrid gives 2.06x at 36 tokens and 1.60x at 76 tokens but
**no benefit at 1K tokens** (1261.2 ms both paths). Small-N projections (`k/v/a/b`) and
large prefills are faster on the reference path. The server defaults to the hybrid policy
(`JEV_NATIVE_FUSED_W4A16=true`); set it to `false` to force the reference path.

API-level behavior: short benchmark on the live server (16 unique requests per level,
hybrid default) measured c=1 **8.440 req/s**, p50 **123.7 ms**; c=4 **8.484 req/s**,
p50 **459.3 ms**, ~**587 tok/s**. Throughput is flat because the engine serializes
forwards; c=4 latency is queueing, not batching. Do not claim concurrency scaling. Do not
reuse the previous serving-stack numbers. Record fresh numbers with:

```bash
./scripts/run-eval.sh
```

The benchmark writes `eval/out/bench_combined_rust.json`; reports land in `eval/out/`.
Wide strategies perform multiple serialized forwards per request (16 group + 1 final for a
256-option tournament, 4 for permute), so their API latency scales with pass count.
