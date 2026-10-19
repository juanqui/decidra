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
  `jev_native::Engine`, `preload_all`, `set_marlin`, `set_cache_weights`,
  `set_fused_w4a16`, and `decision_logits` API.
- `--check-config` validates the production tokenizer/head labels without loading the GPU.
- `scripts/native-smoke.sh` runs one real forward and asserts two finite head logits.

### Live server vs CPU AWQ golden (primary)

Measured with the release binary on GPU 1 (resident weights, Marlin W4A16 default
`JEV_NATIVE_MARLIN=true`) against `eval/data/awq-cpu-suite28.json`, 28 rows:

| metric | value |
|---|---|
| rows compared | 28/28, 0 errors |
| top-1 agreement | **1.0 (28/28)** |
| mean KL | **3.597e-5** |
| mean MAE | **1.0418e-3** |
| max delta | **1.4213e-2** (`r22`, noul) |

Per kind:

| kind | rows | top-1 | mean KL | mean MAE | max delta |
|---|---|---|---|---|---|
| choice | 15 | 1.0 | 1.160e-5 | 2.956e-4 | 3.418e-3 |
| noul | 8 | 1.0 | 6.751e-5 | 2.280e-3 | 1.4213e-2 |
| score | 5 | 1.0 | 5.795e-5 | 1.300e-3 | 6.663e-3 |

Acceptance gates: `top1 1.0`, `max_kl 0.001`, `mean_mae 0.001`, `max_delta 0.011`.
Top-1 and KL pass. Mean MAE misses by **4.2e-5** and max delta by **3.2e-3**, so the
final run is a **FAIL** against the stated gates. Row `r22` is a two-way noul shift
(0.7191 vs 0.7049 on `false`) with the same top-1; this is the largest head-logit
difference in the suite. The strict `mean_mae 0.0005` / `max_delta 0.001` run also fails.

### Historical production-v2 baseline (context only)

`eval/data/awq-v2-suite.json` was collected on a vLLM-based stack. It is not an
acceptance gate: that baseline itself differs from CPU AWQ golden by max **0.01377** /
mean MAE **0.00156**. The CPU golden is the correct reference for the native engine. Its
measurement files are kept only for drift context.

### Live endpoint verification

- `/v1/decide` noul/score/choice: 200 with normalized probabilities.
- `/v1/decisions`: predicate + choice + score, 200.
- `/v1/systemone`: noul + choice, 200.
- 256-option choice: 200 in 1.56 s.
- Image state: 400 `native vision not implemented`.
- Generation routes: 501 `not_implemented_error`.
- HMAC signing: `x-jev-signature` matches the exact body; `/health` unsigned.
- `/metrics`: HTTP and backend counters/histograms exposed.
- Preload with Marlin: 10.1 s clean, 13.1 GiB resident (10.99 s observed in the final
  run); 13,358 MiB in-process.
- 16,444-token `/v1/decide`: **200 in 8.65 s**, choice `true`; no timeout/OOM.
- Short benchmark (16 unique requests/level, final build): c=1 **21.687 req/s**, p50
  **46.3 ms**, p95 **57.0 ms**, p99 **62.1 ms**, ~**1,507 tok/s**; c=4 **21.845 req/s**,
  p50 **173.9 ms**, p95 **201.3 ms**, ~**1,518 tok/s**. Throughput does not scale with
  concurrency because the native engine serializes forwards; c=4 latency is queueing,
  not batching.

## Performance

Resident native engine on GPU 1 (`--preload`, Marlin repacked layout 13.1 GiB), measured
in `native/README.md`:

| prompt | reference (dequant + cuBLAS) | hybrid fused fallback | Marlin (default) |
|---|---|---|---|
| 36 tokens | 174.7 ms | 84.8 ms | **45.1 ms** (3.87x) |
| 76 tokens | 202.2 ms | 126.4 ms | **73.0 ms** (2.77x) |
| 1024 tokens | 1261.2 ms | 1261.2 ms | **486.1 ms** (2.59x) |

Marlin preload/repack is 10.1 s and uses 13.1 GiB; the fallback paths preload in 5.8 s
and use ~12.7 GiB. Init 0.2 s. At 36 tokens Marlin runs at 100% GPU utilization. Marlin
golden checks: layer 0 mean 1.58e-4 / rel_rms 0.347%; final norm rel_rms 1.23%,
cosine 0.999924.

Long-context latency (Marlin resident, tiled FA2 + split-dk GDN, default):

| tokens | full forward | previous warp | peak GPU mem |
|---|---|---|---|
| 1K | **0.486 s** | 0.959 s | 13.1 GiB resident |
| 4K | **2.12-2.18 s** | 4.65 s | ~17.2 GiB |
| 8K | **4.23-4.32 s** | 9.88 s | ~17.6 GiB |
| 16K | **8.62-8.85 s** | 22.88 s | 18.86 GiB |

The standalone attention kernel is 7-12x faster than warp on identical buffers
(1K **0.276 ms**, 4K **2.85 ms**, 8K **9.68 ms**). The split-dk GDN recurrence is 3.9x
faster on its hot kernel: 1K per-layer **13.35 -> 3.41 ms**, 1K forward **958 -> 486 ms**.
The full forward is now **GEMM-bound** at ~0.49-0.55 ms/token; attention is <1% at 1K and
still small at 16K. GPU utilization is 100% at ~455 W.

Long-context quality gate against CPU AWQ/SDPA probes (Marlin + tiled FA2 + split GDN):

| probe | final rel_rms | cosine | two-way prob delta | top-1 |
|---|---|---|---|---|
| 4K | 0.808% | 0.999967 | 3.5e-4 | true |
| 8K | 0.795% | 0.999968 | 3.0e-4 | true |
| 16K | 0.692% | 0.999976 | 6.7e-4 | true |

All **10/10** long-context suite cases are correct. Deltas are BF16 accumulation noise.

At 1K the gain over the previous 1144.9 ms path is 2.59x after the GDN work (Marlin alone
was 1.32x). Two large-M Marlin tile variants were A/B'd at 1K (982.3 / 997.1 vs 984.3 ms)
with bit-identical outputs, so the default 256-thread config stays.

API-level behavior: short benchmark on the live final server (16 unique requests per
level) measured c=1 **21.687 req/s**, p50 **46.3 ms**; c=4 **21.845 req/s**, p50
**173.9 ms**, ~**1,507-1,518 tok/s**. Throughput is flat because the engine serializes
forwards; c=4 latency is queueing, not batching. Do not claim concurrency scaling. Do not
reuse the previous serving-stack numbers. Record fresh numbers with:

```bash
./scripts/run-eval.sh
```

The benchmark writes `eval/out/bench_combined_rust.json`; reports land in `eval/out/`.
Wide strategies perform multiple serialized forwards per request (16 group + 1 final for a
256-option tournament, 4 for permute), so their API latency scales with pass count.
