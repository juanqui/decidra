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

## Final shipped performance (mixed W4A8-INT8)

Default config: `mlp.gate_proj` uses per-token INT8 scales; `in_proj_z/b/a`, `up_proj`,
`down_proj`, `in_proj_qkv`, `out_proj`, `q_proj` use group-64 INT8 scales; `o_proj`,
`v_proj`, `k_proj` stay W4A16 outside layers 56-63; layers 56-63 are all-linear INT8.
One resident server, exact-decision cache disabled.

Quality (`eval/suite.py` vs CPU AWQ golden): **28/28, KL 3.269e-5, MAE 7.166e-4,
max Δ 1.027e-2** — all strict gates pass, deterministic across runs. The group
epilogue uses a fused multiply-add, so the last-bit ordering differs from the
pre-FMA kernel; the gate is the contract.

| short, 16 unique ~73-token prompts | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| req/s | 25.78 | 40.73 | 53.10 | 56.97 | 56.93 |
| p50 | 39.2 ms | 49.1 ms | 74.4 ms | 123.2 ms | 173.1 ms |
| GPU util | 95% | 95% | 96% | 96% | 97% |

| long single request | 4K | 8K | 16K |
|---|---:|---:|---:|
| latency | **0.833 s** | **1.711 s** | **3.740 s** |
| tok/s | 4,908 | 4,779 | 4,380 |
| GPU util / power | 100% / 384-428 W | 100% / 422 W | 100% / 430 W |

Resident memory is **18.83 GiB** (13.1 GiB weights + 2.37 GiB embedding table + INT8
scratch) and 16K peaks at **20.11 GiB**, flat across repeated requests. Engine-only single forward (cache off):
73 tok ~48 ms mean / 37.5 ms min, 4K 834 ms, 16K 3,689 ms. See
[`VLLM-COMPARISON.md`](VLLM-COMPARISON.md) for the same-GPU vLLM comparison.

## Server/API evidence

- `cargo test`: 47 unit tests and 10 integration tests pass against `MockBackend`. These
  cover exact prompt bytes, tournament/permute pass counts, `/v1/decisions`,
  `/v1/systemone`, HMAC signing, OpenAI-compatible `/v1/chat/completions` and
  `/v1/completions`, and 400/502/504 semantics.
- `cargo check --all-targets --features native` compiles the server against the public
  `decidra_native::Engine`, `preload_all`, `set_marlin`, `set_cache_weights`,
  `set_fused_w4a16`, and `decision_logits` API.
- `--check-config` validates the production tokenizer/head labels without loading the GPU.
- `scripts/native-smoke.sh` runs one real forward and asserts two finite head logits.

### Live server vs CPU AWQ golden (primary)

Measured with the release binary on GPU 1 (resident weights, Marlin W4A16 default
`DECIDRA_NATIVE_MARLIN=true`) against `eval/data/awq-cpu-suite28.json`, 28 rows:

| metric | value |
|---|---|
| rows compared | 28/28, 0 errors |
| top-1 agreement | **1.0 (28/28)** |
| mean KL | **3.281e-5** |
| mean MAE | **7.365e-4** |
| max delta | **1.010e-2** (`r17`, score) |

Per kind:

| kind | rows | top-1 | mean KL | mean MAE | max delta |
|---|---|---|---|---|---|
| choice | 15 | 1.0 | 1.803e-5 | 2.350e-4 | 5.121e-3 |
| noul | 8 | 1.0 | 1.752e-5 | 1.237e-3 | 6.368e-3 |
| score | 5 | 1.0 | 1.013e-4 | 1.441e-3 | 1.010e-2 |

Acceptance gates: `top1 1.0`, `max_kl 0.001`, `mean_mae 0.001`, `max_delta 0.011`.
**PASS.** The gated split-GDN policy keeps short-prompt numerics on the accurate shared
recurrence, so `r22` returns to a 6.43e-3 two-way delta; the suite maximum is now
1.010e-2 (`r17`, score).

### Historical production-v2 baseline (context only)

`eval/data/awq-v2-suite.json` was collected on a vLLM-based stack. It is not an
acceptance gate: that baseline itself differs from CPU AWQ golden by max **0.01377** /
mean MAE **0.00156**. The CPU golden is the correct reference for the native engine. Its
measurement files are kept only for drift context.

### Live endpoint verification

- `/v1/decide` noul/score/choice: 200 with normalized probabilities.
- `/v1/decisions`: predicate + choice + score, 200.
- `/v1/systemone`: noul + choice, 200.
- 256-option choice: 200 in 1.36 s.
- Image state: 400 `native vision not implemented`.
- OpenAI-compatible routes: `/v1/chat/completions` and `/v1/completions` parse a JSON
  decision request from the user text, run the native decision, and return the compact
  decision JSON in the OpenAI shape (SSE when `stream: true`). Non-JSON user text is 400.
- HMAC signing: `x-jev-signature` matches the exact body; `/health` unsigned.
- `/metrics`: HTTP and backend counters/histograms exposed.
- Preload with Marlin: 10.1 s clean, 13.1 GiB resident (10.06 s observed in the gated
  run); 13,358 MiB in-process.
- Long context, single request (exceeds the 16,384-token batch cap, so it runs alone),
  chunked GDN on (default): **4,032 tokens -> 200 in 1.63 s**, **8,029 -> 3.27 s**,
  **16,024 -> 6.77-6.80 s** (5 consecutive runs), **35,310 -> 16.67 s** (3 consecutive
  runs); all choice `true` (0.9894 at 16K, 0.9884 at 35.3K). The verified maximum is
  **35,310 tokens** on the 24 GB GPU. The 10-row long-context suite is **10/10 top-1**
  (including choice rows `billing` and `301`) at ~1.2-1.6 s for 3-4K and 3.26 s for 8K.
  Short-request latency is unchanged because those prompts stay on the shared/split
  path: c=1 **16.05 req/s / 60.9 ms**, c=4 **30.88 / 130.5 ms**, c=8
  **30.35-31.41 / 218.9-254.3 ms**. One 16K probe returned a single 502 during the
  first long-context run; it did not reproduce in seven subsequent 16K attempts or the
  five prior 17.7K runs, and the root cause was not captured
  (`decidra_backend_requests_total{status="error"} 1`).
- Short benchmark, serialized build (16 unique requests/level, gated): c=1
  **16.444 req/s**, p50 **60.9 ms**; c=4 **16.826 req/s**, p50 **225.7 ms**. Throughput
  was flat because forwards were serialized; c=4 latency was queueing.
- Short benchmark, batched build (16 unique requests of ~71 prompt tokens/level;
  `scripts/bench-batch.sh`, median of 3 runs): c=1 **15.84 req/s**, p50 **61.9 ms**,
  p95 **75.8 ms**, ~**1,122 tok/s**; c=2 **23.38 req/s**, p50 **83.2 ms**, p95
  **98.7 ms**, ~**1,656 tok/s**; c=4 **30.13 req/s**, p50 **133.7 ms**, p95
  **139.1 ms**, ~**2,133 tok/s**; c=8 **29.25-30.26 req/s**, p50 **233-256 ms**,
  ~**2,051-2,141 tok/s**; c=16 **29.28-30.47 req/s**, p50 **388-442 ms**, p95
  **520-539 ms**, ~**2,107-2,192 tok/s**. Versus the serialized build: c=4
  **1.79x throughput** and p50 **225.7 -> 133.7 ms**; versus c=1 on this build, c=4 is
  **1.90x**. c=1 is unchanged within run-to-run noise. Throughput saturates near
  **30 req/s and 2,100-2,150 tok/s at batch 4**; extra concurrency only queues.
- Tuning matrix (same 16x71-token sweep): `WINDOW_US` 2000/5000/10000 with `MAX_BATCH`
  8 gave c=8 **29.25-30.26 / 30.41-30.59 / 30.47-30.56 req/s** and c=1 p50
  **61.9 / 64.9 / 69.9 ms**. The 5/10 ms windows fill 8-wide batches at c=8 but do not
  raise throughput, and they add the window to a lone request; 2 ms is kept because it
  keeps c=1 p50 <=65 ms. `MAX_BATCH` 12/16 at the 2 ms window gave c=16 p50 **502-554 ms**
  with **27.4-29.6 req/s**, no gain over 8.
- Batch-of-4 vs batch-of-8: at c=8, `MAX_BATCH=4` did **30.83 req/s / 2,182 tok/s**
  (p50 255 ms) versus **29.25-30.26 req/s / 2,051-2,141 tok/s** for 8; at c=16 it did
  **30.16 req/s / 2,169 tok/s** (p50 324 ms) versus 29.3-30.5 req/s (p50 388-442 ms).
  A batch of 8 is therefore **not** more efficient per token than 4 in this sweep;
  `MAX_BATCH=8` is retained for burst draining, and the saturation point is batch 4.
- Batch correctness: batch-of-one matches the CPU golden within the strict gate
  (max **1.027e-2**, MAE **7.17e-4**); the fused-multiply-add group epilogue
  changes last-bit ordering versus the pre-FMA kernel, so byte-identity is not
  claimed. Batch>1 has deterministic BF16/M-tiling composition noise (top-1
  stable). With the shipped mixed W4A8-INT8 config, the 28-row suite at **c8**
  vs the CPU golden is top-1 **28/28**, MAE **7.23e-4**, max **1.305e-2**; at
  **c16** it is 28/28, MAE **1.115e-3**, max **1.251e-2**. The strict gate is
  defined at batch=1 (MAE **7.17e-4**, max **1.027e-2**). W4A16 alone measured
  worse batched deltas (max **1.78e-2**), so the batched spread is the shared
  batch GEMM path, not the INT8 selection.
  Two concurrent 4K requests differ from their single-run probabilities by at most
  **4.2e-5** with identical top-1.
- Batch metrics observed during the sweep (`decidra_native_batch_size` mean per level):
  c=1 **1.0**, c=2 **2.0**, c=4 **4.0**, c=8 **5.33**, c=16 **5.33-8.0**;
  `decidra_native_batch_queue_depth` peaked at **8** at c=16.

## Exact-decision cache

`DECIDRA_DECISION_CACHE_SIZE` (default **256**, `0` disables) adds a thread-safe LRU in
`src/cache.rs` keyed by the exact `(token ids, head ids)` pair and holding raw head
logits. `NativeBackend::logits` checks it after tokenization and before enqueueing;
successful, length-checked forwards insert, errors are never cached, and hits return a
clone without touching the engine.

- **Unique prompts unchanged**: suite28 PASS 28/28, KL 3.281e-5, MAE 7.365e-4, max
  delta 1.010e-2; unique c=1/c=4/c=8 measured **16.04 req/s / 61.1 ms**,
  **30.78 / 131.4 ms**, **30.26 / 225.8 ms**.
- **Repeated exact prompt, c=1, 16 requests**: **0.083 s wall, 193 req/s, p50 0.37 ms**
  (first miss 76.9 ms) versus 16.04 req/s / 61.1 ms uncached.
- **Repeated exact prompt, c=8, 16 concurrent**: cold round **72.4 req/s, p50 70.9 ms,
  p95 216.5 ms** (the first concurrent wave misses); warm rounds **1,560-2,000 req/s,
  p50 2.5-3.1 ms** versus 30.26 req/s / 225.8 ms uncached.
- Cache metrics: `decidra_decision_cache_hits_total`, `decidra_decision_cache_misses_total`,
  `decidra_decision_cache_entries` (81 entries after the validation run). Memory is
  negligible next to the 13.8 GiB process RSS: a 70-token entry is a few hundred bytes,
  and the absolute worst case (256 x 35K-token prompts) is ~36 MB.
- **Limitation**: exact prompt+ids only. Shared-prefix reuse (KV cache plus GDN state
  and conv tail) remains future work and is not provided by this cache.

## Chunked GDN prefill

`DECIDRA_NATIVE_GDN_CHUNKED=true` (default) selects the TF32 WMMA chunked delta-rule
prefill (native commit `7c78a3f`) for sequences **>= 64 tokens**
(`DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ`, default 64); shorter prompts keep the
shared/split recurrence, and `false` restores the legacy path everywhere.

- GDN stage **2.1x** faster; full forward **1K 488 -> 395 ms (-17.7%)** and
  **16K 8.80 -> 6.94 s (-16.4%)** in the native benchmark.
- Short prompts now use it too: suite28 **28/28**, KL 3.278e-5, MAE 7.362e-4,
  max delta **1.010e-2** (`r17`, a 57-token row on the shared fallback); the
  chunked rows are all below that (`r22` 6.37e-3). 36-token golden head logits
  top-1 36/36, max 0.125 (shared 0.109).
- Native forward at 76 tokens **69.5 -> 52.0 ms** (GDN 27.6 -> 7.27 ms per
  prompt); 128 tokens **97.8 -> 64.9 ms**.
- The optimized chunked kernel (vectorized loads, barrier-free 4-column solve,
  4x4 register-tiled state update) cuts GDN **1K 75.3 -> 30.6 ms** and
  **16K 1200 -> 485 ms (2.47x)**, full forward **1K 400 -> 355 ms** and
  **16K 8.26 -> 7.53 s** (same-session A/B, old chunked kernel).
- Batched chunked GDN (`gdn_chunked_tc_batched`, grid `z=batch`, one launch
  for all sequences >= the threshold) then cuts the 8x73-token GDN recurrence
  **54.6 -> 19.6 ms** and the warm forward **247 -> 212 ms** (37.7 req/s
  native). Outputs are bit-identical to per-sequence launches.
- Server uncached sweep, 16 unique ~71-token requests: c=1 **22.8-22.95 req/s,
  p50 40.8-40.9 ms** (was 21.7 / 43.7); c=2 **28.4-28.6 / 69 ms**, c=4
  **32.7-32.8 / 120 ms**, c=8 **32.8-33.2 / 235-239 ms** (was 29.9-31.3),
  c=16 **33.6-33.7 / 356-357 ms** (was 31.2-31.5). The c8 >=32 and c16 >=33
  targets now pass. `MAX_BATCH=16` still makes c16 worse, so the default 8
  stays. The remaining 212 ms batch is MLP GEMMs already at tensor-core peak.
- suite28 **28/28**, KL 3.278e-5, MAE 7.362e-4, max delta 1.010e-2; suite
  chunked outputs are bit-identical to the previous chunked kernel.
- long-context suite 10/10; the 4K/8K/16K probes are bit-identical to the
  previous chunked kernel. Server single-request long-context latency improves
  by the GDN delta (previous measurements: 4K 1.63 s, 8K 3.27 s, 16K 6.77 s,
  35.3K 16.67 s).

## FlashInfer long-context attention

Single-request full-attention prefill for `seq >= 1024` uses vendored
FlashInfer 0.7.0.post1 (`native/third_party/flashinfer`, built into
`libfi_prefill.so`); batched requests and shorter prompts keep the native
tiled/warp kernels. Attention per forward, FI off -> on (same weights, clean
env): 1K 3.98 -> 2.21 ms; 4K 40.5 -> 22.6; 8K 148.5 -> 82.2; 16K 567 -> 315 ms
(1.80x). Isolated W4A16 integration A/B: full forward 1K 522.07 -> 517.13 ms,
4K 2045.37 -> 2026.26, 8K 4135.33 -> 4068.82, 16K 8557.68 -> 8263.24 ms
(-294 ms, -3.4%). At 16K this is 0.294 s saved and satisfies the 0.2 s target.

Quality: suite28 passes. Current default
(mixed per-module W4A8-INT8 + layers 56-63 all-linear + group-64 scales + FMA
epilogue, full-M) at batch=1: 28/28, KL 3.269e-5, MAE **7.166e-4**, max **1.027e-2**
— all strict gates pass.
4K/8K/16K probe logits: FI on vs
golden [-0.0469,+0.0313], [+0.0156,-0.0313], [+0.0156,0.0], within the
W4A16 envelope with unchanged top-1. Disable with
`DECIDRA_FLASHINFER_ATTENTION=0`; `DECIDRA_FLASHINFER_MIN_SEQ` sets the threshold.

## Marlin full-M launches

The W4A16 and INT8 Marlin paths now launch the whole M once by default
(`DECIDRA_MARLIN_CHUNK=0`, the kernel's internal `parallel` loop) instead of
splitting into <=1024-row launches; `DECIDRA_MARLIN_CHUNK=N` restores a cap. At
16K this is 2 launches instead of 16 (16320 + tail): full forward 5991 ->
5935 ms (-0.9%), 4K 1432 -> 1425 ms (-0.5%), INT8 16K 5184 -> 5133 ms
(-1.0%), 1K unchanged. suite28 is bit-identical and passes; 4K/8K/16K probe
deltas vs golden are [-0.0469,0], [+0.0156,+0.0313], [0,-0.0313] with INT8,
inside the existing envelope and with unchanged top-1.

## Marlin layer-wise INT8 and tile probes

Layer subsets that run all-linear INT8 on top of the selective bypass
(`DECIDRA_NATIVE_INT8_ALL_LAYERS`): `last:8` (layers 56-63) is the largest passing
set (28/28, KL 2.37e-5, MAE 6.91e-4, max 7.48e-3); `first:N` and `alt:N` fail
for N=8..32, and `last:10/12/14/16/24/32` fail. Held-out 500 rows: 481/500
top-1 vs the selective config, all flips being <=0.008-margin ties. The speed
gain is under 0.1% (16K 5181.9 -> 5177.7 ms), so it stays opt-in.

Marlin tile probes: the engine already runs BM=64/BN=256/stages=4 with
101,376 B of dynamic shared memory (the exact limit, no spill), so the
vendor's BM=48 fallback does not apply here. New BM=64/BN=256/stages=3
(84,480 B) and BM=64/BN=128/stages=3 (55,808 B) instantiations measured
16.53/17.09 ms for up_proj and 16.47/16.35 ms for down_proj at 16K, versus
16.61/16.50 for the current config: neutral to worse, reverted. `q_proj` is
N=12288 (48 N tiles, 768 tiles over 128 blocks) and measures 11.84 ms/op =
174 TFLOP/s, so its grid is not idle; `in_proj_z` is the N=6144 INT8 module.

## Request batching

The native backend runs one engine thread. HTTP handlers tokenize on the reactor,
enqueue `{tokens, head_ids, oneshot reply}`, and await the reply; the worker owns the
CUDA context. Batch formation:

- block for the first request;
- drain more until the window (`DECIDRA_NATIVE_BATCH_WINDOW_US`, default **2000 µs**),
  `DECIDRA_NATIVE_MAX_BATCH` (default **8**), or `DECIDRA_NATIVE_MAX_BATCH_TOKENS`
  (default **16384**) is reached;
- strict FIFO; a request never spans batches; a request that does not fit is pushed
  back to the head for the next batch;
- a single request larger than the token cap (e.g. 16K/32K) runs alone;
- closed client receivers are skipped; executor errors are returned per request;
- `DECIDRA_NATIVE_BATCH=false` sets max batch 1 and window 0.

Lone requests pay the window; `DECIDRA_NATIVE_BATCH_WINDOW_US=0` removes it. The default
2 ms window is the smallest that keeps a lone request near 62 ms p50. The worker
executes one batch at a time; there is no separate in-flight semaphore, so extra
concurrency only queues. Throughput saturates at batch 4; `MAX_BATCH=8` allows a burst
to drain in one pass, while 12/16 measured no gain and worse p50.

Metrics: `decidra_native_batch_size`, `decidra_native_batch_queue_depth`,
`decidra_native_batch_wait_seconds`, `decidra_decision_cache_hits_total`,
`decidra_decision_cache_misses_total`, `decidra_decision_cache_entries`, plus the existing
`decidra_backend_*` counters.

## Performance

For a controlled same-GPU comparison against vLLM 0.31.0 (short concurrency sweep,
long prefill, repeated exact prompts, startup, memory, KV cache, kernel stacks, and
quantization probes), see [`VLLM-COMPARISON.md`](VLLM-COMPARISON.md). Final summary:
with 73-token prompts the shipped native engine wins **every** concurrency:
**25.78 / 40.73 / 53.10 / 56.97 / 56.93 req/s** at c1/2/4/8/16 (median of 3 runs)
versus vLLM **18.81 / 20.24 / 29.32 / 34.52 / 35.17**, and cold long prefill by
**32-36%** (4K 0.833 vs 1.302 s, 8K 1.711 vs 2.660, 16K 3.740 vs 5.533;
vLLM's earlier 5.052 s 16K number was nested-prefix-cache-assisted). Native
still wins repeated exact prompts (exact cache), startup (**~9.2 s vs 145 s**),
GPU memory (**18.83 GiB resident / 20.11 GiB at 16K vs 21.8 GiB**), and verified
35,310-token context. W4A8-INT8 uses per-module scale granularity: `gate_proj` is
per-token, the expanded set is group-64, and the residual attention writers stay
W4A16. Current server suite at batch=1 is 28/28, KL 3.269e-5, **MAE 7.166e-4**,
max 1.027e-2 — every strict gate passes. All-linear per-token INT8 fails
(max 0.0482, top-1 0.9643); all-linear group-64 fails only max (0.01349) and is
not the default. FP8 remains off. For the ranked, gated plan to close the
remaining gaps, see [`OPTIMIZATION-PROPOSAL.md`](OPTIMIZATION-PROPOSAL.md).

Resident native engine on GPU 1 (`--preload`, Marlin repacked layout 13.1 GiB), measured
in `native/README.md`:

| prompt | reference (dequant + cuBLAS) | hybrid fused fallback | Marlin (default) |
|---|---|---|---|
| 36 tokens | 174.7 ms | 84.8 ms | **45.1 ms** (3.87x) |
| 76 tokens | 202.2 ms | 126.4 ms | **73.0 ms** (2.77x) |
| 1024 tokens | 1261.2 ms | 1261.2 ms | **488 ms** (2.58x) |

Marlin preload/repack is 10.1 s and uses 13.1 GiB; the fallback paths preload in 5.8 s
and use ~12.7 GiB. Init 0.2 s. At 36 tokens Marlin runs at 100% GPU utilization. Marlin
golden checks: layer 0 mean 1.58e-4 / rel_rms 0.347%; final norm rel_rms 1.23%,
cosine 0.999924.

Long-context latency (Marlin resident, mixed W4A8-INT8, chunked GDN, FlashInfer):

| tokens | engine CLI (warm) | resident server | GPU util | peak GPU mem |
|---|---:|---:|---:|---:|
| 4K (4,090) | **0.834 s** | **0.833 s** | 100% | 18.83 GiB |
| 8K (8,178) | — | **1.711 s** | 100% | 20.11 GiB |
| 16K (16,383) | **3.689 s** | **3.740 s** | 100% | 20.11 GiB |
| 32K (32,763) | not rerun | not rerun | — | — |

The pre-chunked W4A16 stack measured 2.12 s (4K), 4.23-4.32 s (8K), 8.62-8.97 s (16K)
and 18.60-19.03 s (32K); the current stack is ~2.2-2.3x faster. It is **GEMM-bound** at
~0.22-0.24 ms/token; attention is <1% at 1K and still small at 16K. Long prefill holds
100% GPU utilization at ~420-430 W.

Split GDN is gated to **seq >= 512 tokens** (`gdn_split_min_seq`). Shorter prompts use
the accurate shared recurrence because the four-way f32 reduction reorders the state dot
products enough to shift a rare short-prompt logit: on `r22` (54 tokens) split measured a
two-way delta of 1.43e-2 (fail) versus shared 6.43e-3 (pass). Long sequences keep the
split path and pass.

For **seq >= 12,288** a low-memory buffer-pool mode releases dropped large tensors and
frees large MLP/attention/GDN intermediates as soon as they are consumed. This keeps
repeated 16K requests flat at **20.11 GiB** (four consecutive runs, 3.84 s each) and makes
**32K fit on 24 GB** (19.09 GiB peak, ~4.4 GiB headroom); 1K/8K timings are unchanged.

Long-context quality gate against CPU AWQ/SDPA probes (Marlin + tiled FA2 + split GDN):

| probe | final rel_rms | cosine | two-way prob delta | top-1 |
|---|---|---|---|---|
| 4K | 0.808% | 0.999967 | 3.5e-4 | true |
| 8K | 0.795% | 0.999968 | 3.0e-4 | true |
| 16K | 0.692% | 0.999976 | 6.7e-4 | true |
| 32K | 0.727% | 0.999974 | 6.0e-4 | true |

All **10/10** long-context suite cases are correct; the 32K probe is the verified maximum
context on 24 GB (expected `true`, golden P(true)=0.98315). The 4K/8K/16K/32K rows are
BF16 accumulation noise.

At 1K the gain over the previous 1144.9 ms path is 2.58x after the GDN work (Marlin alone
was 1.32x). Two large-M Marlin tile variants were A/B'd at 1K (982.3 / 997.1 vs 984.3 ms)
with bit-identical outputs, so the default 256-thread config stays.

API-level behavior: short benchmark on the live gated server (16 unique requests per
level) measured c=1 **16.444 req/s**, p50 **60.9 ms**; c=4 **16.826 req/s**, p50
**225.7 ms**, ~**1,143-1,169 tok/s**. Throughput is flat because the engine serializes
forwards; c=4 latency is queueing, not batching. Do not claim concurrency scaling. Do not
reuse the previous serving-stack numbers. Record fresh numbers with:

```bash
./scripts/run-eval.sh
```

The benchmark writes `eval/out/bench_combined_rust.json`; reports land in `eval/out/`.
Wide strategies perform multiple serialized forwards per request (16 group + 1 final for a
256-option tournament, 4 for permute), so their API latency scales with pass count.
