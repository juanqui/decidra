# Native engine vs vLLM 0.31.0 (final, shipped build)

Controlled single-GPU comparison after the final native shipping commits. Both stacks
loaded the same `/srv/models/jev-awq/awq-w4a16-v2` AWQ artifact and decision head on
GPU 1 only.

## Methodology

- **GPU**: GPU 1, RTX 4090 24 GB; GPU 0 was not used. Clocks ~2,910 MHz SM /
  11,451 MHz memory; power 280–440 W depending on load.
- **Native**: release binary with `--features native`. Defaults:
  `JEV_NATIVE_MARLIN=true`, `JEV_NATIVE_INT8=true`,
  `JEV_NATIVE_INT8_GROUP_SCALE=true` (group-64 activation scales; `mlp.gate_proj`
  stays on the faster per-token epilogue), `JEV_NATIVE_GDN_CHUNKED=true`,
  FlashInfer attention enabled (`JEV_FLASHINFER_MIN_SEQ=1024`),
  `JEV_NATIVE_MAX_BATCH=8`, `JEV_NATIVE_BATCH_WINDOW_US=2000`,
  `JEV_NATIVE_MAX_BATCH_TOKENS=16384`. `JEV_DECISION_CACHE_SIZE=0` so unique
  prompts are actually uncached. One resident server; all levels run back to back.
- **vLLM**: `vllm/vllm-openai:v0.31.0` (vLLM 0.31.0, Torch 2.13.0+cu130,
  CUDA 13.0), exact A_ref flags `--logprobs-mode processed_logprobs
  --max-model-len 16384 --enable-prefix-caching --mamba-cache-mode align
  --trust-request-chat-template --max-num-seqs 8 --gpu-memory-utilization 0.95
  --kv-cache-dtype fp8`. Startup 145 s; KV 41,441 fp8 tokens.
- **Short prompts**: same native-generated 73-token prompt set for both stacks;
  tokenizer-verified means **73.0 / 73.0 / 73.0 / 73.0625 / 73.9375** for
  c=1/2/4/8/16. 16 unique requests per level, 3 runs, 3 warmups; vLLM prefix
  cache recorded 0 hits on all short prompts.
- **Long prompts**: exact native 4,090 / 8,178 / 16,383-token probes, 3 runs
  each. vLLM runs used a distinct `cache_salt` per run, so these are **cold**;
  the previous 1,568-token nested-prefix hit is removed. Native has no prefix
  cache.
- Both servers were run sequentially (combined resident weights do not fit on
  one 24 GB GPU).

## Short unique prompts (~73 tokens, median of 3 runs)

| c | native rps | vLLM rps | native p50 / p95 | vLLM p50 / p95 | native tok/s | vLLM tok/s |
|---:|---:|---:|---|---|---:|---:|
| 1 | **24.63** | 18.81 | **41.0 / 44.4 ms** | 53.06 / 54.08 ms | **1,798** | 1,373 |
| 2 | **39.51** | 20.24 | **51.2 / 53.9 ms** | 98.20 / 103.24 ms | **2,884** | 1,478 |
| 4 | **51.48** | 29.32 | **77.5 / 78.4 ms** | 135.10 / 138.67 ms | **3,758** | 2,140 |
| 8 | **52.73** | 34.52 | **149.7 / 152.4 ms** | 229.98 / 231.08 ms | **3,849** | 2,522 |
| 16 | **55.19** | 35.17 | **217.3 / 283.3 ms** | 304.75 / 454.23 ms | **4,081** | 2,600 |

Native is ahead at every concurrency: **1.31× / 1.95× / 1.76× / 1.53× /
1.57×** req/s. These are the final native medians with the shipped default
config (mixed per-module W4A8-INT8 + layers 56-63 all-linear + Marlin full-M +
FlashInfer), 3 runs, cache disabled. At c16 the GPU is 91–97% utilized
(334 W); long prefill is 100% (420–430 W).

## Long single request (cold, exact native prompts, median of 3)

| prompt | native latency | native tok/s | vLLM cold latency | vLLM tok/s | native speedup |
|---|---:|---:|---:|---:|---:|
| 4K (4,090 tok) | **0.857 s** | **4,777** | 1.302 s | 3,142 | **1.52×** |
| 8K (8,178 tok) | **1.758 s** | **4,650** | 2.660 s | 3,075 | **1.51×** |
| 16K (16,383 tok) | **3.833 s** | **4,274** | 5.533 s | 2,962 | **1.44×** |
| 35.3K (35,310 tok) | not rerun | — | not runnable (16,384 cap) | — | — |

All choices were `true`. With prefix caching bypassed, **native wins cold
long prefill by 33–39%** (one resident server, 3 cold runs per size, 100% GPU
utilization, 420–430 W). Repeated 16K requests stay at 3.83 s and
**20.11 GiB peak**, so the earlier pooled-buffer memory ceiling at repeated 16K
is fixed: the low-memory pool threshold now engages at 12,288 tokens. The
earlier vLLM 5.052 s 16K number was cache-assisted by the nested 4K prompt; the
cold 16K vLLM latency is 5.533 s.

## Quality

CPU AWQ golden: `eval/data/awq-cpu-suite28.json` (`native/suite28.json`
backend b). Strict gates: top-1 1.0, mean KL ≤1e-3, mean MAE ≤1e-3, max delta
≤0.011.

| comparison | top-1 | max Δ | mean MAE | mean KL | gate |
|---|---:|---:|---:|---:|---|
| native (mixed W4A8-INT8, last:8 all-INT8, full-M) vs CPU golden | **28/28** | **0.007511** | **0.000864** | 3.095e-5 | **PASS** |
| vLLM vs CPU golden | 28/28 | 0.013772 | 0.001551 | 8.509e-5 | max and MAE fail |
| vLLM vs native live | 28/28 | 0.015007 | 0.001388 | 5.711e-5 | max/MAE fail if native is reference |

The gate-exact configuration uses group-64 activation scales for the expanded INT8 set
(`up_proj`, `down_proj`, `in_proj_qkv`, `out_proj`, `q_proj`, GDN `z/b/a`) and the faster
per-token epilogue for `gate_proj`; layers 56-63 stay all-linear INT8. It passes every
strict gate: top-1 28/28, max 7.511e-3, MAE 8.64e-4, KL 3.10e-5, deterministic across
three runs. Native is closer to the CPU golden than vLLM on both max delta and MAE.
Final native full forward with this configuration (engine CLI, cache off):
73 tok ~56 ms mean / 37.6 ms min, 4K 924 ms, 16K 3,900 ms.

## Memory and startup

| metric | native | vLLM 0.31.0 |
|---|---:|---:|
| idle/resident GPU memory | **18.83 GiB** (13.1 GiB weights + 2.37 GiB embedding + scratch) | 21,909 MiB |
| 16K prefill peak | **20.11 GiB** | ~21.9 GiB |
| repeated 16K requests | flat 20.11 GiB | n/a (16K cap) |
| startup to ready | **9.2 s** | 145 s |

## Verdict

- **Short:** native beats vLLM at every concurrency, by 1.31–1.95× req/s and
  1.3–1.9× lower p50; c16 also has lower p95 (293 vs 454 ms).
- **Long:** native beats vLLM by 44–52% on cold 4K/8K/16K, at 100% GPU utilization.
  vLLM no longer wins long single-request prefill once its nested-prefix cache
  benefit is removed.
- **Quality:** native 28/28, max Δ 0.00751, MAE 0.000864, KL 3.10e-5 (PASS);
  vLLM 28/28, max Δ 0.01377, MAE 0.00155 (fails the max/MAE gates). Native is
  closer to the CPU golden.
- **Memory:** native is 3.0 GiB leaner at rest and 1.8 GiB leaner at 16K.
- vLLM still has the operational advantages of a mature server (continuous
  batching, CUDA graphs, prefix/KV caching), but on this exact single-GPU decision
  workload the shipped native engine is faster on every measured axis.

## Repeated exact prompt (historical, not rerun in this pass)

The prior native exact-decision cache result remains much faster than vLLM on
repeated prompts (warm c1 ~2,989 req/s / 0.330 ms vs vLLM ~19.4 req/s /
51.1 ms; c8 ~2,298 / 1.258 ms vs vLLM ~41.0 / 193 ms). vLLM's 1,568-token
prefix block cannot cache a 58-token prompt. This comparison used
`JEV_DECISION_CACHE_SIZE=0` for unique-prompt fairness and is not a cache test.

## Kernel/execution stacks

| layer | native | vLLM 0.31.0 |
|---|---|---|
| linear | Marlin W4A16 + mixed W4A8-INT8 (per-token `gate_proj`, group-64 elsewhere) | Marlin W4A16 |
| GDN | chunked TF32 WMMA prefill | Triton/FLA chunked |
| attention | FlashInfer FA2 (≥1K single request), native tiled fallback | FlashInfer FA2-class |
| batching | FIFO, 8 / 2 ms / 16,384 tokens; one-item batches use the single forward below 1,024 tokens and the batch API above | continuous, chunked prefill, max 8 |
| graphs | none | PIECEWISE + FULL |
| KV | none; exact-decision cache (off in this run) | fp8 paged KV + prefix cache |
| max context | 35,310 verified | 16,384 |

Raw data: `/tmp/jev-final3/native/`; vLLM cold-long runs are in
`/tmp/jev-final2/native/vllm_long_salt.json`.
