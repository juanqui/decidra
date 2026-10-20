# Native engine vs vLLM 0.31.0 (final, shipped build)

Controlled single-GPU comparison after the final native shipping commits:
FlashInfer prefill (`78b68cb`), layer-wise INT8 opt-in (`b43ddbc`), full-M Marlin
default (`722ed8e`), and selective INT8 default. Both stacks loaded the same
`/srv/models/jev-awq/awq-w4a16-v2` AWQ artifact and decision head on GPU 1 only.

## Methodology

- **GPU**: GPU 1, RTX 4090 24 GB; GPU 0 was not used. Clocks ~2,910 MHz SM /
  11,451 MHz memory; power 280–410 W depending on load.
- **Native**: current HEAD `722ed8e`, release binary with `--features native`.
  Defaults: `JEV_NATIVE_MARLIN=true`, `JEV_NATIVE_INT8=true` (selective default
  exclusion list), `JEV_NATIVE_GDN_CHUNKED=true`, FlashInfer attention enabled
  (`JEV_FLASHINFER_MIN_SEQ=1024`), `JEV_NATIVE_MAX_BATCH=8`,
  `JEV_NATIVE_BATCH_WINDOW_US=2000`, `JEV_NATIVE_MAX_BATCH_TOKENS=16384`.
  `JEV_DECISION_CACHE_SIZE=0` so unique prompts are actually uncached.
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
| 1 | **24.28** | 18.81 | **40.73 / 46.99 ms** | 53.06 / 54.08 ms | **1,772** | 1,373 |
| 2 | **32.81** | 20.24 | **62.46 / 66.89 ms** | 98.20 / 103.24 ms | **2,395** | 1,478 |
| 4 | **39.24** | 29.32 | **102.16 / 102.99 ms** | 135.10 / 138.67 ms | **2,865** | 2,140 |
| 8 | **41.47** | 34.52 | **190.77 / 205.25 ms** | 229.98 / 231.08 ms | **3,030** | 2,522 |
| 16 | **41.73** | 35.17 | **279.03 / 376.95 ms** | 304.75 / 454.23 ms | **3,086** | 2,600 |

Native is ahead at every concurrency: **1.29× / 1.62× / 1.34× / 1.20× /
1.19×** req/s. c16 native still uses a second wave (p95 377 ms), but vLLM's
`max_num_seqs=8` second wave is worse (p95 454 ms).

## Long single request (cold, exact native prompts, median of 3)

| prompt | native latency | native tok/s | vLLM cold latency | vLLM tok/s | native speedup |
|---|---:|---:|---:|---:|---:|
| 4K (4,090 tok) | **1.219 s** | **3,355** | 1.302 s | 3,142 | **1.068×** |
| 8K (8,178 tok) | **2.479 s** | **3,299** | 2.660 s | 3,075 | **1.073×** |
| 16K (16,383 tok) | **5.113 s** | **3,204** | 5.533 s | 2,962 | **1.082×** |
| 35.3K (35,310 tok) | not rerun | — | not runnable (16,384 cap) | — | — |

All choices were `true`. With prefix caching bypassed, **native wins long
prefill by 6.8–8.2%**. The earlier vLLM 5.052 s 16K number was cache-assisted
by the nested 4K prompt; the cold 16K vLLM latency is 5.533 s.

## Quality

CPU AWQ golden: `eval/data/awq-cpu-suite28.json` (`native/suite28.json`
backend b). Strict gates: top-1 1.0, mean KL ≤1e-3, mean MAE ≤1e-3, max delta
≤0.011.

| comparison | top-1 | max Δ | mean MAE | mean KL | gate |
|---|---:|---:|---:|---:|---|
| native (selective INT8, last:8 all-INT8, full-M) vs CPU golden | **28/28** | **0.007479** | **0.000691** | 2.371e-5 | **PASS** |
| vLLM vs CPU golden | 28/28 | 0.013772 | 0.001551 | 8.509e-5 | max and MAE fail |
| vLLM vs native live | 28/28 | 0.015007 | 0.001388 | 5.711e-5 | max/MAE fail if native is reference |

The gate-exact configuration is selective INT8 with layers 56-63 switched to
all-linear INT8 (`JEV_NATIVE_INT8_ALL_LAYERS=last:8`, default), Marlin full-M
launches and FlashInfer long attention. It passes every strict gate:
top-1 28/28, max 7.479e-3, MAE 6.908e-4, KL 2.371e-5. Native is closer to the
CPU golden than vLLM on both max delta and MAE. Final native full forward with
this configuration: 1K 295 ms, 4K 1231 ms, 8K 2507 ms, 16K 5174 ms.

## Verdict

- **Short:** native beats vLLM at every concurrency, by 1.19–1.62× req/s and
  1.2–1.6× lower p50; c16 also has lower p95 (377 vs 454 ms).
- **Long:** native beats vLLM by 6.8–8.2% on cold 4K/8K/16K. vLLM no longer
  wins long single-request prefill once its nested-prefix cache benefit is
  removed.
- **Quality:** native 28/28, max Δ 0.00748, MAE 0.000691, KL 2.37e-5 (PASS);
  vLLM 28/28, max Δ 0.01377, MAE 0.00155. Native is closer to the CPU golden
  and passes every gate with the last:8 layer-wise config.
- vLLM still has the operational advantages of a mature server (continuous
  batching, CUDA graphs, prefix/KV caching) and lower startup/host-RSS costs,
  but on this exact single-GPU decision workload the shipped native engine is
  faster on both short and long paths.

## Repeated exact prompt (historical, not rerun in this pass)

The prior native exact-decision cache result remains much faster than vLLM on
repeated prompts (warm c1 ~2,989 req/s / 0.330 ms vs vLLM ~19.4 req/s /
51.1 ms; c8 ~2,298 / 1.258 ms vs vLLM ~41.0 / 193 ms). vLLM's 1,568-token
prefix block cannot cache a 58-token prompt. This comparison used
`JEV_DECISION_CACHE_SIZE=0` for unique-prompt fairness and is not a cache test.

## Kernel/execution stacks

| layer | native | vLLM 0.31.0 |
|---|---|---|
| linear | Marlin W4A16 + selective W4A8-INT8 | Marlin W4A16 |
| GDN | chunked TF32 WMMA prefill | Triton/FLA chunked |
| attention | FlashInfer FA2 (≥1K single request), native tiled fallback | FlashInfer FA2-class |
| batching | FIFO, 8 / 2 ms / 16,384 tokens | continuous, chunked prefill, max 8 |
| graphs | none | PIECEWISE + FULL |
| KV | none; exact-decision cache (off in this run) | fp8 paged KV + prefix cache |
| max context | 35,310 verified | 16,384 |

Raw data: `/tmp/jev-final2/native/`, `/tmp/jev-final2/vllm/`; vLLM cold-long
runs are in `vllm_long_salt.json`.
