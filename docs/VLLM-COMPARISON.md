# Native engine vs vLLM 0.31.0

A controlled single-GPU comparison of the in-process native engine with a
production-style vLLM deployment of the same AWQ artifact. This is a snapshot, not
a recurring benchmark: each server was measured once on an otherwise idle GPU 1
(RTX 4090, 24 GB), so small differences at c=4/c=8 (a few percent) are within
run-to-run noise. Raw vLLM data came from `/tmp/jev-vllm-compare/comparison.json`;
native numbers are the final 71-token sweep and long-context probes in
[`RESULTS.md`](RESULTS.md).

## Methodology

- **Hardware**: GPU 1, RTX 4090 24 GB. GPU 0 was not used. Both servers loaded the
  **same** `/srv/models/jev-awq/awq-w4a16-v2` AWQ W4A16 artifact and decision head.
- **Fresh server per stack**: native preloaded before each sweep; vLLM was started
  from a clean container and measured after health.
- **Short load**: 16 unique prompts per level (~70.5 prompt tokens), c = 1, 2, 4,
  8, 16, three warmup requests, one completion token per decision. Native rows are
  the median of three runs at ~70.8 prompt tokens. No prompt is repeated across
  levels, so no prefix/KV reuse flatters either server.
- **Long load**: single-stream document decisions at ~4K, ~8K, and ~15.5K prompt
  tokens, one completion token.
- **vLLM A_ref configuration**: `vllm/vllm-openai:v0.31.0` (Torch 2.13.0+cu130,
  CUDA 13.0), `max_model_len=16384`, `max_num_seqs=8`,
  `gpu_memory_utilization=0.95`, `kv_cache_dtype=fp8`, `quantization=compressed-tensors`,
  CUDA graphs enabled (`enforce_eager=false`), chunked prefill on, V2 runner.
- **Native configuration**: defaults in this repo (`JEV_NATIVE_MAX_BATCH=8`,
  `JEV_NATIVE_BATCH_WINDOW_US=2000`, `JEV_NATIVE_MAX_BATCH_TOKENS=16384`), Marlin
  resident weights.
- `ms/token` is p50 latency divided by mean prompt tokens (per request); `tok/s` is
  aggregate prompt-token throughput.

## Short requests (~70.5 tokens, 16 per level)

| c | vLLM rps | native rps | vLLM p50 | native p50 | vLLM tok/s | native tok/s | vLLM ms/token | native ms/token | vLLM/native rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.02 | 15.84 | 52.1 ms | 61.9 ms | 1,338.6 | 1,122 | 0.744 | 0.874 | **1.20x** |
| 2 | 20.34 | 23.38 | 96.9 ms | 83.2 ms | 1,436.5 | 1,656 | 1.384 | 1.175 | 0.87x (**native 1.15x**) |
| 4 | 29.42 | 30.13 | 134.3 ms | 133.7 ms | 2,073.8 | 2,133 | 1.882 | 1.888 | 0.98x (tie) |
| 8 | 35.85 | 29.25-30.26 | 217.6 ms | 233-256 ms | 2,529.7 | 2,051-2,141 | 3.054 | 3.45 (mid) | **1.20x** |
| 16 | 34.93 | 29.28-30.47 | 318.5 ms | 388-442 ms | 2,464.7 | 2,107-2,192 | 4.962 | 5.86 (mid) | **1.17x** |

Native c8/c16 are ranges over three runs. Native tok/s plateaus at ~2,100-2,150
because the single FIFO worker executes one batch at a time; vLLM's continuous
batching keeps adding throughput through c=8/c=16. Native peaks at c=2/c=4, where
the 2 ms FIFO window captures both requests without queueing.

## Long prefill (single stream)

| prompt | vLLM p50 | vLLM tok/s | native latency | native tok/s | vLLM/native latency |
|---|---:|---:|---:|---:|---:|
| ~4K (4,097 vLLM / 4,096 native) | 1,305.7 ms | 3,135.7 | 2.12-2.18 s | 1,879-1,932 | **1.60-1.67x** |
| ~8K (8,085 / 8,192) | 2,637.0 ms | 3,064.8 | 4.23-4.32 s | 1,871-1,911 | **1.60-1.64x** |
| ~15.5K (15,595 / 16,384) | 5,248.2 ms | 2,970.3 | 8.62-8.97 s (16K row) | 1,739-1,809 | **1.64-1.71x** |

Native also completed a **35,310-token** decision in 20.15-20.16 s (verified
maximum; 10x 17.7K and 3x 35.3K runs, all 200). vLLM was capped at
`max_model_len=16384`, so it was not exercised beyond ~15.6K in this pass.

## Startup, memory, KV cache, context

| metric | vLLM 0.31.0 | native engine |
|---|---|---|
| startup to health | **139 s** (91.8 s engine init + 22.6 s compile) | **10.1 s** (7.6 s observed on final build) |
| peak GPU memory | **21,909 MiB** | **13.1 GiB** resident; **19.1 GiB** peak at 32K |
| KV cache | 41,441 tokens / **2.08 GiB** (fp8) | none: every decision re-runs full prefill |
| configured context | 16,384 tokens | no batch cap on a single request; **35,310 tokens verified** |
| max concurrent sequences | `max_num_seqs=8` | FIFO batch (`MAX_BATCH=8`), one batch at a time |

## Kernel/execution stacks

| layer | vLLM 0.31.0 | native engine |
|---|---|---|
| linear | Marlin (`MarlinLinearKernel`, compressed-tensors WNA16) | Marlin W4A16 (resident repacked layout) |
| GDN | Triton/FLA prefill + CUDA decode | custom split-dk GDN recurrence |
| attention | FlashInfer decode, FlashAttention ViT | tiled FA2 |
| batching | continuous batching, chunked prefill | FIFO request batch, 2 ms window |
| graphs | CUDA graphs (PIECEWISE + FULL) | none |
| KV cache | fp8 LBNHC, paged | none (full recompute per pass) |
| output head | FlashInfer top-p/top-k sampler | decision head logits |

## Why the curves differ

- **c=1**: vLLM wins ~20% because CUDA graphs remove per-forward kernel-launch
  overhead; the native path launches eager kernels and pays the 2 ms batch window.
- **c=2**: native wins ~15% because both requests usually share one 2-wide batch,
  halving forward cost; vLLM's scheduler does not gain as much from two sequences.
- **c=4**: a tie. The native batch is already full at 4 and vLLM is still
  launch-bound.
- **c=8/c=16**: vLLM wins ~17-20% through continuous batching and graph replay;
  the native worker is a single batch at a time and saturates at ~30 req/s.
- **Long prefill**: vLLM is ~1.6-1.7x faster due to chunked prefill, FlashInfer
  attention, FLA GDN, and fused graphs. Native uses full recompute (no KV cache)
  with custom GDN and tiled FA2, which is competitive but not faster.
- **Startup and memory**: native skips graph capture/compilation and keeps no KV
  cache, so it starts 13.8x sooner and uses ~40% less peak memory, at the cost of
  recomputing prefill every pass. That trade is what allows the 35.3K context on
  the same 24 GB card while vLLM is configured for 16K.

## Quality caveat

- Native vs the CPU AWQ golden suite28: top-1 **28/28**, max delta **1.010e-2**,
  mean MAE **7.365e-4**, mean KL **3.281e-5** (passes the 0.011/0.001 gates).
- vLLM vs the saved production artifacts in this pass: suite28 **28/28**, max
  delta **2.61e-4**, mean delta **9.33e-6**. This is a comparison to the saved
  production outputs, not to the CPU AWQ golden.
- Historical vLLM-vs-CPU-AWQ numbers were max **0.01377** / mean MAE **0.00156**.
  That comparison was **not rerun** in this controlled pass, so it remains context
  only. Do not read the vLLM 2.61e-4 as vLLM being closer to CPU AWQ than native;
  the reference files differ.

## Verdict

Under this load shape vLLM is faster at **c=1 (1.20x)**, **c=8 (1.20x)**,
**c=16 (1.17x)**, and on **long prefill (1.6-1.7x)**. The native engine is faster
at **c=2 (1.15x)**, ties at **c=4**, and wins decisively on **startup (139 s vs
10.1 s)**, **peak memory (21.9 GiB vs 13.1 GiB resident / 19.1 GiB peak)**, and
**verified context (35.3K vs a 16K configured cap)**. Neither result is a defect:
they reflect CUDA graphs, chunked prefill, FlashInfer/FLA, and continuous batching
in vLLM versus a simpler FIFO batch with no KV cache in the native engine.
