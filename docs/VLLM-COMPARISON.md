# Native engine vs vLLM 0.31.0 (final)

A controlled single-GPU comparison of the in-process native engine with a
production-style vLLM deployment of the same AWQ artifact. This is a snapshot, not
a recurring benchmark. Raw vLLM data: `/tmp/jev-final-compare/vllm/REPORT.md`
(plus its JSON/log artifacts); native data: `/tmp/jev-final-compare/native/`.
Both servers were measured on GPU 1 only.

## Methodology

- **Hardware**: GPU 1, RTX 4090 24 GB. GPU 0 was not used. Both stacks loaded the
  **same** `/srv/models/jev-awq/awq-w4a16-v2` AWQ W4A16 artifact and decision head.
- **vLLM 0.31.0 A_ref**: `vllm/vllm-openai:v0.31.0` (Torch 2.13.0+cu130, CUDA 13.0),
  exact flags `--model /models/awq-w4a16-v2 --served-model-name jev-awq
  --logprobs-mode processed_logprobs --max-model-len 16384
  --enable-prefix-caching --mamba-cache-mode align --trust-request-chat-template
  --max-num-seqs 8 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8`.
- **Native**: final build with chunked GDN and the exact-decision cache enabled
  (`JEV_NATIVE_GDN_CHUNKED=true`, `JEV_DECISION_CACHE_SIZE=256`, batch defaults 8 /
  2000 us / 16384 tokens).
- **Prompts**: regenerated from the native suite/bench generator with the same seed
  and tags. Locally verified means are **73.0 / 73.0 / 73.0 / 73.0625 / 73.9375**
  tokens for c=1/2/4/8/16, matching native's server-counted means. Short runs: 16
  unique prompts per level, 3 runs, 3 warmups. vLLM `max_num_seqs=8`, so c=16 runs
  as two scheduler waves.
- **Long prompts**: 4,090 / 8,178 / 16,383 tokens; the 16K prompt fits prompt+1 in
  vLLM's 16,384 cap. Native additionally ran the verified 35,310-token request.
- **Repeated exact**: 16 identical requests (58 native tokens / 73 vLLM tokens) at
  c=1 and c=8, cold and warm.
- **Prefix cache**: vLLM recorded **0 hits** on all short and repeated requests
  (58–74-token prompts cannot fill its 1,568-token block); the nested 8K/16K long
  probes received one 1,568-token hit, so vLLM's long numbers are slightly
  cache-assisted. Native has no prefix cache; it does have the exact-decision cache.
- `ms/token` is p50 divided by mean prompt tokens (per request); `tok/s` is
  aggregate prompt-token throughput.

## Short unique prompts (~73 tokens)

| c | vLLM rps | native rps | vLLM p50 | native p50 | vLLM tok/s | native tok/s | vLLM ms/token | native ms/token | vLLM/native rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **18.91** | 15.63 | **52.67 ms** | 61.8 ms | **1,380** | 1,141 | 0.722 | 0.847 | **1.21x** |
| 2 | 20.38 | **23.28** | 97.57 ms | **87.3 ms** | 1,487 | **1,699** | 1.337 | 1.196 | 0.88x (**native 1.14x**) |
| 4 | 29.36 | **29.96** | **134.43 ms** | 134.8 ms | 2,142 | **2,187** | 1.842 | 1.847 | 0.98x (tie) |
| 8 | **34.70** | 29.61 | **229.20 ms** | 269.7 ms | **2,546** | 2,164 | 3.140 | 3.691 | **1.17x** |
| 16 | **35.09** | 28.86 | **305.53 ms** | 431.1 ms | **2,590** | 2,134 | 4.185 | 5.831 | **1.22x** |

vLLM run ranges: c1 rps 18.12-19.08; c2 20.23-20.51; c4 29.34-29.58; c8
34.58-34.85; c16 35.03-35.38. p50 ranges were tight (e.g. c1 52.45-52.85,
c8 228.48-230.32).

## Long single request (cold, exact native prompts)

| prompt | vLLM latency | vLLM tok/s | native latency | native tok/s | vLLM/native |
|---|---:|---:|---:|---:|---:|
| 4K (4,090 tok) | **1.304 s** | **3,136** | 1.670 s | 2,449 | **1.28x** |
| 8K (8,178 tok) | **2.661 s** | **3,073** | 3.327 s | 2,459 | **1.25x** |
| 16K (16,383 tok) | **5.052 s** | **3,243** | 6.932 s | 2,363 | **1.37x** |
| 35.3K (35,310 tok) | not runnable (`max_model_len=16,384`) | - | 16.662 s | 2,120 | - |

All choices were `true`. Native scales linearly; vLLM's 8K/16K rows include one
nested-prefix cache hit. Even allowing for that, vLLM is **1.25-1.37x faster** on
long single-request prefill.

## Repeated exact prompt (16 identical requests)

| case | vLLM cold | vLLM warm | native cold | native warm |
|---|---:|---:|---:|---:|
| c1 rps / p50 | 19.12 / 52.12 ms | 19.40 / 51.11 ms | 254.7 / 0.337 ms* | **2,989 / 0.330 ms** |
| c8 rps / p50 | 41.32 / 191.46 ms | 41.02 / 193.56 ms | 69.3 / 60.9 ms | **2,298 / 1.258 ms** |

`*` native c1 cold includes a 57 ms first request; the warm number is steady state.
vLLM prefix-cache hits: **0** in both cold and warm runs (928 queries each); a
58-token prompt cannot fill the 1,568-token block. vLLM cold and warm are identical
within noise. Native's exact-decision cache sees a hit rate of 1.0 after the first
request, so it wins repeated-prompt workloads by **~150x at c1 and ~56x at c8**.
vLLM has no equivalent response cache.

## Startup, memory, KV

| metric | vLLM | native |
|---|---:|---:|
| startup to ready | 145 s | **7.65 s** |
| GPU idle after startup | 21,781 MiB (21.8 GiB) | **13,383 MiB (13.4 GiB)** |
| GPU peak during suite/long | 22,069 MiB (22.07 GiB) | 22,349 MiB (22.35 GiB) |
| host RSS | **8.58 GiB** (container) | 12.89 GiB |
| KV cache | 41,441 fp8 tokens / 2.08 GiB | none (recurrent state) |
| configured context | 16,384 tokens | 35,310 tokens verified |

vLLM starts ~19x slower and idles ~8.4 GiB higher on GPU because it preallocates
weights plus KV; native peaks slightly higher on GPU and uses more host RSS.

## Kernel/execution stacks

| layer | vLLM 0.31.0 | native engine |
|---|---|---|
| linear | Marlin (compressed-tensors WNA16) | Marlin W4A16 resident |
| GDN | Triton/FLA chunked prefill + CUDA decode | chunked TF32 WMMA prefill (>=512), shared/split fallback |
| attention | FlashInfer (FA2-class on SM89) | tiled FA2 |
| batching | continuous batching, chunked prefill, `max_num_seqs=8` | FIFO batch, 8 / 2 ms / 16384 tokens |
| graphs | CUDA graphs (PIECEWISE + FULL) | none |
| KV | fp8 LBNHC, paged, prefix caching | none; exact-response logits cache only |
| output head | FlashInfer top-p/top-k | decision head logits |

## Why the curves differ

- **c=1**: vLLM wins ~21% via CUDA graphs and a tighter single-request path; native
  still launches eager kernels and pays the 2 ms batch window.
- **c=2**: native wins ~14% because both requests share one 2-wide batch.
- **c=4**: a tie; the native batch is full and vLLM is still launch-bound.
- **c=8/c=16**: vLLM wins 17-22% through continuous batching and graph replay; its
  `max_num_seqs=8` forces a second wave at c=16, raising its p95 to 455 ms, but
  throughput still leads. Native's single FIFO worker saturates near 30 req/s.
- **Long prefill**: vLLM is 1.25-1.37x faster (chunked prefill, FlashInfer, FLA GDN,
  graphs); native's chunked GDN closed the earlier 1.6-1.7x gap but did not erase it.
- **Repeated prompts**: the native exact-decision cache bypasses the engine entirely;
  vLLM's prefix cache cannot help because these prompts are shorter than one block.

## Quality caveat

| comparison | top-1 | max delta | mean MAE | note |
|---|---:|---:|---:|---|
| native vs CPU AWQ golden suite28 | **28/28** | **0.01010** | **0.000736** | native's shipping gate (<=0.011 / <=0.001) |
| vLLM vs saved production reference | **28/28** | **0.00493** | **0.000651** | vLLM sits closer to the saved production outputs |
| vLLM vs native live | 28/28 | 0.02264 | 0.001603 | max fails if native is treated as the reference |

The vLLM-vs-native max delta is concentrated in **r22** (noul): vLLM
`[0.68860, 0.31140]`, production `[0.69353, 0.30647]`, native live
`[0.71124, 0.28876]`. vLLM is closer to production on that row; native's own shift
from production is ~0.0177 versus vLLM's 0.0049. The three comparisons use different
references and must not be read as a single ranking: native is gated against CPU AWQ,
vLLM against saved production, and the vLLM-vs-native delta is mostly native's
row-22 shift.

Historical context, unchanged: the earlier vLLM-vs-CPU-AWQ measurement was max
**0.01377** / mean MAE **0.00156**; it was not rerun in this controlled pass.

## Quantization probes

Both probes were offline kernel/layer experiments on GPU1; neither is enabled in the
shipped server, and the default remains **W4A16**.

- **W4A8-FP8 Marlin** (`kFE4M3fn x kU4` + zero points): kernel works on SM89 and
  gives **1.59-1.61x** at M>=1024 (~280 vs ~176 TFLOPS), but per-token E4M3
  activation quantization adds ~**2.6% normalized layer MAE** (vs ~0.21% for
  W4A16). Quality is a **no-go** for the strict decision gate. Default OFF.
- **W4A8-INT8 Marlin** (`kS8 x kU4` + zero points): kernel works and gives
  **2.6-2.76x** at M=1024/16384 (~457-486 TFLOPS). The old full-model numbers
  (26/28, max 0.657) were dominated by a launcher bug: the tail `pm` group offset
  the activation pointer by `row_off` bytes instead of `row_off * k`, so every
  non-multiple-of-64 row count computed its last group from the wrong tokens
  (visible as the last request of an 8-request batch collapsing toward zero).
  With that fixed the suite is **27/28, max delta 0.0589, MAE 0.00453**, still
  outside the gate (top-1 1.0, max 0.011, MAE 0.001). Layer error is lower than
  FP8 (~0.90-0.96% normalized MAE). Default OFF.
- The shipped default stays **W4A16**: suite28 28/28, max delta 0.0101, MAE
  0.000736. A future W4A8 attempt needs a full calibration/quality project before it
  can be considered.

## Verdict

Under this load shape **vLLM wins c=1 (1.21x), c=8 (1.17x), c=16 (1.22x), and long
prefill (1.25-1.37x)**. The native engine **wins c=2 (1.14x), ties c=4**, and wins
**repeated exact prompts (~150x c1 / ~56x c8)**, **startup (7.65 s vs 145 s)**,
**GPU idle memory (13.4 vs 21.8 GiB)**, and **verified context (35,310 vs a 16,384
cap)**. vLLM has slightly lower GPU peak (22.07 vs 22.35 GiB) and host RSS
(8.58 vs 12.89 GiB), and its production-reference suite passes.

**Do not claim the native engine is faster overall.** The gaps are explained by
CUDA graphs, continuous batching, chunked prefill/FlashInfer/FLA, and prefix caching
in vLLM versus a simpler FIFO batch with no KV cache but an exact-decision cache in
the native engine.
