# Decidra 0.2 benchmarks

All numbers are end-to-end `POST /v1/systemone` on the same 15.5 GB / 49 GB
d3-turbo package families, weights preloaded, one state per request unless
noted. The native server is Decidra (in-process CUDA, state-once); the baseline
is vLLM-SR `vllm-srun` 0.2.0 engine mode with `--max-num-seqs 64
--max-num-batched-tokens 32768 --enable-shared-context`.

## Hardware and precision

| device | VRAM | arch | package | attention |
|---|---:|---|---|---|
| RTX 4090 | 24 GB | SM89 | W4A8 g64 v4, 15 GB, 450 W | FA2 |
| H100 80GB HBM3 | 81.6 GB | SM90 | BF16, 49 GB | FA2 |
| H200 | 143.8 GB | SM90 | BF16, 49 GB | FA2 |

FA3 is implemented for SM90 but is opt-in; on the 2K–24K d3 workload the FA2
cascade was equal or faster on H200 (64q 800 vs 834 ms; 24K 2.62 vs 3.06 s).
The H100/H200 recommendation is BF16: the 80 GB H100 has 30.9 GiB free after
preload, and a re-quantized W4A8 H200 artifact failed the 32-row gate.

## Side-by-side summary

Same checkpoint, same request (2,076-token state, 64 questions), same GPU,
warm weights. Baseline is vLLM-SR `vllm-srun` 0.2.0 engine mode with
`--max-num-seqs 64 --enable-shared-context`.

| GPU | test | Decidra | vLLM-SR | advantage |
|---|---|---:|---:|---|
| RTX 4090, 450 W | 64q c1 | **936 ms** | **3.89 s** | **4.2× lower latency** |
| H100 SXM | 64q c1 | **824 ms** | **2.43 s** | **2.9× lower latency** |
| H100 SXM | 64q c8 | **5.10 s / 100 q/s** | **13.66 s / 23 q/s** | **2.7× / 4.3×** |
| H200 | 64q c1 | **800 ms** | **2.90 s** | **3.6× lower latency** |
| H200 | 64q c8 | **5.47 s / 94 q/s** | **13.22 s / 22 q/s** | **2.4× / 4.3×** |
| RTX PRO 6000, 600 W | FP8 64q c1 | **623 ms** | **3.23 s** | **5.2× lower latency** |
| RTX PRO 6000, 600 W | FP8 64q c8 | **2.80 s / 103 q/s** | **14.57 s / 20 q/s** | **5.2× q/s** |
| H200 | 16q c8 | **1.32 s / 97 q/s** | **4.30 s / 17 q/s** | **3.3× / 5.7×** |

A 4090 vLLM-SR c8 baseline was not measured; no 4090 throughput speedup is
claimed. The 4090 comparison chose the same top answer with max Δp 0.005.

## RTX PRO 6000 Blackwell (SM120, 96 GB)

Two power profiles were measured, plus the 450 W RTX 4090 reference.

| device | power | BF16 64q c1 | BF16 q/s | FP8 64q c1 | FP8 q/s | best 16q c8 |
|---|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 WS | 600 W | 775 ms | 83 | **623 ms** | **103** | **754 ms / 106 q/s** |
| RTX PRO 6000 Max-Q | 300 W | 1,045 ms | 61 | **859 ms** | **74** | **1.06 s / 75 q/s** |
| RTX 4090 (W4A8 reference) | 450 W | n/a (24 GB) | — | — | — | 1.15 s / 69 q/s |

Hosts: Vast.ai RTX PRO 6000 Blackwell WS, 188 SMs, 97.9 GB, 600 W cap,
driver 615.71/580.126, CUDA 13.1; Max-Q, 300 W cap, 192 vCPU. The WS host
sustained **2445–2482 MHz / ~605 W**; the Max-Q sustained **~1.68 GHz /
~300 W**. The earlier 547 W WS host sustained only 1695 MHz / 376 W and is
not a valid hardware comparison.

### Decidra vs vLLM-SR on the same 600 W WS host

Same BF16 checkpoint, same 2K state + 64-question request, vLLM-SR
`vllm-srun` 0.2.0 with `--max-num-seqs 64 --max-num-batched-tokens 65536
--enable-shared-context --dtype bfloat16`.

| test | Decidra BF16 | Decidra FP8 | vLLM-SR BF16 | best advantage |
|---|---:|---:|---:|---:|
| 1q c1 | 51.1 ms | **46.8 ms** | 551.8 ms | **11.8×** |
| 8q c1 | 124.2 ms | **88.8 ms** | 729.5 ms | **8.2×** |
| 16q c1 | 211.5 ms | **155.9 ms** | 955.8 ms | **6.1×** |
| 64q c1 | 775.3 ms | **623.2 ms** | 3.23 s | **5.2×** |
| 64q c8 | 3.42 s / 84 q/s | **2.80 s / 103 q/s** | 14.57 s / 20 q/s | **5.2× q/s** |
| 64q c8 p50 | 3.42 s | **2.80 s** | 14.57 s | **5.2×** |

vLLM-SR reports `approximate` numerics with shared context. Decidra BF16 has
a state-once parity of **0.0102**; the opt-in FP8 path has **0.0362** on the
64-row probe and keeps the top choice on **64/64 rows**. State capture always stays BF16.

### SM120 FP8 implementation

The FP8 path is opt-in (`DECIDRA_NATIVE_FP8=1 DECIDRA_FP8_MODE=all_linear`).
At layer load, each BF16 MLP/linear weight is converted once to E4M3 with a
per-output-row scale. Question passes quantize activations per token, run
cuBLASLt TN FP8 GEMMs, and apply the two scale vectors in a fused epilogue.
The M dimension is padded to a multiple of 16. `capture_state` sets a
`capturing` guard, so state capture remains BF16 even when the question path
is FP8.

All-linear FP8 is the best measured mode. Gate/up-only FP8 measured
**973 ms** on Max-Q with **0.0248** parity; all-MLP FP8 measured 918 ms with
**0.0497**. All-linear FP8 is both faster (**859 ms Max-Q**) and lower-error
(**0.0362**) than those variants on this probe; it keeps the top choice on
**64/64 rows**.

### Rejected SM120 experiments

| change | result |
|---|---|
| BF16 on 547 W host | 998 ms; invalid hardware comparison, 1695 MHz/376 W |
| Fused gate/up + fused SiLU | 616.8 ms vs 623.2 ms, but parity rose to **0.0547**; removed |
| FP8 fast accumulation | 624 ms, no gain |
| L8 register GDN (8 lanes × 16 rows) | correctness failed; removed |
| W8A8 per-row expansion | 923 ms on 547 W, parity **0.281**; removed |
| W8A8 transposed layout | 1.88 s; rejected |
| chunked GDN for short questions | 973 ms; rejected |
| GDN vc32 vs vc64 | no measurable difference |

The remaining non-FP8 time is dominated by the f32 GDN recurrence (~86 ms at
600 W), attention (~27 ms), RMSNorm/residual/epilogue kernels, and the
state-cache/readout machinery. The FP8 MLP GEMMs are at roughly 80% of the
SM120 FP8 tensor peak at the sustained clock.

## RTX 4090 W4A8, 450 W cap (local GPU1)

Both GPUs are hard-capped at 450 W. GPU1 draws ~449 W and runs at ~2.82 GHz.
SM89 defaults: W4A8 group-64, gate/up/down/o/q INT8, k/v W4A16, register GDN,
suffix-only tokenization, 32-row scheduler budget, and a 16-entry / 5 GiB
state cache, an 8 GiB host L2 and a 2 GiB scratch reserve.

| questions | c1 | c8 | c16 | c32 | c8 q/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.0 ms | 127.7 ms | 251.3 ms | — | 62 |
| 16 | 236.0 ms | 1.15 s | 2.07 s | 3.91 s | 69–70 |
| 64 | 936 ms | 4.18 s | 7.91 s | 15.38 s | 69 |

Cold 2K state: 1q **539 ms**, 64q **1.375 s**. State capture is ~400–500 ms.

| state | 16q c1 | 64q c1 | 64q c8 | c8 q/s |
|---:|---:|---:|---:|---:|
| 2K | 236 ms | 936 ms | 4.18 s | 69 |
| 8K | 260 ms | 1.02 s | 4.56 s | 63 |
| 24K | 338 ms | 1.29 s | 5.67 s | 51 |

The 16-entry pinned GPU cache plus an 8 GiB host L2 is the largest scheduler
win for the target working set. An 8-state set at 2K produced **0/8 hits and
23.5 q/s** with 4 entries; 8 entries gave **8/8 hits and 68.9 q/s**; 16 entries
gives **16/16 hits and 69 q/s**. With 32 distinct states, the GPU keeps 16
resident and the host L2 holds the overflow: **32/32 hits at 63 q/s**, versus
16 hits/35 q/s pinned-only and 0 hits/23.5 q/s pure LRU. Memory with 16 GPU
states is ~20.6 GB (3.46 GiB free); the 32-state run leaves 2.69 GiB, and c32
scratch leaves 4.48 GiB after the long-state eviction. A 2 GiB reserve trims
before capture. 24K settles at 19.6 GB used / 4.48 GiB free.

Mixed 1q/8q/64q traffic (small 256-token state, medium 2K, large 8K):

| concurrency | small p50 | medium p50 | large p50 | q/s |
|---:|---:|---:|---:|---:|
| 8 | 63.5 ms | 292 ms | 2.42 s | 64.1 |
| 16 | 98.0 ms | 441 ms | 3.68 s | 64.1 |
| 32 | 164 ms | 1.20 s | 6.97 s | 64.2 |

At 450 W GPU1 is again power-saturated (449.5 W / ~2.82 GHz). A 30-batch c16
soak at 300 W previously completed 480 requests with zero errors and stable
memory. A final 10-batch c16 soak at 450 W completed 160 more requests with
zero errors, 20.3 GB used / 3.78 GB free and 448 W; the 16-state and c32
450 W runs also completed with zero errors. The
earlier vLLM-SR 4090 baseline (3.89 s for 64q) was measured at 450 W.

## H100 SXM 80GB (588 W cap), BF16, register GDN

| questions | native c1 | native c8 | native q/s | vLLM-SR c1 | vLLM-SR c8 | vLLM-SR q/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.9 ms | 154.9 ms | 50 | 475.7 ms | 2.13 s | 2 |
| 8 | 118.2 ms | 660.9 ms | 96 | 893.2 ms | 3.85 s | 9 |
| 16 | 225.4 ms | 1.28 s | 100 | — | — | — |
| 32 | 410.2 ms | 2.54 s | 100 | 1.89 s | 9.00 s | 17 |
| 64 | **824.3 ms** | **5.10 s** | **100** | 2.43 s | 13.66 s | 23 |

Long state (native, c1): 64q at 8K **1.17 s**, at 24K **3.10 s**.
State-once self-test Δp 0.0104 (register recurrence; entropy reduction order).
Preload 30 s; 30.9 GiB free.

The **H100 NVL 95GB** (400 W cap) runs the same binary at 1q 34.6 ms,
64q 916 ms, and **80 q/s** at 16q/c8; SXM's extra power is worth ~25% at high
concurrency. H100 NVL is the cheaper H100, SXM is the fastest H100.

## H200, BF16

| questions | native c1 | native c8 | native q/s | vLLM-SR c1 | vLLM-SR c8 | vLLM-SR q/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.8 ms | 140.5 ms | 56 | 443.8 ms | 1.98 s | 2 |
| 8 | 126.0 ms | 715.3 ms | 89 | 631.0 ms | 2.92 s | 12 |
| 16 | 243.5 ms | 1.32 s | 97 | 951.3 ms | 4.30 s | 17 |
| 64 | 800 ms | 5.47 s | 94 | 2.90 s | 13.22 s | 22 |

Long state (native, c1): 64q at 8K **1.27 s**, at 24K **2.62 s**. State-once
self-test Δp 0.00119. Preload 39 s.
Distinct-state cold c8 at 64q: **4.57 s, 64 q/s** (state cache cannot share).

## Correctness gates

| check | result |
|---|---|
| 32-row HF parity, shipped SM89 defaults | 31/32 top-1, MAE 0.0158, max Δp 0.0837, pass |
| 32-row HF parity, old gate-W4A16 config | 31/32 top-1, MAE 0.0156, max Δp 0.0942, pass |
| 32-row HF parity, W4A8 v4 register GDN | 31/32 top-1, MAE 0.0150, max Δp 0.0854, pass |
| state-once vs per-row block | max Δp 0.00645 |
| batch-vs-single, production config | max Δp 0.02277 (<0.05) |
| zero GDN seed negative control | 0.00834 |
| zero conv-tail negative control | 0.00474 |
| H100 / H200 state-once self-test | 0.00186 / 0.00119 |
| confidence formula | matches vLLM-SR System One (choice/score) |

## Ceiling and the 127 q/s target

The requested 127 q/s at 8 concurrent 2K-state/16-question requests was
**not reached**: best measured is **100 q/s on H100 SXM** and 97 q/s on H200,
21–24% short. On SM89 at 450 W the best is 69 q/s at 2K, 63 at 8K and 51 at 24K; the GPU is
power-saturated for the whole run. Native still beats vLLM-SR by 4.3–5.9× in question throughput on
those cards.

The 2K/64q BF16 question-pass profile (H200) is: GDN recurrence 155 ms (32.5%),
full attention 58 ms (12%), MLP GEMMs 140 ms (29%), other projections/norms/
copies 173 ms. The aggregate is ~411 BF16 TFLOP/s at c8, ~42% of the H200
dense peak. The compute-only ceiling is ~366 q/s at ~2.7 TFLOP per question;
launch/occupancy and the GDN scan set the current gap. A single-launch GDN
seed broadcast replaced 64 per-row copies and was correctness-neutral but
performance-neutral. The register-resident GDN recurrence is now the default
for short question blocks and adds ~4%; it is the best measured single-request
path. The remaining gap is dominated by the GDN scan's arithmetic intensity and
the per-row cascade attention launches. The next work is a batched ragged
cascade and a fused/tensor-core GDN scan, plus an SM90 W4A8 WGMMA kernel if the
15 GB artifact must be used on H100/H200.


## NVFP4 experimental (RTX PRO 6000, 600 W)

FP4 W4A4 is faster in its full form but does **not** meet the 32-row HF
accuracy gate on d3-turbo. It is not a default or recommended serving path.

| Configuration | 64q c1 | HF top-1 | Max Δp |
|---|---:|---:|---:|
| Full MLP + attention W4A4, GDN BF16 | **500.8 ms / 128 q/s** | 28/32 | 0.279 |
| MLP W4A4, attention BF16 | — | 29/32 | 0.284 |
| gate/up W4A4, down/attention BF16 | 611 ms / 105 q/s | 29/32 | 0.113 |
| FP8 all-linear (recommended) | 623 ms / 103 q/s | **32/32** | **0.0115** |

The full W4A4 checkpoint used 1,024 d3 `prefix_bidir` calibration rows and
GPTQ; see `docs/ENGINE.md` and `quant/quantize_nvfp4.py`. A Max-Q NVFP4 run was
not possible on this host: its 600 W power limit did not change when asked for
300 W, so the numbers above are 600 W only. The FP4 kernel and
checkpoint loader are retained for experiments; quality-sensitive deployments
should use BF16 or the validated FP8 path.
