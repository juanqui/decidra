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
| H200 | 16q c8 | **1.32 s / 97 q/s** | **4.30 s / 17 q/s** | **3.3× / 5.7×** |

A 4090 vLLM-SR c8 baseline was not measured; no 4090 throughput speedup is
claimed. The 4090 comparison chose the same top answer with max Δp 0.005.

## RTX PRO 6000 Blackwell WS (SM120, 96 GB)

Host: Vast.ai RTX PRO 6000 Blackwell Workstation Edition, 188 SMs, 97.9 GB
VRAM, power limit 547 W, driver 615.71, CUDA 13.1. The card ran at P1
(1695 MHz SM / 13365 MHz GDDR7) and drew ~376 W under the 64-question
benchmark; no software clock or power adjustment was available in the
container.

| package | 64q c1 | 64q c8 | 16q c8 | 8q c1 | state-once parity |
|---|---:|---:|---:|---:|---:|
| BF16 merged, 49 GB | **992 ms / 64 q/s** | **4.44 s / 65 q/s** | **1.31 s / 61 q/s** | 165 ms / 49 q/s | **0.0102** |
| W4A8 g64, Marlin | 964 ms / 66 q/s | 4.32 s / 67 q/s | 1.22 s / 65 q/s | 141 ms / 57 q/s | 0.081 |

BF16 is the recommended SM120 path: it passes the state-once parity gate at
0.0102 and keeps 46.7 GiB free after preload. W4A8 is slightly faster but its
single-vs-batch parity is 0.081 on this stack; it is not recommended where
per-row equivalence matters. The 4090 remains the fastest W4A8 part because its
clock window is better matched to the SM80-style Marlin kernel.

### SM120 iterations

| change | 64q c1 | result |
|---|---:|---|
| BF16 baseline | 998 ms | cuBLAS BF16 |
| `BATCH_ROWS=512`, 128-row state chunks | 998 ms | rejected; c8 p50 7.66 s, no throughput gain |
| W4A8 Marlin | 964 ms | faster, but 0.081 single-vs-batch parity; not recommended |
| W8A8 per-row expansion | 923 ms | rejected: single-vs-batch max Δp **0.281** (W4A8: 0.017); per-row activation scales were too coarse |
| FP8 E4M3 via cuBLASLt | 939 ms | rejected; slower than W4A8 |
| W8A8 transposed layout | 1,882 ms | rejected; cuBLAS OP_N path slower |
| chunked GDN for 50-token questions | 973 ms | rejected; register GDN is faster |

The BF16 and W4A8 GEMMs both land near 190 TFLOP/s on the MLP linears at this
card's P1 clock. The register GDN recurrence is the next-largest region
(~137 ms for 48 layers) and is bandwidth-bound. FP8 and INT8 large-M variants
were benchmarked through cuBLAS/cuBLASLt but did not beat the W4A8 or BF16
paths at the measured shapes.

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
