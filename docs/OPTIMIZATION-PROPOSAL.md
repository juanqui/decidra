# Deep optimization proposal: native prefill engine

Analysis only. No GPU runs and no kernel changes were made for this document. It
synthesizes `/tmp/jev-native-opt/PLAN.md`, `/tmp/jev-kernel-research/REPORT.md`,
and the controlled results in [`VLLM-COMPARISON.md`](VLLM-COMPARISON.md).

## Executive answer

The controlled same-GPU comparison is the frame for everything below: **vLLM is
1.20x faster at c=1, 1.20x at c=8, and 1.17x at c=16, and 1.6-1.7x faster on long
prefill (4K/8K/15.5K)**. The native engine ties at c=4 and wins c=2 (~1.15x), and
it wins decisively on startup (10.1 s vs 139 s), peak memory (13.1 GiB resident /
19.1 GiB peak vs 21,909 MiB), and verified context (35,310 tokens vs vLLM's
16K configured cap). **Do not claim the native engine is faster overall.** The
proposal below is about closing the vLLM gap where it exists without giving up
the native advantages.

The highest-value work is the GDN recurrence, not the linear layers alone: GDN is
~42% of a 70-token forward and ~28.8% of a 16K prefill while running at ~0.5% of
BF16 peak. Chunking that scan is the only change with a credible double-digit
end-to-end gain at every length.

## Corrected roofline

The earlier brief overstated both the GPU ceiling and Marlin's efficiency.

- **BF16 dense peak on SM89 (RTX 4090): ~330 TFLOPS.** FP8 peak is ~660 TFLOPS,
  but the shipped artifact and current kernels are W4A16/BF16.
- **MLP reaches ~176 TFLOPS = ~53% of BF16 peak**, not "near peak". The practical
  Marlin ceiling is roughly 220-240 TFLOPS; a large-M kernel might recover
  1.25-1.4x on that stage.
- **1K forward (488 ms)**: GDN **33.5%**, MLP **41.9%**, other GEMMs **18.0%**,
  attention **0.8%**, conv **1.8%**.
- **16K forward (8.80 s)**: GDN **28.8%**, MLP **35.8%**, other GEMMs **15.5%**,
  attention **6.4%**, conv+prepare **3.8%**.
- **70-token forward (~62.7 ms, warm)**: GDN **~42%**, MLP ~32%, other GEMMs ~20%,
  attention <1%. The kernel is **latency/weight-bound**, not compute-saturated:
  3.43 TFLOP in 62.7 ms is ~59 TFLOPS (~18% peak), at roughly 210 GB/s; the
  weight-only lower bound is ~13 ms.

Two corrections matter for planning:

1. Do not budget the 72-80 ms `gemm.b` row as recoverable host overhead. It is a
   profiler event-gap artifact (~1.5 ms/op x 48) independent of sequence length.
2. The ~0.6 s gap between 16K region sum and wall time is not proven host
   overhead; it mixes embedding, f32 scratch allocation churn, and profiling.
   Measure with CUPTI/nsys before optimizing it.

## Ranked candidates

Gains are end-to-end latency reductions. They overlap; do not add rows.

| rank | candidate | expected end-to-end gain | effort | risk | confidence |
|---:|---|---|---|---|---|
| 1 | **FLA-style chunked/parallel-scan GDN prefill** (chunk 64, WY/UT, f32 state) | c1 **+10-20%**; 1K **+10-17%**; 16K **+10-19%**; c4/c8 +8-15% | L (2-3 wk) | M (numerics; r22/suite gate) | High |
| 2 | **Marlin W4A8-FP8 bounded probe** (AWQ uint4 + zp, per-token FP8 A) | short 0-5%; 16K **up to +15-30%** if the MLP runs 1.4-1.7x | L (4-6 wk) | H (E4M3 quality) | Low-Medium |
| 3 | **FlashInfer / varlen / split-KV attention** (or FA2-class library path) | 4K/8K/16K **+2-8%**; short <1% | M (1-2 wk) | L-M (mature, fp32 softmax) | Medium-High |
| 4 | **GDN scratch pooling + conv/prepare rewrite** | 16K **+3-5%**; 1K +2-3%; short +1-2% | S (2-5 d) | L (no math change) | High |
| 5 | **CUDA graphs with bucketed shapes** | c1 **+3-8%**; c4/c8 +2-4%; 1K +1-4%; 16K +0-2% | M (1-2 wk) | L-M (static metadata/pools) | Medium |
| 6 | **Prefix cache for long shared prefixes** (KV + GDN state + conv tail) | up to **2x** on an 8K doc with a 4K shared prefix; 0 on unique prompts | L (3-4 wk) | M (workload-dependent) | Medium |
| 7 | **Larger-M Marlin tile / custom CUTLASS W4A16** (raise MLP ~176 -> 220+ TF) | 1K/16K **+8-12%**; short ~0 | L (3-5 wk) | M-H (new kernel) | Medium |

**Marlin is already persistent, weight-stationary, and split-K.** Its grid is
`SMs x blocks_per_sm`, it loops M inside the tile to overlap dequant with MMA, and
it splits K with an fp32 global reduce (atomic-add is disabled for BF16 on SM8x).
"Do a new persistent Marlin" is not a win; only the larger-M tile/kernel ceiling
in rank 7 is.

Ranks 1-2 are the two experiments with the highest expected value. Rank 4 pays for
itself in instrumentation and should land before either. Rank 3 is the safest
attention step; rank 5 must wait until kernel shapes and pools stop changing.

## Mechanism: chunked GDN (rank 1)

The current `gdn_recurrent_split` kernel walks T serially inside one CTA per
(head slice, sequence), keeping the f32 state in shared memory. At 1K that is
1,024 dependent steps; at 16K, 16,384. It costs 163.5 ms at 1K and 2,566 ms at
16K for ~0.5% of peak.

The FLA/vLLM formulation reassociates the same delta rule into chunk-parallel
tensor-core work plus a short state scan, with **chunk size 64**:

1. `A = strict_lower(diag(beta) K K^T)` and solve `(I - A) u = diag(beta) v`
   (WY/UT representation, forward substitution in f32).
2. Intra-chunk output `O_intra = (Q K^T * decay_mask) u`, lower-triangular.
3. State propagation `h_{c+1} = exp(g_cum_c) h_c + K^T (u exp(-g_cum_c))`.
4. Inter-chunk output `O_inter = (Q exp(g_cum)) h_c`; `O = O_intra + O_inter`.

This gives a dependency chain of **T/64 chunk-state updates** instead of T token
updates, and the heavy work is shaped for tensor cores. The FLA implementation is
MIT-licensed (vLLM's copy is Apache-2.0 with MIT attribution), so porting is
legally clean: either vendor/AOT the Triton kernels or rewrite the five kernel
families in CUDA/C++.

Numerics: keep the state `h`, beta/decay, the triangular solve, and both dot
products in **f32**. Do **not** use a BF16 state. FLA itself runs the chunked path
in BF16/FP16 and allocates the inter-chunk state in the input dtype, with only the
final state in f32, so the chunked result is not bit-identical to the sequential
f32 reference even when ported carefully. The existing 4-lane split already broke
suite row r22 at 54 tokens, so the gate must include r22, suite28, and 36/76
golden head logits.

## Mechanism and gate: W4A8-FP8 (rank 2, bounded probe only)

The vendored Marlin sources show the kernel path already exists:

- FP8 activations are gated to Ada, and then **SM89/SM12x only** (`marlin.cu`).
- Instantiated kernels include `kFE4M3fn x kU4` (AWQ with zero points) and
  `kFE4M3fn x kU4B8` (GPTQ), using `mma.m16n8k32.e4m3.e4m3.f32` - twice the K per
  instruction of BF16 `m16n8k16`.
- Zero-point + FP8 dequant is implemented (`dequant.h`, `marlin_template.h`).

What is missing for this checkpoint:

1. **Python rejects it.** `apply_gptq_marlin_linear` asserts `uint4b8` for FP8
   input and rejects `uint4`/zero points. `compressed_tensors_wNa16` can pass
   `act_type=fp8` once the environment enables it.
2. **Per-token FP8 activation quantization** and per-token scales, plus the
   epilogue scaling.
3. **8-bit repack/preprocess**: B packing, scale permutation, and the B stride
   change for `is_a_8bit`; a plain W4A16 repack is not reusable. The symmetric
   `uint4b8` path also runs `marlin_int4_fp8_preprocess` and scales by 512; for
   `uint4` + zero points that requirement must be verified empirically.
4. No new CUDA kernel is needed for the probe.

Risk is high: E4M3 has 3 mantissa bits and a few-percent per-element relative
error; this model already shifts probabilities under batching. A top-1-only pass
is not enough. Gate on suite28 (top-1 28/28, max delta <=0.011, MAE <=0.001),
r22, corpus-20 KL, and a fallback to W4A16. Run it as a bounded probe on one layer
and one prefill shape, in parallel with ranks 1/3, not as a commitment.

## SM90-only and inapplicable options

Do not spend time on these on SM89:

- **Machete** - CC 9.0 only.
- **vLLM CUTLASS W4A8** - SM90 only, rejects zero points, group 128 only.
- **CUTLASS 3.x mixed-input int4 x BF16** - Hopper example/kernels; no ready Ada
  kernel.
- **FlashAttention-3/4** - SM90/SM100; SM89 resolves to FA2.
- **FlashInfer GDN prefill** - SM90/Blackwell (and SM100/120 for this head dim);
  SM89 falls back to Triton/FLA.
- **CuteDSL GDN** - SM100 only.
- **TMA/warp-specialized persistent kernels, XQA decode** - SM90+; decode-only
  paths are irrelevant to a prefill server.

External/requantization options are not drop-in:

- **BitBLAS** - removed from vLLM 0.31; external TVM runtime, no vendored cubins.
- **QuIP#** - 2-bit lattice format, not AWQ; requires requantization.
- **llama.cpp MMQ** - GGUF quant formats, not AWQ; requires requantization.
- **Conch** - accepts groups -1/128 only, so group 64 is rejected without a
  kernel change.
- **TensorRT-LLM** - credible Ada W4A16 AWQ support, but a separate engine and
  build/runtime with no Qwen3.5 support listed; very high integration effort.

## Phased plan and gates

Each phase must pass the shared acceptance gate before the next starts.

**Phase 0 (2-5 days) - cheap fixes, low risk**
- Pool/reuse GDN f32 scratch; zero only `state`.
- Tile conv over (channel, time) and fuse the prepare epilogue; stop the
  ~2.8/16 GB/s launches.
- Add profiling that survives `forward_dev` clearing so embedding and the 16K
  wall-vs-region gap are visible.
- Gate: bit-identical outputs on 36/76 and all probes; 16K >=3% faster; 1K no
  regression.

**Phase 1 (2-3 weeks) - chunked GDN**
- C=64, WY/UT solve, f32 state and accumulators; A/B flag plus the current shared
  kernel as the short/sensitive fallback.
- Gate: suite28 28/28, max delta <=0.011, MAE <=0.001; r22 passes on the
  production path; 36/76 top-1 identical; 4K/8K/16K probes no worse; c1 >=10%,
  1K >=10%, 16K >=10%. Target c1 ~50-56 ms, 1K ~405-438 ms, 16K ~7.1-7.9 s.

**Phase 2 (1-2 weeks) - attention**
- First varlen/batched grid (one launch per batch, GQA-aware KV loads); then
  split-KV for single-request 4K-32K. Prefer integrating a mature FA2-class or
  FlashInfer path over writing a new head_dim-256 tile kernel.
- Gate: fp32 softmax reduction; identical top-1 and probability delta within the
  existing BF16 noise; c8 >=1%, multi-long >=3%, 4K/8K/16K >=2%.

**Phase 3 (1-2 weeks) - CUDA graphs**
- Capture/replay per (batch, bucketed sequence length) after pools are stable;
  eager fallback on a shape miss.
- Gate: c1 >=3%, c8 >=1%, outputs identical per shape; no capture failures or
  recapture in the steady-state sweep.

**Phase 4 (4-6 weeks) - W4A8-FP8 bounded probe**
- Patch the one-layer allow-list and 8-bit repack for uint4+zp; per-token FP8
  quantization; compare against BF16 Marlin on suite28, r22, and corpus KL; one
  short and one 16K shape.
- Gate: suite28 and corpus gates pass, else stop and keep W4A16 default.

**Phase 5 (research) - larger-M CUTLASS W4A16 and prefix cache**
- Larger-M/custom CUTLASS kernel targeting 220+ TFLOPS; prefix cache keyed on the
  exact token prefix storing full-attn K/V, GDN state, and conv tail.
- Gates: CUTLASS - 1K/16K >=8%, suite passes; prefix cache - exact top-1 and
  delta 0 on a repeated-prefix benchmark, >=1.8x on 4K-prefix + 8K doc, peak
  memory <=21 GiB.

### Acceptance tests (all phases)

- `suite28`: top-1 **28/28**, max probability delta **<=0.011**, mean MAE
  **<=0.001** (existing gates); include **r22** explicitly.
- Batch-of-one bit-identical; batch>1 top-1 stable.
- Long probes **4K / 8K / 16K / 32K** and the **35,310-token** verified case:
  200, choice true, no OOM.
- Short sweep c1/c4/c8 at ~70-token prompts, median of three runs, plus the c16
  plateau check.

### Targets

If Phases 0-2 land, the credible native targets are c1 **~50-56 ms** (from 61.9),
1K **~405-438 ms** (from 488), 16K **~7.1-7.9 s** (from 8.62-8.97), and a small
c8 gain. That would narrow, not erase, the vLLM long-prefill lead (1.6-1.7x) and
would still trail vLLM at c16. W4A8-FP8 is the only candidate with a chance of a
larger 16K jump, and only if its quality gate passes.

## Short-prompt caveat

A 70-token prompt is only **2 chunks** at C=64. The chunked path adds fixed
overhead (`solve_tril`, WY, an extra output pass) and may be neutral or slower
than the current shared kernel at that length. Do not assume the 1K/16K gain
applies to c1. Keep the current shared kernel for **T < 128 tokens** until a
measured A/B says otherwise, and judge the chunked path on the full gate, not on
a microbenchmark.

## What not to do

- **Do not atomically/globally split the GDN state** to parallelize T.
  Nondeterministic reduction order will very likely break suite/r22.
- **Do not make FP8 the default** before a calibration + suite28 + corpus-20
  quality gate with a W4A16 fallback.
- **Do not raise `MAX_BATCH` to chase vLLM's c8/c16 throughput.** The single FIFO
  worker is the ceiling; larger batches only lengthen each forward (measured:
  12/16 increased p50 with no throughput gain).
- **Do not port Machete, FA3/FA4, or other SM90-only kernels.**
- **Do not chase the `gemm.b` profile row**; it is profiler event-gap overhead.
- **Do not CUDA-graph capture while kernels or allocation pools are still
  changing**; capture per bucketed shape after Phase 0.
- **Do not claim an overall native win** from the c2/startup/memory/35K results;
  vLLM remains faster at c1, c8, c16, and long prefill.
