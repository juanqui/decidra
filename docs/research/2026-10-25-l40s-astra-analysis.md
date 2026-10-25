**The best L40S plan starts with memory and batching, then attacks the group-64 GEMM implementation.** Batched attention, cross-state execution, and a different GDN algorithm are substantial follow-ups. CUDA graphs should follow evidence of launch gaps, not precede that work.

I inspected the requested reports, hot-path sources, quantization documentation, benchmark harnesses, and local checkpoint metadata at commit `7291a337e489cdce5471e9bc2c1de39df45fce0b`. I changed no files. The NVIDIA driver was unavailable in this environment, so all GPU timings below are repository-reported measurements; new performance ranges are estimates.

**L40S is SM89, and the dense TF32 correction matters.**

| Resource | L40S target | Implication |
|---|---:|---|
| Compute capability | 8.9, Ada | Compile and dispatch SM89 |
| SMs | 142 expected; verify runtime | 10.9% more than 4090’s 128 |
| Memory | 48 GB GDDR6, ECC | Spend capacity on embedding, caches, selected expanded weights |
| Bandwidth | 864 GB/s | 14.3% below 4090’s 1,008 GB/s |
| Maximum power | 350 W | Historical 600 W 4090 results are not matched-power comparisons |
| FP32 CUDA cores | 91.6 TFLOPS | Relevant to recurrence and group-scale processing |
| TF32 tensor | **183 dense / 366 sparse TFLOPS** | 366 is not the dense ceiling |
| FP8 tensor | 733 dense / 1,466 sparse TFLOPS | Requires suitable representation and measured accumulation behavior |
| INT8 tensor | 733 dense / 1,466 sparse TOPS | Relevant reference ceiling for current W4A8 |
| NVLink / MIG | Neither | One GPU, shared-weight execution and software admission control |

These specifications come from [NVIDIA’s L40S page](https://www.nvidia.com/en-us/data-center/l40s/) and [compute-capability table](https://developer.nvidia.com/cuda/gpus). NVIDIA’s current table lists BF16/FP16 as **362.05 dense / 733 sparse**, an internally imperfect doubling; use approximately 362–366 TFLOPS as the advertised dense range and measure the actual instruction/accumulator path.

Do not apply the 4090’s floating-point tensor limits blindly: NVIDIA explicitly lists its BF16/F32-accumulate ceiling as 165.2 TFLOPS and FP8/F32-accumulate as 330.3 TFLOPS. Consequently, a 4090 rejection is useful evidence, but not a conclusive L40S result for a materially different floating-point GEMM implementation. [Ada whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

SM89 retains **48 warps, 64K registers, 100 KiB shared memory per SM, and 99 KiB usable shared memory per block**. More GPU memory does not enlarge these resources. Use `mma.sync`, `ldmatrix`, and `cp.async`; Hopper WGMMA/TMA/FA3 and Blackwell NVFP4/TMEM/tcgen05 designs are not L40S implementations. SM120 and SM100 are also distinct targets. [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

**The historical baseline needs two corrections before comparison.**

The journal’s 879.8 ms HTTP result uses duplicate-control questions. The distinct-question HTTP result is 901.4 ms. Also, the 328/95/44.7/112 ms region profile was collected before the final accepted optimization set; it is prioritization evidence, not a fresh decomposition of 708.91 ms.

| Reported 4090 measurement, 600 W configured cap | Result |
|---|---:|
| Distinct engine, original | 752.69 ms |
| Distinct engine, accepted, host embedding | 708.91 ms |
| Distinct engine, accepted, GPU embedding | **689.63 ms** |
| Duplicate-control HTTP, accepted | 879.8 ms; approximately 73 q/s |
| Distinct HTTP, accepted | **901.4 ms; approximately 71 q/s** |
| Distinct HTTP c8 | 4.04 s reported request latency |
| Eight warm distinct states × eight questions, c8 | 55.5–59.0 q/s |
| Approximately 8K state, duplicate-control 64q | 2,936 ms cold / 998 ms warm |
| Approximately 24K state, duplicate-control 64q | 8,206 ms cold / 1,242 ms warm |
| Text / image decisions | 32/32 and 13/13 |

Sources: [final accepted matrix](/home/juanqui/Git/jev-engine/docs/research/2026-10-25-rtx4090-optimization-journal.md:210), [long-context and mixed-state results](/home/juanqui/Git/jev-engine/docs/research/2026-10-25-rtx4090-optimization-journal.md:280).

I independently confirmed that the distinct engine fixture has 2,121 state tokens, 64 unique questions, and 3,334 question-token rows. The checked-in 256/8K/24K HTTP controls each repeat one question 64 times. Thus **64 questions is approximately M=3,334 for these linears, not M=64**.

The checkpoint contains approximately 24.351 billion packed linear weights. At 3,334 rows, that is approximately **162.37 tera-operations**. Even an impossible all-INT8, peak-throughput execution needs approximately **222 ms** at 733 TOPS, before attention, GDN, activation processing, or serving overhead. That is a lower bound, not a latency prediction.

Using the old region times only illustratively:

- A 1.5× MLP improvement saves approximately 109 ms.
- A 2× recurrence improvement saves approximately 48 ms.
- A 2× attention improvement saves approximately 22 ms.

This is why the major compute investment belongs in linears.

**Several current choices are directly relevant to L40S; two are actual correctness hazards.**

| Finding | Evidence and consequence |
|---|---|
| **Cache eviction can invalidate a saved index** | `run_group` finds `cached`, then the `PIN=0` branch removes entries before using that index. Its `cached_bytes` value also remains stale during eviction. This can over-evict, select the wrong entry, or panic. Default pinning avoids this branch; fix it before eviction-policy experiments. [batch.rs:292](/home/juanqui/Git/jev-engine/src/batch.rs:292) |
| **FlashInfer workspace is unsafe for concurrent ownership** | Function-static workspace, size, SM count, and stream; a stream change frees the allocation. Safe use currently depends on serialization. Concurrent forwards require engine/device/stream-owned workspace. [fi_prefill.cu:46](/home/juanqui/Git/jev-engine/native/third_party/flashinfer/fi_prefill.cu:46) |
| L40S inherits a 24 GB-oriented cache budget | Defaults are 5 GiB GPU cache, 16 entries, 8 GiB host cache, and GPU embedding disabled. [config.rs:23](/home/juanqui/Git/jev-engine/src/config.rs:23), [config.rs:125](/home/juanqui/Git/jev-engine/src/config.rs:125) |
| Long states fragment question passes | `MAX_KEYS=262144` counts logically duplicated prefix keys. With 58-token questions, approximately 8K permits 31 questions/pass; 24K permits ten. A 64q request becomes three or seven passes. These are proxies, not actual FlashAttention allocation sizes. [engine.rs:4784](/home/juanqui/Git/jev-engine/native/src/engine.rs:4784) |
| Scheduler admission counts questions, not token work | A soft 32-question budget and 2 ms window govern collection; distinct states then execute serially. State length, question-token count, and required scratch are not jointly budgeted. [batch.rs:108](/home/juanqui/Git/jev-engine/src/batch.rs:108), [batch.rs:257](/home/juanqui/Git/jev-engine/src/batch.rs:257) |
| GDN exposes only 96 CTAs for one sequence | VC=64 gives 48 heads × two value slices, below L40S’s 142 SMs. This affects one-question recurrence and cold-state chunked GDN; large question batches have many more CTAs. [kernels.rs:1874](/home/juanqui/Git/jev-engine/native/src/kernels.rs:1874), [kernels.rs:2496](/home/juanqui/Git/jev-engine/native/src/kernels.rs:2496) |
| One long question disables the compact short path for the batch | `all(length <= 256)` controls compact Q/K and direct seeded recurrence. A 257-token outlier can restore expanded buffers and state-producing execution. [engine.rs:2149](/home/juanqui/Git/jev-engine/native/src/engine.rs:2149) |
| Marlin’s resource use warrants investigation | Hardcoded specializations, 256 threads, 99 KiB reservation, one persistent CTA/SM. Historical disassembly found 230 registers/thread for the large-M specialization. That gives eight resident warps, approximately 16.7% theoretical warp occupancy; it does **not** prove low tensor utilization. [kernels.rs:273](/home/juanqui/Git/jev-engine/native/src/kernels.rs:273), [kernels.rs:1178](/home/juanqui/Git/jev-engine/native/src/kernels.rs:1178) |
| Repeated quantization and metadata work remain | Gate/up independently quantize the same input. Question RoPE tables are regenerated inside each attention layer. [engine.rs:2712](/home/juanqui/Git/jev-engine/native/src/engine.rs:2712), [engine.rs:1864](/home/juanqui/Git/jev-engine/native/src/engine.rs:1864) |
| Cold capture always selects allocation-conservative behavior | Scratch is released even when 48 GB could support reuse. Test a measured capacity-aware policy. [engine.rs:4561](/home/juanqui/Git/jev-engine/native/src/engine.rs:4561) |
| GDN diagnostic microseconds assume 2.91 GHz | Dividing `clock64` cycles by `2.91e3` is wrong for an L40S running another clock. Use events or measured clock-aware reporting. [engine.rs:1173](/home/juanqui/Git/jev-engine/native/src/engine.rs:1173) |
| Existing benchmark gates are insufficient for this task | Comparator defaults allow 95% agreement and compare only overlapping IDs. HTTP `--runs` repeats c1, while each c8 level is one burst. Require complete fixture coverage and explicit 32/32. [d3_compare.py:34](/home/juanqui/Git/jev-engine/eval/d3_compare.py:34), [d3_bench.py:167](/home/juanqui/Git/jev-engine/eval/d3_bench.py:167) |

Two things are already correct: GEMM launches use the queried SM count, and FA3 activation is restricted to SM90. There is no 128-SM GEMM launch bug to fix. The fixed 128-block repack launch is startup work. [cuda.rs:72](/home/juanqui/Git/jev-engine/native/src/cuda.rs:72), [engine.rs:815](/home/juanqui/Git/jev-engine/native/src/engine.rs:815)

The SM89 profile’s `shared_memory_bytes=100 KiB` is ambiguous rather than an active launch failure: current Marlin correctly reserves 99 KiB. Separate per-SM capacity from per-block limits before using this field in an autotuner.

**Memory should have an explicit allocation budget.**

From the actual dimensions:

```text
state payload =
    64 KiB × state_tokens
  + 144 MiB of F32 GDN seeds
  + 2.8125 MiB of convolution history
```

The scheduler conservatively estimates the fixed portion as 160 MiB.

| State tokens | Actual tensor payload | Capacity under current 5 GiB / 16-entry limits | Under proposed 16 GiB / 64-entry limits* |
|---:|---:|---:|---:|
| 256 | 162.8 MiB | 16 | 64 |
| 2,048 | 274.8 MiB | 16 | 56 |
| 8,192 | 658.8 MiB | 7 | 24 |
| 24,576 | 1,682.8 MiB | 3 | 9 |

\*Calculated using the conservative scheduler estimate; these are budget capacities, not demonstrated residency.

Local metadata also gives:

- GPU embedding: **2.368 GiB**.
- BF16 vision tensors: **0.858 GiB**.
- All linear weights expanded to one byte: **22.678 GiB**.
- All linear weights expanded to BF16: **45.356 GiB**, before everything else.

Start with a 16 GiB state-cache experiment, GPU embedding, and at least the measured peak scratch requirement plus a 2 GiB margin. The present 24K reserve is 5.5 GiB, but larger question batches need a newly measured reserve. Preserve an **eight-distinct-24K-state working-set target** before spending remaining memory on graphs or alternate weight copies.

**Round 0 establishes the rental baseline and acceptance contract.**

Run one isolated L40S at its supported 350 W cap, ECC enabled. Record actual sustained clocks and power, not only configured limits.

| Baseline dimension | Required cases |
|---|---|
| Hardware | L40S 350 W; matched 4090 350 W if available; historical 4090 600 W separately |
| Question count | **1, 8, 16, 32, 64** |
| State length | **256, approximately 2K, 8K, 24K**, recording actual token counts |
| Cache condition | **Cold capture, GPU-resident warm, verified host restore** |
| Request concurrency | **c1 and c8** |
| State identity | Same state; eight distinct states; mixed-length state stream |
| Images | Text-only; image-bearing states; repeated image across changed text; changed image with identical text |
| Embedding | Host baseline first; GPU variant separately |
| Measurements | Engine and HTTP p50/p95/p99, completed q/s, token rows/s, queue time, peak VRAM, cache hits/restores, joules/question |

This is the mandatory **full matrix** referenced below: all five question counts × four state sizes × three cache conditions × c1/c8, for both shared-state and mixed-state traffic, plus the image cases and parity suite. Screen microkernel candidates narrowly, but run the full matrix before promoting a candidate.

Qualification details:

- Capture GPU UUID, CC, SM count, L2 size, shared-memory limits, register limits, ECC, PCIe negotiated width/speed, driver/toolkit/compiler versions, power, clocks, temperatures, and throttling reasons.
- Freeze checkpoint, tokenizer, readout, binary, cubin, and fixture hashes. Build compiler experiments into separate output directories: the timestamp shortcut can otherwise reuse old cubins. [build.rs:73](/home/juanqui/Git/jev-engine/native/build.rs:73)
- Reprofile the accepted L40S baseline with Nsight Systems and selected Nsight Compute kernels. Collect tensor activity, achieved occupancy, eligible warps, spills, barrier stalls, L2/DRAM traffic, and CPU submission gaps.
- Measure capture completion with CUDA events or explicit boundary synchronization. The existing `d3_perf` capture stopwatch has no explicit completion synchronization.
- Use at least three interleaved A/B sessions, 20 engine repetitions per cell, and sustained HTTP runs. Treat a one-burst c8 result as preliminary; require at least 1,000 completed requests for meaningful tail acceptance.
- For cold tests, use unique run identities or disable persistent state caching. Existing deterministic cold nonces can become hits across repeated invocations.
- Force host eviction and verify restore events; never infer host restore merely from a slow request.

An existing engine baseline invocation, intended for the rental environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
DECIDRA_CUDA_ARCH=sm_89 \
DECIDRA_MARLIN=1 DECIDRA_INT8=1 DECIDRA_INT8_GROUP=1 \
DECIDRA_INT8_EXCLUDE=self_attn.v_proj,self_attn.k_proj \
DECIDRA_INT8_GROUP_EXCLUDE='' DECIDRA_INT8_CLIP=1 \
DECIDRA_PRELOAD=1 DECIDRA_GPU_EMBED=0 \
DECIDRA_GDN_CHUNKED=1 DECIDRA_GDN_CHUNK_MIN=16 \
DECIDRA_MARLIN_CHUNK=0 \
DECIDRA_D3_MODEL=/srv/models/d3-turbo-w4a8-vision-v2 \
DECIDRA_D3_PROBE="$PWD/eval/data/d3_distinct_2k_64q.json" \
DECIDRA_D3_REPEAT=20 \
cargo test --release -p decidra-native --test d3_perf -- --ignored --nocapture
```

For the isolated production server:

```bash
python3 eval/d3_bench.py \
  --url http://127.0.0.1:8899 \
  --request eval/data/d3_distinct_http_2k_64q.json \
  --questions 1,8,16,32,64 --concurrency 1,8 \
  --warmup 5 --runs 30 --out /tmp/l40s-distinct-baseline.json
```

The latter remains a smoke benchmark until concurrency repetition is corrected. The server flag for embedding is **`DECIDRA_NATIVE_GPU_EMBED`**; the native performance-test flag is **`DECIDRA_GPU_EMBED`**.

Every subsequent experiment inherits these acceptance requirements:

- **32/32 text decisions**, exact coverage of all 32 IDs, mean MAE ≤0.02, non-near-tie maximum Δp ≤0.10, and zero unsafe flips. Use `--min-agree 1.0`; independently verify coverage.
- **13/13 image decisions**, existing image probability/score parity, cache-identity checks, and BF16 vision preserved.
- State-once, batch/single, seed/history negative controls, permutation, isolation, eviction/restore, and replay tests. Preserve the existing INT8 probability envelope of 0.05.
- Reduction/precision changes additionally need 256–1,024 held-out decisions spanning long states, images, near ties, and choice/noul/score outputs. No degradation hidden by averaging.
- Performance promotion requires repeatable improvement beyond noise: normally **≥2% end-to-end**, or a specifically declared capacity/tail benefit. Complex kernels also need **≥5% targeted-region improvement**.
- Reject unexplained >2% throughput regression or >5% p95 regression in protected cells; use shape-specific dispatch where warranted.
- Preserve long-context capacity and zero OOM/retry regressions under the eight-state 24K target.

The baseline itself must pass on L40S before tuning. Changing from 128 to 142 SMs can change persistent/split-K reduction scheduling even with identical source.

**Round 1: spend memory on proven work removal and eliminate artificial batching limits.**

**Experiment 1 — GPU embedding and a larger state-cache budget.**

- **Hypothesis:** the measured embedding win survives on L40S without forcing long states into host memory.
- **Implementation:** use existing `Engine::preload_all` and `embed`; sweep embedding off/on independently from cache budgets of 5/12/16/20 GiB and entry limits of 16/32/64. Select capacity from measured free memory and scratch, not the SM89 label alone. [engine.rs:1265](/home/juanqui/Git/jev-engine/native/src/engine.rs:1265), [engine.rs:1364](/home/juanqui/Git/jev-engine/native/src/engine.rs:1364), `src/config.rs`.
- **Benchmark:** full matrix; add working-set sweeps of 1/8/16/32/64 states, with all-eight-24K residency explicitly checked.
- **Expected ROI:** approximately 2–4% warm engine improvement from embedding, anchored by the measured 19.28 ms final-path saving on 4090. Cache gains depend on avoided restores and can dominate mixed workloads.
- **Risk/rollback:** low numerical risk; moderate memory pressure. Revert flags and budgets independently.
- **Gate:** ≥2% engine improvement for embedding without long-context regressions; for larger cache, demonstrate the intended resident capacity and ≥10% mixed-working-set p95 improvement.

**Experiment 2 — token-aware batch limits and measured scratch reservation.**

- **Hypothesis:** 8K/24K throughput improves substantially when 64 questions no longer require three/seven complete model passes.
- **Implementation:** revise `state_once_batch_rows`, `pass_reserve`, and worker collection. Budget flattened token rows, largest question, attention workspace, and cache residency separately. Sweep state-once question caps 16/32/64/128 and scheduler budgets 32/64/128/256. Internal coalescing may exceed 64 questions while the API remains capped at 64/request.
- As an initial bounded configuration test, raise `MAX_KEYS` to `1<<21`; keep score limits explicit and sweep them separately. At 24K with ordinary 58-token questions, 64 questions fit that key budget. Do not remove all safeguards.
- **Benchmark:** full matrix; emphasize 8K/24K 32q/64q and c8. Add 255/256/257-token questions and one long outlier among short questions.
- **Expected ROI:** estimated 5–25% in currently fragmented long-state cells; little benefit expected for ordinary 2K/64q.
- **Risk/rollback:** medium: larger GEMM shapes alter reductions; memory estimates may fail. Retain old caps and dispatch fallback.
- **Gate:** ≥5% warm 8K/24K HTTP improvement, ≤5% c1/p95 regression elsewhere, and no reduced resident-state target.

The 32-row accuracy fixture is independent of these scheduler limits. Larger batches never excuse losing its 32/32 result.

**Round 2: improve projection execution without changing the quantized artifact.**

**Experiment 3 — share activation quantization and hoist immutable metadata.**

- **Hypothesis:** gate/up and compatible GDN projections can share quantized activation bytes; repeated RoPE construction is unnecessary.
- **Implementation:** add a prepared-activation object spanning multiple calls to `marlin_int8_launch`. Quantize once, retaining separate module scale arrays. In `quant_int8_group_bf16`, quantized bytes do not depend on `a_global`; stored scales do. Preserve that multiplication and rounding rather than silently sharing incompatible scales. Hoist question positions/RoPE once per pass. [engine.rs:4216](/home/juanqui/Git/jev-engine/native/src/engine.rs:4216), [kernels.cu:64](/home/juanqui/Git/jev-engine/native/kernels/kernels.cu:64), [weights.rs:718](/home/juanqui/Git/jev-engine/native/src/weights.rs:718).
- **Benchmark:** full matrix plus quantize→gate/up and quantize→QKV/z timing at actual flattened M values.
- **Expected ROI:** estimated 1–4% end-to-end; inexpensive relative to new GEMM development.
- **Risk/rollback:** low-to-medium; scale contracts and scratch lifetimes matter. Keep original per-module quantization available.
- **Gate:** ≥5% quantization/metadata-region improvement and ≥2% end-to-end, unless the simpler implementation independently enables accepted graph capture.

**Experiment 4 — rebuild Marlin dispatch around validated SM89 specializations.**

- **Hypothesis:** group-scale processing, register pressure, and persistent scheduling leave recoverable performance on 142 SMs.
- **Implementation:** change the actual Rust-selected symbols and launch metadata in `native/src/kernels.rs`, instantiations in `native/kernels/marlin.cu`, `matmul_a8` in `marlin_template.h`, and matching repack/scale logic in `weights.rs`.
- Sweep supported M/N/K tiles, 128/256 threads, two/three/four stages, and one/two resident CTAs where registers and shared memory permit. Compute actual shared-memory needs; adjust grid, lock indexing, and reduction workspace together.
- Keep the current N=256 path as the first reference. An N=128 retry requires a demonstrably different, validated layout. The invalid 401 ms result supplies **no usable speed evidence**; the validated restricted K8 version was slower.
- Test real captured activations, basis patterns, every nibble/zero-point case, group boundaries, M tails, and current integer scale interpretation. These scales contain integer bit patterns, not ordinary BF16 scale values.
- **Benchmark:** full matrix plus every listed projection shape: MLP `5120→17408→5120`, GDN `5120→10240/6144/48`, and attention `5120→12288/1024`, `6144→5120`. Include M near 48–58, 256/512/1024/1667/3334, and cold-state sizes.
- **Expected ROI:** estimated 5–15% end-to-end if a 10–25% dominant-linear improvement exists; high development cost.
- **Risk/rollback:** high layout/reduction risk. Version repacks and dispatch by specialization; retain the current kernel per shape.
- **Gate:** ≥10% improvement on dominant projection families and ≥5% engine improvement, with full parity. Reject lower occupancy claims unsupported by elapsed-time improvement.

Editing only C++ `determine_exec_config` will not tune the current direct Rust launch path.

**Round 3: batch attention while preserving block visibility.**

**Experiment 5 — shared-prefix, private-question attention.**

- **Hypothesis:** remove approximately 1,024 attention calls and 2,080 K/V copies per unsplit 64q pass, while improving prefix reuse.
- **Implementation:** extend `full_attention_batch`, `native/src/flashinfer.rs`, and `fi_prefill.cu` with an engine-owned workspace and one of two implementations:
  - A segmented/paged ragged kernel referencing shared state and private suffix directly.
  - Shared-state attention over flattened queries, separate private-question attention, then LSE-weighted merge.
- For the latter, use `L=logaddexp(Ls,Lq)` and `O=exp(Ls-L)Os+exp(Lq-L)Oq`, with the implementation’s logarithm convention. Preserve noncausal visibility within each question, 24:4 GQA, head dimension 256, and image-derived positions. [FlashInfer recursive-attention formulation](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
- Plan splitting across the whole batch. Reusing each single-question split factor can create excessive partial outputs.
- **Benchmark:** full matrix, especially 1q versus 64q at 2K/24K; ragged lengths and adversarial cross-question leakage tests.
- **Expected ROI:** estimated 2–5% at 2K; potentially 10–25% at 24K, subject to the new profile. A 2× attention kernel alone cannot deliver a large 2K whole-pass gain.
- **Risk/rollback:** medium-to-high reduction-order and masking risk. Retain current concatenation/single-request attention.
- **Gate:** ≥20% attention-region improvement including packing/merge, and ≥2% end-to-end at 2K or ≥5% at 24K.

**Round 4: recover concurrency by batching across states.**

**Experiment 6 — one shared-weight execution across multiple state IDs.**

- **Hypothesis:** mixed 1–8q traffic loses throughput because each state group runs separate linears.
- **Implementation:** change `run_jobs`/`run_group`, `BatchInfo`, `LayerIo`, `run_layer_batch`, and seeded convolution/GDN/attention interfaces. Carry per-sequence state ID, seed/history pointers, KV ranges, and `pos_end`; concatenate token rows for linears and route state-dependent work afterward.
- Hold participating caches alive for the pass. Budget aggregate token rows and scratch. Bucket short versus >256-token GDN work without splitting every projection. Preserve response order and cancellation.
- Sweep the batching window through 0/250/500/1000/2000 µs with bounded age-based scheduling.
- **Benchmark:** full matrix; reproduce eight distinct warm states × eight questions, then mix 256/2K/8K/24K and cold/image jobs with warm short jobs.
- **Expected ROI:** estimated 10–25% mixed-state throughput gain; larger gains are plausible for sparse 1q traffic. The measured 55.5–59 versus 73 q/s gap motivates this, but is not a perfectly matched-work bound.
- **Risk/rollback:** medium numerical risk, high routing/lifetime risk. Fall back to exact-state groups.
- **Gate:** ≥15% c8 mixed-state q/s improvement, ≥10% mixed p95 improvement, no starvation, and ≤2% same-state throughput loss.

Do this before multiple compute streams. The engine shares scratch, Marlin locks, BLAS handles, pools, and profiling state; extra streams alone cannot safely create concurrent forwards.

**Round 5: make GDN use L40S effectively, then test an algorithmic replacement.**

**Experiment 7 — repair small-grid and mixed-length inefficiencies.**

- **Hypothesis:** VC=64 underfills L40S for one sequence, and a long outlier unnecessarily penalizes short questions.
- **Implementation:** specialize the seeded register kernel and `gdn_chunked_tc_body` for VC=32 and appropriate thread counts, producing 192 CTAs instead of 96. Inspect actual registers/spills. Dispatch short and long sequence subsets separately; retain compact Q/K and immutable seeds for eligible questions. [kernels.cu:2220](/home/juanqui/Git/jev-engine/native/kernels/kernels.cu:2220), [kernels.cu:1647](/home/juanqui/Git/jev-engine/native/kernels/kernels.cu:1647).
- **Benchmark:** full matrix, emphasizing 1q cold/warm, and lengths 1/15/16/31/32/33/63/64/65/255/256/257.
- **Expected ROI:** estimated 5–20% targeted GDN improvement in underfilled cells; approximately 0–3% for already well-filled 64q passes.
- **Risk/rollback:** medium. Keep VC64 and existing dispatch per shape.
- **Gate:** ≥10% GDN-region improvement and ≥2% end-to-end in the selected cells. Reject spills or extra launch costs that erase the gain.

**Experiment 8 — output-only short-block WY GDN; separately optimize cold state updates.**

- **Hypothesis:** question outputs can be computed using seed projections, a triangular solve, and matrix products without constructing a final recurrent state.
- **Implementation:** add a new inference-only path beside `gdn_batch`; use C=16/32/64, compact key heads, and grouped `K·S0`/`Q·S0` products. Compute decay ratios from differences of cumulative log-decays. Start with an FP32 reference, then evaluate TF32 or BF16 operands with F32 accumulation.
- For cold capture, investigate the scalar outer-product update at `kernels.cu:1849`, preserving the final state. Do not omit capture states or blindly materialize dense transitions for every time chunk. [state-update implementation](/home/juanqui/Git/jev-engine/native/kernels/kernels.cu:1849), [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464).
- **Benchmark:** full matrix; real nonzero seeds, extreme decays, long-state captures, and kernel time including packing, solves, projections, and outputs.
- **Expected ROI:** estimated 1.3–2× recurrence improvement if successful, roughly 3–7% warm whole-pass gain using the old profile. Cold gains require separate measurement.
- **Risk/rollback:** high numerical risk; use a measured crossover and retain register recurrence.
- **Gate:** ≥25% complete GDN-region improvement and ≥3% end-to-end; cold variant additionally needs ≥5% cold-request improvement. Extended parity is mandatory.

The already-rejected “enable chunked GDN for short questions” and global FMA switches are not substitutes for this experiment.

**Round 6: capture stable question execution in CUDA graphs.**

**Experiment 9 — bounded graph cache with stable buffers and state indirection.**

- **Hypothesis:** after embedding and attention cleanup, remaining CPU submission gaps can be removed.
- **Implementation:** preallocate per-execution scratch, metadata, Marlin locks, and FlashInfer workspace. Capture common question-pass buckets through last-row gathering; initially leave CPU readout conversion outside capture. Keep model/precision policy in graph identity, and use stable device pointer tables for changing states.
- Avoid a graph per cached state. Begin with a ≤1 GiB graph/workspace budget, shared arenas where safe, explicit cache lifetimes, and uncaptured fallback.
- **Benchmark:** full matrix; first use versus replay, 10,000 repeated replays, pointer changes, eviction/restore, changing ragged lengths, and image position offsets.
- **Expected ROI:** estimated 0–5% for large batches, 5–15% for launch-bound small requests. The journal’s CPU profile is not a measured graph-saving ceiling. NVIDIA’s launch study supports the mechanism, not these estimates. [CUDA graph study](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- **Risk/rollback:** low arithmetic risk, high pointer/lifetime risk. Disable capture per bucket.
- **Gate:** ≥3% end-to-end or ≥10% p95 improvement in affected cells, with capture cost amortized at observed reuse and no long-context capacity loss.

The rejected GPU readout cast may become a capture enabler later; it should not be presented again as an independent latency win.

**Round 7: improve restore-heavy and image-heavy traffic.**

**Experiment 10 — bounded pinned staging and asynchronous state transfer.**

- **Hypothesis:** pageable vectors and many small transfers unnecessarily lengthen host restores.
- **Implementation:** replace the transfer representation around `StateCache::offload`/`restore` with contiguous slabs and offset metadata. Use bounded pinned staging, a transfer stream, and completion events; prefetch an admitted next state when memory permits. Keep the bulk host L2 pageable if pinning the whole cache harms the host. [engine.rs:545](/home/juanqui/Git/jev-engine/native/src/engine.rs:545), [engine.rs:605](/home/juanqui/Git/jev-engine/native/src/engine.rs:605), `batch.rs::host_insert`.
- **Benchmark:** full matrix plus alternating resident/host states, NUMA placement, and concurrent transfer/compute interference.
- **Expected ROI:** estimated 20–50% restore-stage reduction; no warm-resident benefit. A 24K payload is approximately 1.64 GiB, so transfer time remains substantial.
- **Risk/rollback:** low numerical risk, medium lifetime and host-resource risk. Restore the vector path.
- **Gate:** ≥20% restore-stage and ≥5% restore-heavy HTTP improvement, with ≤2% warm throughput loss and an explicit pinned-memory cap.

**Experiment 11 — cache BF16 vision features across different text states.**

- **Hypothesis:** repeated images with changed text should reuse vision encoding even when exact-state caching misses.
- **Implementation:** cache `VisionOutput.features` before language-state capture in `run_group`. Key by canonical image/preprocessing identity, grids, model/vision-weight identity, and precision policy. Store BF16 features, not recurrent language states. Feature payload is 10 KiB per merged image token. Start with a bounded 512 MiB budget. [vision.rs:435](/home/juanqui/Git/jev-engine/native/src/vision.rs:435), [batch.rs:366](/home/juanqui/Git/jev-engine/src/batch.rs:366).
- **Benchmark:** full matrix and image parity, distinguishing exact-state hits, feature-only hits, and complete misses. Include changed image/same text, two-image order changes, formats, and resolutions.
- **Expected ROI:** potentially large for reused-image traffic; zero benefit for unique images. The historical 88→34 ms difference includes more than vision, so it is not the expected feature-cache saving.
- **Risk/rollback:** low arithmetic risk, medium identity risk. Disable feature caching.
- **Gate:** ≥15% feature-hit HTTP improvement, ≤2% miss-path regression, and no loss of the eight-state 24K target.

If unique-image profiling still makes vision material, extend the experiment to ragged batched vision attention and BF16-preserving epilogues; the current tower already batches projection GEMMs.

**Round 8: investigate different representations only after establishing the optimized W4A8 ceiling.**

**Experiment 12 — selected expanded weights, then QServe-style quantization if justified.**

- **Hypothesis:** spending memory can remove enough unpacking/group-scale work to beat the packed path at large M.
- **Implementation:** introduce bounded alternate representations in `LinearWeight`, `WeightProvider::linear`, and `gemm_core`. First test exact INT4→INT8 unpacking with the existing group-scale mathematics. It removes unpacking but doubles weight traffic, so L40S’s lower bandwidth may make it lose.
- Next test selected FP8/dense GEMMs, recognizing that changing group scaling is a new numerical approximation. Keep capture, sensitive modules, and small-M fallbacks explicit. The existing SM120 FP8 flag cannot simply be enabled on L40S: it is architecture-gated and retains BF16 weights. [engine.rs:1212](/home/juanqui/Git/jev-engine/native/src/engine.rs:1212), [weights.rs:472](/home/juanqui/Git/jev-engine/native/src/weights.rs:472).
- Only if group-scale processing remains dominant, prototype QoQ/QServe-style quantization and recalibrate. QServe has L40S evidence on other models, not this hybrid decision model or its GPTQ artifact. [QServe paper](https://arxiv.org/abs/2405.04532)
- **Benchmark:** full matrix plus conversion-inclusive projection timings, memory residency, cold capture, and the extended decision corpus.
- **Expected ROI:** speculative 10–25% end-to-end for a successful representation change; exact byte expansion may yield no gain.
- **Risk/rollback:** high for FP8/requantization, lower for exact unpacking. Retain the original checkpoint and module dispatch.
- **Gate:** ≥15% selected-linear and ≥8% end-to-end improvement, all quality gates, and preserved long-context residency. A microbenchmark-only win is insufficient.

**Experiment 13 — compressed read-only KV, then quantized attention.**

- **Hypothesis:** long-state bandwidth and cache capacity justify INT8/FP8 KV after attention batching.
- **Implementation:** quantize cached K/V at capture, initially keeping private question K/V BF16 and all GDN seeds F32. Dequantize inside attention; count scales and temporary storage. Extend `StateCache`, capture/restore, and the new FlashInfer wrapper.
- Any head-dimension-256 quantized QK implementation must combine both 128-dimensional partial dot products **before one softmax**. Two independent attentions are mathematically wrong.
- **Benchmark:** full matrix, emphasizing 8K/24K, cache churn, images, long-state answer margins, and dequantization overhead.
- **Expected ROI:** approximately halve the KV portion, saving about 768 MiB per 24K state before scales; estimated 5–15% long-state speed improvement if bandwidth-bound, possibly none at 2K.
- **Risk/rollback:** high numerical risk. Retain BF16 KV and allow per-layer fallback.
- **Gate:** ≥40% KV-memory reduction plus either ≥5% long-state HTTP improvement or ≥25% useful resident-state capacity improvement; no text/image degradation.

Do not pursue sparse advertised peaks through uncalibrated pruning, native NVFP4 on Ada, arbitrary prefix reuse in bidirectional states, or speculative decoding for this non-generative decision path.

**The first three experiments I would run, after Round 0 qualifies the L40S, are:**

1. **GPU embedding with a 16 GiB cache-budget trial.** It already has a measured benefit, needs little implementation, and directly exploits the extra memory.
2. **Long-state batch-cap and scratch-budget sweep.** The current code demonstrably fragments 8K/24K requests; this is the clearest unmeasured opportunity to avoid repeated whole-model passes.
3. **Actual-shape Marlin profiling and validated tile/scale-loop experiments.** MLP and GDN projections dominate the measured work. This determines whether the largest remaining speed gain lies in kernel scheduling or requires a different quantized representation.