# RTX 4090 performance research: d3-turbo native inference

Date: 2026-10-24. Reviewed commit: `51757462104342921caf2a4f0c44101d59fad596`.

**Recommendation:** keep the Rust runtime and state-once design. Concentrate on the large-M W4A8 kernels, eliminating per-question data movement, and an inference-only GDN implementation. Batched attention and CUDA Graphs are worthwhile, but neither alone can close the throughput gap. There is concrete headroom in the implementation; there is not yet evidence for a particular end-to-end speedup.

This review combines source inspection, local checkpoint metadata, existing benchmark reports, upstream research, and a new **compile-only** experiment. No inference benchmark, GPU-counter profile, or numerical parity run was performed. Both local 4090s had resident workloads, including the running Decidra server. Those processes and their GPU settings were left unchanged. `nvcc`, `ncu`, and `nsys` were absent from the host PATH; the existing CUDA 13.1.1 development container supplied the compiler and disassembler without GPU access.

## Findings that should drive the next work

| Priority | Finding | Recommended experiment | Evidence / limitation |
|---|---|---|---|
| P0 | Current measurements mix HTTP, engine, instrumented-region, precision, and power configurations | Establish one reproducible SM89 baseline and a distinct-question workload | Existing 685 ms region sum, 733 ms engine p50, and 936 ms HTTP result are different measurements |
| P1 | Large-M Marlin uses 230 registers/thread, 256 threads/block, 99 KiB dynamic shared memory, and one persistent block per SM | Tune tiles, stages, register pressure, and persistent grid together | Source plus disassembly; low occupancy is established, its performance cost needs counters |
| P1 | GDN scalar arithmetic is built with `-fmad=false` | Selectively enable FMA in recurrent/state-update arithmetic | Fresh compilation reduced GDN static instructions from 561 to 443; no runtime or accuracy result |
| P1 | Seeded convolution performs three device copies per question per GDN layer | Read the shared history directly and write compact question outputs | 9,216 copy calls for a 64-question pass, derived from source |
| P1 | GDN creates and writes temporary per-question final states that the question path discards | Load the immutable seed directly into registers and omit final-state stores | Removes a 192 MiB scratch state at B=64 and 18 GiB of explicit writes across 48 layers |
| P1 | Question RoPE tables and metadata are rebuilt repeatedly; readout stages hidden vectors through the CPU | Hoist metadata and RoPE, gather before final norm, convert to F32 on GPU | Directly visible in the hot path; individually modest wins |
| P2 | Full attention invokes single-request FlashInfer independently for each question | Batched noncausal cascade or shared-prefix attention plus private-suffix attention and LSE merge | 1,024 C ABI attention calls for B=64; roughly 39 ms in the historical SM89 region profile |
| P2 | Short GDN blocks use a sequential register recurrence; existing chunked GDN still performs scalar state updates | Prototype a short-block WY solve that never materializes the final question state | Largest algorithmic GDN experiment; numerical risk and development effort are substantial |
| P2 | Different states are serialized, even when each has few questions | Batch linears across state IDs, then route attention and recurrence to their own caches | Particularly relevant to mixed-state 1–8-question traffic |
| P3 | Group-64 scaling consumes scalar instructions inside W4A8 GEMMs | Compare a QServe-style quantization/kernel design, selected-module W4A16, and transient expanded-weight GEMMs | Requires measured shape dispatch; changing quantization needs new calibration and quality gates |

P1 work can proceed as separate, reviewable experiments. Do not combine all changes before measuring: several alter reduction order, and their gains overlap.

## 1. What is actually running

The model is a dense Qwen3.5 hybrid: hidden size 5,120, MLP width 17,408, 64 layers, 48 GDN layers, and 16 full-attention layers. Full attention uses 24 query heads, four KV heads, and head dimension 256. GDN has 16 key heads and 48 value heads with 128-dimensional keys and values. The readout is F32 `[255,5120]`. These dimensions were checked against the local shipped `config.json` and safetensors header, as well as [the architecture document](../ENGINE.md).

The important distinction is **questions versus token rows**. The checked-in [engine performance fixture](../../eval/data/d3_once_64_2k.json) contains a 2,048-token state and 64 question blocks totaling 3,236 tokens, with block lengths 48–55. A projection GEMM therefore has M≈3,236, not M=64. This is a substantial prefill GEMM, unlike the tiny decode batches targeted by many weight-only kernel benchmarks.

| Projection | Input K | Output N | Layers |
|---|---:|---:|---:|
| MLP gate / up, separately | 5,120 | 17,408 | 64 each |
| MLP down | 17,408 | 5,120 | 64 |
| GDN QKV | 5,120 | 10,240 | 48 |
| GDN z | 5,120 | 6,144 | 48 |
| GDN a / b, separately | 5,120 | 48 | 48 each |
| GDN output | 6,144 | 5,120 | 48 |
| Full-attention Q plus gate | 5,120 | 12,288 | 16 |
| Full-attention K / V, separately | 5,120 | 1,024 | 16 each |
| Full-attention output | 6,144 | 5,120 | 16 |

The server selects W4A8 group-64 Marlin for most large linears. K/V exclusions and the `last:8` layer override make the actual policy more nuanced than “all INT8”. The 48-wide GDN a/b projections fail the loader's `out % 64 == 0` Marlin condition and take the packed-weight fallback. Consult [the loader](../../native/src/weights.rs), [server setup](../../src/native_backend.rs), and [configuration](../../src/config.rs) when constructing a benchmark.

Optimizations already present include exact-state reuse, packed resident weights, ragged projection GEMMs, register GDN for short questions, chunked tensor-core GDN for long sequences, fused residual/RMSNorm, fused SiLU/multiply/INT8 quantization, suffix-only tokenization, and GPU/host state caches. Recommending these as unimplemented features would miss the current baseline.

### Benchmark representativeness

`d3_ticket_2k_64q.json`, `semel_short256_64q.json`, `semel_short512_64q.json`, `semel_long8k_64q.json`, and `semel_long24k_64q.json` each contain **one unique question object repeated under 64 keys**. The tokenized engine fixture has **28 unique token blocks among 64 questions**. This was checked directly, independently of the reported timings.

These fixtures are useful for controlled scaling, but add a second suite with 64 genuinely distinct questions, varied lengths, choice/noul/score types, and varied answer margins. Report duplicate and distinct suites separately. Exact duplicate token blocks within an identical state can be computed once and their logits fanned out, subject to batch-shape parity checks. That is valid application work elimination, but it must not be advertised as a 64-question kernel-throughput improvement. Cache keys must include model/artifact and precision policy as well as the complete state/image identity and question tokens.

## 2. Saturation, ceilings, and the right denominator

The historical 450 W runs were **power-limited**, according to [BENCHMARKS.md](../BENCHMARKS.md). That does not imply full tensor-core, memory-bandwidth, or instruction-issue utilization. A kernel can consume its power budget while spending many cycles on unpacking, scalar scaling, dependencies, and memory traffic. The local observation during this review showed **600 W configured limits**, so future results need an explicitly matched power/clock configuration.

Use these dense reference peaks at NVIDIA's 2.52 GHz boost specification, distinguishing accumulator type and sparsity:

| RTX 4090 resource | Dense peak / capacity |
|---|---:|
| INT8 tensor operations | 660.6 TOPS |
| FP8 tensor operations, F32 accumulation | 330.3 TFLOP/s |
| BF16 tensor operations, F32 accumulation | 165.2 TFLOP/s |
| TF32 tensor operations | 82.6 TFLOP/s |
| FP32 CUDA cores | 82.6 TFLOP/s |
| DRAM bandwidth | 1,008 GB/s |
| L2 | 72 MiB |

These are specifications, not measured sustainable ceilings. In particular, the 1,321 figure is not the dense W4A8 INT8 ceiling, and switching INT8 to FP8 with F32 accumulation does not double Ada's peak. [NVIDIA Ada whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

Ada supplies 48 resident warps, 64K 32-bit registers, and 100 KiB shared memory per SM, with a 99 KiB per-block shared-memory limit. [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

### What the historical profile suggests

The SM89 profile in [ENGINE.md, §5](../ENGINE.md#5-performance-on-one-rtx-4090) reports:

| Region | Time | Fraction of 685 ms region sum |
|---|---:|---:|
| MLP gate + up + down | 320 ms | 46.7% |
| GDN recurrence | 93 ms | 13.6% |
| Full attention | 39 ms | 5.7% |
| GDN QKV + z + output projections | 110 ms | 16.1% |
| Remaining measured regions | 123 ms | 18.0% |

These are repository-reported historical measurements, not a profile collected here. Region instrumentation has overhead and excludes some surrounding work; neither the 685 ms sum nor the 733 ms engine p50 should be substituted for the 936 ms HTTP result. The HTTP benchmark also uses a different state/question fixture.

The local packed-linear metadata contains approximately 24.351B matrix weights. A shape-based estimate is:

```text
linear operations = 2 × token_rows × sum(K × N)
                  ≈ 2 × 3236 × 24.351B
                  ≈ 157.6 tera-operations per 64-question engine pass
                  ≈ 2.46 tera-operations per average question
```

This excludes attention, recurrence, norms, quantization, and padding; not all operations execute as INT8. Dividing by 685 ms gives approximately 230 effective TOPS, consistent with the document's approximate 232. The MLP portion alone is about 110.8 tera-operations / 320 ms = **346 effective TOPS**. Thus the large GEMMs are considerably closer to their ceiling than an end-to-end utilization estimate suggests.

An impossible best case that executes *all* linear work at 660.6 TOPS takes about 239 ms, before any other work. It establishes room for improvement but is not a latency prediction. Recalculate at measured clocks and compare each kernel to its own achievable roofline.

For the 685 ms region profile, Amdahl's law gives useful experiment bounds:

| Hypothetical change | Speedup of those measured regions |
|---|---:|
| 1.5× faster MLP region | 1.18× |
| 2× faster GDN recurrence | 1.07× |
| 2× faster attention region | 1.03× |
| All three together | 1.34× |

These are arithmetic scenarios, not measured improvements, and exclude any additional savings in uninstrumented work. Reaching 127 q/s from 69 q/s requires **1.84× throughput**, or about 46% less time per question at fixed work. Modest attention/GDN tuning alone cannot provide that.

## 3. Kernel and execution-path review

### 3.1 Marlin: tune the actual Rust launch path

Relevant code: [configuration symbols](../../native/src/kernels.rs#L267), [INT8 launcher](../../native/src/kernels.rs#L1136), [row dispatch](../../native/src/engine.rs#L3676), and [group-scale inner loop](../../native/third_party/marlin/marlin_template.h#L1100).

The Rust path loads hard-coded kernel specializations and launches them directly. Editing the C++ `determine_exec_config` heuristic in `marlin.cu` alone will not tune this hot path. For the large-M group-scale specialization, the template describes a 64×256×64 tile, four stages, and 256 threads. The inspected local cubin uses **230 registers per thread**, with no reported local memory or stack. The Rust launcher reserves **101,376 bytes = 99 KiB** of dynamic shared memory and launches **exactly `sms` persistent blocks**.

Consequences:

* One block supplies eight warps per SM: at most 8/48 = **16.7% theoretical warp occupancy** for that launch, before scheduling effects.
* Registers alone consume 58,880 per block before allocation granularity; a second identical block cannot fit. Reducing the shared-memory reservation by itself cannot fix this specialization.
* Increasing grid size alone also cannot make two such blocks resident. Conversely, changing tiles to allow two resident blocks while keeping a grid of only 128 blocks leaves that capacity unused.
* Low occupancy does not prove a bad kernel: persistent GEMMs can hide latency through instruction-level parallelism. Measure tensor issue activity, eligible warps, stalls, and instruction mix before deciding which constraint matters.

**Experiment:** expose a generated configuration table keyed by `(M bucket, N, K, activation/weight scaling mode)`. Test M tiles 16/32/64, N tiles 64/128/256, K tiles compatible with the group boundaries, 2/3/4 stages, and one versus multiple resident CTAs where resources permit. Calculate each specialization's actual shared-memory layout, including group-scale staging and reduction scratch. Expand the grid and lock workspace coherently with Marlin's persistent scheduler; do not change `sms`, locks, or scratch independently.

Measure the entire quantize→GEMM path on the listed model shapes and M values around 48–64, 256, 512, 1,024, 1,618, 3,236, and cold-state lengths. Check split/reduction boundaries explicitly. `DECIDRA_MARLIN_CHUNK=0` already enables the whole-M internal loop; “stop launching once per 64 rows” is not a new optimization. Split-K is more likely to help underfilled small-M shapes than already large M, and introduces reduction cost and numerical changes.

The original [Marlin implementation](https://github.com/IST-DASLab/marlin) is an important design reference, but its headline FP16×INT4, small/medium-M results are not directly transferable to this group-scaled INT8 prefill path.

#### The group-scale tax is real

The modified Marlin loop stages per-group activation scales, synchronizes, converts INT32 partial sums to F32, and scales/accumulates them at group boundaries. For K=5,120 there are 80 groups; for K=17,408 there are 272. Those scalar operations are not represented by the INT8 tensor peak. Shared-memory barriers, conversion throughput, and the large live accumulator set are first-class optimization targets.

An architectural alternative is a QServe/QoQ-style representation that reduces group dequantization/partial-sum overhead through quantization and kernel co-design. This is a different representation, not a compatible kernel swap for the current GPTQ artifact. Benchmark its kernel on the actual shapes before paying for a new calibration run. [QServe paper](https://arxiv.org/abs/2405.04532), [implementation](https://github.com/mit-han-lab/omniserve).

### 3.2 GDN: selective FMA is a concrete first experiment

[build.rs](../../native/build.rs#L56) passes `-fmad=false` to the native and Marlin cubins. The recurrent kernel's dot products and state updates use ordinary multiply/add expressions. NVCC documents this flag as disabling automatic contraction; it is narrower than `--use_fast_math`, which changes division, square root, and denormal handling too. [NVCC documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fmad).

I compiled the unchanged current `kernels.cu` twice for SM89 using CUDA 13.1.1, changing only `-fmad=false` to `-fmad=true`, then disassembled `gdn_recurrent_reg_split_batched`:

| Static property | FMA disabled | FMA enabled |
|---|---:|---:|
| Registers/thread | 127 | 128 |
| Disassembled instructions | 561 | 443 |
| `FFMA` | 4 | 100 |
| `FMUL` | 130 | 34 |
| `FADD` | 102 | 6 |
| Local-memory load/store instructions | 0 / 0 | 0 / 0 |

That is **21.0% fewer static instructions**, not a 21% speedup claim. Instruction execution frequency, dependency chains, cache behavior, and scheduling remain unmeasured. Register usage does not improve. At the current 128-thread block size, roughly 128 registers/thread permits at most four such blocks, or 16 warps/SM, from the register budget.

Use explicit `fmaf` at selected GDN accumulation sites or compile a separate GDN specialization, leaving BF16 rounding boundaries intact. Evaluate dot-product contraction and state-update contraction independently. The code deliberately emulates intermediate BF16 operations in several kernels; blanket fast math can invalidate that contract. FMA also changes rounding even when it improves mathematical accuracy, so test decision probabilities, not just kernel output norms.

The compile results and hashes are in [the static evidence file](2026-10-24-sm89-static-evidence.json). Both compiles emitted the existing shared-memory symbol linkage warning; neither was run on a GPU.

**Build trap:** the timestamp shortcut in `build.rs` can reuse cubins when only compiler flags in the build script change. Use a separate target/output directory or explicitly regenerate the cubins, then inspect SASS. A Cargo rebuild message alone is not proof that the FMA experiment was compiled.

### 3.3 GDN: eliminate disposable states before rewriting the algorithm

[gdn_batch](../../native/src/engine.rs#L2062) allocates `[B,48,128,128]` F32 state, broadcasts the cached seed, runs recurrence, and discards that state. [The register kernel](../../native/kernels/kernels.cu#L2064) reads the initial state into registers and writes the final state back.

For inference question blocks, add a separate entry point with:

```text
immutable seed pointer + question offsets/lengths → BF16 output rows
```

Load each head/value slice directly from the cache into registers; omit the final state store. Preserve the existing state-producing kernel for capture and any caller that actually needs continuation state. The existing single-launch broadcast was already measured as performance-neutral; removing the *buffer and writes* is a different experiment.

Each layer's seed is 3 MiB. B=64 expands it to 192 MiB. Broadcast writes plus final-state writes total 384 MiB/layer, or **18 GiB over 48 GDN layers**. These are logical store volumes, not measured DRAM traffic. Direct seed loads may also reuse L2 because every question reads the same immutable 3 MiB seed. The current recurrence's arithmetic cost remains, so do not convert this byte count directly into a latency claim.

Additional GDN experiments:

* Keep normalized Q/K at 16 key heads instead of materializing identical values for 48 value heads. Index `key_head = value_head / 3`. Compute normalization once per key head, retaining F32 outputs initially.
* For the short recurrent path, compute `exp(g)` once per token/head in preparation instead of repeatedly inside every value-slice CTA. Keep log-decays for chunked implementations.
* Inspect partial unrolling and Q/K staging to reduce long live ranges and repeated loads. Do not force a register cap that introduces spills without measuring it.
* Benchmark warp-shuffle reductions in preparation and RMSNorm against the current block-wide shared-memory reduction. Preserve BF16 casts and validate the changed summation order.

The upstream [FLA recurrent implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py) supports variable lengths, initial state, optional final state, and alternative state layout. It is a useful reference/prototype comparator; adopting its Python runtime is unnecessary.

### 3.4 Seeded convolution: remove 9,216 copies

The cascade branch of `gdn_batch` creates `[three history rows; question rows]` for every question, convolves the gapped buffer, then copies question output rows back into a compact buffer. That is:

```text
48 layers × 64 questions × (history copy + input copy + output copy)
= 9,216 device-to-device copy API calls
```

Implement a seeded ragged convolution that selects its four inputs directly from the immutable history or current question. Write only compact question rows. Keep the existing convolution weight order, SiLU, and BF16 rounding. This eliminates the gapped input/output buffers and the copies; fusing preparation can be a second step after measuring register pressure.

This is not all accounted for by the `gdn.conv` timing label: several copies and metadata uploads surround the timed kernel. An Nsight Systems timeline is needed to determine the actual contribution of submission overhead and device copies.

### 3.5 Full attention: batch the shared prefix correctly

[full_attention_batch](../../native/src/engine.rs#L1790) copies state K/V once per layer into temporary contiguous buffers. For each question it appends private K/V with two copies and calls `prefill_ext`. At B=64 and 16 full layers this means **1,024 attention calls plus 2,080 K/V copy calls**. State K/V is not copied 64 times on the host path; the per-question computation rereads it.

The [FA2 wrapper](../../native/third_party/flashinfer/fi_prefill.cu) already supplies a split-KV workspace for small queries over sufficiently long K/V. Therefore, “enable split-KV” is also already done. However, the wrapper allocates/frees that workspace and queries device attributes on each qualifying call. Preallocate reusable workspace and cache attributes first. A batched kernel needs a batch-aware splitting policy; applying the single-question split factor to all 64 questions may create excessive partials and merges.

Two compatible designs merit comparison:

1. A paged/segmented ragged kernel with each question's logical K/V range pointing to `[shared state; private question]`, consuming cache pointers without concatenation.
2. One shared-state attention problem for all flattened query tokens, plus batched private-question attention, followed by a merge of normalized outputs using log-sum-exp statistics.

For the second design, if the two disjoint key partitions produce `(O_s,L_s)` and `(O_q,L_q)`, then:

```text
L = logaddexp(L_s, L_q)
O = exp(L_s - L) * O_s + exp(L_q - L) * O_q
```

Use a consistent logarithm base with FlashInfer's returned LSE. Averaging normalized outputs without LSE weighting is incorrect. Every query still performs its own attention; sharing K/V does not make attention results query-independent. The layout aims to reuse K/V tiles and reduce launches, copies, and redundant splitting. [FlashInfer recursive-attention tutorial](https://docs.flashinfer.ai/tutorials/recursive_attention.html), [cascade API](https://docs.flashinfer.ai/api/cascade.html).

Both partitions must be **noncausal** within the allowed block: question tokens see their entire own question and the entire state, never another question. Preserve 24:4 GQA and head dimension 256. Use the vendored C++ templates behind a C ABI; the currently consulted documentation is newer than the vendored 0.7.0.post1 headers, so pin and validate any imported API.

### 3.6 Reuse projection inputs and remove host work

These are incremental changes with smaller individual ceilings:

| Location | Current work | Proposed change |
|---|---|---|
| `mlp` / `gemm_core` | Gate and up independently quantize the same normalized input | Share one INT8 activation payload, with correct per-module scale handling; then test concatenated gate/up weights |
| GDN projections | QKV and z consume the same input; a/b are two narrow projections | Reuse activation quantization where compatible; test merging a/b and QKV/z without losing their precision policies |
| `rope_tables_positions` | CPU sin/cos generation and H2D uploads inside each of 16 attention layers | Compute once per question pass and reuse; preserve image-state position offsets |
| Seeded convolution metadata | Gapped lengths/offsets rebuilt and uploaded in each GDN layer | Hoist metadata once, or remove it with direct seeded convolution |
| `state_once_batch_rows_inner` | Final norm over all tokens, then gather | Gather last rows before final norm |
| Readout | BF16 hidden D2H → CPU F32 conversion → F32 H2D → readout GEMM | GPU BF16→F32 conversion and the existing F32 readout, followed by only the small logits D2H |

The down-projection SiLU/multiply/group-quantize fusion already exists. Focus new fusion on removing repeated quantization or output traffic. `a_global` and quantization exclusions can differ per module; identical BF16 input does not automatically mean identical scale arrays.

The old SM120 gate/up fusion failed its parity target. Retesting an SM89 variant can be justified, but retain explicit BF16 rounding stages and do not assume concatenation preserves accumulation order. Fusion should follow a successful kernel-shape benchmark, not precede it.

A model-specific tail optimization is also possible: layer 63 is full attention and only each question's final output token is read. Keep K/V for every question token, but compute final-layer Q/gate, attention output, output projection, residual/MLP, and final norm only for the last rows. Earlier layers still need all token outputs. State capture can stop the final layer after producing its cached K/V. This removes mathematically unused work, with a modest whole-model ceiling and batch-shape rounding changes to validate.

### 3.7 Cold images and host-cache restores need separate profiles

[The vision path](../../native/src/vision.rs) already combines image patches for GEMMs, but runs attention separately for each image in each of 27 blocks. It also launches separate bias, residual, normalization, and activation kernels. For image-heavy cold traffic, measure a batched ragged vision attention kernel and bias/residual/normalization fusion while retaining image boundaries and the current GELU/BF16 rounding. Cache vision features independently of the text state when the same decoded image and preprocessing configuration recur with different text. Key that cache by image identity, grid/preprocessing, and vision weights; budget its memory explicitly. A warm exact-state cache hit already avoids the vision tower, so these changes do not accelerate the reported warm text benchmark.

The host state cache uses ordinary Rust vectors in [offload/restore](../../native/src/engine.rs#L545). Investigate bounded page-locked staging and copy/compute overlap for restore-heavy workloads, measuring the actual PCIe transfers and synchronization first. This needs stable cache ownership and separate staging buffers; it is distinct from simply increasing the host-cache byte limit. Keep cold image encoding, text-state capture, host restore, and warm question execution as separate benchmark categories.

## 4. Higher-risk research experiments

### 4.1 A short-question GDN kernel with no final state

The existing [chunked tensor-core implementation](../../native/kernels/kernels.cu#L1593) already uses TF32 WMMA and a triangular solve. It is not an unimplemented idea. Its state update still performs a scalar tiled outer-product accumulation, and the long-state path exposes only 48 heads × two value slices = 96 CTAs at VC=64, fewer than the 4090's 128 SMs. Those CTAs loop over time chunks. A cold-state optimization must address that dependence and work distribution, rather than merely enabling tensor cores.

For the usual 48–55-token question, test a **single short chunk with no state-update epilogue**. The mathematical proposal below follows the gated delta recurrence; this is a design derived for this inference use case, not an upstream performance result. The [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) establishes the chunkwise formulation and is the starting reference.

For one head, let `S0` be the cached state, `a_t=exp(g_t)`, and `p_t=product(a_1…a_t)`. With row-vector q/k/v and normalized/scaled q/k as in this engine, define the corrected value `u_t`:

```text
u_t = beta_t * (v_t - p_t*k_t*S0
              - sum_{i<t} exp(log(p_t)-log(p_i)) * dot(k_t,k_i) * u_i)

o_t = p_t*q_t*S0
      + sum_{i<=t} exp(log(p_t)-log(p_i)) * dot(q_t,k_i) * u_i
```

All `u_t` follow from a small lower-triangular solve. Outputs then use matrix products, and **the final 128×128 state never needs to be constructed**. Compute decay ratios from differences of cumulative log-decays rather than dividing potentially underflowed products.

Stack `K*S0` and `Q*S0` across questions sharing a seed to expose larger GEMMs; retain separate triangular problems per question. Compare C=16/32/64, FP32 reference arithmetic, TF32, and BF16 operands with F32 accumulation. The latter two are approximate. For long questions, intermediate states or chunk summaries are still required even if the final state is omitted.

This differs materially from the rejected “use the existing chunked GDN for short questions” switch: it removes the state epilogue and exploits the shared seed. Measure packing, solve, seed projections, and output stages together. If these overheads exceed the register recurrence, retain the recurrent path below an empirical crossover.

Recent [TreeWY research (August 2026)](https://arxiv.org/abs/2608.20961) also uses a WY transform to avoid materializing recurrent states across branches. Its speculative-decoding benchmarks are not d3-turbo results. The useful connection is the state-elimination technique; d3 has no generated continuation to speculate over, and question-prefix tree reuse is not generally valid through its bidirectional full-attention layers.

### 4.2 Cross-state batching and CUDA Graphs

The [scheduler](../../src/batch.rs) groups by exact state and serializes different groups. For many small independent requests, concatenate their question tokens for linears and carry a `state_id`/cache pointer per sequence into seeded GDN and attention. This reuses weight reads across states without pretending the states themselves are shared. Account for token count, state length, scratch bytes, and age when scheduling; question-row count alone is a weak cost predictor.

Prefer this single-stream design before multiple concurrent forwards. The current `Engine` shares activation scratch, Marlin locks, buffer pools, profiling state, and cuBLAS handles. Adding streams requires explicit per-stream ownership and cache lifetimes. Large-M GEMMs may already compete for the same resources; multiple streams can hurt them.

After fixing the host round-trip and making allocations/metadata stable, capture a few common question-pass shapes in CUDA Graphs. Keep cache addresses valid for the lifetime of each graph or update pointer parameters safely. Padding requires exact sequence lengths and masks; padded GDN tokens must not modify real state. Graph cache memory must count against the 24 GiB budget. NVIDIA demonstrates reduced repeated launch overhead, but that result is not an end-to-end speedup estimate for this model. [CUDA Graph launch study](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/).

### 4.3 Precision and alternative GEMMs

| Experiment | Why it might help | Constraint / go-no-go condition |
|---|---|---|
| Selected-module W4A16 | Avoid activation/group-scale overhead in unfavorable shapes | Lower BF16/F32 tensor peak; benchmark including quantization and dequantization cost |
| Per-layer transient INT8/FP8/BF16 expansion plus tuned GEMM | For large M, expansion cost may amortize and a dense GEMM may beat in-loop scaling | Full expanded model does not fit; use bounded scratch and include conversion traffic; FP8 introduces another quantization |
| QServe-style W4A8 | Reduce scalar group-scale costs | New representation/calibration; existing GPTQ scales are not directly interchangeable |
| W4 weights with FP8 activations | Marlin FP8 scaffolding already exists | Ada FP8/F32 peak is lower than INT8; require a shape-specific measured win and probability validation |
| Selected low-sensitivity modules with coarser activation scales | Reduce group-scale processing | Measure decision margins/calibration; do not remove g64 globally |
| INT4×INT4 integer tensor-core experiment | Ada has an integer INT4 path | This is not NVFP4; activation range/quality and group-scale overhead may erase the benefit |
| Structured sparsity or smaller distilled decision model | Reduce the actual work needed per answer | Requires retraining/calibration and a separate quality contract; random pruning is not a kernel optimization |

For transient expansion, the largest individual MLP matrix has 89,128,960 weights: about 85 MiB at INT8/FP8 or 170 MiB at BF16, before workspace. That is plausible scratch capacity; expanding the entire 24.35B linear set to bytes already consumes about 24.35 GB before embeddings, vision, caches, and scratch. The current SM120 FP8 loader retains BF16 weights as well, making it especially unsuitable to enable unchanged on a 4090.

[FLUTE](https://github.com/HanGuo97/flute) is a useful alternative lookup-table quantization/kernel reference with RTX 4090 and group-64 support. Its different formats, shape specialization, and predominantly smaller-M use cases make it a lower-priority comparator for this workload than tuning the current W4A8 path.

### 4.4 Quantized attention and cache compression

Test INT8 QK / FP8 PV ideas only after removing the per-question attention launch pattern. [SageAttention](https://github.com/thu-ml/SageAttention) provides relevant Ada quantized-attention research, but the inspected [SM89 wrapper source](https://raw.githubusercontent.com/thu-ml/SageAttention/main/sageattention/core.py) rejects head dimensions above 128. d3 needs **256**, GQA 6, independent Q/KV lengths, and exact block visibility. It is not a drop-in replacement. A port must accumulate both halves of a 256-dimensional QK dot product **before** a single softmax; two independent 128-dimensional attentions are incorrect.

Cache compression has a separate value: more resident states and fewer host restores. Full-attention BF16 K/V consumes 64 KiB per state token across 16 layers; GDN seeds add 144 MiB, plus convolution tails. INT8/FP8 K/V could approximately halve the KV portion. Start with read-only cached K/V and dequantize inside attention. Keep recurrent seeds F32 initially: errors there propagate through every question token. Evaluate cache-hit rate, restore latency, and cold-state latency as well as warm attention speed. At 2K, attention's modest profile fraction limits the direct speedup; at long states or a large working set, memory savings can matter more.

### 4.5 What not to prioritize

* **Hopper/Blackwell-specific kernels on SM89.** FA3's Hopper path, WGMMA/TMA designs, and NVFP4 are not switchable 4090 features. Preserve an SM89 `mma.sync`/`cp.async` path.
* **Speculative decoding.** There is no autoregressive answer-token loop in this decision path.
* **More HTTP concurrency alone.** Existing c8–c32 results already show a throughput plateau; additional queueing does not remove GPU work.
* **Power-limit changes as the primary solution.** Match power for comparisons and optimize joules per answered question. This review did not change power settings.
* **Blindly retrying rejected numerical configurations.** SM120 L8 GDN, short-question chunked GDN, several W8A8 paths, gate/up fusion, and full NVFP4 already have negative evidence in [BENCHMARKS.md](../BENCHMARKS.md). A retry needs a changed hypothesis and architecture-specific validation.
* **Generic prefix caching within a state.** Bidirectional state tokens depend on the complete state, and question tokens see their complete own block. Exact-state caching is valid; arbitrary sub-prefix reuse generally is not.

If both 4090s eventually become available, independent model replicas with state-affinity routing are a simpler aggregate-throughput experiment than tensor parallelism. This is a capacity option, not an improvement in per-GPU efficiency or single-request latency. Measure the actual PCIe topology before considering sharding, and account for duplicated caches.

## 5. Measurement and acceptance plan

### Baseline matrix

Record git commit, model/config/readout hashes, precision/exclusion flags, CUDA/compiler/driver versions, cubin hashes, GPU clocks/power/temperature, GPU sharing, and cache state. Retain raw timing samples and JSON results.

| Dimension | Required cases |
|---|---|
| Question count | 1, 8, 16, 32, 64 |
| Question length | Actual 48–55-token fixture; distinct 16/64/128/256-token blocks and ragged mixes |
| State length | Approximately 256, 2K, 8K, 24K; include the 12K allocation boundary |
| State reuse | Cold capture; GPU hit; host-L2 restore; distinct-state working sets |
| Traffic | c1 latency; c8 sustained throughput; mixed small/large requests and p95/p99 |
| Content | Distinct text decisions, duplicate controls, and image requests |

Use three measurement levels: device kernel timings, engine pass timings, and HTTP wall time. Add NVTX ranges for state capture, question pass, per-layer mixer/MLP, cache restore, and readout. The current `DECIDRA_PROFILE` regions create CUDA events and `prof_report` aggregates only those regions. Keep it separate from uninstrumented headline timing. `DECIDRA_GEMM_DEBUG_SYNC` must be unset for performance runs.

Use Nsight Systems first to count copies, allocations, CPU submission gaps, and synchronizations. Use Nsight Compute on a few isolated representative kernels for tensor-pipeline activity, achieved occupancy, registers/spills, DRAM/L2 traffic, eligible warps, barriers, and dependency stalls. Select supported metrics with the installed tool rather than assuming names are portable. Counter replay changes execution and can disturb caches; end-to-end latency must come from unprofiled runs. [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

### Reproducible starting commands

These are commands for an isolated benchmark window, **not commands executed during this review**. The test harness uses different environment names and defaults from the server. In particular, explicitly set its Marlin, group-scale, K/V exclusion, and preload options:

```bash
CUDA_VISIBLE_DEVICES=1 \
DECIDRA_MARLIN=1 DECIDRA_INT8=1 DECIDRA_INT8_GROUP=1 \
DECIDRA_INT8_EXCLUDE=self_attn.v_proj,self_attn.k_proj \
DECIDRA_PRELOAD=1 DECIDRA_GPU_EMBED=0 \
DECIDRA_GDN_CHUNKED=1 DECIDRA_GDN_CHUNK_MIN=16 DECIDRA_GDN_REG=1 \
DECIDRA_D3_REPEAT=20 \
DECIDRA_D3_MODEL=/srv/models/d3-turbo-w4a8-vision-v2 \
DECIDRA_D3_PROBE="$PWD/eval/data/d3_once_64_2k.json" \
cargo test --release --locked -p decidra-native --test d3_perf \
  -- --ignored --nocapture

python3 eval/d3_bench.py --url http://127.0.0.1:8899 \
  --request eval/data/d3_ticket_2k_64q.json \
  --questions 1,8,16,32,64 --warmup 3 --runs 20 --concurrency 1,8
```

The second command is the duplicate-question control; add the distinct-question suite before interpreting it as production throughput. The first also benchmarks single-question passes after the batch loop, so isolate its question-batch range when collecting a trace. [Harness source](../../native/tests/d3_perf.rs), [HTTP benchmark](../../eval/d3_bench.py).

For kernel work, build a targeted harness that can replay captured real activations and states. Sweep shape and ragged boundaries, warm up each candidate, use CUDA-event timing without a host synchronization inside the measured loop, and report median plus variation. Synthetic inputs alone do not reproduce quantization outliers or GDN decay distributions.

### Experiment order and decision criteria

| Stage | Work | Continue when |
|---|---|---|
| 0 | New distinct-question suite, matched SM89 baseline, Systems/Compute traces | Region totals reconcile with engine/HTTP timing and the bottleneck is reproducible |
| 1 | Readout/metadata cleanup; direct seeded convolution; direct GDN seed with no final output | Copies/bytes decrease as predicted and latency or memory capacity improves without quality regression |
| 2 | Selective GDN FMA and Marlin tile/resource tuning, independently | Target-region improvement survives whole-pass testing and the quality gates |
| 3 | Batched attention with reusable workspace; quantization reuse / selected projection fusion | All block-isolation and batch-shape tests pass and the full region improves, including packing/merge |
| 4 | Short-block output-only WY GDN; cross-state batching; common-shape graphs | Wins on the actual workload distribution, not just a favorable microbenchmark |
| 5 | New quantization, quantized attention/cache, sparsity/distillation | Earlier changes are measured and the additional quality/performance tradeoff is justified |

Set thresholds before running the experiments. A practical starting policy is to require a repeatable improvement greater than run-to-run noise, approximately ≥5% for a targeted hot region, and either ≥2% end-to-end improvement or a material memory/latency-tail benefit to justify added maintenance. These are proposed engineering criteria, not measured outcomes. Keep useful small changes simple; do not retain a complex specialization for an unrepeatable fractional gain.

### Correctness gates

Retain the current state-once, single-versus-batch, seed/tail negative-control, and HF gates. For new precision or reduction schemes, extend to 256–1,024 held-out representative decisions, including long states, image inputs, near ties, extreme decays, and option-order changes. Preserve choice/noul/score probability semantics and the F32 head.

For transformations intended to preserve computation, compare per-layer intermediate results and logits as well as decisions. For approximate changes, report top-1, mean and maximum probability error, high-margin flips, score error, calibration, and batch-shape drift. The existing 32-row HF gate is a smoke gate, not strong statistical evidence for a new quantizer. Do not loosen its acceptance thresholds to make a kernel pass.

Specifically test ragged lengths at 1/15/16/31/32/33/63/64/65/255/256/257, unequal question lengths, state pointer changes, cache eviction/restore, repeated graph replay, and absence of cross-question leakage. Recurrent output must agree after seeded convolution, not only from a zero-state synthetic input.

## 6. Evidence and reproducibility notes

The compile experiment used the unchanged current CUDA source and these flags:

```text
nvcc -std=c++17 -O3 -fmad={false,true} -lineinfo -cubin -arch=sm_89 kernels.cu
cuobjdump -res-usage -fun gdn_recurrent_reg_split_batched <cubin>
cuobjdump -sass -fun gdn_recurrent_reg_split_batched <cubin>
```

Disassembled instruction counts include the whole function once, including setup and loop bodies; they are not dynamic executed-instruction counts. The Marlin resource result came from the local cached `decidra-native-9e53196644b80002` SM89 build and is labeled separately from the freshly compiled GDN variants. Exact compiler/source/cubin hashes and the selected kernel symbol are retained in [the JSON evidence](2026-10-24-sm89-static-evidence.json).

Documentation caveats discovered during review: the README's “best measured 100 q/s” sentence is stale relative to its own SM120 FP8 results; the Marlin notice says there are no local kernel changes although `group_a_scale` and related logic are present. Treat source, artifact hashes, and dated raw results as the authorities for this work, and update these descriptions when publishing new measurements.

The most valuable next result is an isolated SM89 before/after measurement for **direct seeded convolution + disposable-state removal**, alongside an independent **Marlin tile sweep** and **selective-FMA parity test**. Those experiments directly address demonstrated implementation costs and will establish whether the larger attention and WY rewrites are the best next investment.
