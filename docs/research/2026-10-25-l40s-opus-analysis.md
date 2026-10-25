I've written an independent L40S plan below. Nothing was edited or run. No L40S has been measured. Numbers labelled **measured** come from the 4090 journal or from source/disassembly; numbers labelled **estimate** are my arithmetic from NVIDIA specs and should be treated as predictions to test.

## Bottom line

On the 4090 at 600 W, the work splits roughly 46% MLP GEMMs, 16% GDN projection GEMMs, 13% GDN recurrence and 6% attention (**measured**: 710 ms region sum, 709 ms engine p50). The L40S changes three things that matter:

1. **Datacenter Ada runs BF16, FP8 and TF32 tensor math with FP32 accumulation at full rate.** GeForce Ada (the 4090) runs it at half rate. Per the spec sheets, dense peaks are:

   | Precision | 4090 | L40S |
   |---|---:|---:|
   | INT8 | 660.6 TOPS | 733 TOPS |
   | FP8, FP32 accumulate | 330 | 733 |
   | BF16, FP32 accumulate | 165 | 362 |
   | TF32 | 83 | 183 |

   Several 4090 conclusions rested on the 4090's 4:1 INT8:BF16 ratio and do not carry over. These include rejection E12 (keeping the MLP on W4A16), the research doc's "FP8 is lower than INT8 on Ada", and the cost of attention and the vision tower.
2. **48 GB of memory.** Every memory policy in the repo is sized for 24 GiB.
3. **350 W, 142 SMs, 864 GB/s.** For INT8-bound work I expect roughly 0.8–0.95× the 4090@600W (**estimate**). The 350 W cap also means fewer instructions per answer translates into higher clocks.

The best-value kernel item is GPU-independent, so it can be built and verified on tinylife's 4090 now: the Marlin W4A8 group-scale epilogue can be made bit-exact and much cheaper.

## Currently wrong (or unsafe to assume) for L40S

1. **One profile for both GPUs.** `native/src/arch/sm89.rs:5-17` serves "RTX 4090 / L40S" from one profile. Nothing distinguishes GeForce from datacenter Ada (accumulate rates) or 24 GB from 48 GB. Policy needs to key on device memory and SKU.
2. **24 GiB cache defaults.** `src/config.rs:23-28`: 16 state entries, a 5 GiB GPU state cache ("keeps large states from exhausting 24 GiB") and an 8 GiB host L2. On L40S that leaves roughly 15–20 GiB idle (**estimate**: about 46 GiB usable, minus about 14 GiB of weights/context, minus a 7 GiB pass reserve at 32K).
3. **GPU embedding is off by default** (`src/config.rs:125`). It was rejected only for memory (E13) after measuring −2.44% / −18.4 ms on the 4090.
4. **L40S results won't be bit-identical to the 4090.**
   - Marlin's split-K stripe partition depends on `gridDim.x` = SM count (`marlin_template.h:366-372`).
   - Partial sums are reduced through the BF16 output buffer, because `use_fp32_reduce=false` (`kernels.rs:1177`; `marlin_template.h:1838-1845`).
   - So the 32/32 text gate, 13/13 image parity and state-once self-test must all be re-run on L40S. They are not inherited from the 4090.
5. **Capture-path GDN leaves SMs idle.** The chunked kernel launches a 96-CTA grid with 84.5 KiB of shared memory, so only one CTA fits per SM (`kernels.rs:1757-1759, 1867-1876`). That leaves 46 of 142 SMs idle (32%) on every long-state capture (**derived**).
6. **Seeded GDN at B=1** launches only 96 CTAs (`kernels.rs:2496-2500`). Single-question latency underfills 142 SMs.
7. **24 GiB scratch heuristics.** The `low_mem` thresholds (`engine.rs:1525, 1706, 4867`) and the forced low-mem capture (`engine.rs:4564-4569`) exist to fit 24 GiB.
8. **Server CPU risk (unverified).** The 4090 host is a 7950X. L40S servers usually have lower single-thread clocks.
   - The journal measured 147 ms of host embedding in the CPU profile of a question pass.
   - The per-question attention loop issues 1,024 FlashInfer calls plus 2,080 copies per 64q pass.
   - Both could become exposed on a slower CPU.
   - The HTTP–engine gap is already unexplained: 901.4 ms distinct HTTP vs 708.91 ms engine, about 192 ms (**measured**, different measurement levels).
9. **Long-context fixtures are duplicates.** `semel_short256/long8k/long24k_64q.json` repeat one question 64 times (research doc §1). Long-context results on them are not distinct-question throughput.

## Shared benchmark protocol (all rounds)

- **Matrix:**
  - Questions: 1/8/16/32/64.
  - States: 256/2K/8K/24K. Create distinct-question 256/8K/24K fixtures first.
  - Cache: cold (capture), warm (GPU hit), restore (host L2 hit).
  - Concurrency: c1/c8.
  - Content: distinct text, duplicate control, 13 image cases (`eval/image_api.py`), mixed states (`eval/sm89_mixed.py`, 8 states × 8q).
- **Core subset per round:** Q∈{1,8,64} × S∈{2K,24K} × warm × c1, plus 64q/2K c8. Run the full matrix only before accepting a round.
- **Three levels, never mixed:**
  - Kernels: ncu on isolated shapes, or `decidra-native matmul --bench` with captured activations (`native/src/main.rs:409-420`).
  - Engine: `native/tests/d3_perf.rs`.
  - HTTP: `eval/d3_bench.py`. Use the existing `semel_pass_seconds` / `semel_queue_wait_seconds` metrics to attribute HTTP time.
- **Record every run:** SM clocks, power, and throttle reasons (`nvidia-smi -q -d PERFORMANCE`), ECC state, L2 size (`cudaDevAttrL2CacheSize`), and J/question. Power-cap behaviour is now first-order.
- **Gate A (bit-exact changes):** byte-identical 255-row readouts on the core matrix plus ragged lengths 1/15/16/31/32/33/63/64/65/255/256/257. Then ≥5% on the target region or ≥2% end-to-end.
- **Gate B (changes that alter numerics):**
  - 32/32 text, 13/13 image parity, max Δp ≤ 0.0984, state-once self-test, seed/tail negative controls.
  - Precision changes additionally need a 256–1,024-decision held-out set with near-ties, long states and images.
  - Never loosen thresholds to pass.

## Rounds, in priority order

### R0 — L40S bring-up, numerics re-validation, measured ceilings (required first)
- **Hypothesis:** the sm_89 cubins run unchanged, but numerics shift (item 4 above) and the real ceilings differ from spec at 350 W.
- **Code areas:** none for the baseline. Write a small throughput microbenchmark: `mma.sync` INT8 m16n8k32, BF16/FP8 m16n8k16 with F32 accumulation, a DRAM copy, and an L2 sweep.
- **Sketch:**
  - Build with `DECIDRA_CUDA_ARCH=sm_89`, inspecting SASS so stale cubins can't slip through (the research doc's build trap).
  - Run all gates, then the core matrix, then an nsys timeline of one 64q/2K pass and one 1q pass.
  - Per-module shape sweep: W4A8-g64 vs Marlin W4A16 vs W4-FP8 at M = 52, 416, 832, 1,664, 3,334, 6,668.
- **ROI:** no direct speedup. It decides R3: if the FP8/BF16 full-rate claim fails on hardware, R3 drops to last.
- **Risk:** a gate fails on L40S because the split-K reduction partition changed. Fallback: test `use_fp32_reduce=true` with a `C_tmp` workspace (it also makes numerics less dependent on SM count).
- **Accept:** gates pass; region totals reconcile with engine p50 within the same ~5% gap seen on the 4090 (710 vs 753); peaks recorded.

### R1 — Spend the 48 GB (configuration first, then small code)
- **Hypothesis:** the 24 GiB policies cost warm latency (embedding), hit rate (restores and recaptures), and cold time (capture scratch churn).
- **Code areas:**
  - `src/config.rs:16-28,125`
  - `src/batch.rs:63-79, 115, 303-315`
  - `engine.rs:1525/1706/4564-4569/4867`
  - `engine.rs:4788-4791` (64-row cap)
  - `engine.rs:605-648` (pageable `clone_htod` restore)
- **Sketch:**
  - (a) Environment only: `DECIDRA_NATIVE_GPU_EMBED=1`; set `DECIDRA_STATE_CACHE_BYTES`/`_ENTRIES` from post-preload `mem_info` minus `pass_reserve(32K)`; sweep `DECIDRA_BATCH_ROWS` 32/64/128 and `DECIDRA_STATE_ONCE_MAX_ROWS` 64/128.
  - (b) Code: derive the defaults from total device memory instead of arch.
  - (c) Code: skip forced low-mem in capture when free memory exceeds `pass_reserve(seq)` by 2×.
  - (d) Code: pinned, reused host staging for restore. Only if restore stays frequent after (a).
- **Matrix:** warm Q∈{1,8,64} × S∈{2K,24K}; restore vs warm at 8K/24K; mixed-state c8 with a working set of 20 × 2K and 8 × 24K states; cold capture at 2K/24K.
- **ROI:**
  - About −2.4% warm from the GPU embedding (**measured on 4090**, may be larger on a slower server CPU).
  - Restore avoidance at 24K: an estimated 70–200 ms per miss (1.66 GB over PCIe Gen4, pinned vs pageable).
  - Capture: unknown.
- **Risk:** low, except OOM at 32K with images. Keep `ensure_headroom`. **Rollback:** environment or defaults. **Accept:** no OOM across the matrix, ≥2% warm improvement or a measured tail/hit-rate gain, gates unchanged.
- **Also:** attribute the ~192 ms HTTP–engine gap with existing metrics before optimizing any further CPU-side work.

### R2 — Bit-exact Marlin W4A8 group-epilogue rewrite (highest-value kernel work; developable on the 4090)
- **Hypothesis:** the INT32→F32 conversion at group boundaries is a co-bottleneck with the IMMA work. Per warp per 64-K group, the tmb4 kernel issues 32 `IMMA.16832` and 64 `__int2float_rn` (`marlin_template.h:1185-1197`).
  - At Ada's documented 16 I2F/clk/SM against 1,024 INT8 MAC/clk/SM, both cost about 512 pipe-cycles per warp per group (**static derivation, needs ncu**).
  - In addition, each stage does a synchronous global load of the activation scales followed by an extra `__syncthreads()` (`:1114-1129`), with only 8 warps per SM (230 registers, 1 CTA/SM: **measured static**).
- **Sketch (each step bit-exact):**
  1. **Magic-bias accumulator.** Initialize `frag_c_tmp` to the int32 pattern `0x4B400000` instead of 0 (it is reset every group anyway). After the IMMA, reinterpret as float and subtract `12582912.0f`.
     - This is exact because |c| ≤ 64·127·8 = 65,024 < 2²², so the value stays in [2²³, 2²⁴).
     - It replaces 64 I2F per warp per group with 64 FADD. Do the same for the uint16 weight scales.
  2. **Pipelined activation scales.** Stage the per-group activation scales through the existing `cp.async` pipeline (256 B per stage) and drop the synchronous load and barrier.
  3. **Shared activation quantization.** Gate+up (64 layers) and qkv+z (48 layers) each quantize the same input twice (`engine.rs:2714-2715, 2075/2145`; `gemm_core` at `:4293-4306`).
     - Produce one INT8 payload plus two scale arrays (`scale*a_global` per module) to keep exactness.
     - Then fuse the MLP-input quantization into `add_rmsnorm`, rounding to BF16 in registers before quantizing.
     - Keep the per-element `x/scale` division (changing it to a multiply breaks exactness).
- **Code areas:** `native/third_party/marlin/marlin_template.h:1112-1200` (`matmul_a8`), the stage-fetch lambda near `:780-840`, `native/kernels/kernels.cu:28-80`, `engine.rs:2712-2760, 4276-4332`.
- **Matrix:**
  - First a SASS diff: I2F count and BAR count.
  - Then ncu on gate/up/down/qkv/o at M = 52/416/3,334 and the 24K capture M.
  - Then the core matrix on both 4090 and L40S.
- **ROI (estimate):** 0–30% on the INT8 group GEMMs depending on what ncu shows is actually limiting. Those GEMMs are about 62% of the pass, so about 3–15% end-to-end. Step 3 is about 1–2% (one less read/write of a 3,334×5,120 activation per pair).
- **Risk:** low numerically, moderate in templates (the m8 and tmb1–4 variants must all change).
- **Rollback:** keep the old mangled symbols in `kernels.rs` behind `DECIDRA_MARLIN_EPILOGUE=0`.
- **Accept:** Gate A, and ≥5% on the Marlin INT8 region.

### R3 — Re-evaluate precision now that FP8/BF16 accumulate at full rate (L40S-only upside)
- **Hypothesis:** E12 (W4A16 MLP measured 993 vs 709 ms) reflects the 4090's 4:1 ratio. On L40S (2:1), and with FP8 equal to INT8, alternatives that drop the INT8 group epilogue may win in some shape ranges.
- **Sketch, cheapest first, using R0 microbenchmark data:**
  - (a) Dispatch by M bucket: Marlin W4A16 for small passes (M ≤ ~416, i.e. 1–8 questions). It skips activation quantization and is *more* accurate.
  - (b) The existing W4 + FP8-activation Marlin path (`marlin_fp8`, `marlin_permute_scales_a8`). It has an FP32 epilogue with no I2F.
  - (c) Highest upside: dense FP8 W8A8 (per-channel weights, per-token activations) through cuBLASLt for the **MLP only**.
    - About 17.1 GB of FP8 replacing about 8.5 GB of W4, which only fits on 48 GB.
    - Requantize from the BF16 source weights, not from the dequantized W4. Check whether those weights exist before scheduling this.
- **Code areas:** `engine.rs:4216-4479` (`gemm_core` dispatch), `weights.rs:338-439` (the `use_int8` and `group_a` policy), `engine.rs:4062` (`fp8_gemm`), `src/config.rs:35-39`.
- **Matrix:** module microbenchmarks per M; then Q∈{1,8,64} × S∈{2K,24K} warm; held-out quality set.
- **ROI (estimate):** (a) 1q/8q latency 5–20%; (c) MLP 1.2–1.6× → 7–15% end-to-end at 64q, minus the lost cache capacity.
- **Risk:** (a) low; (b) and (c) high quality risk (Gate B with the extended set). (c) also costs about 8.6 GB of state cache and needs calibration.
- **Rollback:** per-module policy flags. **Accept:** Gate B, plus ≥2% end-to-end on the target traffic class, with no regression on the other.

### R4 — Shared-prefix attention for long states
- **Hypothesis:** at 24K, each question re-reads the state's K/V.
  - That is about 100 MB per layer → about 102 GB per 64q pass, roughly 119 ms at 864 GB/s if DRAM-bound (**derived**).
  - The 4090's warm cost rose from 2K to 24K by about 362 ms (HTTP: 879.8 → 1,242 ms; **measured**).
  - L40S's doubled BF16 rate makes the compute side cheap, so traffic and launches dominate.
- **Sketch:**
  1. Run one noncausal `SinglePrefill` per layer over all flattened question rows (qo = 3,334) against the state K/V. Rows are already post-RoPE, so this is exact apart from the softmax split.
  2. Run per-question private attention (52×52), as a batched or tiny custom kernel.
  3. Combine with `MergeState`, which is already vendored in `cascade.cuh`.
  4. Wire the LSE output through `fi_prefill.cu:17`.
  5. Delete the state K/V copies and per-question appends (`engine.rs:1893-1962`).
- **Matrix:** S∈{2K,8K,24K} × Q∈{8,64} warm c1; 13 image cases (mRoPE offsets).
- **ROI (estimate):** 2K 1–3%; 8K 5–10%; 24K 8–18%.
- **Risk:** medium. The LSE merge changes the softmax summation order; block-visibility tests must hold (no question sees another question).
- **Rollback:** keep the per-question loop behind a flag. **Accept:** Gate B, and ≥5% at 24K with no regression at 2K.

### R5 — Mixed-state and small-batch throughput on 142 SMs
- **Hypothesis:** serialized state groups underfill the GPU.
  - On the 4090, 8 states × 8 questions measured 55.5–59.0 q/s vs 73 q/s for a single state (**measured**, about −19%).
  - Wider L40S is likely to underfill more.
- **Sketch:**
  - (a) Capacity check with no code: two server replicas on one L40S, with and without MPS. MPS is a host change, so confirm with the box owner first; L40S has no MIG.
  - (b) Cross-state batching. Most of it is pointer plumbing:
    - Per-sequence `hist` pointer arrays in the seeded convolution (`engine.rs:2083-2109`) and `seed` pointer arrays in the seeded GDN (`:2282-2305`).
    - A per-question `StateCache` in the already per-question attention loop.
    - Per-sequence `pos_start` in `BatchInfo` (`:682-735`).
    - Scheduler merge across groups in `src/batch.rs:257-269`.
  - (c) For B ≤ 4, run the seeded GDN with vc=32 to fill SMs.
  - (d) CUDA Graphs for 1q/8q passes, only after allocations are stable.
- **Matrix:** mixed 8×8 c8, 32×2 c8, Q∈{1,8} c1 at 256/2K.
- **ROI (estimate):** recover most of the 19% on mixed traffic; 1q latency 5–15%.
- **Risk:** medium (cross-state leakage). Add a negative control: perturb one state's cache and assert only its questions change.
- **Rollback:** scheduler flag. **Accept:** Gate A if each sequence's math is unchanged (the seeded kernels are per-sequence; GEMM rows are independent), plus ≥10% on mixed c8.

### R6 — Cold path: capture, images, restore
- **Hypothesis:** capture GDN underfills L40S (item 5 above). TF32 and BF16 are full-rate on L40S, so cold capture and the vision tower shift in cost. Cold 24K on the 4090 was 8,206 ms (**measured**).
- **Sketch:**
  - Chunked GDN with C=16/VC=32 (about 38 KiB shared memory → 2 CTAs/SM, so 192 CTAs fit in one wave on 142 SMs).
  - A per-layer capture profile at 24K.
  - A vision-feature cache with an explicit memory budget (48 GB makes this affordable).
  - The pinned restore from R1(d).
- **Code areas:** `kernels.rs:1757-1759, 1918-1980`; `kernels.cu:1910-1931`; `engine.rs:4534-4681`; `native/src/vision.rs`.
- **ROI (estimate):** cold 24K 5–15%; repeated-image cold requests large if cached.
- **Risk:** the chunk size changes reduction order (Gate B). **Accept:** Gate B, plus ≥5% on cold capture.

### R7 — GDN recurrence micro-work (lower priority)
- 128-bit loads of `kt`/`qt` in `gdn_recurrent_reg_split_seeded_bf16` (`kernels.cu:2251-2268`). Each CTA re-reads 64 KB of q/k from L1 per token. Bit-exact.
- Targeted `fmaf` in the recurrence only (global FMA was rejected in E2).
- The output-only WY formulation from the research doc §4.1.
- **ROI (estimate):** GDN recurrence is about 11–13% of the pass, so ≤3% end-to-end. Gate A for the loads, Gate B for the rest.

### Do not do on L40S
- FA3/WGMMA/TMA and NVFP4: these are Hopper/Blackwell paths.
- MIG: L40S doesn't support it.
- Tensor parallelism: no NVLink.
- Re-running rejections E4/E11 (Marlin chunking), E9, E17 or E19 (N=128 layout) without a new hypothesis.

## First three experiments

1. **L40S bring-up and ceilings (R0).**
   - Run the unchanged build with every gate, an nsys trace of 64q/2K and 1q/2K, and the core matrix.
   - Microbenchmark INT8 vs FP8/BF16 F32-accumulate MMA rates, DRAM bandwidth and L2 at the 350 W cap, logging clocks and throttle reasons.
   - Run the per-module W4A8/W4A16/W4-FP8 sweep with `matmul --bench`.
   - **Decides:** whether R3 moves ahead of R4, and whether the BF16-reduce numerics pass.
2. **48 GB configuration (R1a, environment only).**
   - Set `DECIDRA_NATIVE_GPU_EMBED=1`, a memory-derived `DECIDRA_STATE_CACHE_BYTES`/`_ENTRIES`, and sweep `DECIDRA_BATCH_ROWS` 32/64/128.
   - Measure warm/restore/mixed at 2K and 24K, and attribute the HTTP–engine gap with the existing metrics.
   - Zero code risk; establishes the real production baseline.
3. **Magic-bias Marlin epilogue (R2 step 1).**
   - Build it on tinylife's 4090 now: SASS diff (I2F should go to about 0), ncu on gate/up at M=3,334, byte-identical readouts across fixtures, then the engine p50.
   - Port it to L40S unchanged.
   - Accept only on Gate A plus ≥5% on the Marlin INT8 region.
