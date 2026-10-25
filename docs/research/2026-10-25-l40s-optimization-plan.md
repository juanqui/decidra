# L40S maximum-performance plan

Date: 2026-10-25. Starting commit: `7291a33` (accepted SM89 work is
`b28be0d`). Target: one **NVIDIA L40S**.

## Target correction: L40S is SM89, not SM90

The request said "L40S GPU (sm90)". L40S is Ada Lovelace, compute capability
**8.9 / SM89**, exactly like the local RTX 4090; H100/H200 are SM90. This plan
therefore builds `DECIDRA_CUDA_ARCH=sm_89`. If the intended target is actually
an H100, this plan's SM89 tile/occupancy work and the existing `sm_90` binary
must be treated as a separate programme.

| Resource | L40S | RTX 4090 (local) | Consequence |
|---|---:|---:|---|
| Architecture | Ada SM89 | Ada SM89 | Same ISA; SM89 cubins, no WGMMA/TMA/FA3 |
| SMs | 142 | 128 | +10.9% grids; split-K/persistent kernels see a different grid |
| Memory | 48 GB GDDR6 ECC | 24 GB | 2x capacity; cache/embedding/expansion trades become possible |
| Bandwidth | 864 GB/s | 1,008 GB/s | -14.3%; memory-bound regions may regress |
| Power | 350 W | 600 W configured | not a matched-power comparison |
| FP32 | 91.6 TFLOPS | 82.6 | modest |
| INT8 tensor | 733 dense TOPS | 660.6 dense | +11% |
| BF16 tensor, F32 acc | ~362 dense TFLOPS | 165.2 dense | L40S datacenter Ada has a materially different BF16 rate |
| FP8 tensor, F32 acc | 733 dense TFLOPS | 330.3 dense | L40S makes FP8 re-evaluation worthwhile |
| NVLink/MIG | none / none | none / none | no tensor parallel; process-level multi-tenancy only |

NVIDIA's L40S table marks 366/733/1466 as sparse; use 183 TF32 / 362 BF16 /
733 FP8 dense as advertised ceilings, then measure. The 350 W and 864 GB/s
limits mean every result needs an L40S baseline; do not extrapolate 4090
numbers.

## Current baseline to port

The current best measured SM89 engine path is from the prior journal:

- 2,121-state, 64 distinct-question engine fixture: **752.69 -> 708.91 ms**
  at 600 W 4090, with accepted seeded convolution, state-free GDN, compact Q/K,
  and cached FlashInfer workspace.
- HTTP duplicate control: **929 -> 879.8 ms**, 69 -> 73 q/s.
- Distinct HTTP: 901.4 ms c1, 71 q/s; c8 4.04 s.
- 8K warm 998 ms; 24K warm 1,242 ms.
- 32/32 text gate, 13/13 image parity.

L40S numbers are not inherited. Marlin's stripe partition depends on the
runtime SM count, BF16 reduction is used in the output path, and the L40S
numerical result must pass all gates independently.

### Primary measured bottlenecks on the 4090

| Region | Time | Share |
|---|---:|---:|
| MLP gate/up/down | 328 ms | 46.1% |
| GDN projections (qkv/z/out) | 112 ms | 15.8% |
| GDN recurrence | 95 ms | 13.4% |
| Full attention | 44.7 ms | 6.3% |
| Embedding | 18.3 ms | 2.6% |
| GDN prepare/norm/conv | 30.6 ms | 4.3% |

The L40S plan attacks these in ROI order, but the first requirement is a
matched L40S profile. The current profile predates the accepted seeded
convolution/state-free changes and must be recollected.

## Shared benchmark and acceptance protocol

### Fixtures

- Engine: existing `eval/data/d3_distinct_2k_64q.json`; add distinct-question
  256/8K/24K fixtures (do not use the duplicate-control long fixtures for
  distinct throughput claims).
- Text gates: `d3_probe_split32.json` / `d3_hf_eval.json`, complete 32-ID
  coverage, `--min-agree 1.0`, zero unsafe flips, max Δp <= 0.0984.
- Images: 13-case image corpus, 13/13, cache identity, BF16 vision preserved.
- Long/ragged: lengths 1/15/16/31/32/33/63/64/65/255/256/257, mixed 255/256/257
  in one batch.

### Measurement matrix

Every promotion needs the full matrix; microkernel screening can be narrow.

| Dimension | Cases |
|---|---|
| questions | 1, 8, 16, 32, 64 |
| state | 256, ~2K, 8K, 24K |
| cache | cold capture, GPU hit, verified host restore |
| traffic | c1, c8; sustained HTTP, >=1000 requests for tails |
| content | distinct, duplicate control, image, mixed-state |
| instruments | engine p50/p95/p99, HTTP p50/p95/p99, q/s, rows/s, queue wait, peak VRAM, cache hits/restores, J/question |

Record GPU UUID, CC, SM count, L2, ECC, clocks, power, throttle reasons,
driver/toolkit/compiler, cubin hash, checkpoint/readout hashes, GPU sharing.
Use three interleaved A/B sessions and 20 engine repeats per cell.

### Gates

- **Gate A (bit-exact transformation):** byte-identical 255-row readouts on
  the core matrix/ragged cases; then >=5% target-region or >=2% end-to-end.
- **Gate B (numerical change):** 32/32 text, 13/13 image, state-once self-test,
  seed/tail negative controls, extended 256–1,024-row held-out set for any
  precision or reduction change.
- No OOM/retry regressions; preserve an eight-distinct-24K-state working set.
- Reject repeatable >2% regression in protected cells. Complex changes need a
  declared capacity/tail benefit even if latency is flat.

Memory budget on 48 GB: weights/vision ~16.4 GB, embedding 2.37 GB, leave
>=8 GB for long-state caches and >=2 GB scratch margin; measure
`mem_info` after preload rather than hard-coding 24 GB assumptions.

## Round 0 — L40S bring-up and ceilings (required)

**Hypothesis:** SM89 cubins execute, but SM-count-dependent Marlin partitions,
350 W clocks, and BF16/FP8 peak rates change the bottleneck and possibly the
gates.

Experiments:

1. Rent one L40S at 350 W, ECC on. Record actual clocks/power/SM/L2.
2. Build/run unchanged `sm_89` binary; run state-once, 32-row and image gates.
3. `d3_perf` and HTTP core matrix above; capture a 1q and 64q Nsight Systems
   trace and selected Nsight Compute kernels.
4. Microbench: INT8/BF16/FP8 `mma.sync` with F32 accumulation, DRAM copy,
   L2 sweep, at sustained clocks.
5. Per-module shape sweep (captured activations): W4A8-g64, W4A16, W4-FP8 at
   M=52/416/832/1664/3334/6668.

Accept: all gates pass; region sum reconciles with engine; measured ceilings
saved. If a gate changes with SM count, investigate Marlin reduction order
first (`use_fp32_reduce`/`C_tmp`), do not loosen thresholds.

Estimated direct speedup: 0%; this round decides the next three.

## Round 1 — Spend 48 GB and remove 24 GB-era policy limits

**Hypothesis:** GPU embedding, larger caches, and larger per-pass row caps
recover warm latency, restores, and long-state pass fragmentation.

Experiments:

1. GPU embedding on/off (2.37 GiB) with cache budgets 5/12/16/20 GiB and
   entries 16/32/64.
2. Memory-derived defaults in `src/config.rs` from post-preload free VRAM,
   gate 32K pass reserve.
3. Long-state batch caps: sweep `STATE_ONCE_MAX_ROWS` 64/128,
   `DECIDRA_BATCH_ROWS` 32/64/128 and `MAX_KEYS` 2^18/2^19/2^20/2^21;
   current 24K/64q fragments into 7 passes.
4. Capacity-aware capture scratch reuse; remove unconditional low-mem when
   free >2x pass reserve.
5. Attribute the HTTP-engine gap with existing metrics; optimize only after
   evidence.

Accept: >=2% warm improvement or measured hit-rate/tail gain; no OOM; long
24K distinct matrix improves >=5%; preserve eight 24K states.

## Round 2 — Marlin W4A8 group-epilogue rewrite (highest kernel ROI)

**Hypothesis:** group-boundary INT32→F32 conversions plus synchronous scale
loads/barriers are a co-bottleneck with IMMA.

Steps (bit-exact):

1. Magic-bias accumulators: initialize INT32 fragments to `0x4B400000`,
   reinterpret as float and subtract `12582912.0f`; replace per-group I2F
   with FADD. Prove |acc| < 2^24 empirically/test.
2. Stage activation group scales through the existing `cp.async` pipeline;
   remove synchronous global load and extra barrier.
3. Share one INT8 activation payload across gate/up and qkv/z only when
   `a_global`/clip/group policy match; retain per-module scale arrays.
4. Fuse MLP-input quantization into `add_rmsnorm` only after (3) works.
5. Sweep N=128 only with a matching repack/layout; the prior N128 K8 was
   valid only on selected layers and slower; no retry without a new contract.

Benchmark: ncu gate/up/down/qkv/o at M=52/416/3334/long; Gate A; per-module
shape sweep; end-to-end same. Expected 3–15% if epilogue is co-limiting.

## Round 3 — Re-evaluate precision on datacenter Ada

**Hypothesis:** L40S BF16/FP8 F32-accumulate rates are much higher relative
to INT8 than on the 4090, so previously rejected precision paths can win.

Cheapest first:

1. Small-M dispatch: W4A16 for M <= ~416 (1–8 questions), where activation
   quantization/epilogue overhead dominates. It is also more accurate.
2. W4 + FP8-activation Marlin path, shape-gated.
3. Dense FP8 W8A8 MLP only (~17.1 GB weights replacing ~8.5 GB W4), via
   cuBLASLt; requantize from BF16 source, not dequantized W4. Only after
   R0/R2 establish that MLP remains dominant.

Accept: Gate B on extended 256–1,024 rows; >=2% target traffic; no cache
capacity regression beyond declared memory trade.

## Round 4 — Batched shared-prefix attention

**Hypothesis:** at 24K each question rereads the state K/V; 864 GB/s makes
launch/traffic dominate while L40S BF16 compute is cheaper.

Implement shared-state attention over all flattened query rows, private
per-question attention, and LSE-weighted merge; or a paged/segmented ragged
kernel. Preserve noncausal block visibility, 24:4 GQA, head_dim 256, image
mRoPE offsets. Preallocate workspace per engine, not process-static.

Accept: >=20% attention-region improvement including packing/merge; >=2% at
2K or >=5% at 24K; no cross-question leakage; Gate B.

## Round 5 — Recover mixed-state and small-batch throughput

**Hypothesis:** serialized states underfill 142 SMs; measured mixed 8x8 c8 is
55–59 q/s versus 73 single-state on the 4090.

Experiments:

1. Cross-state linears: concatenate rows across state groups, carry per-row
   state IDs and cache pointers through convolution/GDN/attention; keep
   response order and cache lifetimes.
2. VC=32 seeded GDN specialization for B<=4 / one-question latency.
3. Scheduler token/work-aware admission and bounded age.
4. Only then CUDA Graphs for common 1q/8q shapes, with stable device pointer
   tables, graph cache memory budget, and replay tests.

Accept: >=10–15% mixed c8; no same-state regression; graphs only on measured
launch-gap evidence.

## Round 6 — Cold capture, vision, restore

- Cold GDN chunked VC=32/C=16/32 to fit 2 CTAs/SM and fill 142 SMs.
- Vision feature cache keyed by image/preprocess/grid/weights/precision,
  BF16 features, bounded 512 MiB; distinguish exact-state/feature-only/miss.
- Pinned, bounded host staging for 24K restore and copy/compute overlap.
- Layer-63 tail optimization only after cold profile justifies <=1% ceiling.

Accept: >=5% cold 24K or >=15% feature-hit HTTP; <=2% warm regression; no
OOM; gates unchanged.

## Round 7 — GDN recurrence micro-work and WY (lower priority)

- 128-bit q/k loads in the seeded register kernel; targeted FMA only in
  recurrence (global FMA was rejected); measure spill/regs.
- Output-only short-block WY solve with shared seed; FP32 reference, then
  TF32/BF16 operands; preserve 1/15/.../257 ragged cases.

Accept: Gate A for loads; Gate B and >=25% complete GDN-region improvement or
>=3% end-to-end for WY; retain register fallback.

## Execution order

| Round | Focus | Decision gate |
|---|---|---|
| 0 | L40S baseline, gates, microbench, per-shape | gates + ceilings |
| 1 | 48 GB policy/cache/batch caps | >=5% long-state, no OOM |
| 2 | Marlin epilogue (bit-exact) | >=5% kernel, >=2% end-to-end |
| 3 | Precision re-evaluation | extended quality + >=2% |
| 4 | Batched attention | >=5% 24K |
| 5 | Cross-state/small-batch/graphs | >=10% mixed |
| 6 | Cold/vision/restore | >=5% cold |
| 7 | GDN micro/WY | >=3% end-to-end |

Round 0 and 1 are mandatory before any kernel acceptance. A round may be
rejected; the next round proceeds with the journal updated.

## Immediate first three experiments

1. **L40S bring-up** (Round 0): unchanged binary, all gates, core matrix,
   nsys/ncu, microbench and per-shape sweep.
2. **48 GB configuration** (Round 1): GPU embedding, memory-derived cache
   budgets, batch-row cap sweep, long-state distinct matrix.
3. **Magic-bias Marlin epilogue** (Round 2): implement on the 4090 first,
   verify byte-identical 255-row readouts, then L40S.

## Open hazards

- `run_group` cache eviction with pinning disabled has a stale index/bytes
  path; fix and test before eviction experiments.
- FlashInfer workspace is process-static; safe only under one stream. Make it
  engine/stream-owned before any concurrent-forward or graph work.
- L40S and 4090 are not matched power/clock configurations.
- The 32-row gate is a smoke gate; no threshold may be loosened for L40S.
- Existing long HTTP fixtures repeat one question; build distinct 256/8K/24K
  fixtures before claiming long-context throughput.
