# Warp-specialized group-INT8 GEMM — working journal

Goal: replace the per-group INT8 epilogue cost in Marlin with a
warp-specialized / software-pipelined schedule, targeting the per-token
kernel's 463 TOPS (1.616 ms/op at 4K) instead of the group kernel's ~330 TOPS
(2.135 ms/op).

Hardware: RTX 4090 (SM89, Ada), CUDA 13.2 driver, kernels built to `sm_89`
with `-O3 -fmad=false`. No local CUTLASS checkout; network access is
available. CUTLASS 3.x's warp-specialized (TMA/WGMMA) collectives target
SM90; on SM89 the applicable design is the SM80 `cp.async` + `mma.sync`
pipeline, which Marlin already implements. The missing piece is the
per-group epilogue, not the MMA pipeline.

## 2026-10-20 — measurements

All timings are 4K-token forwards (`--repeat 4-6`, profile region sums),
median-of-runs, same session; the per-token `gate` region and `gdn.recurrent`
are the clock references.

| variant | up ms/op | down | in_qkv | notes |
|---|---:|---:|---:|---|
| shipped group kernel | 2.229 | 2.155 | 1.324 | HEAD, strided scales + shared staging |
| per-token template, group data | **1.616** | **1.526** | **0.959** | wrong numerics, valid timing; the ceiling |
| group, blocked scales + shared staging | 2.577 | 2.522 | 1.436 | bit-identical |
| group, blocked scales + direct loads | 2.628 | 2.553 | 1.558 | bit-identical, no barrier |
| group, FMA epilogue `c0*(bs*a_s)` | 2.122 | 2.050 | 1.263 | +1.9%, max Δ 1.027e-2 |
| group, FMA epilogue `(c0*bs)*a_s` | — | — | — | fails gate: max 1.731e-2 |

Diagnostics:

- **Registers**: group 236 vs per-token 233 at tmb=4 (driver
  `CU_FUNC_ATTRIBUTE_NUM_REGS`). Both are one CTA/SM at 256 threads. The +3
  registers are not an occupancy cliff.
- **Quantizer**: full vs `DECIDRA_SKIP_QUANT` differs by <0.01 ms/op. Scale writes
  are not the cost, in either layout.
- **Blocked scale layout** (`[64-row tile][group][64]`, contiguous 64-row
  loads): bit-identical and correct, but slower in both staging and direct
  forms. Reverted. The strided `[row][group]` layout benefits from L1 reuse
  across the 80 K stages.
- **No-staging baseline** (a_s = 1, staging removed): 1.891 ms/op. So the
  staging/barrier costs ~0.24 ms/op and the group epilogue structure costs
  ~0.27 ms/op even with trivial scales.
- The per-token kernel applies its row scale once after the K loop; the group
  kernel must scale and accumulate every 64-K group into f32. That serialized
  epilogue is the 25% gap.

## Design: software-pipelined per-group epilogue

The MMA and the epilogue are serialized per group: the next group's MMA writes
`frag_c_tmp` while the previous group's epilogue reads it (WAR hazard). The
fix is to overlap them:

1. **Double-buffer the int32 group accumulators.** `frag_c_tmp` is 64
   registers/thread at tmb=4 (2 M tiles × 2 N halves × 4 k-groups × 2 int
   words). A second buffer does not fit (236 + 64 > 255). At **tmb=2** the
   base kernel is 198 registers; +32 = 230, which fits one CTA/SM. Start
   there: MMA group g+1 into buffer B while the epilogue consumes buffer A,
   alternating.
2. **Producer/consumer alternative**: 4 MMA warps + 4 epilogue warps. The MMA
   warps publish int32 partials to shared (64×128×4 B = 32 KB per buffer,
   double-buffered 64 KB). The current dynamic shared budget is tight; a
   BN=64 tile halves it to 16 KB/buffer. The epilogue warps apply
   `a_s * b_s`, accumulate f32, and write the BF16 output.
3. **Scale fetch in the existing pipeline**: with the old `[row][group]`
   layout, stage the next K-stage's 64 scales with `cp.async` alongside the
   A/B tiles and let `wait_for_stage()` publish them. No extra barrier. (The
   earlier piggyback attempt read the wrong stage because the pipeline is one
   stage ahead; the fetch must target the same stage as `fetch_to_shared`.)
4. **Keep bit-identical numerics**: all of the above must reproduce the
   current order `acc += (c0*bs0)*a_s` (`-fmad=false`). The FMA forms are a
   separate, margin-costing tradeoff and are not part of the kernel rewrite.

Ceiling: the per-token template's 1.616 ms/op. A successful pipelined epilogue
should land 1.75–1.85 ms/op, i.e. **8–15% on the five group GEMMs** and
**4–7% end-to-end**, with the 28/28 suite bit-identical.

## 2026-10-20 — shipped: FMA group epilogue; pipeline scale fetch rejected

- **FMA epilogue shipped** (`fmaf(c0, bs0*a_s, acc)`): 4K `up` ~2.14 ms/op
  (from 2.23), server 4K/8K/16K **0.833 / 1.711 / 3.740 s** (from
  0.854/1.757/3.838). Suite **28/28, MAE 7.166e-4, max 1.027e-2, KL 3.269e-5**
  (gate 0.011). Batched: c8 max 1.305e-2 / MAE 7.23e-4, c16 max 1.251e-2 /
  MAE 1.115e-3, top-1 28/28. The exact `+= c0*bs0*a_s` order remains the
  bit-identical option but was 1.9% slower.
- **Pipelined scale fetch** (two shared halves, fill after the wait, fetch the
  next stage after the fill, tmb≥2): bit-level wrong (4K logits delta
  -0.203/+0.094) and only ~1% faster. The staging barrier is not the dominant
  cost. Reverted.
- **Double-buffered int32 accumulators**: `frag_c_tmp` is 64 registers at tmb=2
  (not 32); 208+64=272 > 255, so tmb=2 does not fit. tmb=1 fits (191) but
  quadruples B traffic. A producer/consumer epilogue needs shared partials;
  `MR_SHARED` is already 101,376 B (99 KB), so it needs a smaller BN tile.

## Next checklist

- [x] FMA epilogue: +1.9% long, gate passes.
- [x] Pipeline scale fetch: attempted, wrong and not faster; reverted.
- [x] Double-buffered accumulators: does not fit at tmb=2; documented.
- [ ] Producer/consumer epilogue with a BN=64 tile and shared partials.
- [ ] Only then consider the full CUTLASS/CuTe SM80 collective port.

## Status

The FMA group epilogue is shipped (1.9% faster, gate passes). The exact-order
kernel is the documented fallback; the deeper warp-specialized epilogue remains
the BN=64 producer/consumer design above.
