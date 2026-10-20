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
- **Quantizer**: full vs `JEV_SKIP_QUANT` differs by <0.01 ms/op. Scale writes
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

## Next checklist

- [ ] Double-buffered accumulators at tmb=2; measure up/down/in_qkv.
- [ ] If that wins, try tmb=4 with a shared partial buffer (BN=64 if needed).
- [ ] Pipeline the scale fetch with `cp.async`; remove the staging barrier.
- [ ] Re-run suite (28/28, MAE 8.642e-4, max 7.511e-3) after each step.
- [ ] Only then consider the full CUTLASS/CuTe SM80 collective port.

## Status

All experiments above were reverted. HEAD remains the verified kernel
(`ec1ace4` + `e9a49f3` docs); no uncommitted kernel changes.
