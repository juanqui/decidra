# L40S optimization journal

Program plan: [2026-10-25-l40s-optimization-plan.md](2026-10-25-l40s-optimization-plan.md).
Start commit: `7291a33`; accepted SM89 engine work: `b28be0d`.

## 2026-10-25 — Research inputs and target correction

- User target: L40S, one GPU at a time; at least five optimization rounds.
- **L40S is Ada SM89 (CC 8.9), not SM90.** H100/H200 are SM90. This programme
  builds `DECIDRA_CUDA_ARCH=sm_89`; if the real target is H100, stop and
  retarget.
- L40S facts used: 48 GB GDDR6 ECC, 864 GB/s, 350 W, ~142 SMs, FP32 91.6 TF,
  dense tensor ceilings approximately 183 TF32 / 362 BF16 / 733 FP8 / 733 INT8
  (NVIDIA lists the doubled values with sparsity). Measure at runtime.
- Independent analyses were requested through:
  - **Codex Astra** (`gpt-6-astra`, `model_reasoning_effort=xhigh`),
    read-only.
  - **Claude Opus 5.5** (`claude --model opus`, high effort), read-only.
- Both independently agreed the first work is a matched L40S baseline, then
  48 GB policy tuning, then Marlin group-epilogue/occupancy work; attention,
  cross-state, precision and graphs follow evidence.

### External-analysis synthesis

Astra emphasized: 48 GB cache/embedding budget; long-state fragmentation
(8K/24K passes); Marlin register/shared-memory occupancy and group-scale
epilogue; 96-CTA GDN underfill on 142 SMs; cache-eviction and static
FlashInfer-workspace hazards; full A/B matrix and exact 32/32 gate.

Opus emphasized: L40S datacenter Ada materially changes BF16/FP8 rates versus
GeForce Ada, so prior SM89 precision rejections must be re-evaluated; 24 GB
policy defaults are stale; magic-bias INT32→F32 epilogue is a bit-exact
high-value kernel experiment; shared-prefix attention matters at 24K;
cross-state batching targets the measured 55–59 vs73 q/s mixed-state gap.

## Round 0 — L40S baseline

Status: planned. Required evidence:

- Actual L40S UUID/CC/SM/L2/ECC/clocks/power/throttle reasons.
- Build/cubin hash and runtime architecture match.
- State-once self-test, complete 32/32 text gate, 13/13 image parity.
- Engine distinct 2K 64q, HTTP distinct 2K, cold/warm 8K/24K, mixed-state c8.
- nsys timeline and selected ncu kernel metrics.
- INT8/BF16/FP8 MMA, DRAM, L2 microbench at sustained clocks.

The baseline is not complete until these are recorded and compared with the
matched 4090 accepted path (708.91 ms engine, 879.8 ms duplicate HTTP,
901.4 ms distinct HTTP, 8K/24K warm 998/1242 ms).

## 2026-10-25 — R0 preparation

- Added distinct-question fixtures for 256/8K/24K states:
  - engine: state 301/8,234/24,968 tokens, 64 unique questions, 4,039 rows.
  - HTTP: same states and unique questions.
- Fixed the `run_group` unpinned-cache eviction hazard identified by Astra:
  the hit entry is preserved while older entries are removed, the byte total
  is recomputed after each removal, and the cached index is rebound after
  reinsertion. Default pinned behavior is unchanged. This is required before
  the L40S cache-budget round.
- Built the current engine on the L40S (22 MB binary, sm_89).
- First hardware microbench at sustained 2520 MHz:
  - BF16/F32-accumulate 8192^3: 262.8 TFLOPS, 156 W, C[0]=8192 correct.
  - INT8/INT32 cuBLAS: 151.1 TOPS (cuBLAS path; not the Marlin kernel).
  - 1 GiB copy: 625.6 GB/s (r+w).
  - L40S reports 142 SMs, 46,068 MiB, ECC on, 350 W limit.
- Artifact upload is still in progress; L40S baseline pending.

## 2026-10-25 — L40S Round 0 baseline

Rental: one L40S 48 GB, 46,068 MiB, ECC on, 350 W, 142 SMs, boost 2520
MHz, 48 host CPUs, 236 GB RAM. Built exact `sm_89` binary from `c7569c1`;
artifact SHA-256 prefix `8321ed451da7f5af` matches the 4090 v2 artifact.

### Gates

- Text gate: **32/32**, mean MAE 0.0158, max Δp 0.0984 (same envelope).
- Image API smoke: 20/20; image decisions **13/13**.
- State-once self-test passes.

### Engine baseline, distinct questions

| state | capture | question p50 | 4090 accepted p50 |
|---|---:|---:|---:|
| 2K (2,121) | 483 ms | **853.2 ms** | 708.9 ms |
| 2K + GPU embedding | 480 ms | **845.4 ms** | — |
| 8K (8,234) | 1,748 ms | **1,119.3 ms** | — |
| 24K (24,968) | 5,868 ms | **1,386.5 ms** | — |

L40S is ~20% slower than the 600 W 4090 on the 2K INT8 path despite 142 SMs;
measured BF16 tensor was 262.8 TFLOPS and 1 GiB copy 625.6 GB/s at 2520 MHz,
with only ~156 W during the BF16 microbench. The 350 W cap is not the immediate
limiter; Marlin/INT8 and memory behavior are.

### HTTP baseline (distinct)

| state | c1 | q/s | c8 |
|---|---:|---:|---:|
| 2K | 1,060.9 ms | 60.3 | 4,737.5 ms, 60.8 q/s |
| 8K cold / warm | 3,573.6 / 1,165.9 ms | 18 / 55 | — |
| 24K cold / warm | 9,976.4 / 1,474.8 ms | 6 / 43 | — |

Fresh post-preload free VRAM was 30.39 GiB (host embedding) / 28.05 GiB (GPU
embedding); 24K warm cache left 21.0 GiB. This is the capacity headroom the
L40S plan targets.

## Round 1 — 48 GB configuration

Configuration: GPU embedding, 16 GiB/64-entry GPU state cache, 16 GiB host
cache, batch rows 64, state-once rows 128, max keys 2^21, max scores 2^29.

| path | baseline | R1 | delta |
|---|---:|---:|---:|
| 2K engine | 853.2 | 845.4 (GPU embed only) | -0.9% |
| 2K HTTP c1 64q | 1,060.9 | **1,033.0** | -2.6% |
| 2K HTTP c8 | 4,737.5 | **4,624.2** | -2.4% |
| 8K warm HTTP | 1,165.9 | **1,133.5** | -2.8% |
| 24K warm HTTP | 1,474.8 | **1,424.0** | -3.4% |
| 24K engine (caps only) | 1,379.5 | **1,353.7** | -1.9% |

No OOM. The 5% long-state gate was not met; capacity headroom remains large.
R1 is retained as an L40S production candidate but not yet accepted as a round
win.

## Round 2 — Marlin epilogue and thread-resource experiments

- **Magic-bias bit-exact group epilogue** (INT32 fragment biased by `2^23`,
  FADD instead of I2F): local 4090 logits were **byte-identical** to the
  reference on all 32 rows. Performance: 710.4 ms versus 708.9 ms accepted;
  no gain. Reverted.
- **512-thread Marlin specialization** (N16/K4/tmb4 group-a8): compiled but
  launch failed with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`; registers/shared
  prevent 512 threads. Reverted.
- Conclusion: L40S Marlin work needs a different resource/tile design, not
  these two knobs.

## Round 3 — Precision alternatives

Static W4A16 MLP on L40S (first 56 layers; last:8 remains INT8):

| workload | W4A8 R1 | W4A16 MLP |
|---|---:|---:|
| 1q c1 | 43.6 ms | **41.6 ms** (-4.6%) |
| 8q c1 | 135.8 ms | 145.7 ms (+7.3%) |
| 64q c1 | 1,060.9 ms | 1,125.8 ms (+6.1%) |

W4A16 only wins at 1q; a static switch loses production throughput. A shape-
dispatched dual representation is possible but costs another MLP weight copy
and is not accepted without a 1–4q traffic case.

## Round 4 — Current L40S profile

Distinct 2K region sum **842.8 ms** (engine p50 858.8): MLP 394.2 ms (46.8%),
GDN recurrence 112.2 ms (13.3%), GDN projections 158.7 ms (18.8%), attention
37.6 ms (4.5%), GDN prepare/conv/norm 33.9 ms. Attention is not slower than the
4090; the INT8/Marlin and recurrence portions are. This profile is the target
for the next kernel round.
