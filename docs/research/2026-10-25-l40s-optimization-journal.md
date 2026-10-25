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
