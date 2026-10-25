# Changelog

## Unreleased

- Publish per-architecture images (`sm89-latest`, `sm90-latest`,
  `sm120-latest`, plus `sm<arch>-<version>` and `sm<arch>-<sha>`); the
  unsuffixed tags remain SM89.
- Fixed state-once OOMs under coalesced and long requests: the batch row
  budget now sums queued question rows, scratch reserves scale with state
  length, cached states are offloaded when needed, and the CUDA
  async-allocator pool is trimmed with `cuMemPoolTrimTo`. State capture now
  releases scratch for the whole capture instead of retaining it below the old
  12,288-token low-memory cutoff. Added cache/free-VRAM metrics and one-shot
  retries for both state-capture and question-pass OOMs.
- Added native SM120 NVFP4 checkpoint loading and FP4×FP4 cuBLASLt GEMMs.
- Added dynamic NVFP4 activation quantization with a coalesced one-CTA-per-row
  kernel and block-scaled VEC16_UE4M3 scale repacking.
- Added `quant/quantize_nvfp4.py` for GPTQ NVFP4 with d3 `prefix_bidir`
  calibration and per-scope module selection.
- NVFP4 W4A4 is explicitly experimental: full MLP+attention is 500.8 ms /
  128 q/s on a 600 W RTX PRO 6000 but fails the 32-row HF accuracy gate
  (28/32, max Δp 0.279). FP8 remains the validated approximate path
  (32/32, max Δp 0.0115).

## 0.2.0 — 2026-10-22

Public d3-turbo release.

- Removed the JEV/AutoTrust/pplx/GEV paths; the server, docs, tests, and
  evaluation surface are d3-turbo only.
- Added the exact-state GPU cache plus host-memory L2 for overflow states:
  16 states at 69 q/s, 32 states at 63 q/s on a 450 W RTX 4090.
- Added fused residual + RMSNorm, suffix-only question tokenization, 32-row
  scheduler budgeting, state-cache metrics, and cuBLAS worker-thread rebind.
- Shipped W4A8 group-64 SM89 defaults, BF16 SM90 support, and RTX PRO 6000
  Blackwell (SM120) support with architecture-checked cubins.
- Added an opt-in SM120 all-linear FP8 question path: 623 ms / 103 q/s on a
  600 W RTX PRO 6000 WS and 859 ms / 74 q/s on a 300 W Max-Q, with BF16 state
  capture and approximate (0.0362) per-row parity.
- Refreshed the container image defaults, compose file, and public docs.

## 0.1.0

Initial state-once engine and server path.
