# Changelog

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
