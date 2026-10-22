#!/usr/bin/env bash
# Build the native engine, preload d3-turbo, and run the state-once self-test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo build --release --locked --features native --manifest-path "$ROOT/Cargo.toml"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
DECIDRA_CHECKPOINT="${DECIDRA_CHECKPOINT:-/srv/models/d3-turbo-w4a8-g64-v4}" \
DECIDRA_NATIVE_MODEL="${DECIDRA_NATIVE_MODEL:-${DECIDRA_CHECKPOINT:-/srv/models/d3-turbo-w4a8-g64-v4}}" \
DECIDRA_NATIVE_GPU_EMBED="${DECIDRA_NATIVE_GPU_EMBED:-false}" \
DECIDRA_CUDA_LIBS="${DECIDRA_CUDA_LIBS:-/usr/local/cuda/lib64}" \
LD_LIBRARY_PATH="${DECIDRA_CUDA_LIBS:-/usr/local/cuda/lib64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "$ROOT/target/release/decidra" --native-check
