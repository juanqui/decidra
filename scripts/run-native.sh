#!/usr/bin/env bash
# Build and serve d3-turbo with the in-process native CUDA backend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/target/release/decidra"

if [[ "${DECIDRA_SKIP_BUILD:-0}" != "1" ]]; then
  EXTRA=()
  [[ -n "${DECIDRA_CUDA_ARCH:-}" ]] && EXTRA+=(env "DECIDRA_CUDA_ARCH=${DECIDRA_CUDA_ARCH}")
  "${EXTRA[@]}" cargo build --release --locked --features native --manifest-path "$ROOT/Cargo.toml"
fi
[[ -x "$BIN" ]] || { echo "error: missing $BIN" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DECIDRA_BIND_ADDR="${DECIDRA_BIND_ADDR:-0.0.0.0:8899}"
export DECIDRA_CHECKPOINT="${DECIDRA_CHECKPOINT:-/srv/models/d3-turbo-w4a8-g64-v4}"
export DECIDRA_NATIVE_MODEL="${DECIDRA_NATIVE_MODEL:-$DECIDRA_CHECKPOINT}"
export DECIDRA_TOKENIZER="${DECIDRA_TOKENIZER:-$DECIDRA_CHECKPOINT/tokenizer.json}"
export DECIDRA_NATIVE_GPU_EMBED="${DECIDRA_NATIVE_GPU_EMBED:-false}"
if [[ -z "${DECIDRA_CUDA_LIBS:-}" && -d /usr/local/cuda/lib64 ]]; then
  DECIDRA_CUDA_LIBS=/usr/local/cuda/lib64
fi
if [[ -n "${DECIDRA_CUDA_LIBS:-}" ]]; then
  export LD_LIBRARY_PATH="${DECIDRA_CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
exec "$BIN" "$@"
