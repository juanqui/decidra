#!/usr/bin/env bash
# Build and run the jev-engine release binary with the in-process native CUDA
# backend. The HTTP server is CPU-side; the native engine uses GPU 1 (selected
# with CUDA_VISIBLE_DEVICES) inside the same process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/target/release/jev-engine"

if [[ -z "${CARGO:-}" ]]; then
  if command -v cargo >/dev/null 2>&1; then
    CARGO="$(command -v cargo)"
  elif [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
    CARGO="${HOME}/.cargo/bin/cargo"
  else
    echo "error: cargo not found on PATH or at \$HOME/.cargo/bin/cargo" >&2
    echo "       install Rust (https://rustup.rs) or set CARGO=/path/to/cargo" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export JEV_NATIVE_MODEL="${JEV_NATIVE_MODEL:-/srv/models/jev-awq/awq-w4a16-v2}"
export JEV_CHECKPOINT="${JEV_CHECKPOINT:-${JEV_NATIVE_MODEL}}"
export JEV_NATIVE_CACHE="${JEV_NATIVE_CACHE:-true}"
export JEV_NATIVE_MARLIN="${JEV_NATIVE_MARLIN:-true}"
export JEV_NATIVE_FUSED_W4A16="${JEV_NATIVE_FUSED_W4A16:-true}"
export JEV_BIND_ADDR="${JEV_BIND_ADDR:-0.0.0.0:8501}"
export JEV_METRICS_ENABLED="${JEV_METRICS_ENABLED:-true}"

# cuBLAS/cuBLASLt location. The native engine loads them dynamically. Prefer the
# stable directory provisioned next to the model; JEV_CUDA_LIBS overrides it.
if [[ -z "${JEV_CUDA_LIBS:-}" && -d /srv/models/jev-awq/cudalibs ]]; then
  JEV_CUDA_LIBS=/srv/models/jev-awq/cudalibs
fi
if [[ -n "${JEV_CUDA_LIBS:-}" ]]; then
  if ! compgen -G "${JEV_CUDA_LIBS}/libcublas.so.13*" >/dev/null; then
    echo "error: libcublas.so.13 not found in JEV_CUDA_LIBS=${JEV_CUDA_LIBS}" >&2
    exit 1
  fi
  if ! compgen -G "${JEV_CUDA_LIBS}/libcublasLt.so.13*" >/dev/null; then
    echo "error: libcublasLt.so.13 not found in JEV_CUDA_LIBS=${JEV_CUDA_LIBS}" >&2
    exit 1
  fi
  if [[ ":${LD_LIBRARY_PATH:-}:" != *":${JEV_CUDA_LIBS}:"* ]]; then
    export LD_LIBRARY_PATH="${JEV_CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
elif ! (ldconfig -p 2>/dev/null || true) | grep -q 'libcublas\.so\.13'; then
  echo "error: libcublas.so.13 not found; set JEV_CUDA_LIBS to the cuBLAS directory" >&2
  exit 1
fi

if [[ "${JEV_SKIP_BUILD:-0}" != "1" ]]; then
  "$CARGO" build --release --locked --features native \
    --manifest-path "$ROOT/Cargo.toml"
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: release binary not found at $BIN" >&2
  exit 1
fi

exec "$BIN" "$@"
