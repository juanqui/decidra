#!/usr/bin/env bash
# One real native forward on GPU 1: loads the AWQ checkpoint, keeps the packed
# weights resident, and reads the two noul head logits.
#
# This is the only test that needs the GPU. It is compiled only with
# `--features native` and is ignored by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CARGO:-}" ]]; then
  if command -v cargo >/dev/null 2>&1; then
    CARGO="$(command -v cargo)"
  elif [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
    CARGO="${HOME}/.cargo/bin/cargo"
  else
    echo "error: cargo not found on PATH or at \$HOME/.cargo/bin/cargo" >&2
    exit 1
  fi
fi

# The native engine opens ordinal 0 of the visible set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DECIDRA_NATIVE_MODEL="${DECIDRA_NATIVE_MODEL:-/srv/models/jev-awq/awq-w4a16-v2}"
export DECIDRA_CHECKPOINT="${DECIDRA_CHECKPOINT:-${DECIDRA_NATIVE_MODEL}}"
export DECIDRA_NATIVE_CACHE="${DECIDRA_NATIVE_CACHE:-true}"

# cuBLAS/cuBLASLt location (dynamic loading at runtime).
if [[ -z "${DECIDRA_CUDA_LIBS:-}" && -d /srv/models/jev-awq/cudalibs ]]; then
  DECIDRA_CUDA_LIBS=/srv/models/jev-awq/cudalibs
fi
if [[ -n "${DECIDRA_CUDA_LIBS:-}" ]]; then
  if ! compgen -G "${DECIDRA_CUDA_LIBS}/libcublas.so.13*" >/dev/null \
    || ! compgen -G "${DECIDRA_CUDA_LIBS}/libcublasLt.so.13*" >/dev/null; then
    echo "error: libcublas.so.13 / libcublasLt.so.13 not found in ${DECIDRA_CUDA_LIBS}" >&2
    exit 1
  fi
  if [[ ":${LD_LIBRARY_PATH:-}:" != *":${DECIDRA_CUDA_LIBS}:"* ]]; then
    export LD_LIBRARY_PATH="${DECIDRA_CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
elif ! (ldconfig -p 2>/dev/null || true) | grep -q 'libcublas\.so\.13'; then
  echo "error: libcublas.so.13 not found; set DECIDRA_CUDA_LIBS" >&2
  exit 1
fi

exec "$CARGO" test --release --features native --test native_smoke -- \
  --ignored --nocapture "$@"
