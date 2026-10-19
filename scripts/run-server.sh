#!/usr/bin/env bash
# Build and run the jev-engine release binary.
#
# Defaults are deliberately different from the production Python front:
# bind 0.0.0.0:8501 and vLLM on 127.0.0.1:8002 (GPU 1 test backend).
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

if [[ ! -x "$CARGO" ]] && ! command -v "$CARGO" >/dev/null 2>&1; then
  echo "error: CARGO is not executable: $CARGO" >&2
  exit 1
fi

export JEV_VLLM_URL="${JEV_VLLM_URL:-http://127.0.0.1:8002}"
export JEV_VLLM_MODEL="${JEV_VLLM_MODEL:-jev-awq}"
export JEV_CHECKPOINT="${JEV_CHECKPOINT:-/srv/models/jev-awq/awq-w4a16-v2}"
export JEV_BIND_ADDR="${JEV_BIND_ADDR:-0.0.0.0:8501}"
export JEV_MAX_CONCURRENCY="${JEV_MAX_CONCURRENCY:-8}"
export JEV_REQUEST_TIMEOUT_SECS="${JEV_REQUEST_TIMEOUT_SECS:-600}"
export JEV_METRICS_ENABLED="${JEV_METRICS_ENABLED:-true}"

if [[ "${JEV_SKIP_BUILD:-0}" != "1" ]]; then
  "$CARGO" build --release --locked --manifest-path "$ROOT/Cargo.toml"
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: release binary not found at $BIN" >&2
  exit 1
fi

exec "$BIN" "$@"
