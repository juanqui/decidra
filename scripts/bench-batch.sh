#!/usr/bin/env bash
# Run the batching concurrency sweep against an already-running jev-engine.
#
# Requires:
#   - jev-engine on JEV_SMOKE_URL (default http://127.0.0.1:8501)
#   - Python dependencies from eval/requirements.txt (dev tooling only)
#
# This script never starts or stops the server and never touches the GPU.
# The default sweep is c=1,2,4,8,16 with 16 unique ~69-token decision requests
# per level.  Results, including server batch metrics when /metrics exposes
# them, are written to eval/out/bench_batch.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL="$ROOT/eval"
PYTHON="${PYTHON:-python3}"

RUST_URL="${JEV_SMOKE_URL:-${JEV_DECIDE_URL:-http://127.0.0.1:8501}}"
OUT="${JEV_BATCH_OUT:-$EVAL/out/bench_batch.json}"
CONCURRENCY="${JEV_BATCH_CONCURRENCY:-1,2,4,8,16}"
REQUESTS="${JEV_BATCH_REQUESTS:-16}"
TOKENS="${JEV_BATCH_TOKENS:-69}"
CHARS_PER_TOKEN="${JEV_BATCH_CHARS_PER_TOKEN:-3.38}"
WARMUP="${JEV_BATCH_WARMUP:-3}"

RUST_URL="${RUST_URL%/}"
mkdir -p "$(dirname "$OUT")"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing required command: $1" >&2
    exit 1
  }
}

need curl
need "$PYTHON"

if ! curl -fsS --max-time 3 "$RUST_URL/health" >/dev/null 2>&1; then
  echo "error: jev-engine is not healthy at $RUST_URL/health" >&2
  exit 1
fi

echo "batch sweep: url=$RUST_URL concurrency=$CONCURRENCY requests=$REQUESTS tokens=$TOKENS"
cd "$EVAL"
"$PYTHON" bench.py \
  --url "$RUST_URL" \
  --mode batch \
  --batch-concurrency "$CONCURRENCY" \
  --batch-requests "$REQUESTS" \
  --batch-tokens "$TOKENS" \
  --chars-per-token "$CHARS_PER_TOKEN" \
  --warmup "$WARMUP" \
  --out "$OUT"
