#!/usr/bin/env bash
# Run the independent evaluation against an already-running jev-engine.
#
# Requires:
#   - jev-engine on JEV_EVAL_RUST_URL (default http://127.0.0.1:8501)
#   - Python dependencies from eval/requirements.txt (dev tooling only)
#
# This script never starts or stops the server and never touches the GPU.
# Results are written to eval/out/ (git-ignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL="$ROOT/eval"
PYTHON="${PYTHON:-python3}"

RUST_URL="${JEV_EVAL_RUST_URL:-${JEV_SMOKE_URL:-http://127.0.0.1:8501}}"
OUT="${JEV_EVAL_OUT:-$EVAL/out}"
REQUESTS="${JEV_EVAL_REQUESTS:-128}"
LONG_REQUESTS="${JEV_EVAL_LONG_REQUESTS:-8}"
CONCURRENCY="${JEV_EVAL_CONCURRENCY:-1,4,8,12}"
LONG_SIZES="${JEV_EVAL_LONG_SIZES:-3000,9000}"
CHARS_PER_TOKEN="${JEV_EVAL_CHARS_PER_TOKEN:-3.38}"
WARMUP="${JEV_EVAL_WARMUP:-5}"
SKIP_BENCH="${JEV_EVAL_SKIP_BENCH:-0}"
SKIP_FUNCTIONAL="${JEV_EVAL_SKIP_FUNCTIONAL:-0}"
SIGNING_KEY="${JEV_EVAL_SIGNING_KEY:-}"

RUST_URL="${RUST_URL%/}"
mkdir -p "$OUT"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: $1 is required" >&2
    exit 1
  }
}

need curl
need "$PYTHON"

if ! "$PYTHON" - <<'PY'
import sys
try:
    import requests  # noqa: F401
    from PIL import Image  # noqa: F401
except Exception as error:
    print(f"error: missing Python dependency: {error}", file=sys.stderr)
    print("       install with: python3 -m pip install -r eval/requirements.txt", file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 1
fi

wait_health() {
  local name="$1" url="$2" i
  for i in $(seq 1 60); do
    if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
      echo "ok   $name health: $url/health"
      return 0
    fi
    sleep 1
  done
  echo "error: $name is not healthy at $url/health" >&2
  exit 1
}

wait_health jev-engine "$RUST_URL"

cd "$EVAL"

echo
echo "== CPU AWQ golden (primary): strict gates"
"$PYTHON" suite.py \
  --a-url "$RUST_URL" \
  --baseline data/awq-cpu-suite28.json \
  --min-top1 1.0 --max-kl 0.001 --max-mae 0.0005 --max-delta 0.001 \
  --no-strict \
  --out "$OUT/suite_rust_vs_cpu_strict.json"

echo
echo "== CPU AWQ golden (primary): acceptance"
"$PYTHON" suite.py \
  --a-url "$RUST_URL" \
  --baseline data/awq-cpu-suite28.json \
  --min-top1 1.0 --max-kl 0.001 --max-mae 0.001 --max-delta 0.011 \
  --out "$OUT/suite_rust_vs_cpu.json"

echo
echo "== historical production-v2 baseline (previous serving stack; context only)"
# That baseline itself differs from CPU AWQ golden by max 0.01377 / mean MAE
# 0.00156, so it cannot be the native acceptance gate.
"$PYTHON" suite.py \
  --a-url "$RUST_URL" \
  --baseline data/awq-v2-suite.json \
  --min-top1 1.0 --max-kl 0.001 --max-mae 0.0025 --max-delta 0.02 \
  --no-strict \
  --out "$OUT/suite_rust_vs_v2_historical.json"

if [[ "$SKIP_BENCH" != "1" ]]; then
  echo
  echo "== combined short/long benchmark"
  "$PYTHON" bench.py \
    --url "$RUST_URL" \
    --mode both \
    --concurrency "$CONCURRENCY" \
    --requests "$REQUESTS" \
    --long-requests "$LONG_REQUESTS" \
    --long-sizes "$LONG_SIZES" \
    --chars-per-token "$CHARS_PER_TOKEN" \
    --warmup "$WARMUP" \
    --out "$OUT/bench_combined_rust.json"
else
  echo "skip benchmark (JEV_EVAL_SKIP_BENCH=1)"
fi

if [[ "$SKIP_FUNCTIONAL" != "1" ]]; then
  echo
  echo "== functional checks"
  "$PYTHON" functional_checks.py \
    --url "$RUST_URL" \
    --out "$OUT/functional_checks.json"
else
  echo "skip functional checks (JEV_EVAL_SKIP_FUNCTIONAL=1)"
fi

if [[ -n "$SIGNING_KEY" ]]; then
  echo
  echo "== HMAC signing"
  "$PYTHON" sign_check.py \
    --url "$RUST_URL" \
    --key "$SIGNING_KEY" \
    --out "$OUT/signing_check.json"
else
  echo
  echo "skip signing check (set JEV_EVAL_SIGNING_KEY for a server started with that key)"
fi

echo
echo "== reports"
if [[ -f "$OUT/bench_combined_rust.json" ]]; then
  "$PYTHON" report.py \
    --suite "$OUT/suite_rust_vs_cpu.json" \
    --bench "$OUT/bench_combined_rust.json" \
    --out-prefix "$OUT/live-report" \
    --per-row \
    --note "Primary comparison is native vs the CPU AWQ golden, max_delta 0.011 / mean_mae 0.001." \
    --note "See docs/RESULTS.md and docs/KERNELS.md."
else
  "$PYTHON" report.py \
    --suite "$OUT/suite_rust_vs_cpu.json" \
    --out-prefix "$OUT/live-report" \
    --per-row \
    --note "Primary comparison is native vs the CPU AWQ golden, max_delta 0.011 / mean_mae 0.001."
fi

echo
echo "eval complete: $OUT"
find "$OUT" -maxdepth 1 -type f -printf '  %f\n' | sort
