#!/usr/bin/env bash
# Smoke-test a running jev-engine instance (native backend). python3 is used
# only for JSON assertions in this development script.
set -euo pipefail

BASE="${JEV_SMOKE_URL:-${JEV_URL:-http://127.0.0.1:8501}}"
BASE="${BASE%/}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required for JSON assertions" >&2
  exit 1
fi

get() {
  local path="$1" out="$2"
  local status
  status="$(curl -sS --connect-timeout 5 --max-time 30 -o "$out" -w '%{http_code}' "${BASE}${path}")"
  if [[ ! "$status" =~ ^2 ]]; then
    echo "FAIL ${path}: HTTP ${status}" >&2
    cat "$out" >&2
    exit 1
  fi
  echo "ok   GET ${path} (${status})"
}

get /health "$TMP/health.json"
python3 - "$TMP/health.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("status") == "ready", data
assert data.get("model") == "autotrust/JEV-27B-VL", data
PY

get /v1/models "$TMP/models.json"
python3 - "$TMP/models.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
models = data.get("data")
assert isinstance(models, list) and models, data
ids = {model.get("id") for model in models}
assert "autotrust/JEV-27B-VL" in ids, ids
PY

get /v1/decide/info "$TMP/info.json"
python3 - "$TMP/info.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("protocol") == "jev27-bare-v1", data
assert data.get("max_options", 0) >= 2, data
assert data.get("native_choice_options") == 16, data
PY

body='{"kind":"noul","state":"The user requested a refund.","question":"Is this abusive?"}'
status="$(curl -sS --connect-timeout 5 --max-time 600 \
  -H 'content-type: application/json' \
  --data-binary "$body" \
  -o "$TMP/decide.json" -w '%{http_code}' \
  "${BASE}/v1/decide")"
if [[ ! "$status" =~ ^2 ]]; then
  echo "FAIL POST /v1/decide: HTTP ${status}" >&2
  cat "$TMP/decide.json" >&2
  exit 1
fi
python3 - "$TMP/decide.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
probabilities = data.get("probabilities")
assert isinstance(probabilities, list) and len(probabilities) == 2, data
assert abs(sum(probabilities) - 1.0) < 1e-6, probabilities
assert data.get("choice") in ("false", "true"), data
assert data.get("protocol") == "jev27-bare-v1", data
assert isinstance(data.get("usage"), dict), data
assert isinstance(data.get("num_model_requests"), int), data
PY
echo "ok   POST /v1/decide (${status})"

echo "smoke passed: ${BASE}"
