#!/usr/bin/env bash
# Backwards-compatible entry point: the native engine is the only backend.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run-native.sh" "$@"
