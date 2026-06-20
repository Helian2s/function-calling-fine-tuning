#!/usr/bin/env bash
set -euo pipefail

AUTOMODEL_BIN="${AUTOMODEL_BIN:-automodel}"
SMOKE_CONFIG_PATH="${SMOKE_CONFIG_PATH:-configs/exp00_smoke/smoke_qlora.yaml}"

command -v "$AUTOMODEL_BIN" >/dev/null 2>&1 || {
  printf '[train-smoke] ERROR: %s not found in PATH\n' "$AUTOMODEL_BIN" >&2
  exit 1
}

[[ -f "$SMOKE_CONFIG_PATH" ]] || {
  printf '[train-smoke] ERROR: missing config %s\n' "$SMOKE_CONFIG_PATH" >&2
  exit 1
}

printf '[train-smoke] Running %s %s\n' "$AUTOMODEL_BIN" "$SMOKE_CONFIG_PATH"
exec "$AUTOMODEL_BIN" "$SMOKE_CONFIG_PATH"
