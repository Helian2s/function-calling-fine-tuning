#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${SMOKE_RESULTS_DIR:-results/smoke-run}"
DESTINATION_URI="${SMOKE_RESULTS_URI:-}"

if [[ -z "$DESTINATION_URI" ]]; then
  printf '[sync-results] SMOKE_RESULTS_URI is not set; skipping upload.\n'
  exit 0
fi

command -v aws >/dev/null 2>&1 || {
  printf '[sync-results] ERROR: aws CLI is required for uploads.\n' >&2
  exit 1
}

[[ -d "$RESULTS_DIR" ]] || {
  printf '[sync-results] ERROR: results directory not found: %s\n' "$RESULTS_DIR" >&2
  exit 1
}

printf '[sync-results] Uploading %s to %s\n' "$RESULTS_DIR" "$DESTINATION_URI"
exec aws s3 sync "$RESULTS_DIR" "$DESTINATION_URI"
