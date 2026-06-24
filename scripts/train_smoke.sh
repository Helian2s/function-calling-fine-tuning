#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

AUTOMODEL_BIN="${AUTOMODEL_BIN:-automodel}"
TRAINING_LOG_PATH="${TRAINING_LOG_PATH:-/workspace/logs/exp-00/training.log}"
TRAIN_SMOKE_DRY_RUN="${TRAIN_SMOKE_DRY_RUN:-${SMOKE_DRY_RUN:-0}}"
RUN_INFO_DIR="${RUN_INFO_DIR:-/workspace/run-info}"
RESOLVED_CONFIG_PATH="${RUN_INFO_DIR}/resolved_config.yaml"
TRAIN_COMMAND_PATH="${RUN_INFO_DIR}/train_command.json"

log() {
  printf '[train-smoke] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

[[ -f "$SMOKE_CONFIG_PATH" ]] || {
  die "missing config $SMOKE_CONFIG_PATH"
}

python3 scripts/validate_smoke_config.py "$SMOKE_CONFIG_PATH"

cmd=("$AUTOMODEL_BIN" finetune llm -c "$SMOKE_CONFIG_PATH")

log "Command: ${cmd[*]}"
log "Log: $TRAINING_LOG_PATH"

if [[ "$TRAIN_SMOKE_DRY_RUN" == "1" ]]; then
  log "Dry run requested; not writing metadata or executing training."
  exit 0
fi

mkdir -p "$(dirname "$TRAINING_LOG_PATH")" "$RUN_INFO_DIR"
cp "$SMOKE_CONFIG_PATH" "$RESOLVED_CONFIG_PATH"

python3 - "$TRAIN_COMMAND_PATH" "$SMOKE_CONFIG_PATH" "$TRAINING_LOG_PATH" <<'PY'
from __future__ import annotations

import json
import sys

path, config_path, log_path = sys.argv[1:]
payload = {
    "command": ["automodel", "finetune", "llm", "-c", config_path],
    "config_path": config_path,
    "training_log_path": log_path,
}
with open(path, "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, sort_keys=True)
    file.write("\n")
PY

command -v "$AUTOMODEL_BIN" >/dev/null 2>&1 || {
  die "$AUTOMODEL_BIN not found in PATH"
}

set +e
"${cmd[@]}" 2>&1 | tee -a "$TRAINING_LOG_PATH"
status="${PIPESTATUS[0]}"
set -e

exit "$status"
