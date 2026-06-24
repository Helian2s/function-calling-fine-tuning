#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_CHECKPOINT_DIR="${SMOKE_CHECKPOINT_DIR:-$SMOKE_ADAPTER_PATH}"
SMOKE_RESULTS_DIR="${SMOKE_RESULTS_DIR:-/workspace/results/exp-00}"
SMOKE_LOGS_DIR="${SMOKE_LOGS_DIR:-/workspace/logs/exp-00}"
SMOKE_RUN_INFO_DIR="${SMOKE_RUN_INFO_DIR:-/workspace/run-info}"
SMOKE_DRY_RUN="${SMOKE_DRY_RUN:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

SMOKE_PREFLIGHT_CMD="${SMOKE_PREFLIGHT_CMD:-make smoke-preflight}"
SMOKE_BASELINE_CMD="${SMOKE_BASELINE_CMD:-make smoke-baseline}"
SMOKE_TRAIN_CMD="${SMOKE_TRAIN_CMD:-make smoke-train}"
SMOKE_RELOAD_CMD="${SMOKE_RELOAD_CMD:-make smoke-reload-check}"
SMOKE_EVALUATE_CMD="${SMOKE_EVALUATE_CMD:-make smoke-evaluate}"
SMOKE_UPLOAD_CMD="${SMOKE_UPLOAD_CMD:-}"

log() {
  printf '[smoke-run] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

run_step() {
  local step_name="$1"
  local command_string="$2"

  if [[ -z "$command_string" ]]; then
    log "Skipping optional step with no command: $step_name"
    return 0
  fi

  log "Step: $step_name"
  log "Command: $command_string"

  if [[ "$SMOKE_DRY_RUN" == "1" ]]; then
    return 0
  fi

  bash -lc "$command_string"
}

verify_environment() {
  log "Verifying environment"
  require_command "$PYTHON_BIN"
  [[ -f "$SMOKE_CONFIG_PATH" ]] || die "Missing smoke config: $SMOKE_CONFIG_PATH"

  if [[ "$REQUIRE_GPU" == "1" ]]; then
    require_command nvidia-smi
    nvidia-smi
  fi

  mkdir -p "$SMOKE_RESULTS_DIR" "$SMOKE_LOGS_DIR" "$SMOKE_RUN_INFO_DIR"
}

verify_checkpoint_written() {
  if [[ "$SMOKE_DRY_RUN" == "1" ]]; then
    return 0
  fi

  [[ -d "$SMOKE_CHECKPOINT_DIR" ]] || die "Checkpoint directory was not created: $SMOKE_CHECKPOINT_DIR"
}

main() {
  verify_environment
  run_step "preflight" "$SMOKE_PREFLIGHT_CMD"
  run_step "baseline" "$SMOKE_BASELINE_CMD"
  run_step "train" "$SMOKE_TRAIN_CMD"
  verify_checkpoint_written
  log "Checkpoint directory: $SMOKE_CHECKPOINT_DIR"
  run_step "reload-check" "$SMOKE_RELOAD_CMD"
  run_step "full evaluation" "$SMOKE_EVALUATE_CMD"
  run_step "upload results" "$SMOKE_UPLOAD_CMD"
  log "Smoke run completed."
}

main "$@"
