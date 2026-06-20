#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_CONFIG_PATH="${SMOKE_CONFIG_PATH:-configs/exp00_smoke/smoke_qlora.yaml}"
SMOKE_CHECKPOINT_DIR="${SMOKE_CHECKPOINT_DIR:-/workspace/outputs/smoke-qlora}"
SMOKE_RESULTS_DIR="${SMOKE_RESULTS_DIR:-results/smoke-run}"
SMOKE_DRY_RUN="${SMOKE_DRY_RUN:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SMOKE_TRAIN_CMD="${SMOKE_TRAIN_CMD:-./scripts/train_smoke.sh}"
SMOKE_RELOAD_CMD="${SMOKE_RELOAD_CMD:-}"
SMOKE_GENERATE_CMD="${SMOKE_GENERATE_CMD:-}"
SMOKE_SCORE_CMD="${SMOKE_SCORE_CMD:-}"
SMOKE_UPLOAD_CMD="${SMOKE_UPLOAD_CMD:-./scripts/sync_results.sh}"

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

  [[ -n "$command_string" ]] || die "No command configured for step: $step_name"
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

  mkdir -p "$SMOKE_RESULTS_DIR"
}

verify_checkpoint_written() {
  if [[ "$SMOKE_DRY_RUN" == "1" ]]; then
    return 0
  fi

  [[ -d "$SMOKE_CHECKPOINT_DIR" ]] || die "Checkpoint directory was not created: $SMOKE_CHECKPOINT_DIR"
}

main() {
  verify_environment
  run_step "train" "$SMOKE_TRAIN_CMD"
  verify_checkpoint_written
  log "Step: save"
  log "Checkpoint directory: $SMOKE_CHECKPOINT_DIR"
  run_step "reload" "$SMOKE_RELOAD_CMD"
  run_step "generate" "$SMOKE_GENERATE_CMD"
  run_step "score" "$SMOKE_SCORE_CMD"
  run_step "upload results" "$SMOKE_UPLOAD_CMD"
  log "Smoke run completed."
}

main "$@"
