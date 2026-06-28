#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP05A_CONFIG_PATH="${EXP05A_CONFIG_PATH:-configs/exp05a_full_sft/full_sft_pilot.yaml}"
EXP05A_RESULTS_ROOT="${EXP05A_RESULTS_ROOT:-/workspace/results/exp-05a}"
EXP05A_LOGS_ROOT="${EXP05A_LOGS_ROOT:-/workspace/logs/exp-05a}"
EXP05A_CHECKPOINT_ROOT="${EXP05A_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-05a/full-parameter-sft-bf16-pilot}"
EXP05A_CACHE_DIR="${EXP05A_CACHE_DIR:-/root/.cache/huggingface}"
EXP05A_PILOT_STEPS="${EXP05A_PILOT_STEPS:-100}"
EXP05A_VALIDATION_RECORDS="${EXP05A_VALIDATION_RECORDS:-100}"
EXP05A_NO_TOOL_RECORDS="${EXP05A_NO_TOOL_RECORDS:-100}"
EXP05A_RELOAD_BATCH_SIZE="${EXP05A_RELOAD_BATCH_SIZE:-1}"
EXP05A_DRY_RUN="${EXP05A_DRY_RUN:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp05a_full_sft.py
  --config "$EXP05A_CONFIG_PATH"
  --results-root "$EXP05A_RESULTS_ROOT"
  --logs-root "$EXP05A_LOGS_ROOT"
  --checkpoint-root "$EXP05A_CHECKPOINT_ROOT"
  --cache-dir "$EXP05A_CACHE_DIR"
  --pilot-steps "$EXP05A_PILOT_STEPS"
  --validation-records "$EXP05A_VALIDATION_RECORDS"
  --no-tool-records "$EXP05A_NO_TOOL_RECORDS"
  --reload-batch-size "$EXP05A_RELOAD_BATCH_SIZE"
)

if [[ "$EXP05A_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

exec "${cmd[@]}"
