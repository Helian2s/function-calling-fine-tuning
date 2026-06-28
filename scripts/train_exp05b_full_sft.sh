#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP05B_CONFIG_PATH="${EXP05B_CONFIG_PATH:-configs/exp05b_full_sft/full_sft_10k.yaml}"
EXP05B_RESULTS_ROOT="${EXP05B_RESULTS_ROOT:-/workspace/results/exp-05b}"
EXP05B_LOGS_ROOT="${EXP05B_LOGS_ROOT:-/workspace/logs/exp-05b}"
EXP05B_CHECKPOINT_ROOT="${EXP05B_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-05b/full-parameter-sft-bf16-10k-epoch1}"
EXP05B_CACHE_DIR="${EXP05B_CACHE_DIR:-/root/.cache/huggingface}"
EXP05B_CHECKPOINT_INTERVAL="${EXP05B_CHECKPOINT_INTERVAL:-834}"
EXP05B_VALIDATION_INTERVAL="${EXP05B_VALIDATION_INTERVAL:-834}"
EXP05B_TOOL_RECORDS="${EXP05B_TOOL_RECORDS:-1000}"
EXP05B_NO_TOOL_RECORDS="${EXP05B_NO_TOOL_RECORDS:-100}"
EXP05B_RELOAD_BATCH_SIZE="${EXP05B_RELOAD_BATCH_SIZE:-1}"
EXP05B_DRY_RUN="${EXP05B_DRY_RUN:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp05b_full_sft.py
  --config "$EXP05B_CONFIG_PATH"
  --results-root "$EXP05B_RESULTS_ROOT"
  --logs-root "$EXP05B_LOGS_ROOT"
  --checkpoint-root "$EXP05B_CHECKPOINT_ROOT"
  --cache-dir "$EXP05B_CACHE_DIR"
  --checkpoint-interval "$EXP05B_CHECKPOINT_INTERVAL"
  --validation-interval "$EXP05B_VALIDATION_INTERVAL"
  --tool-records "$EXP05B_TOOL_RECORDS"
  --no-tool-records "$EXP05B_NO_TOOL_RECORDS"
  --reload-batch-size "$EXP05B_RELOAD_BATCH_SIZE"
)

if [[ "$EXP05B_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

exec "${cmd[@]}"
