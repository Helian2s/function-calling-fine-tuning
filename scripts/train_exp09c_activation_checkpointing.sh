#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP09C_RESULTS_ROOT="${EXP09C_RESULTS_ROOT:-/workspace/results/exp-09c}"
EXP09C_LOGS_ROOT="${EXP09C_LOGS_ROOT:-/workspace/logs/exp-09c}"
EXP09C_CHECKPOINT_ROOT="${EXP09C_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-09c}"
EXP09C_CACHE_DIR="${EXP09C_CACHE_DIR:-/root/.cache/huggingface}"
EXP09C_MAX_STEPS="${EXP09C_MAX_STEPS:-300}"
EXP09C_RUN_SECONDARY_MICROBATCH="${EXP09C_RUN_SECONDARY_MICROBATCH:-0}"
EXP09C_DISABLE_MEMORY_TRACE="${EXP09C_DISABLE_MEMORY_TRACE:-0}"
EXP09C_DRY_RUN="${EXP09C_DRY_RUN:-0}"
EXP09C_VALIDATE_ONLY="${EXP09C_VALIDATE_ONLY:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp09c_activation_checkpointing.py
  --results-root "$EXP09C_RESULTS_ROOT"
  --logs-root "$EXP09C_LOGS_ROOT"
  --checkpoint-root "$EXP09C_CHECKPOINT_ROOT"
  --cache-dir "$EXP09C_CACHE_DIR"
  --max-steps "$EXP09C_MAX_STEPS"
)

if [[ "$EXP09C_RUN_SECONDARY_MICROBATCH" == "1" ]]; then
  cmd+=(--run-secondary-microbatch)
fi

if [[ "$EXP09C_DISABLE_MEMORY_TRACE" == "1" ]]; then
  cmd+=(--disable-memory-trace)
fi

if [[ "$EXP09C_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP09C_VALIDATE_ONLY" == "1" ]]; then
  cmd+=(--validate-only)
fi

printf '[train-exp09c-activation-checkpointing] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
