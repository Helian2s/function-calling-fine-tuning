#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP09_RESULTS_ROOT="${EXP09_RESULTS_ROOT:-/workspace/results/exp-09a}"
EXP09_LOGS_ROOT="${EXP09_LOGS_ROOT:-/workspace/logs/exp-09a}"
EXP09_CHECKPOINT_ROOT="${EXP09_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-09a}"
EXP09_CACHE_DIR="${EXP09_CACHE_DIR:-/root/.cache/huggingface}"
EXP09_LOCAL_BATCH_SIZE="${EXP09_LOCAL_BATCH_SIZE:-4}"
EXP09_GLOBAL_BATCH_SIZE="${EXP09_GLOBAL_BATCH_SIZE:-4}"
EXP09_GENERATION_BATCH_SIZE="${EXP09_GENERATION_BATCH_SIZE:-16}"
EXP09_BOOTSTRAP_SAMPLES="${EXP09_BOOTSTRAP_SAMPLES:-1000}"
EXP09_MAX_STEPS="${EXP09_MAX_STEPS:-300}"
EXP09_EVAL_VALIDATION_RECORDS="${EXP09_EVAL_VALIDATION_RECORDS:-1000}"
EXP09_DRY_RUN="${EXP09_DRY_RUN:-0}"
EXP09_VALIDATE_ONLY="${EXP09_VALIDATE_ONLY:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp09_loss_masking.py
  --results-root "$EXP09_RESULTS_ROOT"
  --logs-root "$EXP09_LOGS_ROOT"
  --checkpoint-root "$EXP09_CHECKPOINT_ROOT"
  --cache-dir "$EXP09_CACHE_DIR"
  --local-batch-size "$EXP09_LOCAL_BATCH_SIZE"
  --global-batch-size "$EXP09_GLOBAL_BATCH_SIZE"
  --generation-batch-size "$EXP09_GENERATION_BATCH_SIZE"
  --bootstrap-samples "$EXP09_BOOTSTRAP_SAMPLES"
  --max-steps "$EXP09_MAX_STEPS"
  --eval-validation-records "$EXP09_EVAL_VALIDATION_RECORDS"
)

if [[ "$EXP09_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP09_VALIDATE_ONLY" == "1" ]]; then
  cmd+=(--validate-only)
fi

printf '[train-exp09-loss-masking] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
