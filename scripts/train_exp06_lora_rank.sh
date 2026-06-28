#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP06_RESULTS_ROOT="${EXP06_RESULTS_ROOT:-/workspace/results/exp-06}"
EXP06_LOGS_ROOT="${EXP06_LOGS_ROOT:-/workspace/logs/exp-06}"
EXP06_CHECKPOINT_ROOT="${EXP06_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-06}"
EXP06_CACHE_DIR="${EXP06_CACHE_DIR:-/root/.cache/huggingface}"
EXP06_LOCAL_BATCH_SIZE="${EXP06_LOCAL_BATCH_SIZE:-4}"
EXP06_GLOBAL_BATCH_SIZE="${EXP06_GLOBAL_BATCH_SIZE:-4}"
EXP06_GENERATION_BATCH_SIZE="${EXP06_GENERATION_BATCH_SIZE:-16}"
EXP06_BOOTSTRAP_SAMPLES="${EXP06_BOOTSTRAP_SAMPLES:-1000}"
EXP06_REUSE_RANK8="${EXP06_REUSE_RANK8:-1}"
EXP06_DRY_RUN="${EXP06_DRY_RUN:-0}"
EXP06_VALIDATE_ONLY="${EXP06_VALIDATE_ONLY:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp06_lora_rank.py
  --results-root "$EXP06_RESULTS_ROOT"
  --logs-root "$EXP06_LOGS_ROOT"
  --checkpoint-root "$EXP06_CHECKPOINT_ROOT"
  --cache-dir "$EXP06_CACHE_DIR"
  --local-batch-size "$EXP06_LOCAL_BATCH_SIZE"
  --global-batch-size "$EXP06_GLOBAL_BATCH_SIZE"
  --generation-batch-size "$EXP06_GENERATION_BATCH_SIZE"
  --bootstrap-samples "$EXP06_BOOTSTRAP_SAMPLES"
)

if [[ "$EXP06_REUSE_RANK8" == "1" ]]; then
  cmd+=(--reuse-rank8)
else
  cmd+=(--no-reuse-rank8)
fi

if [[ "$EXP06_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP06_VALIDATE_ONLY" == "1" ]]; then
  cmd+=(--validate-only)
fi

printf '[train-exp06-lora-rank] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
