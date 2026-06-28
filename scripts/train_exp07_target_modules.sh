#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP07_RESULTS_ROOT="${EXP07_RESULTS_ROOT:-/workspace/results/exp-07}"
EXP07_LOGS_ROOT="${EXP07_LOGS_ROOT:-/workspace/logs/exp-07}"
EXP07_CHECKPOINT_ROOT="${EXP07_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-07}"
EXP07_CACHE_DIR="${EXP07_CACHE_DIR:-/root/.cache/huggingface}"
EXP07_LOCAL_BATCH_SIZE="${EXP07_LOCAL_BATCH_SIZE:-4}"
EXP07_GLOBAL_BATCH_SIZE="${EXP07_GLOBAL_BATCH_SIZE:-4}"
EXP07_GENERATION_BATCH_SIZE="${EXP07_GENERATION_BATCH_SIZE:-16}"
EXP07_BOOTSTRAP_SAMPLES="${EXP07_BOOTSTRAP_SAMPLES:-1000}"
EXP07_REUSE_ATTENTION="${EXP07_REUSE_ATTENTION:-1}"
EXP07_DRY_RUN="${EXP07_DRY_RUN:-0}"
EXP07_VALIDATE_ONLY="${EXP07_VALIDATE_ONLY:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp07_target_modules.py
  --results-root "$EXP07_RESULTS_ROOT"
  --logs-root "$EXP07_LOGS_ROOT"
  --checkpoint-root "$EXP07_CHECKPOINT_ROOT"
  --cache-dir "$EXP07_CACHE_DIR"
  --local-batch-size "$EXP07_LOCAL_BATCH_SIZE"
  --global-batch-size "$EXP07_GLOBAL_BATCH_SIZE"
  --generation-batch-size "$EXP07_GENERATION_BATCH_SIZE"
  --bootstrap-samples "$EXP07_BOOTSTRAP_SAMPLES"
)

if [[ "$EXP07_REUSE_ATTENTION" == "1" ]]; then
  cmd+=(--reuse-attention)
else
  cmd+=(--no-reuse-attention)
fi

if [[ "$EXP07_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP07_VALIDATE_ONLY" == "1" ]]; then
  cmd+=(--validate-only)
fi

printf '[train-exp07-target-modules] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
