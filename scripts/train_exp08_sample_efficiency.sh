#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP08_RESULTS_ROOT="${EXP08_RESULTS_ROOT:-/workspace/results/exp-08}"
EXP08_LOGS_ROOT="${EXP08_LOGS_ROOT:-/workspace/logs/exp-08}"
EXP08_CHECKPOINT_ROOT="${EXP08_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-08}"
EXP08_CACHE_DIR="${EXP08_CACHE_DIR:-/root/.cache/huggingface}"
EXP08_LOCAL_BATCH_SIZE="${EXP08_LOCAL_BATCH_SIZE:-4}"
EXP08_GLOBAL_BATCH_SIZE="${EXP08_GLOBAL_BATCH_SIZE:-4}"
EXP08_GENERATION_BATCH_SIZE="${EXP08_GENERATION_BATCH_SIZE:-16}"
EXP08_BOOTSTRAP_SAMPLES="${EXP08_BOOTSTRAP_SAMPLES:-1000}"
EXP08_REUSE_TRAIN10K="${EXP08_REUSE_TRAIN10K:-1}"
EXP08_DRY_RUN="${EXP08_DRY_RUN:-0}"
EXP08_VALIDATE_ONLY="${EXP08_VALIDATE_ONLY:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp08_sample_efficiency.py
  --results-root "$EXP08_RESULTS_ROOT"
  --logs-root "$EXP08_LOGS_ROOT"
  --checkpoint-root "$EXP08_CHECKPOINT_ROOT"
  --cache-dir "$EXP08_CACHE_DIR"
  --local-batch-size "$EXP08_LOCAL_BATCH_SIZE"
  --global-batch-size "$EXP08_GLOBAL_BATCH_SIZE"
  --generation-batch-size "$EXP08_GENERATION_BATCH_SIZE"
  --bootstrap-samples "$EXP08_BOOTSTRAP_SAMPLES"
)

if [[ -n "${EXP08_HOURLY_COST_USD:-}" ]]; then
  cmd+=(--hourly-cost-usd "$EXP08_HOURLY_COST_USD")
fi

if [[ "$EXP08_REUSE_TRAIN10K" == "1" ]]; then
  cmd+=(--reuse-train10k)
else
  cmd+=(--no-reuse-train10k)
fi

if [[ "$EXP08_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP08_VALIDATE_ONLY" == "1" ]]; then
  cmd+=(--validate-only)
fi

printf '[train-exp08-sample-efficiency] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
