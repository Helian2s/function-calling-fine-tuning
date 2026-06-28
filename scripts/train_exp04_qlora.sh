#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP04_CONFIG_PATH="${EXP04_CONFIG_PATH:-configs/exp04_lora_vs_qlora/qlora.yaml}"
EXP04_RESULTS_ROOT="${EXP04_RESULTS_ROOT:-/workspace/results/exp-04}"
EXP04_LOGS_ROOT="${EXP04_LOGS_ROOT:-/workspace/logs/exp-04}"
EXP04_CHECKPOINT_ROOT="${EXP04_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-04/reference-nf4-qlora-r8-attention}"
EXP04_CACHE_DIR="${EXP04_CACHE_DIR:-/root/.cache/huggingface}"
EXP04_BATCH_SIZES="${EXP04_BATCH_SIZES:-1,2,4,8}"
EXP04_PROBE_STEPS="${EXP04_PROBE_STEPS:-5}"
EXP04_PILOT_STEPS="${EXP04_PILOT_STEPS:-100}"
EXP04_PROBE_VALIDATION_RECORDS="${EXP04_PROBE_VALIDATION_RECORDS:-32}"
EXP04_PILOT_TRACE_MAX_RESERVED_GB="${EXP04_PILOT_TRACE_MAX_RESERVED_GB:-20}"
EXP04_FULL="${EXP04_FULL:-0}"
EXP04_DRY_RUN="${EXP04_DRY_RUN:-0}"
EXP04_DISABLE_MEMORY_TRACE="${EXP04_DISABLE_MEMORY_TRACE:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp03_reference_lora.py
  --method qlora
  --config "$EXP04_CONFIG_PATH"
  --results-root "$EXP04_RESULTS_ROOT"
  --logs-root "$EXP04_LOGS_ROOT"
  --checkpoint-root "$EXP04_CHECKPOINT_ROOT"
  --cache-dir "$EXP04_CACHE_DIR"
  --batch-sizes "$EXP04_BATCH_SIZES"
  --probe-steps "$EXP04_PROBE_STEPS"
  --pilot-steps "$EXP04_PILOT_STEPS"
  --probe-validation-records "$EXP04_PROBE_VALIDATION_RECORDS"
  --pilot-trace-max-reserved-gb "$EXP04_PILOT_TRACE_MAX_RESERVED_GB"
  --reload-load-in-4bit
)

if [[ "$EXP04_FULL" == "1" ]]; then
  cmd+=(--full)
fi

if [[ "$EXP04_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP04_DISABLE_MEMORY_TRACE" == "1" ]]; then
  cmd+=(--disable-memory-trace)
fi

exec "${cmd[@]}"
