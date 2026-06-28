#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP03_CONFIG_PATH="${EXP03_CONFIG_PATH:-configs/exp03_reference_lora/lora_r8_attention.yaml}"
EXP03_RESULTS_ROOT="${EXP03_RESULTS_ROOT:-/workspace/results/exp-03}"
EXP03_LOGS_ROOT="${EXP03_LOGS_ROOT:-/workspace/logs/exp-03}"
EXP03_CHECKPOINT_ROOT="${EXP03_CHECKPOINT_ROOT:-/workspace/checkpoints/exp-03/reference-bf16-lora-r8-attention}"
EXP03_BATCH_SIZES="${EXP03_BATCH_SIZES:-1,2,4,8}"
EXP03_PROBE_STEPS="${EXP03_PROBE_STEPS:-5}"
EXP03_PILOT_STEPS="${EXP03_PILOT_STEPS:-100}"
EXP03_DRY_RUN="${EXP03_DRY_RUN:-0}"
EXP03_FULL="${EXP03_FULL:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_exp03_reference_lora.py
  --config "$EXP03_CONFIG_PATH"
  --results-root "$EXP03_RESULTS_ROOT"
  --logs-root "$EXP03_LOGS_ROOT"
  --checkpoint-root "$EXP03_CHECKPOINT_ROOT"
  --batch-sizes "$EXP03_BATCH_SIZES"
  --probe-steps "$EXP03_PROBE_STEPS"
  --pilot-steps "$EXP03_PILOT_STEPS"
)

if [[ "$EXP03_DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$EXP03_FULL" == "1" ]]; then
  cmd+=(--full)
fi

printf '[train-reference-lora] Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
