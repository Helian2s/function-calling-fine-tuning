#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

DRY_RUN=0
VERIFY_ONLY=0

log() {
  printf '[sync-results] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/sync_results.sh [--dry-run] [--verify-only] [--stage baseline|baseline-1000|final]
                               [--include-final-adapter|--no-final-adapter]

Synchronizes Experiment 0 results, logs, run-info, and final adapter artifacts
to S3. It intentionally does not upload base model caches, Docker/NGC caches,
or intermediate training checkpoints.
This script never shuts down the instance.
EOF
}

STAGE="${FT_SYNC_STAGE:-final}"
INCLUDE_FINAL_ADAPTER="${FT_SYNC_INCLUDE_FINAL_ADAPTER:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --stage)
      [[ $# -ge 2 ]] || die "--stage requires a value"
      STAGE="$2"
      shift 2
      ;;
    --include-final-adapter)
      INCLUDE_FINAL_ADAPTER=1
      shift
      ;;
    --no-final-adapter)
      INCLUDE_FINAL_ADAPTER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$STAGE" in
  baseline|baseline-1000|final)
    ;;
  *)
    die "Unsupported stage: $STAGE"
    ;;
esac

case "$INCLUDE_FINAL_ADAPTER" in
  auto|0|1)
    ;;
  *)
    die "Unsupported final-adapter setting: $INCLUDE_FINAL_ADAPTER"
    ;;
esac

RESULTS_DIR="${RESULTS_DIR:-${HOST_RESULTS_ROOT}/${FT_EXPERIMENT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOST_CHECKPOINT_ROOT}/${FT_EXPERIMENT}}"
FINAL_ADAPTER_DIR="${FINAL_ADAPTER_DIR:-${CHECKPOINT_DIR}/smoke-lora}"
LOGS_DIR="${LOGS_DIR:-${HOST_LOGS_ROOT}/${FT_EXPERIMENT}}"
RUN_INFO_DIR="${RUN_INFO_DIR:-${HOST_RUN_INFO_ROOT}}"

RESULTS_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/results/${FT_EXPERIMENT}"
CHECKPOINT_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/checkpoints/${FT_EXPERIMENT}"
FINAL_ADAPTER_URI="${CHECKPOINT_URI}/smoke-lora"
LOGS_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/logs/${FT_EXPERIMENT}"
RUN_INFO_URI="${LOGS_URI}/run-info"

required_paths=()

if [[ "$STAGE" == "baseline" ]]; then
  required_paths+=(
    "${RESULTS_DIR}/baseline/predictions.jsonl"
    "${RESULTS_DIR}/baseline/generation_metadata.json"
    "${RESULTS_DIR}/baseline/scored_predictions.jsonl"
    "${RESULTS_DIR}/baseline/parse_failures.jsonl"
    "${RESULTS_DIR}/baseline/scores.json"
    "${LOGS_DIR}/baseline.log"
    "${RUN_INFO_DIR}/bootstrap.env"
  )
elif [[ "$STAGE" == "baseline-1000" ]]; then
  required_paths+=(
    "${RESULTS_DIR}/baseline-1000/predictions.jsonl"
    "${RESULTS_DIR}/baseline-1000/generation_metadata.json"
    "${RESULTS_DIR}/baseline-1000/scored_predictions.jsonl"
    "${RESULTS_DIR}/baseline-1000/parse_failures.jsonl"
    "${RESULTS_DIR}/baseline-1000/scores.json"
    "${LOGS_DIR}/baseline-1000.log"
    "${RUN_INFO_DIR}/bootstrap.env"
  )
else
  required_paths+=(
    "${RESULTS_DIR}/scores.json"
    "${RESULTS_DIR}/predictions.jsonl"
    "${RESULTS_DIR}/scored_predictions.jsonl"
    "${RESULTS_DIR}/parse_failures.jsonl"
    "${RESULTS_DIR}/generation_metadata.json"
    "${RESULTS_DIR}/run_metadata.json"
    "${RESULTS_DIR}/run_manifest.json"
    "${RESULTS_DIR}/training_metrics.json"
    "${RESULTS_DIR}/training_torch_memory.json"
    "${RESULTS_DIR}/environment_report.json"
    "${RESULTS_DIR}/nvidia-smi.txt"
    "${RESULTS_DIR}/package_versions.txt"
    "${RESULTS_DIR}/resolved_config.yaml"
    "${RESULTS_DIR}/requested_metrics.json"
    "${RESULTS_DIR}/case_report.json"
    "${RESULTS_DIR}/case_report.md"
    "${RESULTS_DIR}/checksums.sha256"
    "${LOGS_DIR}/training.log"
    "${LOGS_DIR}/evaluation.log"
    "${RUN_INFO_DIR}/bootstrap.env"
  )
fi

sync_final_adapter=0
if [[ "$INCLUDE_FINAL_ADAPTER" == "1" ]] ||
  [[ "$INCLUDE_FINAL_ADAPTER" == "auto" && "$STAGE" == "final" ]]; then
  sync_final_adapter=1
  required_paths+=("${FINAL_ADAPTER_DIR}")
fi

for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || die "Required artifact is missing: $path"
done

if [[ "$VERIFY_ONLY" == "1" ]]; then
  log "Required artifact verification passed."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run. Planned uploads:"
  log "stage=${STAGE}"
  log "${RESULTS_DIR}/ -> ${RESULTS_URI}/"
  log "${LOGS_DIR}/ -> ${LOGS_URI}/"
  log "${RUN_INFO_DIR}/ -> ${RUN_INFO_URI}/"
  if [[ "$sync_final_adapter" == "1" ]]; then
    log "${FINAL_ADAPTER_DIR}/ -> ${FINAL_ADAPTER_URI}/"
  else
    log "final adapter upload skipped for stage=${STAGE}"
  fi
  exit 0
fi

command -v aws >/dev/null 2>&1 || die "aws CLI is required for uploads"

sync_dir() {
  local source_dir="$1"
  local destination_uri="$2"

  [[ -d "$source_dir" ]] || die "Source directory does not exist: $source_dir"
  log "Uploading ${source_dir}/ to ${destination_uri}/"
  aws s3 sync \
    "${source_dir}/" \
    "${destination_uri}/" \
    --region "$FT_AWS_REGION" \
    --only-show-errors
  aws s3 ls \
    "${destination_uri}/" \
    --region "$FT_AWS_REGION" \
    --recursive \
    --summarize
}

sync_dir "$RESULTS_DIR" "$RESULTS_URI"
sync_dir "$LOGS_DIR" "$LOGS_URI"
sync_dir "$RUN_INFO_DIR" "$RUN_INFO_URI"
if [[ "$sync_final_adapter" == "1" ]]; then
  sync_dir "$FINAL_ADAPTER_DIR" "$FINAL_ADAPTER_URI"
else
  log "Final adapter upload skipped for stage=${STAGE}."
fi
