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
Usage: scripts/sync_results.sh [--dry-run] [--verify-only]

Synchronizes Experiment 0 results, checkpoints, logs, and run-info to S3.
This script never shuts down the instance.
EOF
}

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

RESULTS_DIR="${RESULTS_DIR:-${HOST_RESULTS_ROOT}/${FT_EXPERIMENT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOST_CHECKPOINT_ROOT}/${FT_EXPERIMENT}}"
LOGS_DIR="${LOGS_DIR:-${HOST_LOGS_ROOT}/${FT_EXPERIMENT}}"
RUN_INFO_DIR="${RUN_INFO_DIR:-${HOST_RUN_INFO_ROOT}}"

RESULTS_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/results/${FT_EXPERIMENT}"
CHECKPOINT_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/checkpoints/${FT_EXPERIMENT}"
LOGS_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/logs/${FT_EXPERIMENT}"
RUN_INFO_URI="${LOGS_URI}/run-info"

required_paths=(
  "${RESULTS_DIR}/scores.json"
  "${RESULTS_DIR}/predictions.jsonl"
  "${RESULTS_DIR}/run_metadata.json"
  "${CHECKPOINT_DIR}/smoke-qlora"
  "${LOGS_DIR}/training.log"
  "${RUN_INFO_DIR}/bootstrap.env"
)

for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || die "Required artifact is missing: $path"
done

if [[ "$VERIFY_ONLY" == "1" ]]; then
  log "Required artifact verification passed."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run. Planned uploads:"
  log "${RESULTS_DIR}/ -> ${RESULTS_URI}/"
  log "${CHECKPOINT_DIR}/ -> ${CHECKPOINT_URI}/"
  log "${LOGS_DIR}/ -> ${LOGS_URI}/"
  log "${RUN_INFO_DIR}/ -> ${RUN_INFO_URI}/"
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
sync_dir "$CHECKPOINT_DIR" "$CHECKPOINT_URI"
sync_dir "$LOGS_DIR" "$LOGS_URI"
sync_dir "$RUN_INFO_DIR" "$RUN_INFO_URI"
