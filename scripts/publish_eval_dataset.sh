#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

DRY_RUN=1
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/eval/stratified_1000}"
S3_DATASET_PREFIX="${S3_DATASET_PREFIX:-${FT_S3_PREFIX}/data/eval/stratified_1000}"

log() {
  printf '[publish-eval-dataset] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/publish_eval_dataset.sh [--dry-run|--execute]
                                       [--dataset-dir PATH]
                                       [--s3-prefix PREFIX]

Publishes a prepared evaluation dataset, such as data/eval/stratified_1000,
to S3 without deleting older objects. The script verifies local checksums
before upload and lists uploaded objects after upload.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --execute)
      DRY_RUN=0
      shift
      ;;
    --dataset-dir)
      [[ $# -ge 2 ]] || die "--dataset-dir requires a value"
      DATASET_DIR="$2"
      shift 2
      ;;
    --s3-prefix)
      [[ $# -ge 2 ]] || die "--s3-prefix requires a value"
      S3_DATASET_PREFIX="$2"
      shift 2
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

required_paths=(
  "${DATASET_DIR}/raw/test.jsonl"
  "${DATASET_DIR}/normalized/test.jsonl"
  "${DATASET_DIR}/manifests/summary.json"
  "${DATASET_DIR}/manifests/selection.json"
  "${DATASET_DIR}/manifests/normalization_report.json"
  "${DATASET_DIR}/checksums.sha256"
)

for path in "${required_paths[@]}"; do
  [[ -s "$path" ]] || die "Required dataset artifact is missing or empty: $path"
done

(
  cd "$DATASET_DIR"
  sha256sum -c checksums.sha256 >/dev/null
)

S3_URI="s3://${FT_S3_BUCKET}/${S3_DATASET_PREFIX}"

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run. Planned dataset upload:"
  log "${DATASET_DIR}/ -> ${S3_URI}/"
  exit 0
fi

command -v aws >/dev/null 2>&1 || die "aws CLI is required for uploads"

aws s3 sync \
  "${DATASET_DIR}/" \
  "${S3_URI}/" \
  --region "$FT_AWS_REGION" \
  --only-show-errors

aws s3 ls \
  "${S3_URI}/" \
  --region "$FT_AWS_REGION" \
  --recursive \
  --summarize

log "Uploaded dataset to ${S3_URI}/"
