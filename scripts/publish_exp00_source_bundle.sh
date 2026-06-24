#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

PROFILE="${AWS_PROFILE:-finetuning-local}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/dist/exp00/exp00-release-manifest.json}"
DRY_RUN=1
UPDATE_ALIAS=0

log() {
  printf '[publish-exp00-source] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/publish_exp00_source_bundle.sh [--dry-run|--execute] [--update-alias]

Uploads the revisioned source archive, release manifest, nonempty bootstrap,
and shutdown script to the Experiment 0 S3 source-bundles prefix.
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
    --update-alias)
      UPDATE_ALIAS=1
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

cd "$REPO_ROOT"

[[ -z "$(git status --short)" ]] ||
  die "Working tree is not clean. Refusing publication."

[[ -s "$MANIFEST_PATH" ]] ||
  die "Release manifest is missing or empty: $MANIFEST_PATH"

command -v jq >/dev/null 2>&1 || die "jq is required"

SOURCE_ARCHIVE="$(jq -r '.source_archive' "$MANIFEST_PATH")"
SOURCE_ARCHIVE_SHA256="$(jq -r '.source_archive_sha256' "$MANIFEST_PATH")"
BOOTSTRAP_SHA256="$(jq -r '.bootstrap_sha256' "$MANIFEST_PATH")"
SHUTDOWN_SHA256="$(jq -r '.shutdown_script_sha256' "$MANIFEST_PATH")"

SOURCE_ARCHIVE_PATH="${REPO_ROOT}/dist/exp00/${SOURCE_ARCHIVE}"
BOOTSTRAP_PATH="${REPO_ROOT}/infrastructure/aws/bootstrap/bootstrap_instance.sh"
SHUTDOWN_PATH="${REPO_ROOT}/infrastructure/aws/bootstrap/shutdown_and_sync.sh"

for path in "$SOURCE_ARCHIVE_PATH" "$BOOTSTRAP_PATH" "$SHUTDOWN_PATH" "$MANIFEST_PATH"; do
  [[ -s "$path" ]] || die "Required publication file is missing or empty: $path"
done

python3 scripts/verify_source_archive.py \
  "$SOURCE_ARCHIVE_PATH" \
  --sha256 "$SOURCE_ARCHIVE_SHA256" \
  --expect Makefile \
  --expect scripts/train_smoke.sh

[[ "$(sha256sum "$BOOTSTRAP_PATH" | awk '{print $1}')" == "$BOOTSTRAP_SHA256" ]] ||
  die "Bootstrap checksum does not match release manifest"
[[ "$(sha256sum "$SHUTDOWN_PATH" | awk '{print $1}')" == "$SHUTDOWN_SHA256" ]] ||
  die "Shutdown script checksum does not match release manifest"

BASE_URI="s3://${FT_S3_BUCKET}/${FT_S3_PREFIX}/source-bundles"

upload_file() {
  local source_path="$1"
  local destination_uri="$2"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run upload: ${source_path} -> ${destination_uri}"
    return
  fi

  aws s3 cp \
    "$source_path" \
    "$destination_uri" \
    --profile "$PROFILE" \
    --region "$FT_AWS_REGION" \
    --only-show-errors

  local content_length
  local object_key
  object_key="${destination_uri#s3://}"
  object_key="${object_key#"${FT_S3_BUCKET}"/}"
  content_length="$(
    aws s3api head-object \
      --bucket "$FT_S3_BUCKET" \
      --key "$object_key" \
      --profile "$PROFILE" \
      --region "$FT_AWS_REGION" \
      --query ContentLength \
      --output text
  )"
  [[ "$content_length" != "0" ]] ||
    die "Uploaded zero-byte object: $destination_uri"
  log "Uploaded ${destination_uri} (${content_length} bytes)"
}

upload_file "$SOURCE_ARCHIVE_PATH" "${BASE_URI}/${SOURCE_ARCHIVE}"
upload_file "$BOOTSTRAP_PATH" "${BASE_URI}/bootstrap_instance.sh"
upload_file "$SHUTDOWN_PATH" "${BASE_URI}/shutdown_and_sync.sh"
upload_file "$MANIFEST_PATH" "${BASE_URI}/exp00-release-manifest.json"

if [[ "$UPDATE_ALIAS" == "1" ]]; then
  upload_file "$SOURCE_ARCHIVE_PATH" "${BASE_URI}/exp00-source.tar.gz"
else
  log "Convenience alias exp00-source.tar.gz not updated."
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run complete. Re-run with --execute to publish."
fi
