#!/usr/bin/env bash
set -Eeuo pipefail

readonly BUCKET="finetuning-lab-1-037678282394-us-west-2-an"
readonly PREFIX="finetuning"
readonly REGION="us-west-2"
readonly EXPERIMENT="exp-00"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

sync_directory() {
  local source_directory="$1"
  local destination_uri="$2"

  if [[ ! -d "$source_directory" ]]; then
    log "Skipping missing directory: $source_directory"
    return 0
  fi

  log "Uploading $source_directory to $destination_uri"

  aws s3 sync \
    "${source_directory}/" \
    "${destination_uri}/" \
    --region "$REGION" \
    --only-show-errors
}

main() {
  command -v aws >/dev/null 2>&1 || {
    log "ERROR: AWS CLI is not installed."
    exit 1
  }

  log "Starting final artifact synchronization."

  sync_directory \
    "/mnt/workspace/results" \
    "s3://${BUCKET}/${PREFIX}/results/${EXPERIMENT}"

  sync_directory \
    "/mnt/workspace/checkpoints" \
    "s3://${BUCKET}/${PREFIX}/checkpoints/${EXPERIMENT}"

  sync_directory \
    "/mnt/workspace/logs" \
    "s3://${BUCKET}/${PREFIX}/logs/${EXPERIMENT}"

  log "Artifact synchronization completed successfully."

  sync

  log "Stopping the EC2 instance."
  sudo shutdown -h now
}

main "$@"
