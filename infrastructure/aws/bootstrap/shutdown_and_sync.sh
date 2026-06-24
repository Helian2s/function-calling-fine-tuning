#!/usr/bin/env bash
set -Eeuo pipefail

readonly BUCKET="${FT_S3_BUCKET:-finetuning-lab-1-037678282394-us-west-2-an}"
readonly PREFIX="${FT_S3_PREFIX:-finetuning}"
readonly REGION="${FT_AWS_REGION:-us-west-2}"
readonly EXPERIMENT="${FT_EXPERIMENT:-exp-00}"
readonly DRY_RUN_DEFAULT="${FT_SYNC_DRY_RUN:-0}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<'EOF'
Usage: sudo /usr/local/sbin/ft-exp00-shutdown-and-sync [--dry-run]

Uploads required Experiment 0 artifacts to S3 and then calls shutdown -h now.
The EC2 launch template must be configured with shutdown behavior Stop.
EOF
}

sync_directory() {
  local source_directory="$1"
  local destination_uri="$2"
  local dry_run="$3"

  if [[ ! -d "$source_directory" ]]; then
    die "Missing required directory: $source_directory"
  fi

  if [[ "$dry_run" == "1" ]]; then
    log "Dry run upload: ${source_directory}/ -> ${destination_uri}/"
    return 0
  fi

  log "Uploading $source_directory to $destination_uri"

  aws s3 sync \
    "${source_directory}/" \
    "${destination_uri}/" \
    --region "$REGION" \
    --only-show-errors

  aws s3 ls \
    "${destination_uri}/" \
    --region "$REGION" \
    --recursive \
    --summarize
}

verify_required_artifacts() {
  local required_paths=(
    "/mnt/workspace/results/${EXPERIMENT}/scores.json"
    "/mnt/workspace/results/${EXPERIMENT}/predictions.jsonl"
    "/mnt/workspace/results/${EXPERIMENT}/run_metadata.json"
    "/mnt/workspace/checkpoints/${EXPERIMENT}/smoke-qlora"
    "/mnt/workspace/logs/${EXPERIMENT}/training.log"
    "/mnt/workspace/run-info/bootstrap.env"
  )
  local path

  for path in "${required_paths[@]}"; do
    [[ -e "$path" ]] || die "Required artifact is missing: $path"
  done
}

main() {
  local dry_run="$DRY_RUN_DEFAULT"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        dry_run=1
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

  command -v aws >/dev/null 2>&1 || die "AWS CLI is not installed."

  verify_required_artifacts

  log "Starting final artifact synchronization."

  sync_directory \
    "/mnt/workspace/results/${EXPERIMENT}" \
    "s3://${BUCKET}/${PREFIX}/results/${EXPERIMENT}" \
    "$dry_run"

  sync_directory \
    "/mnt/workspace/checkpoints/${EXPERIMENT}" \
    "s3://${BUCKET}/${PREFIX}/checkpoints/${EXPERIMENT}" \
    "$dry_run"

  sync_directory \
    "/mnt/workspace/logs/${EXPERIMENT}" \
    "s3://${BUCKET}/${PREFIX}/logs/${EXPERIMENT}" \
    "$dry_run"

  sync_directory \
    "/mnt/workspace/run-info" \
    "s3://${BUCKET}/${PREFIX}/logs/${EXPERIMENT}/run-info" \
    "$dry_run"

  log "Artifact synchronization completed successfully."

  if [[ "$dry_run" == "1" ]]; then
    log "Dry run requested; not shutting down."
    return 0
  fi

  sync

  log "Stopping the EC2 instance."
  shutdown -h now
}

main "$@"
