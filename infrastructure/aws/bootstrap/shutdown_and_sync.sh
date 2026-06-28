#!/usr/bin/env bash
set -Eeuo pipefail

readonly BUCKET="${FT_S3_BUCKET:-finetuning-lab-1-037678282394-us-west-2-an}"
readonly PREFIX="${FT_S3_PREFIX:-finetuning}"
readonly REGION="${FT_AWS_REGION:-us-west-2}"
readonly EXPERIMENT="${FT_EXPERIMENT:-exp-00}"
readonly DRY_RUN_DEFAULT="${FT_SYNC_DRY_RUN:-0}"
readonly STAGE_DEFAULT="${FT_SYNC_STAGE:-final}"
readonly INCLUDE_FINAL_ADAPTER_DEFAULT="${FT_SYNC_INCLUDE_FINAL_ADAPTER:-auto}"

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
                                                        [--stage baseline|baseline-1000|final]
                                                        [--include-final-adapter|--no-final-adapter]

Uploads required Experiment 0 artifacts to S3 and then calls shutdown -h now.
Results, logs, run-info, and final adapters are synced. Base model caches,
Docker/NGC caches, and intermediate training checkpoints are intentionally not
uploaded.
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
  local stage="$1"
  local sync_final_adapter="$2"
  local required_paths=()
  local path

  if [[ "$stage" == "baseline" ]]; then
    required_paths+=(
      "/mnt/workspace/results/${EXPERIMENT}/baseline/predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline/generation_metadata.json"
      "/mnt/workspace/results/${EXPERIMENT}/baseline/scored_predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline/parse_failures.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline/scores.json"
      "/mnt/workspace/logs/${EXPERIMENT}/baseline.log"
      "/mnt/workspace/run-info/bootstrap.env"
    )
  elif [[ "$stage" == "baseline-1000" ]]; then
    required_paths+=(
      "/mnt/workspace/results/${EXPERIMENT}/baseline-1000/predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline-1000/generation_metadata.json"
      "/mnt/workspace/results/${EXPERIMENT}/baseline-1000/scored_predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline-1000/parse_failures.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/baseline-1000/scores.json"
      "/mnt/workspace/logs/${EXPERIMENT}/baseline-1000.log"
      "/mnt/workspace/run-info/bootstrap.env"
    )
  else
    required_paths+=(
      "/mnt/workspace/results/${EXPERIMENT}/scores.json"
      "/mnt/workspace/results/${EXPERIMENT}/predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/scored_predictions.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/parse_failures.jsonl"
      "/mnt/workspace/results/${EXPERIMENT}/generation_metadata.json"
      "/mnt/workspace/results/${EXPERIMENT}/run_metadata.json"
      "/mnt/workspace/results/${EXPERIMENT}/run_manifest.json"
      "/mnt/workspace/results/${EXPERIMENT}/training_metrics.json"
      "/mnt/workspace/results/${EXPERIMENT}/training_torch_memory.json"
      "/mnt/workspace/results/${EXPERIMENT}/environment_report.json"
      "/mnt/workspace/results/${EXPERIMENT}/nvidia-smi.txt"
      "/mnt/workspace/results/${EXPERIMENT}/package_versions.txt"
      "/mnt/workspace/results/${EXPERIMENT}/resolved_config.yaml"
      "/mnt/workspace/results/${EXPERIMENT}/requested_metrics.json"
      "/mnt/workspace/results/${EXPERIMENT}/case_report.json"
      "/mnt/workspace/results/${EXPERIMENT}/case_report.md"
      "/mnt/workspace/results/${EXPERIMENT}/checksums.sha256"
      "/mnt/workspace/logs/${EXPERIMENT}/training.log"
      "/mnt/workspace/logs/${EXPERIMENT}/evaluation.log"
      "/mnt/workspace/run-info/bootstrap.env"
    )
  fi

  if [[ "$sync_final_adapter" == "1" ]]; then
    required_paths+=("/mnt/workspace/checkpoints/${EXPERIMENT}/smoke-lora")
  fi

  for path in "${required_paths[@]}"; do
    [[ -e "$path" ]] || die "Required artifact is missing: $path"
  done
}

main() {
  local dry_run="$DRY_RUN_DEFAULT"
  local stage="$STAGE_DEFAULT"
  local include_final_adapter="$INCLUDE_FINAL_ADAPTER_DEFAULT"
  local sync_final_adapter=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        dry_run=1
        shift
        ;;
      --stage)
        [[ $# -ge 2 ]] || die "--stage requires a value"
        stage="$2"
        shift 2
        ;;
      --include-final-adapter)
        include_final_adapter=1
        shift
        ;;
      --no-final-adapter)
        include_final_adapter=0
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

  case "$stage" in
    baseline|baseline-1000|final)
      ;;
    *)
      die "Unsupported stage: $stage"
      ;;
  esac

  case "$include_final_adapter" in
    auto|0|1)
      ;;
    *)
      die "Unsupported final-adapter setting: $include_final_adapter"
      ;;
  esac

  if [[ "$include_final_adapter" == "1" ]] ||
    [[ "$include_final_adapter" == "auto" && "$stage" == "final" ]]; then
    sync_final_adapter=1
  fi

  command -v aws >/dev/null 2>&1 || die "AWS CLI is not installed."

  verify_required_artifacts "$stage" "$sync_final_adapter"

  log "Starting artifact synchronization for stage=${stage}."

  sync_directory \
    "/mnt/workspace/results/${EXPERIMENT}" \
    "s3://${BUCKET}/${PREFIX}/results/${EXPERIMENT}" \
    "$dry_run"

  sync_directory \
    "/mnt/workspace/logs/${EXPERIMENT}" \
    "s3://${BUCKET}/${PREFIX}/logs/${EXPERIMENT}" \
    "$dry_run"

  sync_directory \
    "/mnt/workspace/run-info" \
    "s3://${BUCKET}/${PREFIX}/logs/${EXPERIMENT}/run-info" \
    "$dry_run"

  if [[ "$sync_final_adapter" == "1" ]]; then
    sync_directory \
      "/mnt/workspace/checkpoints/${EXPERIMENT}/smoke-lora" \
      "s3://${BUCKET}/${PREFIX}/checkpoints/${EXPERIMENT}/smoke-lora" \
      "$dry_run"
  else
    log "Final adapter upload skipped for stage=${stage}."
  fi

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
