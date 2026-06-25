#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN=1
UPDATE_ALIAS=1

log() {
  printf '[sync-source-to-s3] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/sync_source_to_s3.sh [--dry-run|--execute] [--no-update-alias]

Builds a source archive from the current clean Git commit and publishes it to
the Experiment 0 S3 source-bundles prefix. Run this before each EC2 experiment
stage so S3 records the exact source revision being executed.
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
    --no-update-alias)
      UPDATE_ALIAS=0
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

if [[ -n "$(git status --short)" ]]; then
  die "Working tree is not clean. Commit changes before publishing source."
fi

"${REPO_ROOT}/scripts/build_exp00_source_bundle.sh"

publish_args=(--dry-run)
if [[ "$DRY_RUN" == "0" ]]; then
  publish_args=(--execute)
fi

if [[ "$UPDATE_ALIAS" == "1" ]]; then
  publish_args+=(--update-alias)
fi

"${REPO_ROOT}/scripts/publish_exp00_source_bundle.sh" "${publish_args[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run complete. Re-run with --execute to publish."
fi
