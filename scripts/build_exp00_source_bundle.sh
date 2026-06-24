#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist/exp00}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

log() {
  printf '[build-source-bundle] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

should_exclude() {
  local path="$1"

  case "$path" in
    infrastructure/aws/bootstrap/bootstrap_instance.sh|\
    infrastructure/aws/bootstrap/shutdown_and_sync.sh|\
    .env|\
    .envrc|\
    data/smoke/raw/*|\
    data/smoke/normalized/*|\
    release/*|\
    dist/*|\
    checkpoints/*|\
    outputs/*|\
    results/exp-00/*)
      return 0
      ;;
  esac

  return 1
}

main() {
  cd "$REPO_ROOT"

  if [[ "$ALLOW_DIRTY" != "1" ]] && [[ -n "$(git status --short)" ]]; then
    die "Working tree is not clean. Commit C0 changes before building."
  fi

  local revision
  local short_revision
  local archive_name
  local archive_path
  local alias_path
  local file_list
  local source_sha
  local bootstrap_sha
  local shutdown_sha
  local manifest_path

  revision="$(git rev-parse HEAD)"
  short_revision="$(git rev-parse --short HEAD)"
  archive_name="exp00-source-${short_revision}.tar.gz"
  archive_path="${OUTPUT_DIR}/${archive_name}"
  alias_path="${OUTPUT_DIR}/exp00-source.tar.gz"
  manifest_path="${OUTPUT_DIR}/exp00-release-manifest.json"
  file_list="$(mktemp)"

  trap 'rm -f "${file_list:-}"' EXIT

  mkdir -p "$OUTPUT_DIR"

  while IFS= read -r -d '' path; do
    if should_exclude "$path"; then
      continue
    fi
    printf '%s\0' "$path" >> "$file_list"
  done < <(git ls-files -z)

  [[ -s "$file_list" ]] || die "Source file list is empty"

  tar --null --files-from "$file_list" -czf "$archive_path"
  cp "$archive_path" "$alias_path"

  python3 scripts/verify_source_archive.py \
    "$archive_path" \
    --expect Makefile \
    --expect scripts/train_smoke.sh \
    --expect scripts/generate_predictions.py \
    --expect scripts/evaluate.py \
    --forbid infrastructure/aws/bootstrap/bootstrap_instance.sh \
    --forbid infrastructure/aws/bootstrap/shutdown_and_sync.sh

  source_sha="$(sha256_file "$archive_path")"
  bootstrap_sha="$(sha256_file infrastructure/aws/bootstrap/bootstrap_instance.sh)"
  shutdown_sha="$(sha256_file infrastructure/aws/bootstrap/shutdown_and_sync.sh)"

  python3 - "$manifest_path" "$revision" "$archive_name" "$source_sha" \
    "$bootstrap_sha" "$shutdown_sha" "$AUTOMODEL_IMAGE" \
    "$SMOKE_MODEL_NAME" "$SMOKE_MODEL_REVISION" <<'PY'
from __future__ import annotations

import json
import sys

(
    manifest_path,
    revision,
    archive_name,
    source_sha,
    bootstrap_sha,
    shutdown_sha,
    container_tag,
    model_name,
    model_revision,
) = sys.argv[1:]

manifest = {
    "git_revision": revision,
    "source_archive": archive_name,
    "source_archive_sha256": source_sha,
    "bootstrap_sha256": bootstrap_sha,
    "shutdown_script_sha256": shutdown_sha,
    "model_name": model_name,
    "model_revision": model_revision,
    "container_tag": container_tag,
    "dataset_checksum_manifest": (
        "finetuning/data/smoke-v1/checksums.sha256"
    ),
}

with open(manifest_path, "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2, sort_keys=True)
    file.write("\n")
PY

  log "archive=${archive_path}"
  log "archive_alias=${alias_path}"
  log "archive_sha256=${source_sha}"
  log "manifest=${manifest_path}"
  log "bootstrap_sha256=${bootstrap_sha}"
  log "shutdown_script_sha256=${shutdown_sha}"
}

main "$@"
