#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CURATOR_DOCKER_IMAGE:-nvcr.io/nvidia/nemo-curator:25.09}"
INPUT_DIR="${1:-data/processed/xlam_curated_v1/curator_input}"
OUTPUT_DIR="${2:-data/processed/xlam_curated_v1/curator_exact}"
INPUT_FILE="${CURATOR_INPUT_FILE:-exact_dedup_input.jsonl}"

if [[ ! -f "${INPUT_DIR}/${INPUT_FILE}" ]]; then
  printf 'Curator input file not found: %s\n' "${INPUT_DIR}/${INPUT_FILE}" >&2
  printf 'Run make xlam-curate first.\n' >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/output" "${OUTPUT_DIR}/log" "${OUTPUT_DIR}/profile"

docker run --rm --gpus all \
  -e CURATOR_INPUT_DIR="/workspace/${INPUT_DIR}" \
  -e CURATOR_OUTPUT_DIR="/workspace/${OUTPUT_DIR}/output" \
  -e CURATOR_LOG_DIR="/workspace/${OUTPUT_DIR}/log" \
  -e CURATOR_PROFILE_DIR="/workspace/${OUTPUT_DIR}/profile" \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    python - <<PY
from __future__ import annotations

from pathlib import Path

from nemo_curator.stages.deduplication.exact.workflow import (
    ExactDeduplicationWorkflow,
)

input_dir = Path("$CURATOR_INPUT_DIR")
output_dir = Path("$CURATOR_OUTPUT_DIR")
output_dir.mkdir(parents=True, exist_ok=True)

workflow = ExactDeduplicationWorkflow(
    input_path=str(input_dir),
    output_path=str(output_dir),
    input_filetype="jsonl",
    input_file_extensions=[".jsonl"],
    input_blocksize="128MiB",
    text_field="text",
    assign_id=False,
    id_field="id",
    perform_removal=False,
)
workflow.run()
PY
  '

printf 'Curator exact-dedup output: %s\n' "$OUTPUT_DIR"
