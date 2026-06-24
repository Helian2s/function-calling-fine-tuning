#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT_LOCAL}/configs/common/exp00.env"

if [[ "${EUID}" -ne 0 ]]; then
  printf '[bootstrap-instance] Run this script with sudo.\n' >&2
  exit 1
fi

DEFAULT_USER="${SUDO_USER:-ubuntu}"
OWNER_GROUP="$(id -gn "${DEFAULT_USER}")"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${HOST_WORKSPACE_ROOT}}"
REPO_ROOT="${REPO_ROOT:-${HOST_PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${HOST_DATA_ROOT}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOST_CHECKPOINT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOST_RESULTS_ROOT}}"
LOGS_ROOT="${LOGS_ROOT:-${HOST_LOGS_ROOT}}"
RUN_INFO_ROOT="${RUN_INFO_ROOT:-${HOST_RUN_INFO_ROOT}}"

log() {
  printf '[bootstrap-instance] %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf '[bootstrap-instance] ERROR: required command not found: %s\n' "$1" >&2
    exit 1
  }
}

apt-get update
apt-get install -y \
  awscli \
  ca-certificates \
  curl \
  docker.io \
  git \
  jq

systemctl enable --now docker
usermod -aG docker "${DEFAULT_USER}" || true

require_command docker
require_command git
require_command aws
require_command nvidia-smi

mkdir -p \
  "${WORKSPACE_ROOT}" \
  "${DATA_ROOT}" \
  "${CHECKPOINT_ROOT}" \
  "${RESULTS_ROOT}" \
  "${LOGS_ROOT}" \
  "${RUN_INFO_ROOT}"
chown -R "${DEFAULT_USER}:${OWNER_GROUP}" "${WORKSPACE_ROOT}"

log "Verifying host GPU visibility"
nvidia-smi

log "Verifying Docker GPU visibility"
docker run --rm --gpus all "${CUDA_TEST_IMAGE}" nvidia-smi

if [[ -n "${NGC_API_KEY:-}" ]]; then
  log "Logging in to nvcr.io with NGC_API_KEY"
  printf '%s' "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
else
  log "NGC_API_KEY not set; skipping nvcr.io login"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  log "HF_TOKEN is set; pass it into the training container for model and dataset access"
else
  log "HF_TOKEN not set; Hugging Face access must be configured before training"
fi

cat <<EOF
[bootstrap-instance] Completed successfully.
[bootstrap-instance] Recommended next steps:
  1. git clone <this-repo> "${REPO_ROOT}"
  2. docker pull ${AUTOMODEL_IMAGE}
  3. Run the container with persistent mounts for:
     - ${REPO_ROOT} -> ${CONTAINER_PROJECT_ROOT}
     - ${DATA_ROOT} -> ${CONTAINER_DATA_ROOT}
     - ${CHECKPOINT_ROOT} -> ${CONTAINER_CHECKPOINT_ROOT}
     - ${RESULTS_ROOT} -> ${CONTAINER_RESULTS_ROOT}
     - ${LOGS_ROOT} -> ${CONTAINER_LOGS_ROOT}
     - ${RUN_INFO_ROOT} -> ${CONTAINER_RUN_INFO_ROOT}
EOF
