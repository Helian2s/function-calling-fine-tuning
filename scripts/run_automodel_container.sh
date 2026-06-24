#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

DRY_RUN=0
PULL_IMAGE=0
LOGIN_NGC=0
VERIFY_CUDA=0

log() {
  printf '[run-automodel-container] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/run_automodel_container.sh [options] <command> [args...]

Options:
  --dry-run       Print the sanitized docker command without running it.
  --pull          Pull the pinned AutoModel image and record its digest.
  --login-ngc     Retrieve NGC key from SSM on EC2 before pulling.
  --verify-cuda   Run nvidia-smi inside the verified CUDA test image.

Environment:
  NGC_API_KEY_SSM_PARAMETER  SSM SecureString name for nvcr.io login on EC2.
  HF_TOKEN_SSM_PARAMETER     SSM SecureString name for Hugging Face on EC2.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

is_ec2() {
  local token=""
  token="$(curl -fsS -m 1 -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    'http://169.254.169.254/latest/api/token' 2>/dev/null || true)"

  [[ -n "$token" ]] || return 1

  curl -fsS -m 1 \
    -H "X-aws-ec2-metadata-token: ${token}" \
    'http://169.254.169.254/latest/meta-data/instance-id' \
    >/dev/null 2>&1
}

get_ssm_parameter_on_ec2() {
  local parameter_name="$1"

  [[ -n "$parameter_name" ]] || return 1

  if ! is_ec2; then
    die "Refusing to retrieve SSM secret outside EC2: ${parameter_name}"
  fi

  aws ssm get-parameter \
    --name "$parameter_name" \
    --with-decryption \
    --query 'Parameter.Value' \
    --output text \
    --region "$FT_AWS_REGION"
}

record_image_digest() {
  local image_ref="$1"
  local output_path="${HOST_RUN_INFO_ROOT}/container_image.txt"
  local digest=""

  mkdir -p "$HOST_RUN_INFO_ROOT"
  digest="$(docker image inspect \
    --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
    "$image_ref" 2>/dev/null || true)"

  {
    printf 'image=%s\n' "$image_ref"
    if [[ -n "$digest" ]]; then
      printf 'repo_digest=%s\n' "$digest"
    else
      docker image inspect --format 'image_id={{.Id}}' "$image_ref"
    fi
  } > "$output_path"

  log "Recorded container image metadata: $output_path"
}

pull_automodel_image() {
  local ngc_key=""

  require_command docker

  if [[ "$LOGIN_NGC" == "1" ]]; then
    require_command aws
    ngc_key="$(get_ssm_parameter_on_ec2 "${NGC_API_KEY_SSM_PARAMETER:-}")"
    [[ -n "$ngc_key" ]] || die "NGC SSM parameter returned an empty value"
    printf '%s' "$ngc_key" |
      docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
    unset ngc_key
  fi

  docker pull "$AUTOMODEL_IMAGE"
  record_image_digest "$AUTOMODEL_IMAGE"

  if [[ "$LOGIN_NGC" == "1" ]]; then
    docker logout nvcr.io >/dev/null || true
  fi
}

verify_cuda_image() {
  require_command docker
  docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi
}

prepare_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    return
  fi

  if [[ -n "${HF_TOKEN_SSM_PARAMETER:-}" ]]; then
    HF_TOKEN="$(get_ssm_parameter_on_ec2 "$HF_TOKEN_SSM_PARAMETER")"
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi
}

print_docker_command() {
  printf 'docker run --rm --gpus all --shm-size=16g'
  printf ' -v %q:%q' "$HOST_PROJECT_ROOT" "$CONTAINER_PROJECT_ROOT"
  printf ' -v %q:%q' "$HOST_DATA_ROOT" "$CONTAINER_DATA_ROOT"
  printf ' -v %q:%q' "$HOST_HF_CACHE_ROOT" "/root/.cache/huggingface"
  printf ' -v %q:%q' "$HOST_NGC_CACHE_ROOT" "/root/.cache/ngc"
  printf ' -v %q:%q' "$HOST_CHECKPOINT_ROOT" "$CONTAINER_CHECKPOINT_ROOT"
  printf ' -v %q:%q' "$HOST_RESULTS_ROOT" "$CONTAINER_RESULTS_ROOT"
  printf ' -v %q:%q' "$HOST_LOGS_ROOT" "$CONTAINER_LOGS_ROOT"
  printf ' -v %q:%q' "$HOST_RUN_INFO_ROOT" "$CONTAINER_RUN_INFO_ROOT"
  printf ' -w %q' "$CONTAINER_PROJECT_ROOT"
  printf ' --env HF_HOME=/root/.cache/huggingface'
  printf ' --env TRANSFORMERS_CACHE=/root/.cache/huggingface'
  printf ' --env HF_TOKEN --env HUGGING_FACE_HUB_TOKEN'
  printf ' --env PYTHONPATH=/workspace/project/src'
  printf ' %q' "$AUTOMODEL_IMAGE"
  printf ' %q' "$@"
  printf '\n'
}

run_container() {
  prepare_hf_token

  if [[ "$DRY_RUN" == "1" ]]; then
    print_docker_command "$@"
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    return
  fi

  mkdir -p \
    "$HOST_PROJECT_ROOT" \
    "$HOST_DATA_ROOT" \
    "$HOST_HF_CACHE_ROOT" \
    "$HOST_NGC_CACHE_ROOT" \
    "$HOST_CHECKPOINT_ROOT" \
    "$HOST_RESULTS_ROOT" \
    "$HOST_LOGS_ROOT" \
    "$HOST_RUN_INFO_ROOT"

  docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v "${HOST_PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    -v "${HOST_DATA_ROOT}:${CONTAINER_DATA_ROOT}" \
    -v "${HOST_HF_CACHE_ROOT}:/root/.cache/huggingface" \
    -v "${HOST_NGC_CACHE_ROOT}:/root/.cache/ngc" \
    -v "${HOST_CHECKPOINT_ROOT}:${CONTAINER_CHECKPOINT_ROOT}" \
    -v "${HOST_RESULTS_ROOT}:${CONTAINER_RESULTS_ROOT}" \
    -v "${HOST_LOGS_ROOT}:${CONTAINER_LOGS_ROOT}" \
    -v "${HOST_RUN_INFO_ROOT}:${CONTAINER_RUN_INFO_ROOT}" \
    -w "$CONTAINER_PROJECT_ROOT" \
    --env HF_HOME=/root/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/root/.cache/huggingface \
    --env HF_TOKEN \
    --env HUGGING_FACE_HUB_TOKEN \
    --env PYTHONPATH=/workspace/project/src \
    "$AUTOMODEL_IMAGE" \
    "$@"

  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --pull)
      PULL_IMAGE=1
      shift
      ;;
    --login-ngc)
      LOGIN_NGC=1
      shift
      ;;
    --verify-cuda)
      VERIFY_CUDA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$DRY_RUN" != "1" || "$PULL_IMAGE" == "1" || "$VERIFY_CUDA" == "1" ]]; then
  require_command docker
fi

if [[ "$VERIFY_CUDA" == "1" ]]; then
  verify_cuda_image
fi

if [[ "$PULL_IMAGE" == "1" ]]; then
  pull_automodel_image
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

run_container "$@"
