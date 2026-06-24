#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=configs/common/exp00.env
source "${REPO_ROOT}/configs/common/exp00.env"

PROFILE="${AWS_PROFILE:-finetuning-local}"
LAUNCH_TEMPLATE_NAME="${LAUNCH_TEMPLATE_NAME:-ft-exp00-g6e-v1}"

log() {
  printf '[audit-launch-template] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

command -v aws >/dev/null 2>&1 || die "aws CLI is required"
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v base64 >/dev/null 2>&1 || die "base64 is required"

template_json="$(
  aws ec2 describe-launch-templates \
    --launch-template-names "$LAUNCH_TEMPLATE_NAME" \
    --profile "$PROFILE" \
    --region "$FT_AWS_REGION" \
    --output json
)"
default_version="$(jq -r '.LaunchTemplates[0].DefaultVersionNumber' <<<"$template_json")"

version_json="$(
  aws ec2 describe-launch-template-versions \
    --launch-template-name "$LAUNCH_TEMPLATE_NAME" \
    --versions "$default_version" \
    --profile "$PROFILE" \
    --region "$FT_AWS_REGION" \
    --output json
)"

log "launch_template=${LAUNCH_TEMPLATE_NAME}"
log "default_version=${default_version}"
jq '.LaunchTemplateVersions[0].LaunchTemplateData | {
  image_id: .ImageId,
  instance_type: .InstanceType,
  key_name: (.KeyName // null),
  subnet_id: .NetworkInterfaces[0].SubnetId,
  security_groups: .NetworkInterfaces[0].Groups,
  associate_public_ip_address: .NetworkInterfaces[0].AssociatePublicIpAddress,
  iam_instance_profile: .IamInstanceProfile,
  shutdown_behavior: .InstanceInitiatedShutdownBehavior,
  disable_api_termination: .DisableApiTermination,
  metadata_options: .MetadataOptions,
  block_device_mappings: .BlockDeviceMappings
}' <<<"$version_json"

security_group_ids="$(
  jq -r '.LaunchTemplateVersions[0].LaunchTemplateData.NetworkInterfaces[0].Groups[]?' \
    <<<"$version_json"
)"

if [[ -n "$security_group_ids" ]]; then
  # shellcheck disable=SC2086
  aws ec2 describe-security-groups \
    --group-ids $security_group_ids \
    --profile "$PROFILE" \
    --region "$FT_AWS_REGION" \
    --output json |
    jq '.SecurityGroups[] | {
      group_id: .GroupId,
      group_name: .GroupName,
      inbound_rule_count: (.IpPermissions | length),
      outbound_rule_count: (.IpPermissionsEgress | length),
      inbound: .IpPermissions,
      outbound: .IpPermissionsEgress
    }'
fi

user_data="$(jq -r '.LaunchTemplateVersions[0].LaunchTemplateData.UserData // empty' <<<"$version_json")"
if [[ -n "$user_data" ]]; then
  log "decoded_user_data_begin"
  printf '%s' "$user_data" | base64 -d
  printf '\n'
  log "decoded_user_data_end"
else
  log "No user data configured on default launch-template version."
fi
