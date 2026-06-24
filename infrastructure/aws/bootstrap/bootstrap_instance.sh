#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

readonly BUCKET="finetuning-lab-1-037678282394-us-west-2-an"
readonly PREFIX="finetuning"
readonly REGION="us-west-2"

readonly RELEASE_MANIFEST="exp00-release-manifest.json"
readonly RELEASE_MANIFEST_URI="s3://${BUCKET}/${PREFIX}/source-bundles/${RELEASE_MANIFEST}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-}"
SOURCE_URI="${SOURCE_URI:-}"

readonly WORKSPACE_MOUNT="/mnt/workspace"
readonly PROJECT_ROOT="${WORKSPACE_MOUNT}/project"
readonly DATA_ROOT="${WORKSPACE_MOUNT}/data"
readonly HF_CACHE_ROOT="${WORKSPACE_MOUNT}/huggingface-cache"
readonly NGC_CACHE_ROOT="${WORKSPACE_MOUNT}/ngc-cache"
readonly CHECKPOINT_ROOT="${WORKSPACE_MOUNT}/checkpoints"
readonly RESULTS_ROOT="${WORKSPACE_MOUNT}/results"
readonly LOGS_ROOT="${WORKSPACE_MOUNT}/logs"
readonly RUN_INFO_ROOT="${WORKSPACE_MOUNT}/run-info"

readonly DATA_URI="s3://${BUCKET}/${PREFIX}/data/smoke-v1"
readonly SHUTDOWN_URI="s3://${BUCKET}/${PREFIX}/source-bundles/shutdown_and_sync.sh"
readonly SHUTDOWN_PATH="/usr/local/sbin/ft-exp00-shutdown-and-sync"

readonly DEFAULT_USER="${DEFAULT_USER:-ubuntu}"

exec > >(tee -a /var/log/ft-exp00-bootstrap.log) 2>&1

log() {
    printf '[%s] [bootstrap] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

on_error() {
    local exit_code=$?
    log "ERROR: command failed at line ${BASH_LINENO[0]} with exit code ${exit_code}"
    exit "$exit_code"
}

trap on_error ERR

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        fail "Required command not found: $1"
}

retry() {
    local attempt=1
    local maximum_attempts=5
    local delay_seconds=5

    until "$@"; do
        if (( attempt >= maximum_attempts )); then
            log "Command failed after ${maximum_attempts} attempts: $*"
            return 1
        fi

        log "Command failed; retrying in ${delay_seconds}s: $*"
        sleep "$delay_seconds"
        attempt=$((attempt + 1))
        delay_seconds=$((delay_seconds * 2))
    done
}

install_required_packages() {
    local packages=()

    command -v aws >/dev/null 2>&1 || packages+=(awscli)
    command -v docker >/dev/null 2>&1 || packages+=(docker.io)
    command -v jq >/dev/null 2>&1 || packages+=(jq)
    command -v mkfs.ext4 >/dev/null 2>&1 || packages+=(e2fsprogs)
    command -v python3 >/dev/null 2>&1 || packages+=(python3)

    if (( ${#packages[@]} > 0 )); then
        log "Installing required packages: ${packages[*]}"
        retry apt-get update
        DEBIAN_FRONTEND=noninteractive \
            retry apt-get install -y "${packages[@]}"
    fi

    systemctl enable --now docker

    if id "$DEFAULT_USER" >/dev/null 2>&1; then
        usermod -aG docker "$DEFAULT_USER"
    else
        fail "Expected operating-system user does not exist: $DEFAULT_USER"
    fi
}

identify_workspace_disk() {
    local root_source
    local root_parent
    local root_disk
    local minimum_size_bytes=257698037760
    local maximum_size_bytes=279172874240
    local disk
    local disk_type
    local disk_size

    root_source="$(findmnt -n -o SOURCE /)"
    [[ -n "$root_source" ]] || fail "Could not identify root filesystem source"

    root_source="$(readlink -f "$root_source")"
    root_parent="$(lsblk -nro PKNAME "$root_source" | head -n 1 || true)"

    if [[ -n "$root_parent" ]]; then
        root_disk="/dev/${root_parent}"
    else
        root_disk="$root_source"
    fi

    log "Root filesystem source: $root_source"
    log "Root disk: $root_disk"

    WORKSPACE_DISK_CANDIDATES=()

    while read -r disk disk_type disk_size; do
        [[ "$disk_type" == "disk" ]] || continue
        [[ "$disk" != "$root_disk" ]] || continue

        if (( disk_size >= minimum_size_bytes &&
              disk_size <= maximum_size_bytes )); then
            WORKSPACE_DISK_CANDIDATES+=("$disk")
        fi
    done < <(lsblk -bdnpo NAME,TYPE,SIZE)

    if (( ${#WORKSPACE_DISK_CANDIDATES[@]} != 1 )); then
        log "Unable to identify exactly one 240–260 GiB non-root disk."
        log "Detected block devices:"
        lsblk -o NAME,PATH,TYPE,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS
        fail "Workspace disk candidate count: ${#WORKSPACE_DISK_CANDIDATES[@]}"
    fi

    WORKSPACE_DISK="${WORKSPACE_DISK_CANDIDATES[0]}"
    log "Workspace disk candidate: $WORKSPACE_DISK"
}

select_workspace_filesystem_device() {
    local partitions=()

    mapfile -t partitions < <(
        lsblk -nrpo NAME,TYPE "$WORKSPACE_DISK" |
            awk '$2 == "part" {print $1}'
    )

    case "${#partitions[@]}" in
        0)
            WORKSPACE_FS_DEVICE="$WORKSPACE_DISK"
            ;;
        1)
            WORKSPACE_FS_DEVICE="${partitions[0]}"
            log "Using existing single partition: $WORKSPACE_FS_DEVICE"
            ;;
        *)
            log "Workspace disk has multiple partitions:"
            printf '  %s\n' "${partitions[@]}"
            fail "Refusing to select or format a multi-partition workspace disk"
            ;;
    esac
}

prepare_workspace_filesystem() {
    local filesystem_type
    local existing_signatures

    filesystem_type="$(blkid -o value -s TYPE "$WORKSPACE_FS_DEVICE" 2>/dev/null || true)"

    if [[ -n "$filesystem_type" ]]; then
        case "$filesystem_type" in
            ext4|xfs)
                log "Reusing existing ${filesystem_type} filesystem on $WORKSPACE_FS_DEVICE"
                ;;
            *)
                fail "Unsupported existing filesystem type on workspace device: $filesystem_type"
                ;;
        esac

        return
    fi

    if [[ "$WORKSPACE_FS_DEVICE" == "$WORKSPACE_DISK" ]]; then
        if lsblk -nrpo NAME,TYPE "$WORKSPACE_DISK" |
            awk '$2 == "part" {found=1} END {exit !found}'
        then
            fail "Workspace disk contains partitions but no filesystem was selected"
        fi
    fi

    existing_signatures="$(wipefs -n "$WORKSPACE_FS_DEVICE" 2>/dev/null || true)"

    if [[ -n "$existing_signatures" ]]; then
        log "Existing signatures were detected:"
        printf '%s\n' "$existing_signatures"
        fail "Refusing to format a device containing existing signatures"
    fi

    log "Creating a new ext4 filesystem on confirmed blank device: $WORKSPACE_FS_DEVICE"
    mkfs.ext4 -F -L ft-workspace "$WORKSPACE_FS_DEVICE"
}

mount_workspace() {
    local filesystem_uuid
    local mounted_source

    mkdir -p "$WORKSPACE_MOUNT"

    if mountpoint -q "$WORKSPACE_MOUNT"; then
        mounted_source="$(findmnt -n -o SOURCE "$WORKSPACE_MOUNT")"
        log "$WORKSPACE_MOUNT is already mounted from $mounted_source"
        return
    fi

    filesystem_uuid="$(blkid -o value -s UUID "$WORKSPACE_FS_DEVICE")"
    [[ -n "$filesystem_uuid" ]] || fail "Workspace filesystem has no UUID"

    if ! grep -Eq \
        "^[[:space:]]*UUID=${filesystem_uuid}[[:space:]]+${WORKSPACE_MOUNT}[[:space:]]" \
        /etc/fstab
    then
        printf 'UUID=%s %s auto defaults,nofail 0 2\n' \
            "$filesystem_uuid" \
            "$WORKSPACE_MOUNT" \
            >> /etc/fstab
    fi

    mount "$WORKSPACE_MOUNT"

    mountpoint -q "$WORKSPACE_MOUNT" ||
        fail "Workspace mount failed: $WORKSPACE_MOUNT"

    log "Mounted workspace:"
    findmnt "$WORKSPACE_MOUNT"
}

prepare_workspace_directories() {
    local owner_group

    owner_group="$(id -gn "$DEFAULT_USER")"

    mkdir -p \
        "$PROJECT_ROOT" \
        "$DATA_ROOT" \
        "$HF_CACHE_ROOT" \
        "$NGC_CACHE_ROOT" \
        "$CHECKPOINT_ROOT" \
        "$RESULTS_ROOT" \
        "$LOGS_ROOT" \
        "$RUN_INFO_ROOT"

    chown "$DEFAULT_USER:$owner_group" \
        "$WORKSPACE_MOUNT" \
        "$PROJECT_ROOT" \
        "$DATA_ROOT" \
        "$HF_CACHE_ROOT" \
        "$NGC_CACHE_ROOT" \
        "$CHECKPOINT_ROOT" \
        "$RESULTS_ROOT" \
        "$LOGS_ROOT" \
        "$RUN_INFO_ROOT"
}

load_release_manifest() {
    local manifest_path="/tmp/${RELEASE_MANIFEST}"

    if [[ -n "$SOURCE_REVISION" &&
          -n "$SOURCE_ARCHIVE" &&
          -n "$SOURCE_ARCHIVE_SHA256" ]]
    then
        SOURCE_URI="s3://${BUCKET}/${PREFIX}/source-bundles/${SOURCE_ARCHIVE}"
        log "Using source archive constants supplied by environment"
        return
    fi

    log "Downloading release manifest: $RELEASE_MANIFEST_URI"
    retry aws s3 cp \
        "$RELEASE_MANIFEST_URI" \
        "$manifest_path" \
        --region "$REGION" \
        --only-show-errors

    [[ -s "$manifest_path" ]] ||
        fail "Release manifest is empty: $RELEASE_MANIFEST_URI"

    SOURCE_REVISION="$(jq -r '.git_revision // empty' "$manifest_path")"
    SOURCE_ARCHIVE="$(jq -r '.source_archive // empty' "$manifest_path")"
    SOURCE_ARCHIVE_SHA256="$(
        jq -r '.source_archive_sha256 // empty' "$manifest_path"
    )"

    [[ -n "$SOURCE_REVISION" ]] ||
        fail "Release manifest is missing git_revision"
    [[ -n "$SOURCE_ARCHIVE" ]] ||
        fail "Release manifest is missing source_archive"
    [[ -n "$SOURCE_ARCHIVE_SHA256" ]] ||
        fail "Release manifest is missing source_archive_sha256"
    [[ "$SOURCE_ARCHIVE" == exp00-source-*.tar.gz ]] ||
        fail "Release manifest source_archive is not revisioned"

    SOURCE_URI="s3://${BUCKET}/${PREFIX}/source-bundles/${SOURCE_ARCHIVE}"
    rm -f "$manifest_path"
}

validate_source_archive() {
    local archive_path="$1"

    python3 - "$archive_path" <<'PY'
from __future__ import annotations

import posixpath
import sys
import tarfile

archive_path = sys.argv[1]

with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive.getmembers():
        name = member.name
        normalized = posixpath.normpath(name)

        if not name or name.startswith("/") or normalized.startswith("../"):
            raise SystemExit(
                f"Unsafe archive member path rejected: {name!r}"
            )

        if normalized == ".." or "/../" in f"/{normalized}/":
            raise SystemExit(
                f"Unsafe archive member path rejected: {name!r}"
            )

        if member.issym() or member.islnk():
            raise SystemExit(f"Archive links are not allowed: {name!r}")
PY
}

download_and_extract_source() {
    local archive_path="/tmp/${SOURCE_ARCHIVE}"
    local marker_path="${PROJECT_ROOT}/.source-revision"
    local existing_revision=""
    local owner_group

    owner_group="$(id -gn "$DEFAULT_USER")"

    if [[ -f "$marker_path" ]]; then
        existing_revision="$(tr -d '[:space:]' < "$marker_path")"
    fi

    if [[ "$existing_revision" == "$SOURCE_REVISION" ]]; then
        log "Pinned source revision is already installed: $SOURCE_REVISION"
        return
    fi

    if find "$PROJECT_ROOT" -mindepth 1 -maxdepth 1 -print -quit |
        grep -q .
    then
        fail "Project directory is nonempty but does not contain the expected source marker"
    fi

    log "Downloading pinned source archive"
    retry aws s3 cp \
        "$SOURCE_URI" \
        "$archive_path" \
        --region "$REGION" \
        --only-show-errors

    printf '%s  %s\n' "$SOURCE_ARCHIVE_SHA256" "$archive_path" |
        sha256sum -c -

    validate_source_archive "$archive_path"
    tar -xzf "$archive_path" -C "$PROJECT_ROOT" --no-same-owner

    printf '%s\n' "$SOURCE_REVISION" > "$marker_path"

    chown -R "$DEFAULT_USER:$owner_group" "$PROJECT_ROOT"

    rm -f "$archive_path"

    log "Installed source revision: $SOURCE_REVISION"
}

download_and_verify_dataset() {
    local owner_group

    owner_group="$(id -gn "$DEFAULT_USER")"

    log "Synchronizing frozen smoke dataset"
    retry aws s3 sync \
        "${DATA_URI}/" \
        "${DATA_ROOT}/" \
        --region "$REGION" \
        --only-show-errors

    [[ -f "${DATA_ROOT}/checksums.sha256" ]] ||
        fail "Dataset checksum manifest is missing"

    (
        cd "$DATA_ROOT"
        sha256sum -c checksums.sha256
    )

    chown -R "$DEFAULT_USER:$owner_group" "$DATA_ROOT"

    log "Dataset checksum verification passed"
}

install_shutdown_script() {
    local temporary_path="/tmp/shutdown_and_sync.sh"

    retry aws s3 cp \
        "$SHUTDOWN_URI" \
        "$temporary_path" \
        --region "$REGION" \
        --only-show-errors

    [[ -s "$temporary_path" ]] ||
        fail "Downloaded shutdown script is empty"

    bash -n "$temporary_path"

    install \
        -o root \
        -g root \
        -m 0750 \
        "$temporary_path" \
        "$SHUTDOWN_PATH"

    rm -f "$temporary_path"

    log "Installed shutdown script: $SHUTDOWN_PATH"
}

write_bootstrap_metadata() {
    {
        printf 'bootstrap_completed_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'source_revision=%s\n' "$SOURCE_REVISION"
        printf 'source_archive=%s\n' "$SOURCE_ARCHIVE"
        printf 'source_archive_sha256=%s\n' "$SOURCE_ARCHIVE_SHA256"
        printf 'release_manifest=%s\n' "$RELEASE_MANIFEST_URI"
        printf 'workspace_disk=%s\n' "$WORKSPACE_DISK"
        printf 'workspace_filesystem_device=%s\n' "$WORKSPACE_FS_DEVICE"
        printf 'workspace_mount=%s\n' "$WORKSPACE_MOUNT"
    } > "${RUN_INFO_ROOT}/bootstrap.env"

    chown "$DEFAULT_USER:$(id -gn "$DEFAULT_USER")" \
        "${RUN_INFO_ROOT}/bootstrap.env"
}

main() {
    [[ "$EUID" -eq 0 ]] ||
        fail "Bootstrap must run as root"

    log "Starting Experiment 0 instance bootstrap"

    require_command apt-get
    require_command findmnt
    require_command lsblk
    require_command blkid
    require_command wipefs
    require_command sha256sum
    require_command tar

    install_required_packages
    require_command aws
    require_command docker
    require_command python3
    identify_workspace_disk
    select_workspace_filesystem_device
    prepare_workspace_filesystem
    mount_workspace
    prepare_workspace_directories
    load_release_manifest
    download_and_extract_source
    download_and_verify_dataset
    install_shutdown_script
    write_bootstrap_metadata

    log "Bootstrap completed successfully"
    log "Project: $PROJECT_ROOT"
    log "Dataset: $DATA_ROOT"
    log "Checkpoints: $CHECKPOINT_ROOT"
    log "Results: $RESULTS_ROOT"
    log "Logs: $LOGS_ROOT"
    log "Shutdown command: sudo $SHUTDOWN_PATH"
}

main "$@"
