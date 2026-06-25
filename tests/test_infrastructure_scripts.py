from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts.verify_source_archive import validate_archive


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_script_is_nonempty_and_rejects_dangerous_patterns() -> None:
    bootstrap = ROOT / "infrastructure/aws/bootstrap/bootstrap_instance.sh"
    text = bootstrap.read_text(encoding="utf-8")

    assert bootstrap.stat().st_size > 0
    assert "set -Eeuo pipefail" in text
    assert "wipefs --all" not in text
    assert "validate_source_archive" in text
    assert 'readonly WORKSPACE_MOUNT="/mnt/workspace"' in text
    assert 'readonly CHECKPOINT_ROOT="${WORKSPACE_MOUNT}/checkpoints"' in text
    assert "exp00-release-manifest.json" in text


def test_container_runner_has_required_mounts_and_secret_safe_env() -> None:
    text = (ROOT / "scripts/run_automodel_container.sh").read_text(
        encoding="utf-8",
    )

    assert "--gpus all" in text
    assert "--shm-size=16g" in text
    assert "${HOST_PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" in text
    assert "${HOST_CHECKPOINT_ROOT}:${CONTAINER_CHECKPOINT_ROOT}" in text
    assert "--env HF_TOKEN" in text
    assert "--password-stdin" in text


def test_shutdown_script_has_dry_run_and_never_terminates() -> None:
    text = (
        ROOT / "infrastructure/aws/bootstrap/shutdown_and_sync.sh"
    ).read_text(encoding="utf-8")

    assert "--dry-run" in text
    assert "shutdown -h now" in text
    assert "terminate-instances" not in text
    assert "--delete" not in text
    assert "/mnt/workspace/run-info" in text
    assert "--stage baseline|final" in text
    assert "/mnt/workspace/checkpoints/${EXPERIMENT}/smoke-qlora" in text
    assert '"/mnt/workspace/checkpoints/${EXPERIMENT}"' not in text


def test_artifact_sync_policy_skips_intermediate_checkpoints() -> None:
    sync_script = (ROOT / "scripts/sync_results.sh").read_text(
        encoding="utf-8",
    )
    shutdown_script = (
        ROOT / "infrastructure/aws/bootstrap/shutdown_and_sync.sh"
    ).read_text(encoding="utf-8")

    assert "--stage baseline|final" in sync_script
    assert "FINAL_ADAPTER_DIR" in sync_script
    assert 'sync_dir "$CHECKPOINT_DIR" "$CHECKPOINT_URI"' not in sync_script
    assert "intermediate training checkpoints" in sync_script

    assert "smoke-qlora" in shutdown_script
    assert "intermediate training checkpoints" in shutdown_script
    assert '"/mnt/workspace/checkpoints/${EXPERIMENT}"' not in shutdown_script


def test_source_archive_verifier_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive member"):
        validate_archive(archive_path)


def test_source_archive_verifier_accepts_safe_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"ok"
        info = tarfile.TarInfo("Makefile")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    report = validate_archive(
        archive_path,
        expected_members=["Makefile"],
    )

    assert report["member_count"] == 1
