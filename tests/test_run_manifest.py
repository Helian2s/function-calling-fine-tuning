from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from function_calling_ft.run_manifest import (
    CANONICAL_ARTIFACT_KEYS,
    build_exp00_run_manifest,
    migrate_smoke_metadata,
    validate_run_manifest,
)


MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _training_config() -> dict[str, Any]:
    return {
        "step_scheduler": {
            "global_batch_size": 4,
            "local_batch_size": 1,
            "max_steps": 30,
        },
        "model": {
            "pretrained_model_name_or_path": MODEL,
            "revision": REVISION,
            "torch_dtype": "bfloat16",
        },
        "peft": {"_target_": "nemo_automodel.components._peft.lora.PeftConfig"},
        "dataset": {"seq_length": 4096},
        "packed_sequence": {"packed_sequence_size": 0},
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": "/workspace/checkpoints/exp-00/smoke-lora",
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-4,
        },
        "rng": {"seed": 42},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_artifacts(results_dir: Path) -> None:
    for name in (
        "resolved_config.yaml",
        "environment_report.json",
        "predictions.jsonl",
        "scored_predictions.jsonl",
        "scores.json",
        "training_torch_memory.json",
        "case_report.md",
    ):
        (results_dir / name).write_text(f"{name}\n", encoding="utf-8")


def test_run_manifest_validation_accepts_complete_manifest(
    tmp_path: Path,
) -> None:
    _seed_artifacts(tmp_path)
    training_log = tmp_path / "training.log"
    training_log.write_text("loss=1.0\n", encoding="utf-8")
    dataset_manifest = tmp_path / "dataset_checksums.sha256"
    dataset_manifest.write_text("abc  train.jsonl\n", encoding="utf-8")

    manifest = build_exp00_run_manifest(
        run_metadata={
            "git_revision": "abc123",
            "git_dirty": False,
            "git_dirty_files": [],
            "container_tag": "nvcr.io/nvidia/nemo-automodel:25.11.00",
            "container_digest": "sha256:c4f613",
            "packages": {"nemo_automodel": "0.2.0rc0"},
            "model_name": MODEL,
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
        },
        generation_metadata={
            "model_name": MODEL,
            "model_revision": REVISION,
            "seed": 42,
            "max_new_tokens": 256,
            "load_in_4bit": False,
        },
        scores={"total_records": 40},
        training_config=_training_config(),
        results_dir=tmp_path,
        training_log=training_log,
        dataset_manifest_path=dataset_manifest,
        run_id="exp-00-test",
        status="succeeded",
    )

    validation = validate_run_manifest(manifest)

    assert validation.ok
    assert set(manifest["artifacts"]) == set(CANONICAL_ARTIFACT_KEYS)
    assert manifest["method"]["name"] == "bf16_lora"
    assert manifest["model"]["tokenizer_revision"] == REVISION


def test_run_manifest_validation_reports_missing_required_fields() -> None:
    validation = validate_run_manifest({"schema_version": "1.0"})

    assert not validation.ok
    assert "manifest.experiment_id must be a non-empty string" in validation.errors
    assert "environment must be a mapping" in validation.errors


def test_migrate_existing_smoke_metadata_writes_manifest(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _seed_artifacts(results_dir)
    training_log = tmp_path / "training.log"
    training_log.write_text("loss=1.0\n", encoding="utf-8")
    dataset_manifest = tmp_path / "dataset_checksums.sha256"
    dataset_manifest.write_text("abc  train.jsonl\n", encoding="utf-8")

    run_metadata = results_dir / "run_metadata.json"
    generation_metadata = results_dir / "generation_metadata.json"
    scores = results_dir / "scores.json"
    config = results_dir / "resolved_config.yaml"
    package_versions = results_dir / "package_versions.txt"
    output = results_dir / "run_manifest.json"

    _write_json(
        run_metadata,
        {
            "git_revision": "abc123",
            "git_dirty": False,
            "container_tag": "nvcr.io/nvidia/nemo-automodel:25.11.00",
            "container_digest": "sha256:c4f613",
            "model_name": MODEL,
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
        },
    )
    _write_json(
        generation_metadata,
        {
            "seed": 42,
            "max_new_tokens": 256,
            "load_in_4bit": False,
        },
    )
    _write_json(scores, {"total_records": 40})
    config.write_text(
        "\n".join(
            [
                "step_scheduler:",
                "  global_batch_size: 4",
                "  local_batch_size: 1",
                "  max_steps: 30",
                "model:",
                f"  pretrained_model_name_or_path: {MODEL}",
                f"  revision: {REVISION}",
                "  torch_dtype: bfloat16",
                "peft:",
                "  _target_: nemo_automodel.components._peft.lora.PeftConfig",
                "dataset:",
                "  seq_length: 4096",
                "packed_sequence:",
                "  packed_sequence_size: 0",
                "checkpoint:",
                "  enabled: true",
                "  checkpoint_dir: /workspace/checkpoints/exp-00/smoke-lora",
                "optimizer:",
                "  _target_: torch.optim.AdamW",
                "  lr: 0.0001",
                "rng:",
                "  seed: 42",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    package_versions.write_text(
        "nemo_automodel=0.2.0rc0\n",
        encoding="utf-8",
    )

    manifest = migrate_smoke_metadata(
        run_metadata_path=run_metadata,
        generation_metadata_path=generation_metadata,
        scores_path=scores,
        resolved_config_path=config,
        package_versions_path=package_versions,
        dataset_manifest_path=dataset_manifest,
        training_log_path=training_log,
        output_path=output,
        run_id="exp-00-migrated-test",
    )

    assert output.is_file()
    assert validate_run_manifest(manifest).ok
    assert manifest["run_id"] == "exp-00-migrated-test"
    assert manifest["environment"]["package_versions"] == {
        "nemo_automodel": "0.2.0rc0",
    }


def test_migrate_smoke_metadata_backfills_container_evidence(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _seed_artifacts(results_dir)
    training_log = tmp_path / "training.log"
    training_log.write_text("loss=1.0\n", encoding="utf-8")

    run_metadata = results_dir / "run_metadata.json"
    generation_metadata = results_dir / "generation_metadata.json"
    scores = results_dir / "scores.json"
    config = results_dir / "resolved_config.yaml"
    package_versions = results_dir / "package_versions.txt"
    container_image = tmp_path / "container_image.txt"
    container_report = tmp_path / "c4_container_report.json"
    output = results_dir / "run_manifest.json"

    _write_json(
        run_metadata,
        {
            "git_revision": "abc123",
            "git_dirty": False,
            "model_name": MODEL,
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
        },
    )
    _write_json(
        generation_metadata,
        {
            "seed": 42,
            "max_new_tokens": 256,
            "load_in_4bit": False,
        },
    )
    _write_json(scores, {"total_records": 40})
    config.write_text(
        "\n".join(
            [
                "step_scheduler:",
                "  global_batch_size: 4",
                "  local_batch_size: 1",
                "  max_steps: 30",
                "model:",
                f"  pretrained_model_name_or_path: {MODEL}",
                f"  revision: {REVISION}",
                "  torch_dtype: bfloat16",
                "peft:",
                "  _target_: nemo_automodel.components._peft.lora.PeftConfig",
                "dataset:",
                "  seq_length: 4096",
                "packed_sequence:",
                "  packed_sequence_size: 0",
                "checkpoint:",
                "  enabled: true",
                "  checkpoint_dir: /workspace/checkpoints/exp-00/smoke-lora",
                "optimizer:",
                "  _target_: torch.optim.AdamW",
                "  lr: 0.0001",
                "rng:",
                "  seed: 42",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    package_versions.write_text(
        "torch=2.9.0a0+50eac811a6.nv25.09\n",
        encoding="utf-8",
    )
    container_image.write_text(
        "\n".join(
            [
                "image=nvcr.io/nvidia/nemo-automodel:25.11.00",
                "repo_digest=nvcr.io/nvidia/nemo-automodel@sha256:c4f613",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        container_report,
        {
            "packages": {
                "nemo_automodel": {
                    "imported": True,
                    "version": "0.2.0rc0",
                },
                "torch": {
                    "imported": True,
                    "version": "container-report-version",
                },
            },
        },
    )

    manifest = migrate_smoke_metadata(
        run_metadata_path=run_metadata,
        generation_metadata_path=generation_metadata,
        scores_path=scores,
        resolved_config_path=config,
        package_versions_path=package_versions,
        container_image_path=container_image,
        container_report_path=container_report,
        training_log_path=training_log,
        output_path=output,
        run_id="exp-00-migrated-container-test",
    )

    assert validate_run_manifest(manifest).ok
    assert (
        manifest["environment"]["container_tag"]
        == "nvcr.io/nvidia/nemo-automodel:25.11.00"
    )
    assert manifest["environment"]["container_digest"] == "sha256:c4f613"
    assert manifest["environment"]["package_versions"] == {
        "nemo_automodel": "0.2.0rc0",
        "torch": "2.9.0a0+50eac811a6.nv25.09",
    }


def test_run_manifest_rejects_unknown_method(tmp_path: Path) -> None:
    _seed_artifacts(tmp_path)
    training_log = tmp_path / "training.log"
    training_log.write_text("loss=1.0\n", encoding="utf-8")
    config = _training_config()
    config.pop("peft")

    manifest = build_exp00_run_manifest(
        run_metadata={
            "git_revision": "abc123",
            "git_dirty": False,
            "container_tag": "container",
            "container_digest": "sha256:c4f613",
            "packages": {"nemo_automodel": "0.2.0rc0"},
            "model_name": MODEL,
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
        },
        generation_metadata={"seed": 42, "max_new_tokens": 256},
        scores={"total_records": 40},
        training_config=config,
        results_dir=tmp_path,
        training_log=training_log,
        dataset_manifest_path=None,
        run_id="exp-00-base",
    )
    manifest["method"]["name"] = "unsupported"

    validation = validate_run_manifest(manifest)

    assert not validation.ok
    assert "method.name 'unsupported' is not allowed" in validation.errors
