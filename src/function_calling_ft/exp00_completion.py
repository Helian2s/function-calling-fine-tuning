from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from function_calling_ft.evaluation import read_jsonl, score_prediction_records
from function_calling_ft.run_manifest import sha256_file


COMPLETION_SCHEMA_VERSION = "1.0"
REQUIRED_FINAL_RESULT_FILES = (
    "predictions.jsonl",
    "scored_predictions.jsonl",
    "parse_failures.jsonl",
    "scores.json",
    "generation_metadata.json",
    "run_metadata.json",
    "run_manifest.json",
    "training_metrics.json",
    "training_torch_memory.json",
    "environment_report.json",
    "nvidia-smi.txt",
    "package_versions.txt",
    "resolved_config.yaml",
    "requested_metrics.json",
    "case_report.json",
    "case_report.md",
    "checksums.sha256",
)
REQUIRED_S3_SUFFIXES = tuple(
    f"finetuning/results/exp-00/{name}"
    for name in REQUIRED_FINAL_RESULT_FILES
) + (
    "finetuning/logs/exp-00/training.log",
    "finetuning/checkpoints/exp-00/smoke-lora/LATEST/model/adapter_config.json",
    "finetuning/checkpoints/exp-00/smoke-lora/LATEST/model/adapter_model.safetensors",
)


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    summary: str
    evidence: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _stage(
    name: str,
    status: str,
    summary: str,
    evidence: Mapping[str, Any] | None = None,
) -> StageResult:
    return StageResult(
        name=name,
        status=status,
        summary=summary,
        evidence=dict(evidence or {}),
    )


def _compare_values(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def verify_scores_from_predictions(
    *,
    dataset_path: Path,
    predictions_path: Path,
    scores_path: Path,
) -> dict[str, Any]:
    if not dataset_path.is_file():
        return {
            "status": "missing",
            "summary": f"dataset is missing: {dataset_path}",
        }
    if not predictions_path.is_file():
        return {
            "status": "missing",
            "summary": f"predictions are missing: {predictions_path}",
        }
    if not scores_path.is_file():
        return {
            "status": "missing",
            "summary": f"scores are missing: {scores_path}",
        }

    dataset_records = read_jsonl(dataset_path)
    prediction_records = read_jsonl(predictions_path)
    _scored, _failures, recomputed = score_prediction_records(
        dataset_records,
        prediction_records,
    )
    stored = load_json(scores_path)
    mismatches = {
        key: {
            "stored": stored_value,
            "recomputed": recomputed.get(key),
        }
        for key, stored_value in stored.items()
        if not _compare_values(stored_value, recomputed.get(key))
    }
    recomputed_additional_keys = sorted(set(recomputed) - set(stored))
    status = "pass" if not mismatches else "fail"
    summary = (
        "stored scores match recomputed scores"
        if not mismatches
        else "stored scores differ from recomputed scores"
    )
    return {
        "status": status,
        "summary": summary,
        "dataset_records": len(dataset_records),
        "prediction_records": len(prediction_records),
        "prediction_sha256": sha256_file(predictions_path),
        "scores_sha256": sha256_file(scores_path),
        "stored_scores": stored,
        "recomputed_scores": recomputed,
        "recomputed_additional_keys": recomputed_additional_keys,
        "mismatches": mismatches,
    }


def _score_stage(
    *,
    name: str,
    dataset_path: Path,
    predictions_path: Path,
    scores_path: Path,
    expected_records: int,
) -> StageResult:
    result = verify_scores_from_predictions(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        scores_path=scores_path,
    )
    if result["status"] != "pass":
        return _stage(name, result["status"], result["summary"], result)

    stored_scores = result["stored_scores"]
    total_records = int(stored_scores.get("total_records", 0) or 0)
    predictions_present = int(
        stored_scores.get("predictions_present", 0) or 0,
    )
    if total_records != expected_records or predictions_present != expected_records:
        return _stage(
            name,
            "fail",
            (
                f"expected {expected_records} scored predictions, got "
                f"total_records={total_records}, "
                f"predictions_present={predictions_present}"
            ),
            result,
        )
    return _stage(
        name,
        "pass",
        f"{expected_records} predictions verified against stored scores",
        result,
    )


def _template_stage(report_path: Path) -> StageResult:
    report = load_json(report_path)
    if not report:
        return _stage(
            "native_template_rendering",
            "missing",
            "template report is missing",
            {"report": _artifact(report_path)},
        )

    rendered = int(report.get("examples_rendered", 0) or 0)
    thinking_enabled = bool(report.get("thinking_mode_enabled"))
    failures = int(report.get("total_failures", 0) or 0)
    status = "pass" if rendered >= 5 and not thinking_enabled and failures == 0 else "fail"
    return _stage(
        "native_template_rendering",
        status,
        (
            "at least five native-template examples rendered with thinking disabled"
            if status == "pass"
            else "template evidence does not satisfy Experiment 0 contract"
        ),
        {
            "report": _artifact(report_path),
            "examples_rendered": rendered,
            "thinking_mode_enabled": thinking_enabled,
            "total_failures": failures,
            "model_name": report.get("model_name"),
            "model_revision": report.get("model_revision"),
        },
    )


def _loss_mask_stage(report_path: Path) -> StageResult:
    report = load_json(report_path)
    if not report:
        return _stage(
            "loss_mask",
            "missing",
            "loss-mask report is missing",
            {"report": _artifact(report_path)},
        )

    examples = []
    if isinstance(report.get("smoke_examples"), list):
        examples.extend(report["smoke_examples"])
    if isinstance(report.get("synthetic_example"), dict):
        examples.append(report["synthetic_example"])

    included_non_tool_regions: list[str] = []
    included_token_count = 0
    for example in examples:
        if not isinstance(example, dict):
            continue
        included_token_count += int(example.get("included_token_count", 0) or 0)
        spans = example.get("spans", [])
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict) or not span.get("include_in_loss"):
                continue
            region = str(span.get("region", ""))
            if region != "assistant_tool_call":
                included_non_tool_regions.append(region)

    status = (
        "pass"
        if included_token_count > 0 and not included_non_tool_regions
        else "fail"
    )
    return _stage(
        "loss_mask",
        status,
        (
            "loss includes assistant tool-call spans only"
            if status == "pass"
            else "loss-mask evidence is incomplete or includes non-tool-call spans"
        ),
        {
            "report": _artifact(report_path),
            "thinking_mode_enabled": report.get("thinking_mode_enabled"),
            "examples_checked": len(examples),
            "included_token_count": included_token_count,
            "included_non_tool_regions": included_non_tool_regions,
        },
    )


def _environment_stage(
    *,
    results_dir: Path,
    run_info_dir: Path,
) -> StageResult:
    environment = load_json(results_dir / "environment_report.json")
    c4_report = load_json(run_info_dir / "c4_container_report.json")
    nvidia_smi = results_dir / "nvidia-smi.txt"
    bootstrap = run_info_dir / "bootstrap.env"
    package_versions = results_dir / "package_versions.txt"
    container_image = run_info_dir / "container_image.txt"
    expected_container = "nvcr.io/nvidia/nemo-automodel:25.11.00"
    expected_digest = (
        "sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2"
    )
    packages_text = (
        package_versions.read_text(encoding="utf-8")
        if package_versions.is_file()
        else ""
    )
    container_text = (
        container_image.read_text(encoding="utf-8")
        if container_image.is_file()
        else ""
    )
    container_tag = environment.get("container_tag")
    container_digest = environment.get("container_digest")
    container_ok = (
        container_tag == expected_container
        or f"image={expected_container}" in container_text
    )
    digest_ok = (
        container_digest == expected_digest
        or expected_digest in container_text
    )
    package_ok = (
        "nemo_automodel=0.2.0rc0" in packages_text
        or environment.get("packages", {}).get("nemo_automodel") == "0.2.0rc0"
        or c4_report.get("packages", {})
        .get("nemo_automodel", {})
        .get("version")
        == "0.2.0rc0"
    )
    evidence_present = nvidia_smi.is_file() or bootstrap.is_file()
    status = "pass" if container_ok and digest_ok and package_ok and evidence_present else "missing"
    return _stage(
        "gpu_driver_docker_cuda_mount_s3_preflight",
        status,
        (
            "environment/package evidence is present"
            if status == "pass"
            else "environment evidence is incomplete"
        ),
        {
            "environment_report": _artifact(results_dir / "environment_report.json"),
            "nvidia_smi": _artifact(nvidia_smi),
            "package_versions": _artifact(package_versions),
            "bootstrap_env": _artifact(bootstrap),
            "container_image": _artifact(container_image),
            "c4_container_report": _artifact(run_info_dir / "c4_container_report.json"),
            "container_tag": container_tag,
            "container_digest": container_digest,
            "nemo_automodel_0_2_0rc0_observed": package_ok,
        },
    )


def _model_load_stage(results_dir: Path) -> StageResult:
    generation = load_json(results_dir / "generation_metadata.json")
    environment = load_json(results_dir / "environment_report.json")
    model_name = generation.get("model_name") or environment.get("model_name")
    model_revision = generation.get("model_revision") or environment.get(
        "model_revision",
    )
    status = (
        "pass"
        if model_name == "Qwen/Qwen3-1.7B"
        and model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        else "missing"
    )
    return _stage(
        "qwen3_1_7b_pinned_load",
        status,
        (
            "generation metadata records pinned Qwen3-1.7B load"
            if status == "pass"
            else "pinned model load evidence is missing"
        ),
        {
            "generation_metadata": _artifact(results_dir / "generation_metadata.json"),
            "model_name": model_name,
            "model_revision": model_revision,
        },
    )


def _training_stage(results_dir: Path, logs_dir: Path) -> StageResult:
    metrics_path = results_dir / "training_metrics.json"
    metrics = load_json(metrics_path)
    if not metrics:
        return _stage(
            "smoke_training_30_steps",
            "missing",
            "training metrics are missing",
            {"training_metrics": _artifact(metrics_path)},
        )

    global_step = metrics.get("global_step")
    finite = metrics.get("losses_are_finite")
    oom_count = int(metrics.get("oom_event_count", 0) or 0)
    trainable_count = metrics.get("trainable_parameter_count")
    frozen_count = metrics.get("frozen_parameter_count")
    adapter_gradients = metrics.get("adapter_gradient_status")
    base_status = metrics.get("base_model_trainability_status")
    checkpoint_exists = bool(metrics.get("checkpoint_exists_after", True))
    step_history = metrics.get("step_history", [])
    completed_step_count = (
        len(step_history) if isinstance(step_history, list) else None
    )
    pass_core = (
        completed_step_count is not None
        and completed_step_count >= 30
        and finite is True
        and oom_count == 0
        and checkpoint_exists
    )
    gradient_evidence = adapter_gradients in {
        "observed_nonzero_grad_norm",
        "observed",
    }
    frozen_evidence = base_status in {
        "frozen_by_ratio",
        "not_detected_as_trainable",
    } and trainable_count not in {None, 0}
    status = "pass" if pass_core and gradient_evidence and frozen_evidence else "missing"
    return _stage(
        "smoke_training_30_steps",
        status,
        (
            "30-step training has finite loss and adapter-only evidence"
            if status == "pass"
            else "training evidence is incomplete for Task 01"
        ),
        {
            "training_metrics": _artifact(metrics_path),
            "training_log": _artifact(logs_dir / "training.log"),
            "global_step": global_step,
            "completed_step_count": completed_step_count,
            "losses_are_finite": finite,
            "oom_event_count": oom_count,
            "trainable_parameter_count": trainable_count,
            "frozen_parameter_count": frozen_count,
            "adapter_gradient_status": adapter_gradients,
            "base_model_trainability_status": base_status,
            "checkpoint_exists_after": checkpoint_exists,
        },
    )


def _adapter_stage(adapter_dir: Path) -> StageResult:
    if not adapter_dir.exists():
        return _stage(
            "adapter_save",
            "missing",
            "adapter directory is missing",
            {"adapter_dir": _artifact(adapter_dir)},
        )
    weight_files = sorted(
        [
            *adapter_dir.rglob("adapter_model*.safetensors"),
            *adapter_dir.rglob("adapter_model*.bin"),
        ],
    )
    config_files = sorted(adapter_dir.rglob("adapter_config.json"))
    status = "pass" if weight_files and config_files else "missing"
    return _stage(
        "adapter_save",
        status,
        (
            "adapter config and weights exist in retained storage"
            if status == "pass"
            else "adapter config or weights are missing"
        ),
        {
            "adapter_dir": _artifact(adapter_dir),
            "weight_files": [_artifact(path) for path in weight_files],
            "config_files": [_artifact(path) for path in config_files],
        },
    )


def _reload_stage(results_dir: Path) -> StageResult:
    report_path = results_dir / "reload-check.json"
    report = load_json(report_path)
    if not report:
        return _stage(
            "clean_process_reload",
            "missing",
            "reload-check report is missing",
            {"reload_check": _artifact(report_path)},
        )
    deterministic = bool(report.get("deterministic"))
    records_checked = int(report.get("records_checked", 0) or 0)
    status = "pass" if deterministic and records_checked > 0 else "fail"
    return _stage(
        "clean_process_reload",
        status,
        (
            "adapter reload report is deterministic"
            if status == "pass"
            else "adapter reload report failed deterministic check"
        ),
        {
            "reload_check": _artifact(report_path),
            "process_id": report.get("process_id"),
            "records_checked": records_checked,
            "deterministic": deterministic,
            "load_in_4bit": report.get("load_in_4bit"),
        },
    )


def _canonical_bundle_stage(results_dir: Path) -> StageResult:
    artifacts = {name: _artifact(results_dir / name) for name in REQUIRED_FINAL_RESULT_FILES}
    missing = [name for name, artifact in artifacts.items() if not artifact["exists"]]
    status = "pass" if not missing else "missing"
    return _stage(
        "canonical_artifact_bundle",
        status,
        (
            "all canonical final result files are present"
            if status == "pass"
            else "canonical final result files are missing"
        ),
        {"artifacts": artifacts, "missing": missing},
    )


def _memory_stage(results_dir: Path, baseline_dir: Path) -> StageResult:
    candidates = {
        "untouched_base_generation": load_json(
            baseline_dir / "generation_metadata.json",
        ),
        "smoke_training": load_json(results_dir / "training_metrics.json"),
        "adapter_loaded_evaluation": load_json(
            results_dir / "generation_metadata.json",
        ),
    }
    missing: list[str] = []
    evidence: dict[str, Any] = {}
    for stage_name, payload in candidates.items():
        allocated = payload.get("peak_allocated_vram_gb")
        reserved = payload.get("peak_reserved_vram_gb")
        evidence[stage_name] = {
            "peak_allocated_vram_gb": allocated,
            "peak_reserved_vram_gb": reserved,
        }
        if allocated is None or reserved is None:
            missing.append(stage_name)
    status = "pass" if not missing else "missing"
    return _stage(
        "peak_allocated_reserved_vram",
        status,
        (
            "peak allocated/reserved VRAM is recorded for all GPU stages"
            if status == "pass"
            else "peak allocated/reserved VRAM is missing for one or more GPU stages"
        ),
        {"measurements": evidence, "missing": missing},
    )


def _s3_stage(
    *,
    s3_inventory: dict[str, Any] | None,
    required_keys: tuple[str, ...],
) -> StageResult:
    if s3_inventory is None:
        return _stage(
            "s3_artifact_upload",
            "blocked",
            "S3 inventory was not available; AWS auth may be required",
            {"required_keys": list(required_keys)},
        )
    contents = s3_inventory.get("Contents", s3_inventory)
    if not isinstance(contents, list):
        return _stage(
            "s3_artifact_upload",
            "fail",
            "S3 inventory has an unexpected shape",
            {"inventory_keys": list(s3_inventory.keys())},
        )
    keys = {
        str(item.get("Key"))
        for item in contents
        if isinstance(item, dict) and item.get("Key")
    }
    missing = [
        required
        for required in required_keys
        if not any(key.endswith(required) for key in keys)
    ]
    status = "pass" if not missing else "missing"
    return _stage(
        "s3_artifact_upload",
        status,
        (
            "required S3 artifact names are present in inventory"
            if status == "pass"
            else "required S3 artifact names are missing from inventory"
        ),
        {
            "object_count": len(keys),
            "missing_required_suffixes": missing,
            "required_suffixes": list(required_keys),
        },
    )


def _instance_stage(instance_state: str | None) -> StageResult:
    if instance_state is None:
        return _stage(
            "ec2_instance_stopped",
            "blocked",
            "instance state was not available; AWS auth may be required",
        )
    status = "pass" if instance_state == "stopped" else "fail"
    return _stage(
        "ec2_instance_stopped",
        status,
        (
            "EC2 instance is stopped"
            if status == "pass"
            else f"EC2 instance is {instance_state!r}, not stopped"
        ),
        {"instance_state": instance_state},
    )


def build_completion_report(
    *,
    dataset_path: Path,
    results_dir: Path,
    baseline_results_dir: Path,
    logs_dir: Path,
    run_info_dir: Path,
    adapter_dir: Path,
    template_report_path: Path,
    loss_mask_report_path: Path,
    s3_inventory: dict[str, Any] | None = None,
    instance_state: str | None = None,
) -> dict[str, Any]:
    stages = [
        _environment_stage(results_dir=results_dir, run_info_dir=run_info_dir),
        _model_load_stage(results_dir),
        _template_stage(template_report_path),
        _loss_mask_stage(loss_mask_report_path),
        _score_stage(
            name="untouched_base_generation_40",
            dataset_path=dataset_path,
            predictions_path=baseline_results_dir / "predictions.jsonl",
            scores_path=baseline_results_dir / "scores.json",
            expected_records=40,
        ),
        _training_stage(results_dir, logs_dir),
        _adapter_stage(adapter_dir),
        _reload_stage(results_dir),
        _score_stage(
            name="post_training_generation_parse_score_40",
            dataset_path=dataset_path,
            predictions_path=results_dir / "predictions.jsonl",
            scores_path=results_dir / "scores.json",
            expected_records=40,
        ),
        _memory_stage(results_dir, baseline_results_dir),
        _canonical_bundle_stage(results_dir),
        _s3_stage(
            s3_inventory=s3_inventory,
            required_keys=REQUIRED_S3_SUFFIXES,
        ),
        _instance_stage(instance_state),
    ]
    stage_dicts = [
        {
            "name": stage.name,
            "status": stage.status,
            "summary": stage.summary,
            "evidence": stage.evidence,
        }
        for stage in stages
    ]
    blockers = [
        stage.summary
        for stage in stages
        if stage.status in {"missing", "fail", "blocked"}
    ]
    complete = not blockers
    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "experiment_id": "exp-00",
        "generated_at": utc_now(),
        "overall_status": "complete" if complete else "incomplete",
        "may_proceed_to_later_experiments": complete,
        "stages": stage_dicts,
        "blockers": blockers,
    }


def write_completion_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Experiment 0 Completion",
        "",
        f"- Status: `{report['overall_status']}`",
        f"- May proceed: `{report['may_proceed_to_later_experiments']}`",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Stage Status",
        "",
        "| Stage | Status | Summary |",
        "| --- | --- | --- |",
    ]
    for stage in report.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.append(
            "| {name} | `{status}` | {summary} |".format(
                name=stage.get("name"),
                status=stage.get("status"),
                summary=str(stage.get("summary", "")).replace("|", "\\|"),
            ),
        )

    blockers = report.get("blockers", [])
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
