#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation_compare import DEFAULT_METRICS
from function_calling_ft.generation import read_jsonl
from function_calling_ft.reference_lora import (
    EXPECTED_VALIDATION_PATH,
    EXP08_SAMPLE_PROFILES,
    LoraSampleProfile,
    clone_training_config_for_stage,
    load_yaml_config,
    validate_lora_sample_efficiency_config,
    validation_to_dict,
)
from function_calling_ft.split_guard import assert_split_allowed
from scripts.run_exp03_reference_lora import (
    _full_epoch_steps,
    _run_reload_check,
    _run_training_stage,
)
from scripts.run_exp06_lora_rank import (
    AGGREGATE_METRICS,
    _compare_pair,
    _controlled_config_view,
    _generate_and_score,
    _read_json,
    _requested_metric_value,
    _run_command,
    _score_value,
    _sha256_file,
    _write_json,
)


DEFAULT_CONFIG_ROOT = Path("configs/exp08_sample_efficiency")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-08")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-08")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-08")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")
DEFAULT_VALIDATION_MANIFEST = Path(
    "/workspace/data/processed/xlam_splits_v1/manifests/validation_manifest.jsonl",
)
DEFAULT_REFERENCE_TRAIN10K_RESULTS_ROOT = Path("/workspace/results/exp-06/rank4-alpha8")
DEFAULT_REFERENCE_TRAIN10K_RESOLVED_CONFIG = (
    DEFAULT_REFERENCE_TRAIN10K_RESULTS_ROOT / "full-epoch" / "resolved_config.yaml"
)
DEFAULT_REFERENCE_TRAIN10K_CHECKPOINT = Path(
    "/workspace/checkpoints/exp-06/rank4-alpha8/full-epoch",
)
SAMPLE_CONFIGS = {
    "train_2k": DEFAULT_CONFIG_ROOT / "train_2k.yaml",
    "train_10k": DEFAULT_CONFIG_ROOT / "train_10k.yaml",
    "train_full": DEFAULT_CONFIG_ROOT / "train_full.yaml",
}
PAIRWISE_METRICS = DEFAULT_METRICS + (
    "tool_call_emitted",
    "no_tool_false_positive",
)
EXEC_TIE_THRESHOLD = 0.01
CALL_F1_TIE_THRESHOLD = 0.005
HARD_BUCKET_LOSS_LIMIT = 0.02
NO_TOOL_FP_WORSENING_LIMIT = 0.05
MIN_HARD_BUCKET_RECORDS = 50
FLOAT_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class SampleRun:
    profile: LoraSampleProfile
    config_path: Path
    run_id: str

    @property
    def stage_name(self) -> str:
        return self.profile.stage_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 08 dataset-size sample-efficiency comparison.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--automodel-bin", default="automodel")
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-train10k",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reference-train10k-resolved-config",
        type=Path,
        default=DEFAULT_REFERENCE_TRAIN10K_RESOLVED_CONFIG,
    )
    parser.add_argument(
        "--reference-train10k-results-root",
        type=Path,
        default=DEFAULT_REFERENCE_TRAIN10K_RESULTS_ROOT,
    )
    parser.add_argument(
        "--reference-train10k-checkpoint",
        type=Path,
        default=DEFAULT_REFERENCE_TRAIN10K_CHECKPOINT,
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path(EXPECTED_VALIDATION_PATH),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST,
    )
    parser.add_argument("--no-tool-dev-dataset", type=Path, default=DEFAULT_NO_TOOL_DEV)
    parser.add_argument(
        "--hourly-cost-usd",
        type=float,
        default=None,
        help="Optional instance hourly price for rough run-cost reporting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _sample_runs() -> list[SampleRun]:
    return [
        SampleRun(
            profile=profile,
            config_path=SAMPLE_CONFIGS[profile.name],
            run_id=f"bf16-lora-r{profile.rank}-alpha{profile.alpha}-attention-{profile.stage_name}",
        )
        for profile in EXP08_SAMPLE_PROFILES
    ]


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("example_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("record must contain a non-empty id or example_id")
    return value


def _jsonl_ids(path: Path) -> set[str]:
    return {_record_id(record) for record in _read_jsonl(path)}


def _sum_manifest_int(path: Path, field: str) -> int | None:
    if not path.is_file():
        return None
    total = 0
    for record in _read_jsonl(path):
        value = record.get(field)
        if isinstance(value, int):
            total += value
        elif isinstance(value, float):
            total += int(value)
    return total


def _manifest_record_count(path: Path) -> int | None:
    return _dataset_count(path) if path.is_file() else None


def _validate_sample_configs(results_root: Path) -> dict[str, Any]:
    validations = {}
    for run in _sample_runs():
        validation = validate_lora_sample_efficiency_config(
            run.config_path,
            sample_profile=run.profile.name,
        )
        validations[run.profile.name] = validation_to_dict(validation)
    _write_json(results_root / "config_validation.json", validations)
    errors = [
        f"{name}: {error}"
        for name, payload in validations.items()
        for error in payload["errors"]
    ]
    if errors:
        raise ValueError("; ".join(errors))
    return validations


def verify_nested_split_paths(
    *,
    train_2k: Path,
    train_10k: Path,
    train_full: Path,
) -> dict[str, Any]:
    paths = {
        "train_2k": train_2k,
        "train_10k": train_10k,
        "train_full": train_full,
    }
    existing = {name: path.is_file() for name, path in paths.items()}
    if not all(existing.values()):
        return {
            "schema_version": "1.0",
            "status": "skipped_missing_files",
            "paths": {name: str(path) for name, path in paths.items()},
            "existing": existing,
            "ok": False,
        }

    ids_2k = _jsonl_ids(train_2k)
    ids_10k = _jsonl_ids(train_10k)
    ids_full = _jsonl_ids(train_full)
    duplicate_counts = {
        "train_2k": _dataset_count(train_2k) - len(ids_2k),
        "train_10k": _dataset_count(train_10k) - len(ids_10k),
        "train_full": _dataset_count(train_full) - len(ids_full),
    }
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "paths": {name: str(path) for name, path in paths.items()},
        "record_counts": {
            "train_2k": len(ids_2k),
            "train_10k": len(ids_10k),
            "train_full": len(ids_full),
        },
        "duplicate_counts": duplicate_counts,
        "train_2k_subset_train_10k": ids_2k.issubset(ids_10k),
        "train_10k_subset_train_full": ids_10k.issubset(ids_full),
    }
    report["ok"] = (
        report["train_2k_subset_train_10k"]
        and report["train_10k_subset_train_full"]
        and all(count == 0 for count in duplicate_counts.values())
    )
    return report


def _reference_train10k_reuse_report(
    *,
    train10k_run: SampleRun,
    reference_resolved_config: Path,
    reference_checkpoint: Path,
    reference_results_root: Path,
    train_count: int | None,
    global_batch_size: int,
    local_batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "sample_profile": "train_10k",
        "reuse_requested": True,
        "reference_resolved_config": str(reference_resolved_config),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_results_root": str(reference_results_root),
        "reference_resolved_config_exists": reference_resolved_config.is_file(),
        "reference_checkpoint_exists": reference_checkpoint.exists(),
        "reference_results_root_exists": reference_results_root.exists(),
        "eligible": False,
        "reasons": [],
    }
    reasons: list[str] = []
    if not reference_resolved_config.is_file():
        reasons.append("missing_reference_resolved_config")
    if not reference_checkpoint.exists():
        reasons.append("missing_reference_checkpoint")
    if not reference_results_root.exists():
        reasons.append("missing_reference_results_root")
    if train_count is None:
        reasons.append("missing_train_count")
    if reasons:
        report["reasons"] = reasons
        return report

    assert train_count is not None
    train10k_config = load_yaml_config(train10k_run.config_path)
    full_steps = _full_epoch_steps(train_count, global_batch_size)
    expected = clone_training_config_for_stage(
        train10k_config,
        checkpoint_dir=str(reference_checkpoint),
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        max_steps=full_steps,
        ckpt_every_steps=max(1, full_steps // 4),
        val_every_steps=max(1, full_steps // 4),
        validation_path=None,
        checkpoint_enabled=True,
    )
    actual = load_yaml_config(reference_resolved_config)
    expected_view = _controlled_config_view(expected)
    actual_view = _controlled_config_view(actual)
    mismatches = {
        key: {
            "expected": expected_view.get(key),
            "actual": actual_view.get(key),
        }
        for key in sorted(expected_view)
        if expected_view.get(key) != actual_view.get(key)
    }
    if mismatches:
        reasons.append("controlled_config_mismatch")

    required_eval = {
        dataset_name: all(
            (reference_results_root / "eval" / dataset_name / filename).is_file()
            for filename in (
                "predictions.jsonl",
                "scored_predictions.jsonl",
                "scores.json",
            )
        )
        for dataset_name in ("validation", "no_tool_dev")
    }
    if not all(required_eval.values()):
        reasons.append("missing_reference_eval_artifacts")

    report.update(
        {
            "expected_full_steps": full_steps,
            "controlled_config_mismatches": mismatches,
            "required_eval_artifacts_present": required_eval,
            "eligible": not reasons,
            "reasons": reasons,
        },
    )
    return report


def _copy_directory(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def _copy_reused_train10k_artifacts(
    *,
    reference_results_root: Path,
    destination_root: Path,
    reuse_report: Mapping[str, Any],
) -> None:
    stage_root = destination_root / "train-10k"
    for relative in (
        Path("full-epoch"),
        Path("eval/validation"),
        Path("eval/no_tool_dev"),
        Path("lora_target_inspection.json"),
    ):
        _copy_directory(reference_results_root / relative, stage_root / relative)
    _write_json(stage_root / "reuse_train10k_reference.json", reuse_report)


def _train_sample(
    *,
    run: SampleRun,
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    train_count: int,
    global_batch_size: int,
    local_batch_size: int,
    cache_dir: Path,
    dry_run: bool,
) -> Path:
    base_config = load_yaml_config(run.config_path)
    full_steps = _full_epoch_steps(train_count, global_batch_size)
    metrics = _run_training_stage(
        stage_name=f"{run.stage_name}/full-epoch",
        base_config=base_config,
        results_root=results_root,
        logs_root=logs_root,
        checkpoint_root=checkpoint_root,
        automodel_bin=automodel_bin,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        max_steps=full_steps,
        ckpt_every_steps=max(1, full_steps // 4),
        val_every_steps=max(1, full_steps // 4),
        validation_path=None,
        checkpoint_enabled=True,
        memory_trace=False,
        dry_run=dry_run,
        patch_qlora_peft_state_dict=False,
        validator=partial(
            validate_lora_sample_efficiency_config,
            sample_profile=run.profile.name,
        ),
    )
    supervised_tokens = _sum_manifest_int(Path(run.profile.manifest_path), "supervised_target_tokens")
    _write_json(
        results_root / run.stage_name / "full-epoch" / "sample_training_summary.json",
        {
            "schema_version": "1.0",
            "sample_profile": run.profile.name,
            "train_records": train_count,
            "supervised_target_tokens": supervised_tokens,
            "full_steps": full_steps,
            "metrics": metrics,
        },
    )
    adapter_path = checkpoint_root / run.stage_name / "full-epoch"
    _run_reload_check(
        adapter_path=adapter_path,
        results_root=results_root / run.stage_name,
        logs_root=logs_root / run.stage_name,
        cache_dir=cache_dir,
        dry_run=dry_run,
        load_in_4bit=False,
        stage_name="full-epoch",
    )
    return adapter_path


def _inspect_targets(
    *,
    run: SampleRun,
    results_root: Path,
    logs_root: Path,
    cache_dir: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/inspect_lora_targets.py",
        "--config",
        str(run.config_path),
        "--output",
        str(results_root / run.stage_name / "lora_target_inspection.json"),
        "--cache-dir",
        str(cache_dir),
        "--sample-profile",
        run.profile.name,
    ]
    _run_command(
        command,
        log_path=logs_root / run.stage_name / "target-inspection.log",
        dry_run=dry_run,
    )


def _training_metrics(results_root: Path, run: SampleRun) -> dict[str, Any]:
    return _read_json(results_root / run.stage_name / "full-epoch" / "training_metrics.json")


def _completed_training_stage(
    *,
    run: SampleRun,
    results_root: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    metrics_path = results_root / run.stage_name / "full-epoch" / "training_metrics.json"
    adapter_path = checkpoint_root / run.stage_name / "full-epoch"
    metrics = _read_json(metrics_path)
    complete = (
        metrics.get("return_code") == 0
        and bool(metrics.get("losses_are_finite", False))
        and bool(metrics.get("checkpoint_exists_after", False))
        and adapter_path.exists()
    )
    return {
        "schema_version": "1.0",
        "sample_profile": run.profile.name,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(adapter_path),
        "metrics_exists": metrics_path.is_file(),
        "checkpoint_exists": adapter_path.exists(),
        "complete": complete,
        "return_code": metrics.get("return_code"),
        "losses_are_finite": metrics.get("losses_are_finite"),
        "checkpoint_exists_after": metrics.get("checkpoint_exists_after"),
    }


def _schema_complexity_bucket_metrics(
    *,
    scored_path: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    if not scored_path.is_file() or not validation_manifest.is_file():
        return {}
    manifest_by_id = {
        str(record.get("example_id")): record
        for record in _read_jsonl(validation_manifest)
        if isinstance(record.get("example_id"), str)
    }
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(scored_path):
        record_id = str(record.get("id", ""))
        manifest = manifest_by_id.get(record_id, {})
        bucket = str(manifest.get("schema_complexity_bucket", "unknown"))
        buckets.setdefault(bucket, []).append(record)
    return {
        bucket: _compact_records(records)
        for bucket, records in sorted(buckets.items())
    }


def _compact_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    expected_calls = sum(
        int(record.get("call_metrics", {}).get("expected_call_count", 0) or 0)
        for record in records
    )
    predicted_calls = sum(
        int(record.get("call_metrics", {}).get("predicted_call_count", 0) or 0)
        for record in records
    )
    strict_complete_calls = sum(
        int(record.get("call_metrics", {}).get("strict_complete_call_count", 0) or 0)
        for record in records
    )
    executable_records = sum(
        int(record.get("headline_scores", {}).get("executable_complete_match", False))
        for record in records
    )
    missing_calls = sum(
        int(record.get("call_metrics", {}).get("missing_call_count", 0) or 0)
        for record in records
    )
    extra_calls = sum(
        int(record.get("call_metrics", {}).get("extra_call_count", 0) or 0)
        for record in records
    )
    precision = (
        strict_complete_calls / predicted_calls
        if predicted_calls
        else None
    )
    recall = (
        strict_complete_calls / expected_calls
        if expected_calls
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "total_records": total_records,
        "expected_call_count": expected_calls,
        "predicted_call_count": predicted_calls,
        "executable_complete_match_count": executable_records,
        "executable_complete_match_rate": (
            executable_records / total_records if total_records else None
        ),
        "complete_call_precision": precision,
        "complete_call_recall": recall,
        "complete_call_f1": f1,
        "missing_call_count": missing_calls,
        "extra_call_count": extra_calls,
    }


def _evaluation_bucket_report(
    *,
    results_root: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for run in _sample_runs():
        validation_root = results_root / run.stage_name / "eval" / "validation"
        scores = _read_json(validation_root / "scores.json")
        metrics_by_group = scores.get("metrics_by_group")
        metrics_by_group = metrics_by_group if isinstance(metrics_by_group, dict) else {}
        report[run.profile.name] = {
            "call_category": metrics_by_group.get("call_category", {}),
            "seen_status": metrics_by_group.get("seen_status", {}),
            "expected_call_count_bucket": metrics_by_group.get("expected_call_count_bucket", {}),
            "tool_count_bucket": metrics_by_group.get("tool_count_bucket", {}),
            "length_bucket": metrics_by_group.get("length_bucket", {}),
            "primary_api_category": metrics_by_group.get("primary_api_category", {}),
            "schema_complexity_bucket": _schema_complexity_bucket_metrics(
                scored_path=validation_root / "scored_predictions.jsonl",
                validation_manifest=validation_manifest,
            ),
        }
    return report


def _extract_bucket_exec(
    bucket_report: Mapping[str, Any],
    *,
    profile: str,
    dimension: str,
    bucket: str,
) -> tuple[float | None, int]:
    profile_report = bucket_report.get(profile)
    if not isinstance(profile_report, Mapping):
        return None, 0
    dimension_report = profile_report.get(dimension)
    if not isinstance(dimension_report, Mapping):
        return None, 0
    bucket_report_value = dimension_report.get(bucket)
    if not isinstance(bucket_report_value, Mapping):
        return None, 0
    value = bucket_report_value.get("executable_complete_match_rate")
    total = int(bucket_report_value.get("total_records", 0) or 0)
    return float(value) if isinstance(value, int | float) else None, total


def _hard_bucket_guardrails(bucket_report: Mapping[str, Any]) -> dict[str, Any]:
    hard_buckets = [
        ("call_category", "multiple"),
        ("call_category", "parallel"),
        ("call_category", "multiple_parallel"),
        ("seen_status", "unseen_family"),
        ("seen_status", "unseen"),
        ("schema_complexity_bucket", "complex"),
        ("schema_complexity_bucket", "very_complex"),
    ]
    profiles = [profile.name for profile in EXP08_SAMPLE_PROFILES]
    checks: list[dict[str, Any]] = []
    ok_by_profile = {profile: True for profile in profiles}
    for dimension, bucket in hard_buckets:
        values = {
            profile: _extract_bucket_exec(
                bucket_report,
                profile=profile,
                dimension=dimension,
                bucket=bucket,
            )
            for profile in profiles
        }
        usable = {
            profile: value
            for profile, value in values.items()
            if value[0] is not None and value[1] >= MIN_HARD_BUCKET_RECORDS
        }
        if len(usable) < 2:
            continue
        best = max(float(value[0]) for value in usable.values() if value[0] is not None)
        for profile, (value, total) in usable.items():
            assert value is not None
            delta_from_best = best - value
            passed = delta_from_best <= HARD_BUCKET_LOSS_LIMIT
            ok_by_profile[profile] = ok_by_profile[profile] and passed
            checks.append(
                {
                    "profile": profile,
                    "dimension": dimension,
                    "bucket": bucket,
                    "total_records": total,
                    "executable_complete_match_rate": value,
                    "best_rate": best,
                    "delta_from_best": delta_from_best,
                    "passed": passed,
                },
            )
    return {
        "schema_version": "1.0",
        "min_bucket_records": MIN_HARD_BUCKET_RECORDS,
        "loss_limit_absolute": HARD_BUCKET_LOSS_LIMIT,
        "checks": checks,
        "ok_by_profile": ok_by_profile,
    }


def _aggregate_table(
    *,
    results_root: Path,
    hourly_cost_usd: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _sample_runs():
        train_metrics = _training_metrics(results_root, run)
        duration_seconds = train_metrics.get("duration_seconds")
        gpu_hours = (
            float(duration_seconds) / 3600.0
            if isinstance(duration_seconds, int | float)
            else None
        )
        estimated_cost = (
            gpu_hours * hourly_cost_usd
            if gpu_hours is not None and hourly_cost_usd is not None
            else None
        )
        train_records = _manifest_record_count(Path(run.profile.manifest_path))
        supervised_tokens = _sum_manifest_int(
            Path(run.profile.manifest_path),
            "supervised_target_tokens",
        )
        for dataset_name in ("validation", "no_tool_dev"):
            eval_root = results_root / run.stage_name / "eval" / dataset_name
            scores = _read_json(eval_root / "scores.json")
            requested = eval_root / "requested_metrics.json"
            row: dict[str, Any] = {
                "sample_profile": run.profile.name,
                "stage_name": run.stage_name,
                "train_records": train_records,
                "supervised_target_tokens": supervised_tokens,
                "dataset": dataset_name,
                "scores_path": str(eval_root / "scores.json"),
                "requested_metrics_path": str(requested),
                "training_duration_seconds": duration_seconds,
                "training_gpu_hours": gpu_hours,
                "estimated_training_cost_usd": estimated_cost,
                "peak_reserved_vram_gb": train_metrics.get("peak_reserved_vram_gb"),
                "peak_allocated_vram_gb": train_metrics.get("peak_allocated_vram_gb"),
                "average_gpu_utilization_pct": train_metrics.get("average_gpu_utilization_pct"),
                "average_step_time_seconds": train_metrics.get("average_step_time_seconds"),
                "trainable_parameter_count": train_metrics.get("trainable_parameter_count"),
            }
            for metric in AGGREGATE_METRICS:
                row[metric] = _score_value(scores, metric)
            row["protocol_clean_response_rate"] = _requested_metric_value(
                requested,
                "protocol_clean_response_rate",
            )
            row["no_tool_false_positive_rate"] = _requested_metric_value(
                requested,
                "no_tool_false_positive_rate",
            )
            rows.append(row)
    return rows


def _select_dataset_size(
    *,
    rows: list[dict[str, Any]],
    hard_bucket_guardrails: Mapping[str, Any],
) -> dict[str, Any]:
    validation_rows = [
        row
        for row in rows
        if row["dataset"] == "validation"
        and isinstance(row.get("executable_complete_match_rate"), int | float)
        and isinstance(row.get("complete_call_f1"), int | float)
    ]
    if not validation_rows:
        return {
            "status": "insufficient_metrics",
            "selected_sample_profile": None,
            "reason": "No validation rows with executable complete and complete-call F1.",
        }
    no_tool_rows = {
        str(row["sample_profile"]): row
        for row in rows
        if row["dataset"] == "no_tool_dev"
    }
    best_exec = max(float(row["executable_complete_match_rate"]) for row in validation_rows)
    best_call_f1 = max(float(row["complete_call_f1"]) for row in validation_rows)
    no_tool_values = [
        float(row["no_tool_false_positive_rate"])
        for row in no_tool_rows.values()
        if isinstance(row.get("no_tool_false_positive_rate"), int | float)
    ]
    best_no_tool_fp = min(no_tool_values) if no_tool_values else None
    hard_ok = hard_bucket_guardrails.get("ok_by_profile")
    hard_ok = hard_ok if isinstance(hard_ok, Mapping) else {}
    candidates: list[dict[str, Any]] = []
    for row in validation_rows:
        profile = str(row["sample_profile"])
        no_tool_row = no_tool_rows.get(profile, {})
        no_tool_fp = no_tool_row.get("no_tool_false_positive_rate")
        no_tool_pass = (
            True
            if best_no_tool_fp is None or not isinstance(no_tool_fp, int | float)
            else float(no_tool_fp) - best_no_tool_fp <= NO_TOOL_FP_WORSENING_LIMIT
        )
        primary_pass = (
            best_exec - float(row["executable_complete_match_rate"])
            <= EXEC_TIE_THRESHOLD + FLOAT_TOLERANCE
            and best_call_f1 - float(row["complete_call_f1"])
            <= CALL_F1_TIE_THRESHOLD + FLOAT_TOLERANCE
        )
        hard_pass = bool(hard_ok.get(profile, True))
        candidate = {
            "sample_profile": profile,
            "train_records": row.get("train_records"),
            "supervised_target_tokens": row.get("supervised_target_tokens"),
            "executable_complete_match_rate": row.get("executable_complete_match_rate"),
            "complete_call_f1": row.get("complete_call_f1"),
            "no_tool_false_positive_rate": no_tool_fp,
            "primary_pass": primary_pass,
            "hard_bucket_pass": hard_pass,
            "no_tool_pass": no_tool_pass,
            "eligible": primary_pass and hard_pass and no_tool_pass,
        }
        candidates.append(candidate)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("executable_complete_match_rate") or 0.0),
                float(item.get("complete_call_f1") or 0.0),
            ),
        )
        status = "no_profile_met_all_guardrails"
    else:
        selected = min(
            eligible,
            key=lambda item: int(item.get("train_records") or 10**12),
        )
        status = "selected"
    return {
        "schema_version": "1.0",
        "status": status,
        "selected_sample_profile": selected["sample_profile"],
        "best_executable_complete_match_rate": best_exec,
        "best_complete_call_f1": best_call_f1,
        "best_no_tool_false_positive_rate": best_no_tool_fp,
        "executable_complete_tie_threshold_absolute": EXEC_TIE_THRESHOLD,
        "complete_call_f1_tie_threshold_absolute": CALL_F1_TIE_THRESHOLD,
        "hard_bucket_loss_limit_absolute": HARD_BUCKET_LOSS_LIMIT,
        "no_tool_false_positive_worsening_limit_absolute": NO_TOOL_FP_WORSENING_LIMIT,
        "selection_rule": (
            "Choose the smallest dataset within 1pp executable-complete accuracy "
            "and 0.5pp complete-call F1 of the best validation result, while "
            "passing hard-bucket and no-tool false-positive guardrails."
        ),
        "candidates": candidates,
    }


def _write_checksum_manifest(root: Path) -> Path:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rel = path.relative_to(root)
            lines.append(f"{_sha256_file(path)}  {rel.as_posix()}")
    output = root / "checksums.sha256"
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def _write_markdown_summary(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    validation_rows = [row for row in rows if row["dataset"] == "validation"]
    no_tool_rows = [row for row in rows if row["dataset"] == "no_tool_dev"]
    lines = [
        "# Experiment 8 Sample Efficiency",
        "",
        f"Selected dataset size: `{decision.get('selected_sample_profile')}`",
        "",
        "## Validation Tool-Calling Metrics",
        "",
        "| Training set | Records | Supervised tokens | Exec complete | Complete-call F1 | Fn F1 | Arg value acc | Missing calls | Extra calls | Malformed | GPU-hours |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(validation_rows, key=lambda item: int(item.get("train_records") or 0)):
        lines.append(
            "| {profile} | {records} | {tokens} | {exec_rate:.4f} | {call_f1:.4f} | "
            "{fn_f1:.4f} | {arg_value:.4f} | {missing} | {extra} | {malformed} | {gpu_hours:.3f} |".format(
                profile=row["sample_profile"],
                records=row.get("train_records"),
                tokens=row.get("supervised_target_tokens"),
                exec_rate=float(row.get("executable_complete_match_rate") or 0.0),
                call_f1=float(row.get("complete_call_f1") or 0.0),
                fn_f1=float(row.get("function_name_f1") or 0.0),
                arg_value=float(row.get("average_argument_value_accuracy") or 0.0),
                missing=row.get("missing_call_count"),
                extra=row.get("extra_call_count"),
                malformed=row.get("malformed_tool_call_count"),
                gpu_hours=float(row.get("training_gpu_hours") or 0.0),
            ),
        )
    lines.extend(
        [
            "",
            "## Development No-Tool Regression",
            "",
            "| Training set | No-tool false positive | Protocol clean | Tool calls emitted |",
            "| --- | ---: | ---: | ---: |",
        ],
    )
    for row in sorted(no_tool_rows, key=lambda item: int(item.get("train_records") or 0)):
        lines.append(
            "| {profile} | {fp:.4f} | {clean:.4f} | {emitted} |".format(
                profile=row["sample_profile"],
                fp=float(row.get("no_tool_false_positive_rate") or 0.0),
                clean=float(row.get("protocol_clean_response_rate") or 0.0),
                emitted=row.get("tool_call_emitted_count"),
            ),
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_artifact_hashes(
    *,
    results_root: Path,
    sample_runs: list[SampleRun],
    validation_dataset: Path,
    validation_manifest: Path,
    no_tool_dev: Path,
) -> None:
    paths = {
        "validation_dataset": validation_dataset,
        "validation_manifest": validation_manifest,
        "no_tool_dev_dataset": no_tool_dev,
    }
    for run in sample_runs:
        paths[f"{run.profile.name}_config"] = run.config_path
        paths[f"{run.profile.name}_train_dataset"] = Path(run.profile.train_path)
        paths[f"{run.profile.name}_manifest"] = Path(run.profile.manifest_path)
    hashes = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for name, path in paths.items()
    }
    _write_json(results_root / "artifact_hashes.json", hashes)


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    sample_runs = _sample_runs()
    split_decisions = {
        run.profile.name: assert_split_allowed(
            run.profile.train_path,
            command_name=f"exp08-sample-efficiency-{run.profile.name}",
        ).__dict__
        for run in sample_runs
    }
    validation_decision = assert_split_allowed(
        args.validation_dataset,
        command_name="exp08-sample-efficiency-validation",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev_dataset,
        command_name="exp08-sample-efficiency-no-tool",
    )
    validations = _validate_sample_configs(args.results_root)
    train_counts = {
        run.profile.name: (
            _dataset_count(Path(run.profile.train_path))
            if Path(run.profile.train_path).is_file()
            else None
        )
        for run in sample_runs
    }
    supervised_tokens = {
        run.profile.name: _sum_manifest_int(
            Path(run.profile.manifest_path),
            "supervised_target_tokens",
        )
        for run in sample_runs
    }
    manifest_counts = {
        run.profile.name: _manifest_record_count(Path(run.profile.manifest_path))
        for run in sample_runs
    }
    validation_count = (
        _dataset_count(args.validation_dataset)
        if args.validation_dataset.is_file()
        else None
    )
    no_tool_count = (
        _dataset_count(args.no_tool_dev_dataset)
        if args.no_tool_dev_dataset.is_file()
        else None
    )
    nesting_report = verify_nested_split_paths(
        train_2k=Path(EXP08_SAMPLE_PROFILES[0].train_path),
        train_10k=Path(EXP08_SAMPLE_PROFILES[1].train_path),
        train_full=Path(EXP08_SAMPLE_PROFILES[2].train_path),
    )
    train10k_run = next(run for run in sample_runs if run.profile.name == "train_10k")
    reuse_report = (
        _reference_train10k_reuse_report(
            train10k_run=train10k_run,
            reference_resolved_config=args.reference_train10k_resolved_config,
            reference_checkpoint=args.reference_train10k_checkpoint,
            reference_results_root=args.reference_train10k_results_root,
            train_count=train_counts["train_10k"],
            global_batch_size=args.global_batch_size,
            local_batch_size=args.local_batch_size,
        )
        if args.reuse_train10k
        else {
            "schema_version": "1.0",
            "sample_profile": "train_10k",
            "reuse_requested": False,
            "eligible": False,
            "reasons": ["reuse_disabled"],
        }
    )

    run_plan = {
        "schema_version": "1.0",
        "experiment_id": "exp-08",
        "task_id": "task-13",
        "selected_peft_method": "bf16_lora",
        "selected_rank": 4,
        "selected_alpha": 8,
        "selected_target_profile": "attention",
        "primary_protocol": "one_epoch_per_dataset_size",
        "fixed_token_secondary_analysis": "skipped",
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "train_splits": split_decisions,
        "validation_split": validation_decision.__dict__,
        "no_tool_split": no_tool_decision.__dict__,
        "train_records": train_counts,
        "manifest_records": manifest_counts,
        "supervised_target_tokens": supervised_tokens,
        "validation_records": validation_count,
        "no_tool_dev_records": no_tool_count,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "sample_profiles": [
            {
                "sample_profile": run.profile.name,
                "config": str(run.config_path),
                "run_id": run.run_id,
                "train_path": run.profile.train_path,
                "manifest_path": run.profile.manifest_path,
            }
            for run in sample_runs
        ],
        "nesting_report": nesting_report,
        "train_10k_reuse": reuse_report,
        "config_validation": validations,
        "selection_rule": {
            "executable_complete_tie_threshold_absolute": EXEC_TIE_THRESHOLD,
            "complete_call_f1_tie_threshold_absolute": CALL_F1_TIE_THRESHOLD,
            "hard_bucket_loss_limit_absolute": HARD_BUCKET_LOSS_LIMIT,
            "no_tool_false_positive_worsening_limit_absolute": NO_TOOL_FP_WORSENING_LIMIT,
        },
    }
    _write_json(args.results_root / "run_plan.json", run_plan)
    _write_artifact_hashes(
        results_root=args.results_root,
        sample_runs=sample_runs,
        validation_dataset=args.validation_dataset,
        validation_manifest=args.validation_manifest,
        no_tool_dev=args.no_tool_dev_dataset,
    )

    if args.validate_only:
        print("exp08_validation_ok=true")
        return
    if not args.dry_run:
        missing_counts = [
            name for name, count in train_counts.items() if count is None
        ]
        if missing_counts:
            raise RuntimeError(f"Missing train dataset counts: {missing_counts}")
        if not bool(nesting_report.get("ok")):
            raise RuntimeError("Nested split verification failed")

    adapters: dict[str, Path] = {}
    for run in sample_runs:
        _inspect_targets(
            run=run,
            results_root=args.results_root,
            logs_root=args.logs_root,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
        )
        if run.profile.name == "train_10k" and bool(reuse_report.get("eligible")):
            adapters[run.profile.name] = args.reference_train10k_checkpoint
            _copy_reused_train10k_artifacts(
                reference_results_root=args.reference_train10k_results_root,
                destination_root=args.results_root,
                reuse_report=reuse_report,
            )
            continue
        completed_stage = _completed_training_stage(
            run=run,
            results_root=args.results_root,
            checkpoint_root=args.checkpoint_root,
        )
        if bool(completed_stage["complete"]):
            adapters[run.profile.name] = Path(str(completed_stage["checkpoint_path"]))
            _write_json(
                args.results_root / run.stage_name / "reuse_existing_exp08_stage.json",
                completed_stage,
            )
            continue
        train_count = train_counts[run.profile.name]
        if train_count is None:
            continue
        adapters[run.profile.name] = _train_sample(
            run=run,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            train_count=train_count,
            global_batch_size=args.global_batch_size,
            local_batch_size=args.local_batch_size,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        _write_json(
            args.results_root / "sample_efficiency_selection.json",
            {
                "schema_version": "1.0",
                "status": "dry_run_complete",
                "train_10k_reuse": reuse_report,
            },
        )
        print("exp08_sample_efficiency_summary=" + json.dumps({"status": "dry_run_complete"}))
        return

    missing_adapters = sorted({run.profile.name for run in sample_runs} - set(adapters))
    if missing_adapters:
        raise RuntimeError(f"Missing adapter paths for sample profiles: {missing_adapters}")

    for run in sample_runs:
        if run.profile.name == "train_10k" and bool(reuse_report.get("eligible")):
            # Eval artifacts were copied with the reused run. Re-score only if
            # the reference bundle was incomplete, which should make the run
            # ineligible before this point.
            continue
        adapter_path = adapters[run.profile.name]
        for dataset_name, dataset_path in (
            ("validation", args.validation_dataset),
            ("no_tool_dev", args.no_tool_dev_dataset),
        ):
            _generate_and_score(
                run=cast(Any, run),
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                adapter_path=adapter_path,
                output_root=args.results_root,
                logs_root=args.logs_root,
                cache_dir=args.cache_dir,
                generation_batch_size=args.generation_batch_size,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                dry_run=args.dry_run,
            )

    runs_by_name = {run.profile.name: run for run in sample_runs}
    for dataset_name in ("validation", "no_tool_dev"):
        _compare_pair(
            baseline=cast(Any, runs_by_name["train_2k"]),
            candidate=cast(Any, runs_by_name["train_10k"]),
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        _compare_pair(
            baseline=cast(Any, runs_by_name["train_10k"]),
            candidate=cast(Any, runs_by_name["train_full"]),
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        _compare_pair(
            baseline=cast(Any, runs_by_name["train_2k"]),
            candidate=cast(Any, runs_by_name["train_full"]),
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    rows = _aggregate_table(
        results_root=args.results_root,
        hourly_cost_usd=args.hourly_cost_usd,
    )
    bucket_report = _evaluation_bucket_report(
        results_root=args.results_root,
        validation_manifest=args.validation_manifest,
    )
    hard_guardrails = _hard_bucket_guardrails(bucket_report)
    decision = _select_dataset_size(
        rows=rows,
        hard_bucket_guardrails=hard_guardrails,
    )
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "train_10k_reuse": reuse_report,
        "nesting_report": nesting_report,
        "aggregate_metrics": rows,
        "bucket_metrics": bucket_report,
        "hard_bucket_guardrails": hard_guardrails,
        "decision": decision,
        "checkpoint_paths": {name: str(path) for name, path in adapters.items()},
        "deletions": [],
    }
    _write_json(args.results_root / "sample_efficiency_selection.json", summary)
    _write_markdown_summary(
        path=args.results_root / "sample_efficiency_selection.md",
        rows=rows,
        decision=decision,
    )
    checksum_path = _write_checksum_manifest(args.results_root)
    print(
        "exp08_sample_efficiency_summary="
        + json.dumps(
            {
                "status": "complete",
                "selected_sample_profile": decision.get("selected_sample_profile"),
                "train10k_reused": bool(reuse_report.get("eligible")),
                "checksums": str(checksum_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
