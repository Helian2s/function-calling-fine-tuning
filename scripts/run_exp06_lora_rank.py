#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation_compare import DEFAULT_METRICS
from function_calling_ft.generation import read_jsonl
from function_calling_ft.reference_lora import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_TRAIN_PATH,
    EXPECTED_VALIDATION_PATH,
    EXP06_RANK_PROFILES,
    LoraRankProfile,
    clone_training_config_for_stage,
    load_yaml_config,
    validation_to_dict,
    validate_lora_rank_config,
    write_yaml_config,
)
from function_calling_ft.split_guard import assert_split_allowed
from scripts.run_exp03_reference_lora import (
    _full_epoch_steps,
    _run_logged,
    _run_reload_check,
    _run_training_stage,
)

DEFAULT_CONFIG_ROOT = Path("configs/exp06_lora_rank")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-06")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-06")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-06")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_REFERENCE_RANK8_CONFIG = Path("configs/exp03_reference_lora/lora_r8_attention.yaml")
DEFAULT_REFERENCE_RANK8_RESULTS_ROOT = Path("/workspace/results/exp-03")
DEFAULT_REFERENCE_RANK8_CHECKPOINT = Path(
    "/workspace/checkpoints/exp-03/reference-bf16-lora-r8-attention/full-epoch",
)
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")

RANK_CONFIGS = {
    4: DEFAULT_CONFIG_ROOT / "rank4_alpha8.yaml",
    8: DEFAULT_CONFIG_ROOT / "rank8_alpha16.yaml",
    16: DEFAULT_CONFIG_ROOT / "rank16_alpha32.yaml",
}
PAIRWISE_METRICS = DEFAULT_METRICS + (
    "tool_call_emitted",
    "no_tool_false_positive",
)
AGGREGATE_METRICS = (
    "strict_complete_match_rate",
    "schema_equivalent_complete_match_rate",
    "executable_complete_match_rate",
    "function_name_precision",
    "function_name_recall",
    "function_name_f1",
    "complete_call_precision",
    "complete_call_recall",
    "complete_call_f1",
    "average_argument_value_accuracy",
    "schema_validation_success_rate",
    "parseable_given_emission_rate",
    "valid_structure_rate",
    "missing_call_count",
    "extra_call_count",
    "malformed_tool_call_count",
    "extra_prose_with_tool_call_count",
    "no_tool_call_emitted_count",
    "tool_call_emitted_count",
)
TIE_THRESHOLD = 0.01
PRACTICAL_LOSS_THRESHOLD = 0.02


@dataclass(frozen=True)
class RankRun:
    profile: LoraRankProfile
    config_path: Path
    run_id: str

    @property
    def stage_name(self) -> str:
        return f"rank{self.profile.rank}-alpha{self.profile.alpha}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 06 LoRA rank/alpha comparison.",
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
    parser.add_argument("--reuse-rank8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reference-rank8-config",
        type=Path,
        default=DEFAULT_REFERENCE_RANK8_CONFIG,
    )
    parser.add_argument(
        "--reference-rank8-resolved-config",
        type=Path,
        default=DEFAULT_REFERENCE_RANK8_RESULTS_ROOT / "full-epoch" / "resolved_config.yaml",
    )
    parser.add_argument(
        "--reference-rank8-checkpoint",
        type=Path,
        default=DEFAULT_REFERENCE_RANK8_CHECKPOINT,
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path(EXPECTED_VALIDATION_PATH),
    )
    parser.add_argument("--no-tool-dev-dataset", type=Path, default=DEFAULT_NO_TOOL_DEV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _rank_runs() -> list[RankRun]:
    return [
        RankRun(
            profile=profile,
            config_path=RANK_CONFIGS[profile.rank],
            run_id=f"bf16-lora-r{profile.rank}-alpha{profile.alpha}-attention",
        )
        for profile in EXP06_RANK_PROFILES
    ]


def _validate_rank_configs(results_root: Path) -> dict[str, Any]:
    validations = {}
    for run in _rank_runs():
        validation = validate_lora_rank_config(
            run.config_path,
            rank=run.profile.rank,
        )
        validations[str(run.profile.rank)] = validation_to_dict(validation)
    _write_json(results_root / "config_validation.json", validations)
    errors = [
        f"rank{rank}: {error}"
        for rank, payload in validations.items()
        for error in payload["errors"]
    ]
    if errors:
        raise ValueError("; ".join(errors))
    return validations


def _controlled_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dist_env",
        "rng",
        "model",
        "compile",
        "peft",
        "distributed",
        "loss_fn",
        "dataset",
        "packed_sequence",
        "dataloader",
        "validation_dataset",
        "validation_dataloader",
        "optimizer",
        "lr_scheduler",
        "step_scheduler",
    )
    return {key: config.get(key) for key in keys}


def _rank8_reuse_report(
    *,
    rank8_run: RankRun,
    reference_resolved_config: Path,
    reference_checkpoint: Path,
    train_count: int | None,
    global_batch_size: int,
    local_batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "rank": 8,
        "reuse_requested": True,
        "reference_resolved_config": str(reference_resolved_config),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_resolved_config_exists": reference_resolved_config.is_file(),
        "reference_checkpoint_exists": reference_checkpoint.exists(),
        "eligible": False,
        "reasons": [],
    }
    reasons: list[str] = []

    if not reference_resolved_config.is_file():
        reasons.append("missing_reference_resolved_config")
    if not reference_checkpoint.exists():
        reasons.append("missing_reference_checkpoint")
    if train_count is None:
        reasons.append("missing_train_count")

    if reasons:
        report["reasons"] = reasons
        return report

    assert train_count is not None
    rank8_config = load_yaml_config(rank8_run.config_path)
    full_steps = _full_epoch_steps(train_count, global_batch_size)
    expected = clone_training_config_for_stage(
        rank8_config,
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

    report.update(
        {
            "expected_full_steps": full_steps,
            "expected_reference_checkpoint_dir": str(reference_checkpoint),
            "controlled_config_mismatches": mismatches,
            "eligible": not reasons,
            "reasons": reasons,
        },
    )
    return report


def _config_for_full_epoch(
    *,
    run: RankRun,
    results_root: Path,
    checkpoint_root: Path,
    train_count: int,
    global_batch_size: int,
    local_batch_size: int,
) -> tuple[dict[str, Any], Path, Path]:
    base_config = load_yaml_config(run.config_path)
    full_steps = _full_epoch_steps(train_count, global_batch_size)
    checkpoint_dir = checkpoint_root / run.stage_name / "full-epoch"
    config_path = results_root / run.stage_name / "full-epoch" / "resolved_config.yaml"
    staged = clone_training_config_for_stage(
        base_config,
        checkpoint_dir=str(checkpoint_dir),
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        max_steps=full_steps,
        ckpt_every_steps=max(1, full_steps // 4),
        val_every_steps=max(1, full_steps // 4),
        validation_path=None,
        checkpoint_enabled=True,
    )
    write_yaml_config(config_path, staged)
    validation = validate_lora_rank_config(config_path, rank=run.profile.rank)
    _write_json(
        results_root / run.stage_name / "full-epoch" / "config_validation.json",
        validation_to_dict(validation),
    )
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    return staged, config_path, checkpoint_dir


def _train_rank(
    *,
    run: RankRun,
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
        validator=partial(validate_lora_rank_config, rank=run.profile.rank),
    )
    _write_json(
        results_root / run.stage_name / "full-epoch" / "rank_training_summary.json",
        {
            "schema_version": "1.0",
            "rank": run.profile.rank,
            "alpha": run.profile.alpha,
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


def _run_command(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    _run_logged(command, log_path=log_path, dry_run=dry_run)


def _inspect_rank_targets(
    *,
    run: RankRun,
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
        "--method",
        "lora",
        "--rank",
        str(run.profile.rank),
    ]
    _run_command(
        command,
        log_path=logs_root / run.stage_name / "target-inspection.log",
        dry_run=dry_run,
    )


def _generate_and_score(
    *,
    run: RankRun,
    dataset_name: str,
    dataset_path: Path,
    adapter_path: Path,
    output_root: Path,
    logs_root: Path,
    cache_dir: Path,
    generation_batch_size: int,
    max_new_tokens: int,
    seed: int,
    bootstrap_samples: int,
    dry_run: bool,
) -> Path:
    eval_root = output_root / run.stage_name / "eval" / dataset_name
    eval_root.mkdir(parents=True, exist_ok=True)
    prediction_path = eval_root / "predictions.jsonl"
    generation_command = [
        sys.executable,
        "scripts/generate_predictions.py",
        "--dataset",
        str(dataset_path),
        "--output",
        str(prediction_path),
        "--model-name",
        EXPECTED_MODEL_NAME,
        "--model-revision",
        EXPECTED_MODEL_REVISION,
        "--adapter-path",
        str(adapter_path),
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
        "--batch-size",
        str(generation_batch_size),
        "--cache-dir",
        str(cache_dir),
        "--metadata-output",
        str(eval_root / "generation_metadata.json"),
        "--stream-output",
        "--resume",
        "--progress-interval",
        "25",
        "--progress-file",
        str(eval_root / "progress.json"),
        "--no-load-in-4bit",
        "--torch-dtype",
        "bfloat16",
    ]
    _run_command(
        generation_command,
        log_path=logs_root / run.stage_name / f"generate-{dataset_name}.log",
        dry_run=dry_run,
    )
    evaluate_command = [
        sys.executable,
        "scripts/evaluate.py",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(prediction_path),
        "--output-dir",
        str(eval_root),
    ]
    _run_command(
        evaluate_command,
        log_path=logs_root / run.stage_name / f"evaluate-{dataset_name}.log",
        dry_run=dry_run,
    )
    summarize_command = [
        sys.executable,
        "scripts/summarize_evaluation_report.py",
        "--output-dir",
        str(eval_root),
    ]
    _run_command(
        summarize_command,
        log_path=logs_root / run.stage_name / f"summarize-{dataset_name}.log",
        dry_run=dry_run,
    )
    if dry_run:
        _write_json(
            eval_root / "scores.json",
            {
                "dry_run": True,
                "total_records": 0,
                "bootstrap_samples": bootstrap_samples,
            },
        )
    return eval_root


def _compare_pair(
    *,
    baseline: RankRun,
    candidate: RankRun,
    dataset_name: str,
    results_root: Path,
    logs_root: Path,
    bootstrap_samples: int,
    seed: int,
    dry_run: bool,
) -> None:
    baseline_scored = (
        results_root
        / baseline.stage_name
        / "eval"
        / dataset_name
        / "scored_predictions.jsonl"
    )
    candidate_scored = (
        results_root
        / candidate.stage_name
        / "eval"
        / dataset_name
        / "scored_predictions.jsonl"
    )
    output_dir = (
        results_root
        / "comparisons"
        / dataset_name
        / f"{baseline.stage_name}_vs_{candidate.stage_name}"
    )
    command = [
        sys.executable,
        "scripts/compare_evaluations.py",
        "--baseline-scored",
        str(baseline_scored),
        "--candidate-scored",
        str(candidate_scored),
        "--output-dir",
        str(output_dir),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
    ]
    for metric in PAIRWISE_METRICS:
        command.extend(["--metric", metric])
    _run_command(
        command,
        log_path=logs_root / "comparisons" / f"{dataset_name}-{baseline.stage_name}-vs-{candidate.stage_name}.log",
        dry_run=dry_run,
    )


def _score_value(scores: Mapping[str, Any], metric: str) -> float | int | None:
    value = scores.get(metric)
    if isinstance(value, int | float):
        return value
    return None


def _requested_metric_value(path: Path, name: str) -> float | None:
    payload = _read_json(path)
    metric = payload.get(name)
    if not isinstance(metric, Mapping):
        return None
    value = metric.get("value")
    return float(value) if isinstance(value, int | float) else None


def _aggregate_table(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _rank_runs():
        for dataset_name in ("validation", "no_tool_dev"):
            eval_root = results_root / run.stage_name / "eval" / dataset_name
            scores = _read_json(eval_root / "scores.json")
            requested = eval_root / "requested_metrics.json"
            row: dict[str, Any] = {
                "rank": run.profile.rank,
                "alpha": run.profile.alpha,
                "dataset": dataset_name,
                "scores_path": str(eval_root / "scores.json"),
                "requested_metrics_path": str(requested),
            }
            for metric in AGGREGATE_METRICS:
                row[metric] = _score_value(scores, metric)
            row["protocol_clean_response_rate"] = _requested_metric_value(
                requested,
                "protocol_clean_response_rate",
            )
            row["complete_call_recall_requested"] = _requested_metric_value(
                requested,
                "complete_call_recall",
            )
            row["complete_call_precision_requested"] = _requested_metric_value(
                requested,
                "complete_call_precision",
            )
            rows.append(row)
    return rows


def _select_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validation_rows = [row for row in rows if row["dataset"] == "validation"]
    usable = [
        row
        for row in validation_rows
        if isinstance(row.get("executable_complete_match_rate"), int | float)
        and isinstance(row.get("complete_call_f1"), int | float)
    ]
    if not usable:
        return {
            "status": "insufficient_metrics",
            "selected_rank": None,
            "reason": "No validation rows with executable complete and complete-call F1.",
        }

    best_exec = max(float(row["executable_complete_match_rate"]) for row in usable)
    best_call_f1 = max(float(row["complete_call_f1"]) for row in usable)
    candidates = [
        row
        for row in usable
        if best_exec - float(row["executable_complete_match_rate"]) <= TIE_THRESHOLD
        and best_call_f1 - float(row["complete_call_f1"]) <= TIE_THRESHOLD
    ]
    selected = min(candidates, key=lambda row: int(row["rank"]))
    rank16 = next((row for row in usable if int(row["rank"]) == 16), None)
    rank16_capacity_gain = False
    if rank16 is not None:
        smaller = [row for row in usable if int(row["rank"]) < 16]
        best_smaller_exec = max(
            float(row["executable_complete_match_rate"]) for row in smaller
        )
        best_smaller_call_f1 = max(float(row["complete_call_f1"]) for row in smaller)
        rank16_capacity_gain = (
            float(rank16["executable_complete_match_rate"]) - best_smaller_exec
            > PRACTICAL_LOSS_THRESHOLD
            and float(rank16["complete_call_f1"]) - best_smaller_call_f1
            > TIE_THRESHOLD
        )
    return {
        "status": "selected",
        "selected_rank": selected["rank"],
        "selected_alpha": selected["alpha"],
        "best_executable_complete_match_rate": best_exec,
        "best_complete_call_f1": best_call_f1,
        "tie_threshold_absolute": TIE_THRESHOLD,
        "practical_loss_threshold_absolute": PRACTICAL_LOSS_THRESHOLD,
        "selection_rule": (
            "Choose the smallest rank within 1pp of the best validation "
            "execution-equivalent record accuracy and complete-call F1; "
            "report no-tool regression but do not select on it for Task 11."
        ),
        "rank16_underfitting_evidence": rank16_capacity_gain,
        "rank16_underfitting_definition": (
            "True only when rank 16 shows a practical validation gain over all "
            "smaller ranks on execution-equivalent accuracy and complete-call F1."
        ),
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
        "# Experiment 6 LoRA Rank Comparison",
        "",
        f"Selected rank: `{decision.get('selected_rank')}`",
        "",
        "## Validation Tool-Calling Metrics",
        "",
        "| Rank | Exec complete | Complete-call F1 | Fn F1 | Arg value acc | Missing calls | Extra calls | Malformed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(validation_rows, key=lambda item: int(item["rank"])):
        lines.append(
            "| {rank} | {exec_rate:.4f} | {call_f1:.4f} | {fn_f1:.4f} | "
            "{arg_value:.4f} | {missing} | {extra} | {malformed} |".format(
                rank=row["rank"],
                exec_rate=float(row.get("executable_complete_match_rate") or 0.0),
                call_f1=float(row.get("complete_call_f1") or 0.0),
                fn_f1=float(row.get("function_name_f1") or 0.0),
                arg_value=float(row.get("average_argument_value_accuracy") or 0.0),
                missing=row.get("missing_call_count"),
                extra=row.get("extra_call_count"),
                malformed=row.get("malformed_tool_call_count"),
            ),
        )
    lines.extend(
        [
            "",
            "## Development No-Tool Metrics",
            "",
            "| Rank | No-tool false positive | Protocol clean | Tool calls emitted |",
            "| ---: | ---: | ---: | ---: |",
        ],
    )
    for row in sorted(no_tool_rows, key=lambda item: int(item["rank"])):
        false_positive = _requested_metric_value(
            Path(str(row["requested_metrics_path"])),
            "no_tool_false_positive_rate",
        )
        lines.append(
            "| {rank} | {fp:.4f} | {clean:.4f} | {emitted} |".format(
                rank=row["rank"],
                fp=false_positive if false_positive is not None else 0.0,
                clean=float(row.get("protocol_clean_response_rate") or 0.0),
                emitted=row.get("tool_call_emitted_count"),
            ),
        )
    lines.extend(
        [
            "",
            "No rank-32 run is scheduled by this task.",
            "",
        ],
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_PATH,
        command_name="exp06-lora-rank",
    )
    validation_decision = assert_split_allowed(
        args.validation_dataset,
        command_name="exp06-lora-rank-validation",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev_dataset,
        command_name="exp06-lora-rank-no-tool",
    )
    validations = _validate_rank_configs(args.results_root)
    train_count = (
        _dataset_count(Path(EXPECTED_TRAIN_PATH))
        if Path(EXPECTED_TRAIN_PATH).is_file()
        else None
    )
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
    rank8_run = next(run for run in _rank_runs() if run.profile.rank == 8)
    reuse_report = (
        _rank8_reuse_report(
            rank8_run=rank8_run,
            reference_resolved_config=args.reference_rank8_resolved_config,
            reference_checkpoint=args.reference_rank8_checkpoint,
            train_count=train_count,
            global_batch_size=args.global_batch_size,
            local_batch_size=args.local_batch_size,
        )
        if args.reuse_rank8
        else {
            "schema_version": "1.0",
            "rank": 8,
            "reuse_requested": False,
            "eligible": False,
            "reasons": ["reuse_disabled"],
        }
    )
    run_plan = {
        "schema_version": "1.0",
        "experiment_id": "exp-06",
        "task_id": "task-11",
        "selected_peft_method": "bf16_lora",
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "train_split": train_decision.__dict__,
        "validation_split": validation_decision.__dict__,
        "no_tool_split": no_tool_decision.__dict__,
        "train_records": train_count,
        "validation_records": validation_count,
        "no_tool_dev_records": no_tool_count,
        "ranks": [
            {
                "rank": run.profile.rank,
                "alpha": run.profile.alpha,
                "config": str(run.config_path),
                "run_id": run.run_id,
            }
            for run in _rank_runs()
        ],
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "rank8_reuse": reuse_report,
        "config_validation": validations,
    }
    _write_json(args.results_root / "run_plan.json", run_plan)
    if args.validate_only:
        print("exp06_validation_ok=true")
        return
    if train_count is None and not args.dry_run:
        raise RuntimeError("Train dataset is required for non-dry-run execution")

    adapters: dict[int, Path] = {}
    for run in _rank_runs():
        _inspect_rank_targets(
            run=run,
            results_root=args.results_root,
            cache_dir=args.cache_dir,
            logs_root=args.logs_root,
            dry_run=args.dry_run,
        )

        if run.profile.rank == 8 and bool(reuse_report.get("eligible")):
            adapters[run.profile.rank] = args.reference_rank8_checkpoint
            _write_json(
                args.results_root / run.stage_name / "reuse_reference_rank8.json",
                reuse_report,
            )
            continue

        if train_count is None:
            continue
        adapters[run.profile.rank] = _train_rank(
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
            args.results_root / "rank_selection.json",
            {
                "schema_version": "1.0",
                "status": "dry_run_complete",
                "rank8_reuse": reuse_report,
            },
        )
        print("exp06_lora_rank_summary=" + json.dumps({"status": "dry_run_complete"}))
        return

    missing_adapters = sorted({4, 8, 16} - set(adapters))
    if missing_adapters:
        raise RuntimeError(f"Missing adapter paths for ranks: {missing_adapters}")

    for run in _rank_runs():
        adapter_path = adapters[run.profile.rank]
        _generate_and_score(
            run=run,
            dataset_name="validation",
            dataset_path=args.validation_dataset,
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
        _generate_and_score(
            run=run,
            dataset_name="no_tool_dev",
            dataset_path=args.no_tool_dev_dataset,
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

    runs_by_rank = {run.profile.rank: run for run in _rank_runs()}
    for dataset_name in ("validation", "no_tool_dev"):
        _compare_pair(
            baseline=runs_by_rank[8],
            candidate=runs_by_rank[4],
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        _compare_pair(
            baseline=runs_by_rank[8],
            candidate=runs_by_rank[16],
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        _compare_pair(
            baseline=runs_by_rank[4],
            candidate=runs_by_rank[16],
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    rows = _aggregate_table(args.results_root)
    decision = _select_rank(rows)
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "rank8_reuse": reuse_report,
        "aggregate_metrics": rows,
        "decision": decision,
        "checkpoint_paths": {str(rank): str(path) for rank, path in adapters.items()},
        "deletions": [],
    }
    _write_json(args.results_root / "rank_selection.json", summary)
    _write_markdown_summary(
        path=args.results_root / "rank_selection.md",
        rows=rows,
        decision=decision,
    )
    checksum_path = _write_checksum_manifest(args.results_root)
    print(
        "exp06_lora_rank_summary="
        + json.dumps(
            {
                "status": "complete",
                "selected_rank": decision.get("selected_rank"),
                "rank8_reused": bool(reuse_report.get("eligible")),
                "checksums": str(checksum_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
