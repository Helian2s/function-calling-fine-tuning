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
    EXPECTED_TRAIN_PATH,
    EXPECTED_VALIDATION_PATH,
    EXP07_TARGET_PROFILES,
    LoraTargetProfile,
    clone_training_config_for_stage,
    load_yaml_config,
    validate_lora_target_config,
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


DEFAULT_CONFIG_ROOT = Path("configs/exp07_target_modules")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-07")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-07")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-07")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")
DEFAULT_REFERENCE_ATTENTION_CONFIG = Path("configs/exp06_lora_rank/rank4_alpha8.yaml")
DEFAULT_REFERENCE_ATTENTION_RESULTS_ROOT = Path("/workspace/results/exp-06/rank4-alpha8")
DEFAULT_REFERENCE_ATTENTION_RESOLVED_CONFIG = (
    DEFAULT_REFERENCE_ATTENTION_RESULTS_ROOT / "full-epoch" / "resolved_config.yaml"
)
DEFAULT_REFERENCE_ATTENTION_CHECKPOINT = Path(
    "/workspace/checkpoints/exp-06/rank4-alpha8/full-epoch",
)
TARGET_CONFIGS = {
    "attention": DEFAULT_CONFIG_ROOT / "attention.yaml",
    "attention_mlp": DEFAULT_CONFIG_ROOT / "attention_mlp.yaml",
}
PAIRWISE_METRICS = DEFAULT_METRICS + (
    "tool_call_emitted",
    "no_tool_false_positive",
)
EXEC_GAIN_THRESHOLD = 0.01
CALL_F1_GAIN_THRESHOLD = 0.005
NO_TOOL_FP_WORSENING_LIMIT = 0.05


@dataclass(frozen=True)
class TargetRun:
    profile: LoraTargetProfile
    config_path: Path
    run_id: str

    @property
    def stage_name(self) -> str:
        return self.profile.stage_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 07 LoRA target-module placement comparison.",
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
    parser.add_argument("--reuse-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reference-attention-resolved-config",
        type=Path,
        default=DEFAULT_REFERENCE_ATTENTION_RESOLVED_CONFIG,
    )
    parser.add_argument(
        "--reference-attention-results-root",
        type=Path,
        default=DEFAULT_REFERENCE_ATTENTION_RESULTS_ROOT,
    )
    parser.add_argument(
        "--reference-attention-checkpoint",
        type=Path,
        default=DEFAULT_REFERENCE_ATTENTION_CHECKPOINT,
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


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _target_runs() -> list[TargetRun]:
    return [
        TargetRun(
            profile=profile,
            config_path=TARGET_CONFIGS[profile.name],
            run_id=f"bf16-lora-r{profile.rank}-alpha{profile.alpha}-{profile.stage_name}",
        )
        for profile in EXP07_TARGET_PROFILES
    ]


def _validate_target_configs(results_root: Path) -> dict[str, Any]:
    validations = {}
    for run in _target_runs():
        validation = validate_lora_target_config(
            run.config_path,
            target_profile=run.profile.name,
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


def _attention_reuse_report(
    *,
    attention_run: TargetRun,
    reference_resolved_config: Path,
    reference_checkpoint: Path,
    reference_results_root: Path,
    train_count: int | None,
    global_batch_size: int,
    local_batch_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "target_profile": "attention",
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
    if train_count is None:
        reasons.append("missing_train_count")
    if reasons:
        report["reasons"] = reasons
        return report

    assert train_count is not None
    attention_config = load_yaml_config(attention_run.config_path)
    full_steps = _full_epoch_steps(train_count, global_batch_size)
    expected = clone_training_config_for_stage(
        attention_config,
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
            "controlled_config_mismatches": mismatches,
            "eligible": not reasons,
            "reasons": reasons,
        },
    )
    return report


def _train_target(
    *,
    run: TargetRun,
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
        validator=partial(validate_lora_target_config, target_profile=run.profile.name),
    )
    _write_json(
        results_root / run.stage_name / "full-epoch" / "target_training_summary.json",
        {
            "schema_version": "1.0",
            "target_profile": run.profile.name,
            "rank": run.profile.rank,
            "alpha": run.profile.alpha,
            "target_modules": list(run.profile.target_modules),
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
    run: TargetRun,
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
        "--target-profile",
        run.profile.name,
    ]
    _run_command(
        command,
        log_path=logs_root / run.stage_name / "target-inspection.log",
        dry_run=dry_run,
    )


def _copy_reused_eval(
    *,
    source_root: Path,
    destination_root: Path,
    dataset_name: str,
) -> bool:
    source = source_root / "eval" / dataset_name
    destination = destination_root / "attention" / "eval" / dataset_name
    required = ("predictions.jsonl", "scored_predictions.jsonl", "scores.json")
    if not all((source / name).is_file() for name in required):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return True


def _write_checksum_manifest(root: Path) -> Path:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rel = path.relative_to(root)
            lines.append(f"{_sha256_file(path)}  {rel.as_posix()}")
    output = root / "checksums.sha256"
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def _aggregate_table(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _target_runs():
        for dataset_name in ("validation", "no_tool_dev"):
            eval_root = results_root / run.stage_name / "eval" / dataset_name
            scores = _read_json(eval_root / "scores.json")
            requested = eval_root / "requested_metrics.json"
            row: dict[str, Any] = {
                "target_profile": run.profile.name,
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
            row["no_tool_false_positive_rate"] = _requested_metric_value(
                requested,
                "no_tool_false_positive_rate",
            )
            rows.append(row)
    return rows


def _select_target_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validation = {row["target_profile"]: row for row in rows if row["dataset"] == "validation"}
    no_tool = {row["target_profile"]: row for row in rows if row["dataset"] == "no_tool_dev"}
    attention = validation.get("attention")
    attention_mlp = validation.get("attention_mlp")
    if attention is None or attention_mlp is None:
        return {
            "status": "insufficient_metrics",
            "selected_target_profile": "attention",
            "reason": "Missing validation rows for one or both target profiles.",
        }
    exec_delta = float(attention_mlp.get("executable_complete_match_rate") or 0.0) - float(
        attention.get("executable_complete_match_rate") or 0.0,
    )
    call_f1_delta = float(attention_mlp.get("complete_call_f1") or 0.0) - float(
        attention.get("complete_call_f1") or 0.0,
    )
    no_tool_attention = float(no_tool.get("attention", {}).get("no_tool_false_positive_rate") or 0.0)
    no_tool_mlp = float(no_tool.get("attention_mlp", {}).get("no_tool_false_positive_rate") or 0.0)
    no_tool_fp_delta = no_tool_mlp - no_tool_attention
    broader_wins = (
        exec_delta >= EXEC_GAIN_THRESHOLD
        and call_f1_delta >= CALL_F1_GAIN_THRESHOLD
        and no_tool_fp_delta <= NO_TOOL_FP_WORSENING_LIMIT
    )
    return {
        "status": "selected",
        "selected_target_profile": "attention_mlp" if broader_wins else "attention",
        "default_profile": "attention",
        "exec_gain_threshold_absolute": EXEC_GAIN_THRESHOLD,
        "complete_call_f1_gain_threshold_absolute": CALL_F1_GAIN_THRESHOLD,
        "no_tool_false_positive_worsening_limit_absolute": NO_TOOL_FP_WORSENING_LIMIT,
        "attention_mlp_minus_attention": {
            "executable_complete_match_rate": exec_delta,
            "complete_call_f1": call_f1_delta,
            "no_tool_false_positive_rate": no_tool_fp_delta,
        },
        "selection_rule": (
            "Keep attention-only unless attention+MLP improves validation "
            "execution-equivalent record accuracy and complete-call F1 by the "
            "predefined thresholds without worsening no-tool false positives "
            "by more than the guardrail."
        ),
    }


def _write_markdown_summary(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    validation_rows = [row for row in rows if row["dataset"] == "validation"]
    no_tool_rows = [row for row in rows if row["dataset"] == "no_tool_dev"]
    lines = [
        "# Experiment 7 LoRA Target-Module Comparison",
        "",
        f"Selected target profile: `{decision.get('selected_target_profile')}`",
        "",
        "## Validation Tool-Calling Metrics",
        "",
        "| Target profile | Exec complete | Complete-call F1 | Fn F1 | Arg value acc | Missing calls | Extra calls | Malformed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(validation_rows, key=lambda item: str(item["target_profile"])):
        lines.append(
            "| {target} | {exec_rate:.4f} | {call_f1:.4f} | {fn_f1:.4f} | "
            "{arg_value:.4f} | {missing} | {extra} | {malformed} |".format(
                target=row["target_profile"],
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
            "| Target profile | No-tool false positive | Protocol clean | Tool calls emitted |",
            "| --- | ---: | ---: | ---: |",
        ],
    )
    for row in sorted(no_tool_rows, key=lambda item: str(item["target_profile"])):
        lines.append(
            "| {target} | {fp:.4f} | {clean:.4f} | {emitted} |".format(
                target=row["target_profile"],
                fp=float(row.get("no_tool_false_positive_rate") or 0.0),
                clean=float(row.get("protocol_clean_response_rate") or 0.0),
                emitted=row.get("tool_call_emitted_count"),
            ),
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_PATH,
        command_name="exp07-target-modules-train",
    )
    validation_decision = assert_split_allowed(
        args.validation_dataset,
        command_name="exp07-target-modules-validation",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev_dataset,
        command_name="exp07-target-modules-no-tool",
    )
    validations = _validate_target_configs(args.results_root)
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
    attention_run = next(run for run in _target_runs() if run.profile.name == "attention")
    reuse_report = (
        _attention_reuse_report(
            attention_run=attention_run,
            reference_resolved_config=args.reference_attention_resolved_config,
            reference_checkpoint=args.reference_attention_checkpoint,
            reference_results_root=args.reference_attention_results_root,
            train_count=train_count,
            global_batch_size=args.global_batch_size,
            local_batch_size=args.local_batch_size,
        )
        if args.reuse_attention
        else {
            "schema_version": "1.0",
            "target_profile": "attention",
            "reuse_requested": False,
            "eligible": False,
            "reasons": ["reuse_disabled"],
        }
    )
    run_plan = {
        "schema_version": "1.0",
        "experiment_id": "exp-07",
        "task_id": "task-12",
        "selected_peft_method": "bf16_lora",
        "selected_rank": 4,
        "selected_alpha": 8,
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "train_split": train_decision.__dict__,
        "validation_split": validation_decision.__dict__,
        "no_tool_split": no_tool_decision.__dict__,
        "train_records": train_count,
        "validation_records": validation_count,
        "no_tool_dev_records": no_tool_count,
        "target_profiles": [
            {
                "target_profile": run.profile.name,
                "target_modules": list(run.profile.target_modules),
                "config": str(run.config_path),
                "run_id": run.run_id,
            }
            for run in _target_runs()
        ],
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "attention_reuse": reuse_report,
        "config_validation": validations,
    }
    _write_json(args.results_root / "run_plan.json", run_plan)
    if args.validate_only:
        print("exp07_validation_ok=true")
        return
    if train_count is None and not args.dry_run:
        raise RuntimeError("Train dataset is required for non-dry-run execution")

    adapters: dict[str, Path] = {}
    for run in _target_runs():
        _inspect_targets(
            run=run,
            results_root=args.results_root,
            cache_dir=args.cache_dir,
            logs_root=args.logs_root,
            dry_run=args.dry_run,
        )
        if run.profile.name == "attention" and bool(reuse_report.get("eligible")):
            adapters[run.profile.name] = args.reference_attention_checkpoint
            _write_json(
                args.results_root / run.stage_name / "reuse_attention_reference.json",
                reuse_report,
            )
            continue
        if train_count is None:
            continue
        adapters[run.profile.name] = _train_target(
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
            args.results_root / "target_module_selection.json",
            {
                "schema_version": "1.0",
                "status": "dry_run_complete",
                "attention_reuse": reuse_report,
            },
        )
        print("exp07_target_modules_summary=" + json.dumps({"status": "dry_run_complete"}))
        return

    missing_adapters = sorted({"attention", "attention_mlp"} - set(adapters))
    if missing_adapters:
        raise RuntimeError(f"Missing adapter paths for target profiles: {missing_adapters}")

    for run in _target_runs():
        adapter_path = adapters[run.profile.name]
        for dataset_name, dataset_path in (
            ("validation", args.validation_dataset),
            ("no_tool_dev", args.no_tool_dev_dataset),
        ):
            copied = False
            if run.profile.name == "attention" and bool(reuse_report.get("eligible")):
                copied = _copy_reused_eval(
                    source_root=args.reference_attention_results_root,
                    destination_root=args.results_root,
                    dataset_name=dataset_name,
                )
            if not copied:
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

    runs_by_name = {run.profile.name: run for run in _target_runs()}
    for dataset_name in ("validation", "no_tool_dev"):
        _compare_pair(
            baseline=cast(Any, runs_by_name["attention"]),
            candidate=cast(Any, runs_by_name["attention_mlp"]),
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    rows = _aggregate_table(args.results_root)
    decision = _select_target_profile(rows)
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "attention_reuse": reuse_report,
        "aggregate_metrics": rows,
        "decision": decision,
        "checkpoint_paths": {name: str(path) for name, path in adapters.items()},
        "deletions": [],
    }
    _write_json(args.results_root / "target_module_selection.json", summary)
    _write_markdown_summary(
        path=args.results_root / "target_module_selection.md",
        rows=rows,
        decision=decision,
    )
    checksum_path = _write_checksum_manifest(args.results_root)
    print(
        "exp07_target_modules_summary="
        + json.dumps(
            {
                "status": "complete",
                "selected_target_profile": decision.get("selected_target_profile"),
                "attention_reused": bool(reuse_report.get("eligible")),
                "checksums": str(checksum_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
