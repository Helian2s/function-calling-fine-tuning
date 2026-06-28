#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_TRAIN_PATH,
    EXPECTED_VALIDATION_PATH,
    EXP09_LOSS_MASK_PROFILES,
    LossMaskAblationProfile,
    load_yaml_config,
    validate_loss_mask_ablation_config,
    validation_to_dict,
)
from function_calling_ft.split_guard import assert_split_allowed
from scripts.run_exp03_reference_lora import (
    _run_training_stage,
)
from scripts.run_exp06_lora_rank import (
    AGGREGATE_METRICS,
    _compare_pair,
    _read_json,
    _requested_metric_value,
    _run_command,
    _score_value,
    _write_checksum_manifest,
    _write_json,
)


DEFAULT_CONFIG_ROOT = Path("configs/exp09_loss_masking")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-09a")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-09a")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-09a")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")
LOSS_MASK_CONFIGS = {
    "assistant_only_short": DEFAULT_CONFIG_ROOT / "assistant_only_short.yaml",
    "full_sequence_short": DEFAULT_CONFIG_ROOT / "full_sequence_short.yaml",
}
PAIRWISE_METRICS = DEFAULT_METRICS + (
    "tool_call_emitted",
    "no_tool_false_positive",
)


@dataclass(frozen=True)
class LossMaskRun:
    profile: LossMaskAblationProfile
    config_path: Path
    run_id: str

    @property
    def stage_name(self) -> str:
        return self.profile.stage_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 09A assistant-only masking proof and full-sequence ablation.",
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
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-validation-records", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path(EXPECTED_VALIDATION_PATH),
    )
    parser.add_argument("--no-tool-dev-dataset", type=Path, default=DEFAULT_NO_TOOL_DEV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _loss_mask_runs() -> list[LossMaskRun]:
    return [
        LossMaskRun(
            profile=profile,
            config_path=LOSS_MASK_CONFIGS[profile.name],
            run_id=f"bf16-lora-r{profile.rank}-alpha{profile.alpha}-{profile.stage_name}",
        )
        for profile in EXP09_LOSS_MASK_PROFILES
    ]


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _write_validation_slice(*, source: Path, output: Path, records: int) -> int:
    selected = read_jsonl(source)[:records]
    _write_jsonl(output, selected)
    return len(selected)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _collect_environment(results_root: Path) -> dict[str, Any]:
    nvidia_smi_path = results_root / "nvidia-smi.txt"
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        nvidia_smi_text = repr(exc)
    else:
        nvidia_smi_text = completed.stdout + completed.stderr
    nvidia_smi_path.write_text(nvidia_smi_text, encoding="utf-8")

    packages = {
        name: _package_version(name)
        for name in (
            "nemo_automodel",
            "transformers",
            "torch",
            "peft",
            "bitsandbytes",
        )
    }
    package_versions_path = results_root / "package_versions.txt"
    package_versions_path.write_text(
        "\n".join(f"{name}=={version}" for name, version in sorted(packages.items()))
        + "\n",
        encoding="utf-8",
    )
    environment = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi_path": str(nvidia_smi_path),
        "package_versions_path": str(package_versions_path),
    }
    _write_json(results_root / "environment_report.json", environment)
    return environment


def _write_run_manifest(
    *,
    results_root: Path,
    run_plan: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, str],
    environment: Mapping[str, Any] | None,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "experiment_id": "exp-09a",
        "task_id": "task-14",
        "run_id": "exp09a-loss-masking-short-ablation",
        "status": status,
        "created_at": _utc_now(),
        "model_id": EXPECTED_MODEL_NAME,
        "model_revision": EXPECTED_MODEL_REVISION,
        "tokenizer_revision": EXPECTED_MODEL_REVISION,
        "method": "bf16_lora",
        "precision": "bfloat16",
        "loss_policies": ["assistant_only", "full_sequence"],
        "decoding": {
            "deterministic": True,
            "seed": run_plan.get("seed", 42),
            "max_new_tokens": run_plan.get("max_new_tokens"),
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "dirty_status": _git_value(["status", "--short"]),
        },
        "run_plan": run_plan,
        "environment": environment or {},
        "artifacts": dict(artifacts),
    }
    _write_json(results_root / "run_manifest.json", manifest)


def _validate_configs(results_root: Path) -> dict[str, Any]:
    validations = {}
    for run in _loss_mask_runs():
        validation = validate_loss_mask_ablation_config(
            run.config_path,
            loss_mask_profile=run.profile.name,
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


def _normalized_config_for_diff(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    for key in ("run_id", "title"):
        config.pop(key, None)
    config["checkpoint"] = dict(config["checkpoint"])
    config["checkpoint"].pop("checkpoint_dir", None)
    config["dataset"] = dict(config["dataset"])
    config["dataset"].pop("loss_mask_policy", None)
    config["validation_dataset"] = dict(config["validation_dataset"])
    config["validation_dataset"].pop("loss_mask_policy", None)
    config.pop("task14_policy", None)
    return config


def _config_diff_report() -> dict[str, Any]:
    assistant = _normalized_config_for_diff(LOSS_MASK_CONFIGS["assistant_only_short"])
    full_sequence = _normalized_config_for_diff(LOSS_MASK_CONFIGS["full_sequence_short"])
    assistant_raw = load_yaml_config(LOSS_MASK_CONFIGS["assistant_only_short"])
    full_raw = load_yaml_config(LOSS_MASK_CONFIGS["full_sequence_short"])
    return {
        "schema_version": "1.0",
        "only_mask_policy_differs": assistant == full_sequence,
        "assistant_dataset_loss_mask_policy": assistant_raw["dataset"].get(
            "loss_mask_policy",
        ),
        "full_sequence_dataset_loss_mask_policy": full_raw["dataset"].get(
            "loss_mask_policy",
        ),
        "assistant_validation_loss_mask_policy": assistant_raw[
            "validation_dataset"
        ].get("loss_mask_policy"),
        "full_sequence_validation_loss_mask_policy": full_raw[
            "validation_dataset"
        ].get("loss_mask_policy"),
    }


def _run_loss_mask_audit(
    *,
    results_root: Path,
    logs_root: Path,
    cache_dir: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/audit_exp09_loss_masks.py",
        "--train-dataset",
        EXPECTED_TRAIN_PATH,
        "--validation-dataset",
        EXPECTED_VALIDATION_PATH,
        "--output-dir",
        str(results_root / "loss_mask_audit"),
        "--cache-dir",
        str(cache_dir),
        "--count",
        "20",
    ]
    _run_command(
        command,
        log_path=logs_root / "loss-mask-audit.log",
        dry_run=dry_run,
    )
    if dry_run:
        _write_json(
            results_root / "loss_mask_audit" / "loss_mask_audit_report.json",
            {"schema_version": "1.0", "status": "dry_run"},
        )


def _run_automodel_probe(
    *,
    results_root: Path,
    logs_root: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/probe_automodel_loss_masking.py",
        "--assistant-config",
        str(LOSS_MASK_CONFIGS["assistant_only_short"]),
        "--full-sequence-config",
        str(LOSS_MASK_CONFIGS["full_sequence_short"]),
        "--output",
        str(results_root / "automodel_loss_mask_probe.json"),
    ]
    _run_command(
        command,
        log_path=logs_root / "automodel-loss-mask-probe.log",
        dry_run=dry_run,
    )
    if dry_run:
        _write_json(
            results_root / "automodel_loss_mask_probe.json",
            {"schema_version": "1.0", "status": "dry_run"},
        )


def _inspect_targets(
    *,
    run: LossMaskRun,
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
        "--loss-mask-profile",
        run.profile.name,
    ]
    _run_command(
        command,
        log_path=logs_root / run.stage_name / "target-inspection.log",
        dry_run=dry_run,
    )


def _train_short_run(
    *,
    run: LossMaskRun,
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    validation_slice: Path,
    local_batch_size: int,
    global_batch_size: int,
    max_steps: int,
    dry_run: bool,
) -> Path:
    metrics = _run_training_stage(
        stage_name=run.stage_name,
        base_config=load_yaml_config(run.config_path),
        results_root=results_root,
        logs_root=logs_root,
        checkpoint_root=checkpoint_root,
        automodel_bin=automodel_bin,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        max_steps=max_steps,
        ckpt_every_steps=max_steps,
        val_every_steps=max(1, max_steps // 2),
        validation_path=validation_slice,
        checkpoint_enabled=True,
        memory_trace=False,
        dry_run=dry_run,
        patch_qlora_peft_state_dict=False,
        validator=partial(
            validate_loss_mask_ablation_config,
            loss_mask_profile=run.profile.name,
        ),
    )
    _write_json(
        results_root / run.stage_name / "short_training_summary.json",
        {
            "schema_version": "1.0",
            "loss_mask_profile": run.profile.name,
            "answer_only_loss_mask": run.profile.answer_only_loss_mask,
            "max_steps": max_steps,
            "metrics": metrics,
        },
    )
    return checkpoint_root / run.stage_name


def _generate_and_score(
    *,
    run: LossMaskRun,
    dataset_name: str,
    dataset_path: Path,
    adapter_path: Path,
    output_root: Path,
    logs_root: Path,
    cache_dir: Path,
    generation_batch_size: int,
    max_new_tokens: int,
    seed: int,
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
        _write_json(eval_root / "scores.json", {"dry_run": True, "total_records": 0})
    return eval_root


def _aggregate_table(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _loss_mask_runs():
        for dataset_name in ("validation_slice", "no_tool_dev"):
            eval_root = results_root / run.stage_name / "eval" / dataset_name
            scores = _read_json(eval_root / "scores.json")
            requested = eval_root / "requested_metrics.json"
            row: dict[str, Any] = {
                "loss_mask_profile": run.profile.name,
                "answer_only_loss_mask": run.profile.answer_only_loss_mask,
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


def _write_markdown_summary(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    config_diff: Mapping[str, Any],
) -> None:
    validation_rows = [row for row in rows if row["dataset"] == "validation_slice"]
    no_tool_rows = [row for row in rows if row["dataset"] == "no_tool_dev"]
    lines = [
        "# Experiment 9A Loss Masking Ablation",
        "",
        "Production policy remains assistant/tool-call-output-only loss.",
        "",
        f"Config diff gate: `{config_diff.get('only_mask_policy_differs')}`",
        "",
        "## Validation Slice",
        "",
        "| Mask policy | Exec complete | Complete-call F1 | Fn F1 | Arg value acc | Missing calls | Extra calls | Malformed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in validation_rows:
        lines.append(
            "| {policy} | {exec_rate:.4f} | {call_f1:.4f} | {fn_f1:.4f} | "
            "{arg_value:.4f} | {missing} | {extra} | {malformed} |".format(
                policy=row["loss_mask_profile"],
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
            "## No-Tool Development",
            "",
            "| Mask policy | No-tool false positive | Protocol clean | Tool calls emitted |",
            "| --- | ---: | ---: | ---: |",
        ],
    )
    for row in no_tool_rows:
        false_positive = row.get("no_tool_false_positive_rate")
        lines.append(
            "| {policy} | {fp:.4f} | {clean:.4f} | {emitted} |".format(
                policy=row["loss_mask_profile"],
                fp=float(false_positive or 0.0),
                clean=float(row.get("protocol_clean_response_rate") or 0.0),
                emitted=row.get("tool_call_emitted_count"),
            ),
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    if not args.validate_only and not args.dry_run:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_PATH,
        command_name="exp09-loss-masking-train",
    )
    validation_decision = assert_split_allowed(
        args.validation_dataset,
        command_name="exp09-loss-masking-validation",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev_dataset,
        command_name="exp09-loss-masking-no-tool",
    )
    validations = _validate_configs(args.results_root)
    config_diff = _config_diff_report()
    _write_json(args.results_root / "config_diff.json", config_diff)
    if not config_diff["only_mask_policy_differs"]:
        raise RuntimeError("Exp09 configs differ beyond masking policy and identity")

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
    validation_slice = args.results_root / "validation_slice_1000.jsonl"
    validation_slice_count = 0
    if not args.dry_run and args.validation_dataset.is_file():
        validation_slice_count = _write_validation_slice(
            source=args.validation_dataset,
            output=validation_slice,
            records=args.eval_validation_records,
        )
    elif args.dry_run:
        validation_slice_count = args.eval_validation_records

    run_plan = {
        "schema_version": "1.0",
        "experiment_id": "exp-09a",
        "task_id": "task-14",
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "train_split": train_decision.__dict__,
        "validation_split": validation_decision.__dict__,
        "no_tool_split": no_tool_decision.__dict__,
        "train_records": train_count,
        "validation_records": validation_count,
        "no_tool_dev_records": no_tool_count,
        "validation_slice_path": str(validation_slice),
        "validation_slice_records": validation_slice_count,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "max_steps": args.max_steps,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "config_validation": validations,
        "config_diff": config_diff,
    }
    _write_json(args.results_root / "run_plan.json", run_plan)
    if args.validate_only:
        print("exp09_validation_ok=true")
        return

    environment = None if args.dry_run else _collect_environment(args.results_root)
    _write_run_manifest(
        results_root=args.results_root,
        run_plan=run_plan,
        status="dry_run_started" if args.dry_run else "started",
        artifacts={
            "run_plan": str(args.results_root / "run_plan.json"),
            "config_diff": str(args.results_root / "config_diff.json"),
        },
        environment=environment,
    )

    _run_loss_mask_audit(
        results_root=args.results_root,
        logs_root=args.logs_root,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )
    _run_automodel_probe(
        results_root=args.results_root,
        logs_root=args.logs_root,
        dry_run=args.dry_run,
    )

    adapters: dict[str, Path] = {}
    for run in _loss_mask_runs():
        _inspect_targets(
            run=run,
            results_root=args.results_root,
            logs_root=args.logs_root,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
        )
        adapters[run.profile.name] = _train_short_run(
            run=run,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            validation_slice=validation_slice,
            local_batch_size=args.local_batch_size,
            global_batch_size=args.global_batch_size,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        _write_json(
            args.results_root / "loss_mask_ablation_summary.json",
            {"schema_version": "1.0", "status": "dry_run_complete"},
        )
        _write_run_manifest(
            results_root=args.results_root,
            run_plan=run_plan,
            status="dry_run_complete",
            artifacts={
                "run_plan": str(args.results_root / "run_plan.json"),
                "config_diff": str(args.results_root / "config_diff.json"),
                "summary": str(args.results_root / "loss_mask_ablation_summary.json"),
            },
            environment=environment,
        )
        print("exp09_loss_masking_summary=" + json.dumps({"status": "dry_run_complete"}))
        return

    for run in _loss_mask_runs():
        adapter_path = adapters[run.profile.name]
        _generate_and_score(
            run=run,
            dataset_name="validation_slice",
            dataset_path=validation_slice,
            adapter_path=adapter_path,
            output_root=args.results_root,
            logs_root=args.logs_root,
            cache_dir=args.cache_dir,
            generation_batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
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
            dry_run=args.dry_run,
        )

    runs_by_name = {run.profile.name: run for run in _loss_mask_runs()}
    for dataset_name in ("validation_slice", "no_tool_dev"):
        _compare_pair(
            baseline=cast(Any, runs_by_name["assistant_only_short"]),
            candidate=cast(Any, runs_by_name["full_sequence_short"]),
            dataset_name=dataset_name,
            results_root=args.results_root,
            logs_root=args.logs_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    rows = _aggregate_table(args.results_root)
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "experiment_id": "exp-09a",
        "task_id": "task-14",
        "production_policy": "assistant_only",
        "ablation_policy": "full_sequence",
        "interpretation": (
            "Short full-sequence run is diagnostic only and is not eligible to "
            "replace the production assistant/tool-call-output-only policy."
        ),
        "aggregate_metrics": rows,
        "checkpoint_paths": {
            name: str(path)
            for name, path in adapters.items()
        },
        "config_diff": config_diff,
    }
    _write_json(args.results_root / "loss_mask_ablation_summary.json", summary)
    _write_markdown_summary(
        path=args.results_root / "loss_mask_ablation_summary.md",
        rows=rows,
        config_diff=config_diff,
    )
    shutil.copy2(
        args.results_root / "loss_mask_ablation_summary.json",
        args.results_root / "report.json",
    )
    _write_run_manifest(
        results_root=args.results_root,
        run_plan=run_plan,
        status="complete",
        artifacts={
            "run_plan": str(args.results_root / "run_plan.json"),
            "config_diff": str(args.results_root / "config_diff.json"),
            "loss_mask_audit": str(
                args.results_root
                / "loss_mask_audit"
                / "loss_mask_audit_report.json",
            ),
            "automodel_loss_mask_probe": str(
                args.results_root / "automodel_loss_mask_probe.json",
            ),
            "summary": str(args.results_root / "loss_mask_ablation_summary.json"),
            "report": str(args.results_root / "report.json"),
        },
        environment=environment,
    )
    checksum_path = _write_checksum_manifest(args.results_root)
    print(
        "exp09_loss_masking_summary="
        + json.dumps(
            {
                "status": "complete",
                "checksums": str(checksum_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
