#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation import write_checksums  # noqa: E402
from function_calling_ft.full_sft import (  # noqa: E402
    EXPECTED_GLOBAL_BATCH_SIZE,
    EXPECTED_LR,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_TRAIN_10K_PATH,
    EXPECTED_VALIDATION_PATH,
    clone_full_sft_config_for_stage,
    load_yaml_config,
    validate_full_sft_config,
    validation_to_dict,
    write_yaml_config,
)
from function_calling_ft.split_guard import assert_split_allowed  # noqa: E402


DEFAULT_CONFIG = Path("configs/exp05b_full_sft/full_sft_10k.yaml")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-05b")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-05b")
DEFAULT_CHECKPOINT_ROOT = Path(
    "/workspace/checkpoints/exp-05b/full-parameter-sft-bf16-10k-epoch1",
)
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_TOOL_DEV = Path("/workspace/data/processed/xlam_splits_v1/dev_eval_1k.jsonl")
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")
DEFAULT_LORA_TOOL_SCORED = Path(
    "/workspace/results/exp-03/full-epoch-eval/tool_dev/scored_predictions.jsonl",
)
DEFAULT_LORA_NO_TOOL_SCORED = Path(
    "/workspace/results/exp-03/full-epoch-eval/no_tool_dev/scored_predictions.jsonl",
)
DEFAULT_QLORA_TOOL_SCORED = Path(
    "/workspace/results/exp-04/full-epoch-eval/tool_dev/scored_predictions.jsonl",
)
DEFAULT_QLORA_NO_TOOL_SCORED = Path(
    "/workspace/results/exp-04/full-epoch-eval/no_tool_dev/scored_predictions.jsonl",
)
COMPARISON_METRICS = (
    "strict_complete_match",
    "schema_equivalent_complete_match",
    "executable_complete_match",
    "tool_call_emitted",
    "no_tool_false_positive",
)
ADVANCED_SCORE_KEYS = (
    "total_records",
    "strict_complete_match_rate",
    "schema_equivalent_complete_match_rate",
    "executable_complete_match_rate",
    "function_name_precision",
    "function_name_recall",
    "function_name_f1",
    "complete_call_precision",
    "complete_call_recall",
    "complete_call_f1",
    "average_argument_name_accuracy",
    "average_argument_value_accuracy",
    "expected_call_count",
    "predicted_call_count",
    "missing_call_count",
    "extra_call_count",
    "malformed_tool_call_count",
    "no_tool_call_emitted_count",
    "tool_call_emitted_count",
    "extra_prose_with_tool_call_count",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 5B controlled 10K full-parameter SFT.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tool-dev", type=Path, default=DEFAULT_TOOL_DEV)
    parser.add_argument("--no-tool-dev", type=Path, default=DEFAULT_NO_TOOL_DEV)
    parser.add_argument("--automodel-bin", default="automodel")
    parser.add_argument("--checkpoint-interval", type=int, default=834)
    parser.add_argument("--validation-interval", type=int, default=834)
    parser.add_argument("--tool-records", type=int, default=1000)
    parser.add_argument("--no-tool-records", type=int, default=100)
    parser.add_argument("--reload-batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--min-free-checkpoint-gb", type=float, default=80.0)
    parser.add_argument("--min-free-results-gb", type=float, default=20.0)
    parser.add_argument("--lora-tool-scored", type=Path, default=DEFAULT_LORA_TOOL_SCORED)
    parser.add_argument(
        "--lora-no-tool-scored",
        type=Path,
        default=DEFAULT_LORA_NO_TOOL_SCORED,
    )
    parser.add_argument(
        "--qlora-tool-scored",
        type=Path,
        default=DEFAULT_QLORA_TOOL_SCORED,
    )
    parser.add_argument(
        "--qlora-no-tool-scored",
        type=Path,
        default=DEFAULT_QLORA_NO_TOOL_SCORED,
    )
    parser.add_argument("--dry-run", action="store_true")
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


def _run_logged(command: list[str], *, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + printable + "\n")
        if dry_run:
            log_file.write("dry_run=true\n")
            return 0
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.monotonic() - started
        log_file.write(f"\nexit_code={return_code} elapsed_seconds={elapsed:.3f}\n")
        return return_code


def _dataset_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _checkpoint_file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _full_epoch_steps(train_count: int, global_batch_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    return math.ceil(train_count / global_batch_size)


def _storage_report(path: Path, min_free_gb: float) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1024**3
    return {
        "path": str(path),
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": free_gb,
        "min_free_gb": min_free_gb,
        "ok": free_gb >= min_free_gb,
    }


def _stage_config(
    *,
    base_config: Mapping[str, Any],
    output_path: Path,
    checkpoint_dir: Path,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
) -> None:
    staged = clone_full_sft_config_for_stage(
        base_config,
        checkpoint_dir=str(checkpoint_dir),
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
        val_every_steps=val_every_steps,
        profile="exp05b",
        checkpoint_enabled=True,
        activation_checkpointing_enabled=False,
        policy_updates={
            "max_steps": max_steps,
            "checkpoint_interval_steps": ckpt_every_steps,
            "validation_interval_steps": val_every_steps,
        },
    )
    write_yaml_config(output_path, staged)
    validation = validate_full_sft_config(
        output_path,
        profile="exp05b",
        checkpoint_root_override=checkpoint_dir.parent,
    )
    _write_json(output_path.parent / "config_validation.json", validation_to_dict(validation))
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))


def _training_command(
    *,
    config_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    metrics_path: Path,
    gpu_log_path: Path,
    torch_memory_path: Path,
    automodel_bin: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_training_with_monitor.py",
        "--log",
        str(log_path),
        "--metrics-output",
        str(metrics_path),
        "--gpu-log",
        str(gpu_log_path),
        "--torch-memory-output",
        str(torch_memory_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--checkpoint-root",
        "/workspace/checkpoints",
        "--require-checkpoint-root-mount",
        "--expected-training-method",
        "full_parameter_sft",
        "--min-full-trainable-ratio",
        "0.95",
        "--",
        automodel_bin,
        "finetune",
        "llm",
        "-c",
        str(config_path),
    ]
    return command


def _run_training_stage(
    *,
    base_config: Mapping[str, Any],
    args: argparse.Namespace,
    max_steps: int,
) -> dict[str, Any]:
    stage_name = "full-epoch"
    stage_root = args.results_root / stage_name
    checkpoint_dir = args.checkpoint_root / stage_name
    config_path = stage_root / "resolved_config.yaml"
    _stage_config(
        base_config=base_config,
        output_path=config_path,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        ckpt_every_steps=args.checkpoint_interval,
        val_every_steps=args.validation_interval,
    )
    log_path = args.logs_root / f"{stage_name}.log"
    metrics_path = stage_root / "training_metrics.json"
    command = _training_command(
        config_path=config_path,
        checkpoint_path=checkpoint_dir,
        log_path=log_path,
        metrics_path=metrics_path,
        gpu_log_path=args.logs_root / f"{stage_name}-gpu.csv",
        torch_memory_path=stage_root / "training_torch_memory.json",
        automodel_bin=args.automodel_bin,
    )
    _write_json(
        stage_root / "train_command.json",
        {
            "schema_version": "1.0",
            "stage_name": stage_name,
            "command": command,
            "max_steps": max_steps,
            "checkpoint_interval_steps": args.checkpoint_interval,
            "validation_interval_steps": args.validation_interval,
            "activation_checkpointing": False,
        },
    )
    return_code = _run_logged(command, log_path=log_path, dry_run=args.dry_run)
    if args.dry_run:
        metrics = {
            "dry_run": True,
            "return_code": 0,
            "stage_name": stage_name,
            "max_steps": max_steps,
            "losses_are_finite": True,
            "oom_event_count": 0,
            "checkpoint_exists_after": True,
        }
        _write_json(metrics_path, metrics)
        return metrics
    metrics = _read_json(metrics_path)
    if not metrics:
        metrics = {"return_code": return_code, "metrics_missing": True}
        _write_json(metrics_path, metrics)
    return metrics


def _stage_success(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("return_code", 0) or 0) == 0
        and not bool(metrics.get("aborted", False))
        and bool(metrics.get("losses_are_finite", True))
        and int(metrics.get("oom_event_count", 0) or 0) == 0
        and bool(metrics.get("checkpoint_exists_after", False))
    )


def _selected_checkpoint_path(checkpoint_dir: Path) -> tuple[Path, str]:
    lowest_val = checkpoint_dir / "LOWEST_VAL"
    latest = checkpoint_dir / "LATEST"
    if lowest_val.exists():
        return lowest_val, "lowest_validation_loss"
    if latest.exists():
        return latest, "latest_checkpoint"
    return checkpoint_dir, "checkpoint_root"


def _run_reload_eval(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> dict[str, Any]:
    output_dir = args.results_root / "selected-checkpoint-eval"
    command = [
        sys.executable,
        "scripts/reload_full_sft_check.py",
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--tool-dataset",
        str(args.tool_dev),
        "--no-tool-dataset",
        str(args.no_tool_dev),
        "--tool-limit",
        str(args.tool_records),
        "--no-tool-limit",
        str(args.no_tool_records),
        "--tokenizer-name",
        EXPECTED_MODEL_NAME,
        "--tokenizer-revision",
        EXPECTED_MODEL_REVISION,
        "--cache-dir",
        str(args.cache_dir),
        "--batch-size",
        str(args.reload_batch_size),
    ]
    return_code = _run_logged(
        command,
        log_path=args.logs_root / "selected-checkpoint-reload-eval.log",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        report = {"dry_run": True, "deterministic": True, "return_code": 0}
        _write_json(output_dir / "reload_check.json", report)
        return report
    report = _read_json(output_dir / "reload_check.json")
    report["return_code"] = return_code
    _write_json(output_dir / "reload_check.json", report)
    return report


def _compare_scored(
    *,
    name: str,
    baseline_scored: Path,
    candidate_scored: Path,
    output_dir: Path,
    logs_root: Path,
    dry_run: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if dry_run:
        payload = {"dry_run": True, "skipped": True, "name": name}
        _write_json(output_dir / "comparison.json", payload)
        return payload
    if not baseline_scored.is_file() or not candidate_scored.is_file():
        payload = {
            "skipped": True,
            "name": name,
            "missing_baseline": not baseline_scored.is_file(),
            "missing_candidate": not candidate_scored.is_file(),
            "baseline_scored": str(baseline_scored),
            "candidate_scored": str(candidate_scored),
        }
        _write_json(output_dir / "comparison.json", payload)
        return payload
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
    ]
    for metric in COMPARISON_METRICS:
        command.extend(["--metric", metric])
    return_code = _run_logged(command, log_path=logs_root / f"{name}.log", dry_run=False)
    summary = _read_json(output_dir / "comparison.json")
    summary["return_code"] = return_code
    _write_json(output_dir / "comparison.json", summary)
    return summary


def _score_subset(scores_path: Path) -> dict[str, Any]:
    scores = _read_json(scores_path)
    return {key: scores.get(key) for key in ADVANCED_SCORE_KEYS if key in scores}


def _write_method_comparison(args: argparse.Namespace) -> dict[str, Any]:
    selected_eval = args.results_root / "selected-checkpoint-eval"
    tool_scored = selected_eval / "tool_reload_eval" / "scored_predictions.jsonl"
    no_tool_scored = selected_eval / "no_tool_reload_eval" / "scored_predictions.jsonl"
    comparison_root = args.results_root / "method-comparisons"
    comparisons = {
        "lora_vs_full_sft_tool": _compare_scored(
            name="lora-vs-full-sft-tool",
            baseline_scored=args.lora_tool_scored,
            candidate_scored=tool_scored,
            output_dir=comparison_root / "lora-vs-full-sft" / "tool_dev",
            logs_root=args.logs_root,
            dry_run=args.dry_run,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "lora_vs_full_sft_no_tool": _compare_scored(
            name="lora-vs-full-sft-no-tool",
            baseline_scored=args.lora_no_tool_scored,
            candidate_scored=no_tool_scored,
            output_dir=comparison_root / "lora-vs-full-sft" / "no_tool_dev",
            logs_root=args.logs_root,
            dry_run=args.dry_run,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "qlora_vs_full_sft_tool": _compare_scored(
            name="qlora-vs-full-sft-tool",
            baseline_scored=args.qlora_tool_scored,
            candidate_scored=tool_scored,
            output_dir=comparison_root / "qlora-vs-full-sft" / "tool_dev",
            logs_root=args.logs_root,
            dry_run=args.dry_run,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "qlora_vs_full_sft_no_tool": _compare_scored(
            name="qlora-vs-full-sft-no-tool",
            baseline_scored=args.qlora_no_tool_scored,
            candidate_scored=no_tool_scored,
            output_dir=comparison_root / "qlora-vs-full-sft" / "no_tool_dev",
            logs_root=args.logs_root,
            dry_run=args.dry_run,
            bootstrap_samples=args.bootstrap_samples,
        ),
    }
    score_table = {
        "full_sft_tool_dev": _score_subset(
            selected_eval / "tool_reload_eval" / "scores.json",
        ),
        "full_sft_no_tool_dev": _score_subset(
            selected_eval / "no_tool_reload_eval" / "scores.json",
        ),
        "lora_tool_dev": _score_subset(args.lora_tool_scored.parent / "scores.json"),
        "lora_no_tool_dev": _score_subset(
            args.lora_no_tool_scored.parent / "scores.json",
        ),
        "qlora_tool_dev": _score_subset(args.qlora_tool_scored.parent / "scores.json"),
        "qlora_no_tool_dev": _score_subset(
            args.qlora_no_tool_scored.parent / "scores.json",
        ),
    }
    payload = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "comparison_metrics": list(COMPARISON_METRICS),
        "paired_comparisons": comparisons,
        "advanced_score_table": score_table,
    }
    _write_json(args.results_root / "method_selection.json", payload)
    return payload


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# Experiment 5B Full-Parameter SFT 10K Run",
        "",
        f"Status: `{decision.get('status')}`",
        f"Selected checkpoint policy: `{decision.get('selected_checkpoint_policy')}`",
        f"Selected checkpoint: `{decision.get('selected_checkpoint_path')}`",
        f"Checkpoint size bytes: `{decision.get('checkpoint_size_bytes')}`",
        "",
        "The task decision is generated from training, clean reload, and paired comparison artifacts.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    validation = validate_full_sft_config(args.config, profile="exp05b")
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint_storage = _storage_report(
        args.checkpoint_root,
        args.min_free_checkpoint_gb,
    )
    results_storage = _storage_report(args.results_root, args.min_free_results_gb)
    if not args.dry_run and (not checkpoint_storage["ok"] or not results_storage["ok"]):
        _write_json(
            args.results_root / "storage_preflight.json",
            {
                "checkpoint_storage": checkpoint_storage,
                "results_storage": results_storage,
            },
        )
        raise SystemExit("insufficient retained storage for full-SFT run")

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_10K_PATH,
        command_name="exp05b-full-sft-train",
    )
    validation_decision = assert_split_allowed(
        EXPECTED_VALIDATION_PATH,
        command_name="exp05b-full-sft-validation",
    )
    tool_dev_decision = assert_split_allowed(
        args.tool_dev,
        command_name="exp05b-full-sft-tool-dev-eval",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev,
        command_name="exp05b-full-sft-no-tool-dev-eval",
    )

    train_path = Path(EXPECTED_TRAIN_10K_PATH)
    validation_path = Path(EXPECTED_VALIDATION_PATH)
    train_count = _dataset_count(train_path) if train_path.is_file() else 10003
    validation_count = _dataset_count(validation_path) if validation_path.is_file() else None
    max_steps = _full_epoch_steps(train_count, EXPECTED_GLOBAL_BATCH_SIZE)

    base_config = load_yaml_config(args.config)
    run_plan = {
        "schema_version": "1.0",
        "started_at": started_at,
        "config": str(args.config),
        "method": "full_parameter_sft",
        "dry_run": args.dry_run,
        "train_split": train_decision.__dict__,
        "validation_split": validation_decision.__dict__,
        "tool_dev_split": tool_dev_decision.__dict__,
        "no_tool_split": no_tool_decision.__dict__,
        "train_records": train_count,
        "validation_records": validation_count,
        "tool_eval_records": args.tool_records,
        "no_tool_eval_records": args.no_tool_records,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
        "local_batch_size": 1,
        "learning_rate": EXPECTED_LR,
        "max_steps": max_steps,
        "checkpoint_interval_steps": args.checkpoint_interval,
        "validation_interval_steps": args.validation_interval,
        "activation_checkpointing": False,
        "checkpoint_storage": checkpoint_storage,
        "results_storage": results_storage,
    }
    _write_json(args.results_root / "run_plan.json", run_plan)

    training_metrics = _run_training_stage(
        base_config=base_config,
        args=args,
        max_steps=max_steps,
    )
    checkpoint_dir = args.checkpoint_root / "full-epoch"
    selected_checkpoint, selected_policy = _selected_checkpoint_path(checkpoint_dir)
    reload_report: dict[str, Any] = {}
    if _stage_success(training_metrics):
        reload_report = _run_reload_eval(args=args, checkpoint_path=selected_checkpoint)
    reload_ok = (
        int(reload_report.get("return_code", 0) or 0) == 0
        and bool(reload_report.get("deterministic", False))
    )
    comparisons = _write_method_comparison(args) if reload_ok or args.dry_run else {}

    checkpoint_size = _directory_size(checkpoint_dir)
    decision = {
        "schema_version": "1.0",
        "started_at": started_at,
        "ended_at": utc_now(),
        "status": "complete" if _stage_success(training_metrics) and reload_ok else "failed",
        "training_succeeded": _stage_success(training_metrics),
        "reload_succeeded": reload_ok,
        "selected_checkpoint_policy": selected_policy,
        "selected_checkpoint_path": str(selected_checkpoint),
        "checkpoint_root": str(checkpoint_dir),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_file_count": _checkpoint_file_count(checkpoint_dir),
        "training_metrics": training_metrics,
        "reload_check": reload_report,
        "method_comparison": comparisons,
    }
    _write_json(args.results_root / "completion_summary.json", decision)
    _write_report(args.results_root / "completion_summary.md", decision)

    files_for_checksums = [
        path
        for path in args.results_root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    if files_for_checksums:
        write_checksums(args.results_root / "checksums.sha256", files_for_checksums)

    print("exp05b_full_sft_decision=" + json.dumps(decision, sort_keys=True))
    if decision["status"] != "complete" and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
