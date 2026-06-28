#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import shutil
import subprocess
import sys
import time
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
    EXPECTED_TRAIN_PATH,
    EXPECTED_VALIDATION_PATH,
    clone_full_sft_config_for_stage,
    load_yaml_config,
    validate_full_sft_config,
    validation_to_dict,
    write_yaml_config,
)
from function_calling_ft.generation import read_jsonl  # noqa: E402
from function_calling_ft.split_guard import assert_split_allowed  # noqa: E402


DEFAULT_CONFIG = Path("configs/exp05a_full_sft/full_sft_pilot.yaml")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-05a")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-05a")
DEFAULT_CHECKPOINT_ROOT = Path(
    "/workspace/checkpoints/exp-05a/full-parameter-sft-bf16-pilot",
)
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_NO_TOOL_DEV = Path("/workspace/data/eval/no_tool_relevance_v1/dev.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 5A full-parameter SFT feasibility pilot.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-tool-dev", type=Path, default=DEFAULT_NO_TOOL_DEV)
    parser.add_argument("--automodel-bin", default="automodel")
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--validation-records", type=int, default=100)
    parser.add_argument("--no-tool-records", type=int, default=100)
    parser.add_argument("--reload-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-with-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_sample(
    *,
    source: Path,
    output: Path,
    records: int,
    dry_run: bool,
) -> None:
    if dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        return
    selected = read_jsonl(source)[:records]
    if len(selected) != records:
        raise ValueError(f"{source} has only {len(selected)} records, need {records}")
    _write_jsonl(output, selected)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _stage_config(
    *,
    base_config: Mapping[str, Any],
    output_path: Path,
    checkpoint_dir: Path,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    validation_path: Path | None,
    checkpoint_enabled: bool,
    activation_checkpointing_enabled: bool,
) -> None:
    staged = clone_full_sft_config_for_stage(
        base_config,
        checkpoint_dir=str(checkpoint_dir),
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
        val_every_steps=val_every_steps,
        validation_path=str(validation_path) if validation_path is not None else None,
        checkpoint_enabled=checkpoint_enabled,
        activation_checkpointing_enabled=activation_checkpointing_enabled,
    )
    write_yaml_config(output_path, staged)
    validation = validate_full_sft_config(
        output_path,
        allow_alternate_validation_path=validation_path is not None,
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
    torch_trace_path: Path | None,
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
    ]
    if torch_trace_path is not None:
        command.extend(["--torch-memory-trace-output", str(torch_trace_path)])
    command.extend(["--", automodel_bin, "finetune", "llm", "-c", str(config_path)])
    return command


def _run_training_stage(
    *,
    stage_name: str,
    base_config: Mapping[str, Any],
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    validation_path: Path | None,
    checkpoint_enabled: bool,
    memory_trace: bool,
    dry_run: bool,
    activation_checkpointing_enabled: bool,
) -> dict[str, Any]:
    stage_root = results_root / stage_name
    checkpoint_dir = checkpoint_root / stage_name
    config_path = stage_root / "resolved_config.yaml"
    _stage_config(
        base_config=base_config,
        output_path=config_path,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
        val_every_steps=val_every_steps,
        validation_path=validation_path,
        checkpoint_enabled=checkpoint_enabled,
        activation_checkpointing_enabled=activation_checkpointing_enabled,
    )
    log_path = logs_root / f"{stage_name}.log"
    metrics_path = stage_root / "training_metrics.json"
    command = _training_command(
        config_path=config_path,
        checkpoint_path=checkpoint_dir,
        log_path=log_path,
        metrics_path=metrics_path,
        gpu_log_path=logs_root / f"{stage_name}-gpu.csv",
        torch_memory_path=stage_root / "training_torch_memory.json",
        torch_trace_path=stage_root / "training_torch_memory_trace.json"
        if memory_trace
        else None,
        automodel_bin=automodel_bin,
    )
    _write_json(
        stage_root / "train_command.json",
        {
            "schema_version": "1.0",
            "stage_name": stage_name,
            "command": command,
            "max_steps": max_steps,
            "checkpoint_enabled": checkpoint_enabled,
            "memory_trace": memory_trace,
            "activation_checkpointing": activation_checkpointing_enabled,
        },
    )
    return_code = _run_logged(command, log_path=log_path, dry_run=dry_run)
    if dry_run:
        metrics = {
            "dry_run": True,
            "return_code": 0,
            "stage_name": stage_name,
            "max_steps": max_steps,
            "losses_are_finite": True,
            "oom_event_count": 0,
            "checkpoint_exists_after": checkpoint_enabled,
        }
        _write_json(metrics_path, metrics)
        return metrics
    metrics = _read_json(metrics_path)
    if not metrics:
        metrics = {"return_code": return_code, "metrics_missing": True}
        _write_json(metrics_path, metrics)
    return metrics


def _run_simple_stage(
    *,
    stage_name: str,
    command: list[str],
    output_path: Path,
    logs_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    return_code = _run_logged(command, log_path=logs_root / f"{stage_name}.log", dry_run=dry_run)
    if dry_run:
        payload = {"schema_version": "1.0", "dry_run": True, "return_code": 0}
        _write_json(output_path, payload)
        return payload
    payload = _read_json(output_path)
    payload.setdefault("return_code", return_code)
    if return_code != 0:
        payload["failed"] = True
        _write_json(output_path, payload)
    return payload


def _stage_success(metrics: Mapping[str, Any], *, require_checkpoint: bool = False) -> bool:
    return (
        int(metrics.get("return_code", 0) or 0) == 0
        and not bool(metrics.get("aborted", False))
        and bool(metrics.get("losses_are_finite", True))
        and int(metrics.get("oom_event_count", 0) or 0) == 0
        and (not require_checkpoint or bool(metrics.get("checkpoint_exists_after", False)))
    )


def _oom_or_memory_failure(metrics: Mapping[str, Any]) -> bool:
    abort_reason = str(metrics.get("abort_reason") or "").lower()
    return int(metrics.get("oom_event_count", 0) or 0) > 0 or "oom" in abort_reason


def _checkpoint_file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _run_sequence(
    *,
    base_config: Mapping[str, Any],
    args: argparse.Namespace,
    validation_sample_path: Path,
    activation_checkpointing_enabled: bool,
    suffix: str,
) -> dict[str, Any]:
    stage_prefix = f"{suffix}-" if suffix else ""
    one_step = _run_training_stage(
        stage_name=f"{stage_prefix}automodel-one-step",
        base_config=base_config,
        results_root=args.results_root,
        logs_root=args.logs_root,
        checkpoint_root=args.checkpoint_root,
        automodel_bin=args.automodel_bin,
        max_steps=1,
        ckpt_every_steps=1000,
        val_every_steps=1000,
        validation_path=validation_sample_path,
        checkpoint_enabled=False,
        memory_trace=False,
        dry_run=args.dry_run,
        activation_checkpointing_enabled=activation_checkpointing_enabled,
    )
    five_step: dict[str, Any] | None = None
    pilot: dict[str, Any] | None = None
    if _stage_success(one_step):
        five_step = _run_training_stage(
            stage_name=f"{stage_prefix}five-step",
            base_config=base_config,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            max_steps=5,
            ckpt_every_steps=1000,
            val_every_steps=1000,
            validation_path=validation_sample_path,
            checkpoint_enabled=False,
            memory_trace=False,
            dry_run=args.dry_run,
            activation_checkpointing_enabled=activation_checkpointing_enabled,
        )
    if five_step is not None and _stage_success(five_step):
        pilot = _run_training_stage(
            stage_name=f"{stage_prefix}pilot",
            base_config=base_config,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            max_steps=args.pilot_steps,
            ckpt_every_steps=args.pilot_steps,
            val_every_steps=max(1, args.pilot_steps // 2),
            validation_path=validation_sample_path,
            checkpoint_enabled=True,
            memory_trace=True,
            dry_run=args.dry_run,
            activation_checkpointing_enabled=activation_checkpointing_enabled,
        )
    return {
        "activation_checkpointing": activation_checkpointing_enabled,
        "stage_prefix": stage_prefix,
        "one_step": one_step,
        "five_step": five_step,
        "pilot": pilot,
    }


def _gate_decision(sequence: Mapping[str, Any], *, reload_ok: bool) -> str:
    one_step = sequence.get("one_step") or {}
    five_step = sequence.get("five_step") or {}
    pilot = sequence.get("pilot") or {}
    for metrics in (one_step, five_step, pilot):
        if metrics and _oom_or_memory_failure(metrics):
            return "FAIL_MEMORY"
    if not _stage_success(one_step):
        return "FAIL_STABILITY"
    if not _stage_success(five_step):
        return "FAIL_STABILITY"
    if not _stage_success(pilot, require_checkpoint=True):
        return "FAIL_CHECKPOINT" if pilot else "FAIL_STABILITY"
    if not reload_ok:
        return "FAIL_CHECKPOINT"
    return (
        "PASS_WITH_CHECKPOINTING"
        if bool(sequence.get("activation_checkpointing"))
        else "PASS"
    )


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# Experiment 5A Full-SFT Feasibility Pilot",
        "",
        f"Gate decision: `{decision.get('gate_decision')}`",
        "",
        f"Activation checkpointing: `{decision.get('activation_checkpointing')}`",
        f"Pilot checkpoint: `{decision.get('pilot_checkpoint_path')}`",
        f"Checkpoint size bytes: `{decision.get('checkpoint_size_bytes')}`",
        "",
        "This report is generated from staged pilot artifacts.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validation = validate_full_sft_config(args.config)
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_PATH,
        command_name="exp05a-full-sft-train",
    )
    validation_decision = assert_split_allowed(
        EXPECTED_VALIDATION_PATH,
        command_name="exp05a-full-sft-validation",
    )
    no_tool_decision = assert_split_allowed(
        args.no_tool_dev,
        command_name="exp05a-full-sft-no-tool-reload",
    )

    validation_sample_path = args.results_root / "validation_sample_100.jsonl"
    no_tool_sample_path = args.results_root / "no_tool_sample_100.jsonl"
    _write_sample(
        source=Path(EXPECTED_VALIDATION_PATH),
        output=validation_sample_path,
        records=args.validation_records,
        dry_run=args.dry_run,
    )
    _write_sample(
        source=args.no_tool_dev,
        output=no_tool_sample_path,
        records=args.no_tool_records,
        dry_run=args.dry_run,
    )

    base_config = load_yaml_config(args.config)
    _write_json(
        args.results_root / "run_plan.json",
        {
            "schema_version": "1.0",
            "config": str(args.config),
            "method": "full_parameter_sft",
            "dry_run": args.dry_run,
            "train_split": train_decision.__dict__,
            "validation_split": validation_decision.__dict__,
            "no_tool_split": no_tool_decision.__dict__,
            "train_records": 2019,
            "validation_records": args.validation_records,
            "no_tool_records": args.no_tool_records,
            "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
            "local_batch_size": 1,
            "learning_rate": EXPECTED_LR,
            "pilot_steps": args.pilot_steps,
            "retry_with_activation_checkpointing": args.retry_with_activation_checkpointing,
        },
    )

    package_report = _run_simple_stage(
        stage_name="automodel-package-inspection",
        command=[
            sys.executable,
            "scripts/inspect_automodel_package.py",
            "--output",
            str(args.results_root / "automodel_package_report.json"),
        ],
        output_path=args.results_root / "automodel_package_report.json",
        logs_root=args.logs_root,
        dry_run=args.dry_run,
    )
    load_probe = _run_simple_stage(
        stage_name="load-probe",
        command=[
            sys.executable,
            "scripts/probe_full_sft_runtime.py",
            "--stage",
            "load",
            "--dataset",
            EXPECTED_TRAIN_PATH,
            "--output",
            str(args.results_root / "load_probe.json"),
            "--model-name",
            EXPECTED_MODEL_NAME,
            "--model-revision",
            EXPECTED_MODEL_REVISION,
            "--cache-dir",
            str(args.cache_dir),
        ],
        output_path=args.results_root / "load_probe.json",
        logs_root=args.logs_root,
        dry_run=args.dry_run,
    )
    forward_probe = _run_simple_stage(
        stage_name="forward-probe",
        command=[
            sys.executable,
            "scripts/probe_full_sft_runtime.py",
            "--stage",
            "forward",
            "--dataset",
            EXPECTED_TRAIN_PATH,
            "--output",
            str(args.results_root / "forward_probe.json"),
            "--model-name",
            EXPECTED_MODEL_NAME,
            "--model-revision",
            EXPECTED_MODEL_REVISION,
            "--cache-dir",
            str(args.cache_dir),
        ],
        output_path=args.results_root / "forward_probe.json",
        logs_root=args.logs_root,
        dry_run=args.dry_run,
    )
    custom_step_probe = _run_simple_stage(
        stage_name="custom-one-step-probe",
        command=[
            sys.executable,
            "scripts/probe_full_sft_runtime.py",
            "--stage",
            "step",
            "--dataset",
            EXPECTED_TRAIN_PATH,
            "--output",
            str(args.results_root / "custom_one_step_probe.json"),
            "--model-name",
            EXPECTED_MODEL_NAME,
            "--model-revision",
            EXPECTED_MODEL_REVISION,
            "--cache-dir",
            str(args.cache_dir),
        ],
        output_path=args.results_root / "custom_one_step_probe.json",
        logs_root=args.logs_root,
        dry_run=args.dry_run,
    )

    sequence = _run_sequence(
        base_config=base_config,
        args=args,
        validation_sample_path=validation_sample_path,
        activation_checkpointing_enabled=False,
        suffix="",
    )
    retry_sequence: dict[str, Any] | None = None
    if (
        args.retry_with_activation_checkpointing
        and _gate_decision(sequence, reload_ok=False) == "FAIL_MEMORY"
    ):
        retry_sequence = _run_sequence(
            base_config=base_config,
            args=args,
            validation_sample_path=validation_sample_path,
            activation_checkpointing_enabled=True,
            suffix="activation-checkpointing",
        )
        active_sequence = retry_sequence
    else:
        active_sequence = sequence

    pilot_stage_name = (
        f"{active_sequence['stage_prefix']}pilot"
        if active_sequence.get("stage_prefix")
        else "pilot"
    )
    pilot_checkpoint = args.checkpoint_root / pilot_stage_name
    reload_report: dict[str, Any] = {}
    reload_return_code = 0
    if _stage_success(active_sequence.get("pilot") or {}, require_checkpoint=True):
        reload_command = [
            sys.executable,
            "scripts/reload_full_sft_check.py",
            "--checkpoint-path",
            str(pilot_checkpoint),
            "--output-dir",
            str(args.results_root / "reload_check"),
            "--tool-dataset",
            str(validation_sample_path),
            "--no-tool-dataset",
            str(no_tool_sample_path),
            "--tokenizer-name",
            EXPECTED_MODEL_NAME,
            "--tokenizer-revision",
            EXPECTED_MODEL_REVISION,
            "--cache-dir",
            str(args.cache_dir),
            "--batch-size",
            str(args.reload_batch_size),
        ]
        reload_return_code = _run_logged(
            reload_command,
            log_path=args.logs_root / "reload-check.log",
            dry_run=args.dry_run,
        )
        reload_report = (
            {"dry_run": True, "deterministic": True}
            if args.dry_run
            else _read_json(args.results_root / "reload_check" / "reload_check.json")
        )

    reload_ok = reload_return_code == 0 and bool(reload_report.get("deterministic", False))
    gate = _gate_decision(active_sequence, reload_ok=reload_ok)
    checkpoint_size = _directory_size(pilot_checkpoint)
    decision = {
        "schema_version": "1.0",
        "gate_decision": gate,
        "activation_checkpointing": bool(active_sequence.get("activation_checkpointing")),
        "pilot_stage_name": pilot_stage_name,
        "pilot_checkpoint_path": str(pilot_checkpoint),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_file_count": _checkpoint_file_count(pilot_checkpoint),
        "package_report_ok": not bool(package_report.get("failed")),
        "load_probe": load_probe,
        "forward_probe": forward_probe,
        "custom_one_step_probe": custom_step_probe,
        "no_checkpoint_sequence": sequence,
        "activation_checkpointing_sequence": retry_sequence,
        "reload_check": reload_report,
    }
    _write_json(args.results_root / "gate_decision.json", decision)
    _write_report(args.results_root / "gate_decision.md", decision)

    files_for_checksums = [
        path
        for path in args.results_root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    if files_for_checksums:
        write_checksums(args.results_root / "checksums.sha256", files_for_checksums)

    if shutil.which("du") and pilot_checkpoint.exists():
        print(f"checkpoint_size_bytes={checkpoint_size}", flush=True)
    print("exp05a_full_sft_decision=" + json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
