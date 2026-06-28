#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import read_jsonl  # noqa: E402
from function_calling_ft.reference_lora import (  # noqa: E402
    EXPECTED_TRAIN_PATH,
    EXPECTED_VALIDATION_PATH,
    clone_training_config_for_stage,
    load_yaml_config,
    validate_reference_lora_config,
    validate_reference_qlora_config,
    validation_to_dict,
    write_yaml_config,
)
from function_calling_ft.split_guard import assert_split_allowed  # noqa: E402


DEFAULT_CONFIG = Path("configs/exp03_reference_lora/lora_r8_attention.yaml")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-03")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-03")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-03/reference-bf16-lora-r8-attention")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 03 reference BF16 LoRA batch probes and pilot.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--method",
        choices=("lora", "qlora"),
        default="lora",
        help="Training method profile used for config validation and reload checks.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--automodel-bin", default="automodel")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--probe-validation-records", type=int, default=32)
    parser.add_argument(
        "--pilot-trace-max-reserved-gb",
        type=float,
        default=38.0,
        help=(
            "When memory tracing is enabled, choose the largest successful "
            "probe whose untraced peak reserved VRAM is at or below this "
            "threshold."
        ),
    )
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help=(
            "Optimizer-step global batch size. Defaults to the selected local "
            "batch size; set this larger than --local-batch-size to preserve "
            "the token budget through gradient accumulation on smaller GPUs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reload-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload saved adapters on an NF4 4-bit base for deterministic checks.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run one full epoch after pilot approval. Default stops after pilot.",
    )
    parser.add_argument(
        "--disable-memory-trace",
        action="store_true",
        help="Disable hook-level pilot memory trace.",
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


def _parse_batch_sizes(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--batch-sizes must contain positive integers")
    return values


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _write_validation_sample(
    *,
    source: Path,
    output: Path,
    records: int,
) -> None:
    if records <= 0:
        raise ValueError("probe validation records must be positive")
    source_records = read_jsonl(source)
    selected = source_records[:records]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in selected:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _run_logged(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + printable + "\n")
        if dry_run:
            log_file.write("dry_run=true\n")
            return
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
    if return_code != 0:
        raise RuntimeError(f"Command failed: {printable}")


def _training_command(
    *,
    config_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    metrics_path: Path,
    gpu_log_path: Path,
    torch_memory_path: Path,
    torch_trace_path: Path | None,
    qlora_patch_report_path: Path | None,
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
    ]
    if torch_trace_path is not None:
        command.extend(["--torch-memory-trace-output", str(torch_trace_path)])
    if qlora_patch_report_path is not None:
        command.extend(
            [
                "--qlora-peft-state-dict-patch-report",
                str(qlora_patch_report_path),
            ],
        )
    command.extend(["--", automodel_bin, "finetune", "llm", "-c", str(config_path)])
    return command


def _inspect_targets(
    *,
    config: Path,
    output: Path,
    cache_dir: Path,
    logs_root: Path,
    dry_run: bool,
    method: str,
) -> None:
    command = [
        sys.executable,
        "scripts/inspect_lora_targets.py",
        "--config",
        str(config),
        "--output",
        str(output),
        "--cache-dir",
        str(cache_dir),
        "--method",
        method,
    ]
    _run_logged(
        command,
        log_path=logs_root / "target-inspection.log",
        dry_run=dry_run,
    )


def _stage_config(
    *,
    base_config: Mapping[str, Any],
    output_path: Path,
    checkpoint_dir: Path,
    global_batch_size: int,
    local_batch_size: int,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    validation_path: Path | None,
    checkpoint_enabled: bool,
) -> dict[str, Any]:
    staged = clone_training_config_for_stage(
        base_config,
        checkpoint_dir=str(checkpoint_dir),
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
        val_every_steps=val_every_steps,
        validation_path=str(validation_path) if validation_path is not None else None,
        checkpoint_enabled=checkpoint_enabled,
    )
    write_yaml_config(output_path, staged)
    return staged


def _run_training_stage(
    *,
    stage_name: str,
    base_config: Mapping[str, Any],
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    local_batch_size: int,
    global_batch_size: int | None = None,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    validation_path: Path | None,
    checkpoint_enabled: bool,
    memory_trace: bool,
    dry_run: bool,
    patch_qlora_peft_state_dict: bool,
    validator: Any = validate_reference_lora_config,
) -> dict[str, Any]:
    stage_root = results_root / stage_name
    checkpoint_dir = checkpoint_root / stage_name
    effective_global_batch_size = global_batch_size or local_batch_size
    config_path = stage_root / "resolved_config.yaml"
    _stage_config(
        base_config=base_config,
        output_path=config_path,
        checkpoint_dir=checkpoint_dir,
        global_batch_size=effective_global_batch_size,
        local_batch_size=local_batch_size,
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
        val_every_steps=val_every_steps,
        validation_path=validation_path,
        checkpoint_enabled=checkpoint_enabled,
    )

    validation = validator(
        config_path,
        allow_alternate_validation_path=validation_path is not None,
    )
    _write_json(stage_root / "config_validation.json", validation_to_dict(validation))
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    log_path = logs_root / f"{stage_name}.log"
    metrics_path = stage_root / "training_metrics.json"
    command = _training_command(
        config_path=config_path,
        checkpoint_path=checkpoint_dir,
        log_path=log_path,
        metrics_path=metrics_path,
        gpu_log_path=logs_root / f"{stage_name}-gpu.csv",
        torch_memory_path=stage_root / "training_torch_memory.json",
        torch_trace_path=(
            stage_root / "training_torch_memory_trace.json"
            if memory_trace
            else None
        ),
        qlora_patch_report_path=(
            stage_root / "qlora_peft_state_dict_patch.json"
            if patch_qlora_peft_state_dict
            else None
        ),
        automodel_bin=automodel_bin,
    )
    _write_json(
        stage_root / "train_command.json",
        {
            "schema_version": "1.0",
            "stage_name": stage_name,
            "command": command,
            "local_batch_size": local_batch_size,
            "global_batch_size": effective_global_batch_size,
            "max_steps": max_steps,
            "memory_trace": memory_trace,
            "qlora_peft_state_dict_patch": patch_qlora_peft_state_dict,
        },
    )
    _run_logged(command, log_path=log_path, dry_run=dry_run)
    metrics = _read_json(metrics_path)
    if dry_run:
        metrics = {
            "dry_run": True,
            "stage_name": stage_name,
            "local_batch_size": local_batch_size,
            "max_steps": max_steps,
        }
        _write_json(metrics_path, metrics)
    return metrics


def _probe_success(metrics: Mapping[str, Any]) -> bool:
    return_code = metrics.get("return_code")
    return (
        return_code == 0
        and not bool(metrics.get("aborted", False))
        and bool(metrics.get("losses_are_finite", False))
        and int(metrics.get("oom_event_count", 0) or 0) == 0
    )


def _select_largest_successful_probe(
    probe_results: list[dict[str, Any]],
    *,
    max_reserved_gb: float | None = None,
) -> dict[str, Any] | None:
    successful = [item for item in probe_results if item["success"]]
    if max_reserved_gb is not None:
        successful = [
            item
            for item in successful
            if float(
                dict(item.get("metrics", {})).get("peak_reserved_vram_gb", math.inf),
            )
            <= max_reserved_gb
        ]
    if not successful:
        return None
    return max(successful, key=lambda item: int(item["local_batch_size"]))


def _run_batch_probes(
    *,
    base_config: Mapping[str, Any],
    batch_sizes: list[int],
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    probe_steps: int,
    validation_sample_path: Path,
    pilot_trace_max_reserved_gb: float,
    dry_run: bool,
    validator: Any,
    patch_qlora_peft_state_dict: bool,
) -> dict[str, Any]:
    probe_results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        stage_name = f"batch-probe-lbs{batch_size}"
        try:
            metrics = _run_training_stage(
                stage_name=stage_name,
                base_config=base_config,
                results_root=results_root,
                logs_root=logs_root,
                checkpoint_root=checkpoint_root,
                automodel_bin=automodel_bin,
                local_batch_size=batch_size,
                global_batch_size=batch_size,
                max_steps=probe_steps,
                ckpt_every_steps=max(probe_steps + 1000, 1000),
                val_every_steps=max(probe_steps + 1000, 1000),
                validation_path=validation_sample_path,
                checkpoint_enabled=False,
                memory_trace=False,
                dry_run=dry_run,
                patch_qlora_peft_state_dict=patch_qlora_peft_state_dict,
                validator=validator,
            )
            success = True if dry_run else _probe_success(metrics)
            error = None
        except Exception as exc:  # pragma: no cover - cloud execution path
            metrics = {}
            success = False
            error = repr(exc)
            print(f"batch_probe_failed batch_size={batch_size} error={error}", flush=True)
        probe_results.append(
            {
                "local_batch_size": batch_size,
                "global_batch_size": batch_size,
                "success": success,
                "error": error,
                "metrics": metrics,
            },
        )
        if not success and not dry_run:
            # Higher batches are unlikely to recover after OOM/config failure.
            break

    selected_training = _select_largest_successful_probe(probe_results)
    selected_pilot = _select_largest_successful_probe(
        probe_results,
        max_reserved_gb=pilot_trace_max_reserved_gb,
    )
    if selected_pilot is None:
        selected_pilot = selected_training
    summary = {
        "schema_version": "1.0",
        "probe_steps": probe_steps,
        "validation_sample_path": str(validation_sample_path),
        "pilot_trace_max_reserved_gb": pilot_trace_max_reserved_gb,
        "batch_sizes_requested": batch_sizes,
        "probes": probe_results,
        "selected_local_batch_size": (
            selected_training["local_batch_size"]
            if selected_training is not None
            else None
        ),
        "selected_training_local_batch_size": (
            selected_training["local_batch_size"]
            if selected_training is not None
            else None
        ),
        "selected_pilot_local_batch_size": (
            selected_pilot["local_batch_size"] if selected_pilot is not None else None
        ),
        "selection_policy": "largest_successful_for_training; traced_pilot_uses_largest_under_reserved_threshold",
    }
    _write_json(results_root / "batch_probe_summary.json", summary)
    if selected_training is None and not dry_run:
        raise RuntimeError("No successful batch probe")
    return summary


def _run_reload_check(
    *,
    adapter_path: Path,
    results_root: Path,
    logs_root: Path,
    cache_dir: Path,
    dry_run: bool,
    load_in_4bit: bool,
    stage_name: str = "pilot",
) -> None:
    command = [
        sys.executable,
        "scripts/reload_check.py",
        "--dataset",
        "/workspace/data/processed/xlam_splits_v1/dev_eval_1k.jsonl",
        "--output",
        str(results_root / stage_name / "reload_check.json"),
        "--model-name",
        "Qwen/Qwen3-1.7B",
        "--model-revision",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "--adapter-path",
        str(adapter_path),
        "--limit",
        "2",
        "--seed",
        "42",
        "--max-new-tokens",
        "128",
        "--cache-dir",
        str(cache_dir),
        "--load-in-4bit" if load_in_4bit else "--no-load-in-4bit",
        "--torch-dtype",
        "bfloat16",
    ]
    _run_logged(
        command,
        log_path=logs_root / f"{stage_name}-reload-check.log",
        dry_run=dry_run,
    )


def _full_epoch_steps(train_count: int, global_batch_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    return math.ceil(train_count / global_batch_size)


def main() -> None:
    args = parse_args()
    validator = (
        validate_reference_qlora_config
        if args.method == "qlora"
        else validate_reference_lora_config
    )
    validation = validator(args.config)
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    train_decision = assert_split_allowed(
        EXPECTED_TRAIN_PATH,
        command_name=f"exp03-reference-{args.method}",
    )
    validation_decision = assert_split_allowed(
        EXPECTED_VALIDATION_PATH,
        command_name=f"exp03-reference-{args.method}",
    )
    train_count = _dataset_count(Path(EXPECTED_TRAIN_PATH)) if Path(EXPECTED_TRAIN_PATH).is_file() else None
    validation_count = (
        _dataset_count(Path(EXPECTED_VALIDATION_PATH))
        if Path(EXPECTED_VALIDATION_PATH).is_file()
        else None
    )

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml_config(args.config)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    validation_sample_path = args.results_root / "validation_sample_32.jsonl"
    if not args.dry_run:
        _write_validation_sample(
            source=Path(EXPECTED_VALIDATION_PATH),
            output=validation_sample_path,
            records=args.probe_validation_records,
        )

    _write_json(
        args.results_root / "run_plan.json",
        {
            "schema_version": "1.0",
            "config": str(args.config),
            "method": args.method,
            "reload_load_in_4bit": args.reload_load_in_4bit,
            "dry_run": args.dry_run,
            "full_epoch_requested": args.full,
            "train_split": train_decision.__dict__,
            "validation_split": validation_decision.__dict__,
            "train_records": train_count,
            "validation_records": validation_count,
            "batch_sizes": batch_sizes,
            "probe_steps": args.probe_steps,
            "pilot_steps": args.pilot_steps,
            "probe_validation_records": args.probe_validation_records,
            "probe_and_pilot_validation_path": str(validation_sample_path),
            "pilot_trace_max_reserved_gb": args.pilot_trace_max_reserved_gb,
            "pilot_requires_user_approval_before_full_epoch": not args.full,
        },
    )

    _inspect_targets(
        config=args.config,
        output=args.results_root / "lora_target_inspection.json",
        cache_dir=args.cache_dir,
        logs_root=args.logs_root,
        dry_run=args.dry_run,
        method=args.method,
    )

    if args.skip_probes:
        if args.local_batch_size is None:
            raise ValueError("--skip-probes requires --local-batch-size")
        selected_batch = args.local_batch_size
        selected_global_batch = args.global_batch_size or selected_batch
        probe_summary = {
            "schema_version": "1.0",
            "probes_skipped": True,
            "selected_local_batch_size": selected_batch,
            "selected_global_batch_size": selected_global_batch,
        }
        _write_json(args.results_root / "batch_probe_summary.json", probe_summary)
    else:
        probe_summary = _run_batch_probes(
            base_config=base_config,
            batch_sizes=batch_sizes,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            probe_steps=args.probe_steps,
            validation_sample_path=validation_sample_path,
            pilot_trace_max_reserved_gb=args.pilot_trace_max_reserved_gb,
            dry_run=args.dry_run,
            validator=validator,
            patch_qlora_peft_state_dict=args.method == "qlora",
        )
        selected_batch = int(probe_summary["selected_pilot_local_batch_size"] or 0)
        selected_training_batch = int(probe_summary["selected_training_local_batch_size"] or 0)
        selected_global_batch = selected_batch

    if selected_batch <= 0:
        raise RuntimeError("No selected batch size")
    if selected_global_batch <= 0:
        raise RuntimeError("No selected global batch size")

    pilot_metrics = _run_training_stage(
        stage_name="pilot",
        base_config=base_config,
        results_root=args.results_root,
        logs_root=args.logs_root,
        checkpoint_root=args.checkpoint_root,
        automodel_bin=args.automodel_bin,
        local_batch_size=selected_batch,
        global_batch_size=selected_global_batch,
        max_steps=args.pilot_steps,
        ckpt_every_steps=args.pilot_steps,
        val_every_steps=max(1, args.pilot_steps // 2),
        validation_path=validation_sample_path,
        checkpoint_enabled=True,
        memory_trace=not args.disable_memory_trace,
        dry_run=args.dry_run,
        validator=validator,
        patch_qlora_peft_state_dict=args.method == "qlora",
    )
    _run_reload_check(
        adapter_path=args.checkpoint_root / "pilot",
        results_root=args.results_root,
        logs_root=args.logs_root,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        load_in_4bit=args.reload_load_in_4bit,
        stage_name="pilot",
    )

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "pilot_complete" if not args.dry_run else "dry_run_complete",
        "selected_local_batch_size": selected_batch,
        "selected_global_batch_size": selected_global_batch,
        "selected_training_local_batch_size": (
            selected_training_batch if not args.skip_probes else selected_batch
        ),
        "selected_training_global_batch_size": (
            selected_training_batch if not args.skip_probes else selected_global_batch
        ),
        "pilot_steps": args.pilot_steps,
        "pilot_metrics": pilot_metrics,
        "full_epoch_requested": args.full,
        "full_epoch_requires_user_approval_before_start": not args.full,
    }

    if train_count is not None:
        estimated_full_batch = (
            selected_training_batch
            if not args.skip_probes and selected_training_batch > 0
            else selected_global_batch
        )
        summary["estimated_full_epoch_optimizer_steps"] = _full_epoch_steps(
            train_count,
            estimated_full_batch,
        )

    if args.full:
        if train_count is None:
            raise RuntimeError("Cannot run full epoch without train dataset count")
        full_batch = (
            selected_training_batch
            if not args.skip_probes and selected_training_batch > 0
            else selected_batch
        )
        full_global_batch = (
            selected_training_batch
            if not args.skip_probes and selected_training_batch > 0
            else selected_global_batch
        )
        full_steps = _full_epoch_steps(train_count, full_global_batch)
        full_metrics = _run_training_stage(
            stage_name="full-epoch",
            base_config=base_config,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            local_batch_size=full_batch,
            global_batch_size=full_global_batch,
            max_steps=full_steps,
            ckpt_every_steps=max(1, full_steps // 4),
            val_every_steps=max(1, full_steps // 4),
            validation_path=None,
            checkpoint_enabled=True,
            memory_trace=False,
            dry_run=args.dry_run,
            patch_qlora_peft_state_dict=args.method == "qlora",
            validator=validator,
        )
        _run_reload_check(
            adapter_path=args.checkpoint_root / "full-epoch",
            results_root=args.results_root,
            logs_root=args.logs_root,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
            load_in_4bit=args.reload_load_in_4bit,
            stage_name="full-epoch",
        )
        summary["status"] = "full_epoch_complete" if not args.dry_run else "dry_run_complete"
        summary["full_epoch_steps"] = full_steps
        summary["full_epoch_local_batch_size"] = full_batch
        summary["full_epoch_global_batch_size"] = full_global_batch
        summary["full_epoch_metrics"] = full_metrics

    _write_json(args.results_root / "pilot_summary.json", summary)
    print("exp03_reference_lora_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
