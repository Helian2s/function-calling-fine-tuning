#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SITE_CUSTOMIZE_DIR = ROOT / "scripts" / "python_sitecustomize"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.training_monitor import (  # noqa: E402
    parse_training_line,
    summarize_training_signals,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smoke training with C7 live monitoring.",
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--torch-memory-output", type=Path)
    parser.add_argument("--torch-memory-trace-output", type=Path)
    parser.add_argument("--qlora-peft-state-dict-patch-report", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/workspace/checkpoints"),
    )
    parser.add_argument("--gpu-poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-oom-events", type=int, default=2)
    parser.add_argument("--max-trainable-ratio", type=float, default=0.10)
    parser.add_argument(
        "--expected-training-method",
        choices=("adapter", "full_parameter_sft"),
        default="adapter",
        help=(
            "Expected trainability profile. Adapter training aborts if most "
            "base parameters appear trainable; full_parameter_sft requires a "
            "near-full trainable ratio."
        ),
    )
    parser.add_argument(
        "--min-full-trainable-ratio",
        type=float,
        default=0.95,
        help="Minimum trainable-parameter ratio expected for full-parameter SFT.",
    )
    parser.add_argument("--low-gpu-utilization-threshold", type=float, default=2.0)
    parser.add_argument("--low-gpu-memory-threshold-mb", type=float, default=512.0)
    parser.add_argument("--low-gpu-grace-seconds", type=float, default=600.0)
    parser.add_argument("--low-gpu-window-samples", type=int, default=60)
    parser.add_argument(
        "--require-checkpoint-root-mount",
        action="store_true",
        help="Fail before training if checkpoint root is not a separate mount.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def ensure_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")
    return command


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def verify_checkpoint_path(
    *,
    checkpoint_path: Path,
    checkpoint_root: Path,
    require_mount: bool,
) -> dict[str, Any]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    root_is_mount = checkpoint_root.is_mount()
    root_writable = False
    write_test = checkpoint_root / ".c7-write-test"
    try:
        write_test.write_text("ok\n", encoding="utf-8")
        root_writable = write_test.read_text(encoding="utf-8") == "ok\n"
    finally:
        try:
            write_test.unlink()
        except FileNotFoundError:
            pass

    under_root = is_relative_to(checkpoint_path, checkpoint_root)
    status = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_path_under_root": under_root,
        "checkpoint_root_exists": checkpoint_root.is_dir(),
        "checkpoint_root_writable": root_writable,
        "checkpoint_root_is_mount": root_is_mount,
        "checkpoint_root_mount_required": require_mount,
    }

    errors: list[str] = []
    if not under_root:
        errors.append("checkpoint path is not under the configured checkpoint root")
    if not root_writable:
        errors.append("checkpoint root is not writable")
    if require_mount and not root_is_mount:
        errors.append("checkpoint root is not a separate mount")

    status["checkpoint_preflight_ok"] = not errors
    status["checkpoint_preflight_errors"] = errors
    if errors:
        raise RuntimeError("; ".join(errors))

    return status


def query_gpu() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        timestamp, index, name, utilization, memory_used, memory_total = parts
        try:
            rows.append(
                {
                    "timestamp": timestamp,
                    "index": int(index),
                    "name": name,
                    "utilization_gpu_pct": float(utilization),
                    "memory_used_mb": float(memory_used),
                    "memory_total_mb": float(memory_total),
                },
            )
        except ValueError:
            continue
    return rows


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def monitor_gpu(
    *,
    process: subprocess.Popen[str],
    gpu_log: Path,
    poll_seconds: float,
    stop_event: threading.Event,
    abort_state: dict[str, str | None],
    low_gpu_utilization_threshold: float,
    low_gpu_memory_threshold_mb: float,
    low_gpu_grace_seconds: float,
    low_gpu_window_samples: int,
) -> None:
    gpu_log.parent.mkdir(parents=True, exist_ok=True)
    recent_low_activity: list[bool] = []
    started = time.monotonic()

    with gpu_log.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sampled_at_utc",
                "gpu_timestamp",
                "index",
                "name",
                "utilization_gpu_pct",
                "memory_used_mb",
                "memory_total_mb",
            ],
        )
        while process.poll() is None and not stop_event.is_set():
            output = query_gpu()
            sampled_at = utc_now()
            rows = parse_gpu_rows(output)
            if rows:
                active = False
                for row in rows:
                    writer.writerow(
                        [
                            sampled_at,
                            row["timestamp"],
                            row["index"],
                            row["name"],
                            row["utilization_gpu_pct"],
                            row["memory_used_mb"],
                            row["memory_total_mb"],
                        ],
                    )
                    active = active or (
                        row["utilization_gpu_pct"] > low_gpu_utilization_threshold
                        or row["memory_used_mb"] > low_gpu_memory_threshold_mb
                    )
                csv_file.flush()
                recent_low_activity.append(not active)
                recent_low_activity = recent_low_activity[-low_gpu_window_samples:]

                elapsed = time.monotonic() - started
                if (
                    elapsed >= low_gpu_grace_seconds
                    and len(recent_low_activity) == low_gpu_window_samples
                    and all(recent_low_activity)
                ):
                    abort_state["reason"] = "low_gpu_activity"
                    terminate_process(process)
                    return
            time.sleep(poll_seconds)


def read_gpu_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "gpu_sample_count": 0,
            "max_gpu_memory_used_mb": None,
            "max_gpu_utilization_pct": None,
            "average_gpu_utilization_pct": None,
        }

    utilizations: list[float] = []
    memory_values: list[float] = []
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                utilizations.append(float(row["utilization_gpu_pct"]))
                memory_values.append(float(row["memory_used_mb"]))
            except (KeyError, ValueError):
                continue

    return {
        "gpu_sample_count": len(utilizations),
        "max_gpu_memory_used_mb": max(memory_values) if memory_values else None,
        "max_gpu_utilization_pct": max(utilizations) if utilizations else None,
        "average_gpu_utilization_pct": (
            sum(utilizations) / len(utilizations) if utilizations else None
        ),
    }


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def child_environment(
    torch_memory_output: Path | None,
    torch_memory_trace_output: Path | None = None,
    qlora_peft_state_dict_patch_report: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if (
        torch_memory_output is None
        and torch_memory_trace_output is None
        and qlora_peft_state_dict_patch_report is None
    ):
        return env

    if torch_memory_output is not None:
        torch_memory_output.parent.mkdir(parents=True, exist_ok=True)
        env["FCFT_TORCH_MEMORY_OUTPUT"] = str(torch_memory_output)
    if torch_memory_trace_output is not None:
        torch_memory_trace_output.parent.mkdir(parents=True, exist_ok=True)
        env["FCFT_TORCH_MEMORY_TRACE_OUTPUT"] = str(torch_memory_trace_output)
    if qlora_peft_state_dict_patch_report is not None:
        qlora_peft_state_dict_patch_report.parent.mkdir(parents=True, exist_ok=True)
        env["FCFT_PATCH_QLORA_PEFT_STATE_DICT"] = "1"
        env["FCFT_QLORA_PEFT_STATE_DICT_PATCH_REPORT"] = str(
            qlora_peft_state_dict_patch_report,
        )
    pythonpath_entries = [str(SITE_CUSTOMIZE_DIR)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def read_torch_memory_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "torch_memory_probe_path": None,
            "torch_memory_probe_present": False,
            "peak_allocated_vram_gb": None,
            "peak_reserved_vram_gb": None,
        }
    if not path.is_file():
        return {
            "torch_memory_probe_path": str(path),
            "torch_memory_probe_present": False,
            "peak_allocated_vram_gb": None,
            "peak_reserved_vram_gb": None,
        }

    loaded = json.loads(path.read_text(encoding="utf-8"))
    report = loaded if isinstance(loaded, dict) else {}
    return {
        "torch_memory_probe_path": str(path),
        "torch_memory_probe_present": True,
        "torch_memory_report": report,
        "peak_allocated_vram_gb": report.get("peak_allocated_vram_gb"),
        "peak_reserved_vram_gb": report.get("peak_reserved_vram_gb"),
    }


def read_torch_memory_trace_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "torch_memory_trace_path": None,
            "torch_memory_trace_present": False,
            "torch_memory_trace_event_count": None,
        }
    if not path.is_file():
        return {
            "torch_memory_trace_path": str(path),
            "torch_memory_trace_present": False,
            "torch_memory_trace_event_count": None,
        }

    loaded = json.loads(path.read_text(encoding="utf-8"))
    report = loaded if isinstance(loaded, dict) else {}
    return {
        "torch_memory_trace_path": str(path),
        "torch_memory_trace_present": True,
        "torch_memory_trace_event_count": report.get("raw_event_count"),
        "torch_memory_trace_summary_keys": sorted(
            str(key) for key in dict(report.get("summary", {})).keys()
        ),
    }


def main() -> None:
    args = parse_args()
    command = ensure_command(args.command)
    started_monotonic = time.monotonic()
    started_at = utc_now()

    checkpoint_status = verify_checkpoint_path(
        checkpoint_path=args.checkpoint_path,
        checkpoint_root=args.checkpoint_root,
        require_mount=args.require_checkpoint_root_mount,
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    learning_rates: list[float] = []
    steps: list[int] = []
    step_times_seconds: list[float] = []
    oom_event_count = 0
    trainable_parameter_count: int | None = None
    total_parameter_count: int | None = None
    frozen_parameter_count: int | None = None
    abort_reason: str | None = None

    with args.log.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"[training-monitor] started_at={started_at}\n")
        log_file.write(f"[training-monitor] command={' '.join(command)}\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=child_environment(
                args.torch_memory_output,
                args.torch_memory_trace_output,
                args.qlora_peft_state_dict_patch_report,
            ),
        )
        assert process.stdout is not None

        stop_event = threading.Event()
        abort_state: dict[str, str | None] = {"reason": None}
        gpu_thread = threading.Thread(
            target=monitor_gpu,
            kwargs={
                "process": process,
                "gpu_log": args.gpu_log,
                "poll_seconds": args.gpu_poll_seconds,
                "stop_event": stop_event,
                "abort_state": abort_state,
                "low_gpu_utilization_threshold": args.low_gpu_utilization_threshold,
                "low_gpu_memory_threshold_mb": args.low_gpu_memory_threshold_mb,
                "low_gpu_grace_seconds": args.low_gpu_grace_seconds,
                "low_gpu_window_samples": args.low_gpu_window_samples,
            },
            daemon=True,
        )
        gpu_thread.start()

        try:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

                parsed = parse_training_line(line)
                losses.extend(parsed.losses)
                learning_rates.extend(parsed.learning_rates)
                steps.extend(parsed.steps)
                step_times_seconds.extend(parsed.step_times_seconds)
                if parsed.trainable_parameter_count is not None:
                    trainable_parameter_count = parsed.trainable_parameter_count
                if parsed.total_parameter_count is not None:
                    total_parameter_count = parsed.total_parameter_count
                if parsed.frozen_parameter_count is not None:
                    frozen_parameter_count = parsed.frozen_parameter_count
                if parsed.oom_event:
                    oom_event_count += 1

                if any(not math.isfinite(loss) for loss in parsed.losses):
                    abort_reason = "non_finite_loss"
                    terminate_process(process)
                    break

                if oom_event_count >= args.max_oom_events:
                    abort_reason = "repeated_cuda_oom"
                    terminate_process(process)
                    break

                if trainable_parameter_count == 0:
                    abort_reason = "zero_trainable_parameters"
                    terminate_process(process)
                    break

                trainable_ratio = (
                    trainable_parameter_count / total_parameter_count
                    if trainable_parameter_count is not None and total_parameter_count
                    else None
                )
                if (
                    args.expected_training_method == "adapter"
                    and trainable_ratio is not None
                    and trainable_ratio > args.max_trainable_ratio
                ):
                    abort_reason = "base_model_probably_trainable"
                    terminate_process(process)
                    break
                if (
                    args.expected_training_method == "full_parameter_sft"
                    and trainable_ratio is not None
                    and trainable_ratio < args.min_full_trainable_ratio
                ):
                    abort_reason = "full_sft_trainable_ratio_too_low"
                    terminate_process(process)
                    break
        except KeyboardInterrupt:
            abort_reason = "keyboard_interrupt"
            os.killpg(process.pid, signal.SIGTERM)
            raise
        finally:
            return_code = process.wait()
            stop_event.set()
            gpu_thread.join(timeout=args.gpu_poll_seconds + 2)

    if abort_reason is None:
        abort_reason = abort_state["reason"]

    ended_at = utc_now()
    duration_seconds = time.monotonic() - started_monotonic
    signal_summary = summarize_training_signals(
        losses=losses,
        learning_rates=learning_rates,
        steps=steps,
        step_times_seconds=step_times_seconds,
    )
    gpu_summary = read_gpu_summary(args.gpu_log)
    torch_memory_summary = read_torch_memory_summary(args.torch_memory_output)
    torch_memory_trace_summary = read_torch_memory_trace_summary(
        args.torch_memory_trace_output,
    )
    checkpoint_exists_after = args.checkpoint_path.exists()

    trainable_ratio = (
        trainable_parameter_count / total_parameter_count
        if trainable_parameter_count is not None and total_parameter_count
        else None
    )
    if frozen_parameter_count is None and total_parameter_count and trainable_parameter_count is not None:
        frozen_parameter_count = total_parameter_count - trainable_parameter_count

    if args.expected_training_method == "full_parameter_sft":
        trainability_status = (
            "expected_full_parameter"
            if trainable_ratio is not None
            and trainable_ratio >= args.min_full_trainable_ratio
            else "full_parameter_not_proven"
        )
    else:
        trainability_status = (
            "probably_trainable"
            if trainable_ratio is not None and trainable_ratio > args.max_trainable_ratio
            else "not_detected_as_trainable"
        )

    metrics = {
        "command": command,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "return_code": return_code,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "oom_event_count": oom_event_count,
        "training_log": str(args.log),
        "gpu_log": str(args.gpu_log),
        "metrics_output": str(args.metrics_output),
        "checkpoint_exists_after": checkpoint_exists_after,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_parameter_ratio": trainable_ratio,
        "adapter_gradient_status": "unknown",
        "expected_training_method": args.expected_training_method,
        "base_model_trainability_status": trainability_status,
        **checkpoint_status,
        **signal_summary,
        **gpu_summary,
        **torch_memory_summary,
        **torch_memory_trace_summary,
    }
    write_metrics(args.metrics_output, metrics)

    if abort_reason is not None and return_code == 0:
        raise SystemExit(2)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
