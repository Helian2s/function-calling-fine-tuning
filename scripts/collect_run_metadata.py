#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


LOSS_PATTERN = re.compile(r"\b(?:loss|train_loss)\b[=:\s]+([0-9]+(?:\.[0-9]+)?)")
LR_PATTERN = re.compile(r"\b(?:lr|learning_rate)\b[=:\s]+([0-9.eE+-]+)")
STEP_PATTERN = re.compile(r"\b(?:step|global_step)\b[=:\s]+([0-9]+)")


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""

    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    return output


def _package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ModuleNotFoundError:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def parse_training_log(path: Path) -> dict[str, Any]:
    losses: list[float] = []
    learning_rates: list[float] = []
    steps: list[int] = []

    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if loss_match := LOSS_PATTERN.search(line):
                losses.append(float(loss_match.group(1)))
            if lr_match := LR_PATTERN.search(line):
                learning_rates.append(float(lr_match.group(1)))
            if step_match := STEP_PATTERN.search(line):
                steps.append(int(step_match.group(1)))

    return {
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
        "learning_rate_history": learning_rates,
        "global_step": max(steps) if steps else None,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect non-secret Experiment 0 runtime metadata.",
    )
    parser.add_argument("--run-info-dir", type=Path, default=Path("/workspace/run-info"))
    parser.add_argument("--results-dir", type=Path, default=Path("/workspace/results/exp-00"))
    parser.add_argument("--training-log", type=Path, default=Path("/workspace/logs/exp-00/training.log"))
    parser.add_argument("--config", type=Path, default=Path("configs/exp00_smoke/smoke_qlora.yaml"))
    parser.add_argument("--adapter-path", default="/workspace/checkpoints/exp-00/smoke-qlora")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--model-revision",
        default="b968826d9c46dd6066d109eabc6255188de91218",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_info_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    nvidia_smi = _run_text(["nvidia-smi"])
    write_text(args.run_info_dir / "nvidia-smi.txt", nvidia_smi)

    package_lines = [
        f"torch={_package_version('torch')}",
        f"transformers={_package_version('transformers')}",
        f"datasets={_package_version('datasets')}",
        f"peft={_package_version('peft')}",
        f"bitsandbytes={_package_version('bitsandbytes')}",
    ]
    write_text(args.run_info_dir / "package_versions.txt", "\n".join(package_lines) + "\n")

    git_revision = _run_text(["git", "rev-parse", "HEAD"]).strip()
    write_text(args.run_info_dir / "git_revision.txt", git_revision + "\n")

    if Path("/workspace/data/checksums.sha256").is_file():
        write_text(
            args.run_info_dir / "dataset_checksums.sha256",
            Path("/workspace/data/checksums.sha256").read_text(encoding="utf-8"),
        )

    if args.config.is_file():
        write_text(
            args.run_info_dir / "resolved_config.yaml",
            args.config.read_text(encoding="utf-8"),
        )

    metrics = parse_training_log(args.training_log)
    training_metrics_path = args.results_dir / "training_metrics.json"
    training_metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment_report = {
        "instance_id": os.environ.get("EC2_INSTANCE_ID"),
        "instance_type": os.environ.get("EC2_INSTANCE_TYPE"),
        "ami_id": os.environ.get("EC2_AMI_ID"),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_output_path": args.adapter_path,
        "config_path": str(args.config),
        "git_revision": git_revision or None,
        "packages": {
            line.split("=", 1)[0]: line.split("=", 1)[1]
            for line in package_lines
        },
    }
    (args.run_info_dir / "environment_report.json").write_text(
        json.dumps(environment_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_metadata = {
        **environment_report,
        **metrics,
        "training_log": str(args.training_log),
    }
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
