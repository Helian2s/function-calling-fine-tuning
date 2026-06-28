#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml  # noqa: E402

from function_calling_ft.run_manifest import build_exp00_run_manifest  # noqa: E402
from function_calling_ft.training_monitor import parse_existing_training_log  # noqa: E402


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


def _exp00_env_default(name: str) -> str | None:
    env_path = ROOT / "configs" / "common" / "exp00.env"
    if not env_path.is_file():
        return None
    prefix = f'{name}="${{{name}:-'
    suffix = '}"'
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and line.endswith(suffix):
            return line[len(prefix) : -len(suffix)]
    return None


def _container_image_metadata(run_info_dir: Path) -> dict[str, str]:
    path = run_info_dir / "container_image.txt"
    if not path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    result: dict[str, str] = {}
    if parsed.get("image"):
        result["container_tag"] = parsed["image"]
    repo_digest = parsed.get("repo_digest")
    if repo_digest and "@sha256:" in repo_digest:
        result["container_digest"] = repo_digest.split("@", 1)[1]
    return result


def _imds_token() -> str | None:
    request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return None


def _imds_value(path: str, token: str | None) -> str | None:
    if token is None:
        return None
    request = urllib.request.Request(
        f"http://169.254.169.254/latest/{path.lstrip('/')}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return None


def _ec2_metadata() -> dict[str, str | None]:
    token = _imds_token()
    return {
        "instance_id": _imds_value("meta-data/instance-id", token),
        "instance_type": _imds_value("meta-data/instance-type", token),
        "ami_id": _imds_value("meta-data/ami-id", token),
    }


def _gpu_name() -> str | None:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
    ).strip()
    if not output:
        return None
    return output.splitlines()[0].strip() or None


def _package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ModuleNotFoundError:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def _git_dirty_files() -> list[str]:
    status = _run_text(["git", "status", "--porcelain"])
    return [
        line[3:]
        for line in status.splitlines()
        if len(line) > 3
    ]


def _host_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if not line.startswith("MemTotal:"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return round(int(parts[1]) / 1024 / 1024, 3)
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def parse_training_log(path: Path) -> dict[str, Any]:
    return parse_existing_training_log(path)


def load_existing_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_artifact_pair(
    *,
    run_info_dir: Path,
    results_dir: Path,
    name: str,
    content: str,
) -> None:
    write_text(run_info_dir / name, content)
    write_text(results_dir / name, content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect non-secret Experiment 0 runtime metadata.",
    )
    parser.add_argument("--run-info-dir", type=Path, default=Path("/workspace/run-info"))
    parser.add_argument("--results-dir", type=Path, default=Path("/workspace/results/exp-00"))
    parser.add_argument("--training-log", type=Path, default=Path("/workspace/logs/exp-00/training.log"))
    parser.add_argument("--config", type=Path, default=Path("configs/exp00_smoke/smoke_lora.yaml"))
    parser.add_argument("--adapter-path", default="/workspace/checkpoints/exp-00/smoke-lora")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--model-revision",
        default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--task-id", default="C9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_info_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    nvidia_smi = _run_text(["nvidia-smi"])
    write_artifact_pair(
        run_info_dir=args.run_info_dir,
        results_dir=args.results_dir,
        name="nvidia-smi.txt",
        content=nvidia_smi,
    )

    package_lines = [
        f"torch={_package_version('torch')}",
        f"transformers={_package_version('transformers')}",
        f"datasets={_package_version('datasets')}",
        f"peft={_package_version('peft')}",
        f"bitsandbytes={_package_version('bitsandbytes')}",
        f"nemo_automodel={_package_version('nemo_automodel')}",
    ]
    write_artifact_pair(
        run_info_dir=args.run_info_dir,
        results_dir=args.results_dir,
        name="package_versions.txt",
        content="\n".join(package_lines) + "\n",
    )

    git_revision = _run_text(["git", "rev-parse", "HEAD"]).strip()
    git_dirty_files = _git_dirty_files()
    write_text(args.run_info_dir / "git_revision.txt", git_revision + "\n")

    if Path("/workspace/data/checksums.sha256").is_file():
        write_text(
            args.run_info_dir / "dataset_checksums.sha256",
            Path("/workspace/data/checksums.sha256").read_text(encoding="utf-8"),
        )

    if args.config.is_file():
        write_artifact_pair(
            run_info_dir=args.run_info_dir,
            results_dir=args.results_dir,
            name="resolved_config.yaml",
            content=args.config.read_text(encoding="utf-8"),
        )

    container_metadata = _container_image_metadata(args.run_info_dir)
    ec2_metadata = _ec2_metadata()
    container_tag = (
        os.environ.get("AUTOMODEL_IMAGE")
        or _exp00_env_default("AUTOMODEL_IMAGE")
        or container_metadata.get("container_tag")
    )
    container_digest = (
        os.environ.get("AUTOMODEL_IMAGE_DIGEST")
        or _exp00_env_default("AUTOMODEL_IMAGE_DIGEST")
        or container_metadata.get("container_digest")
    )

    parsed_metrics = parse_training_log(args.training_log)
    training_metrics_path = args.results_dir / "training_metrics.json"
    metrics = {
        **load_existing_json(training_metrics_path),
        **parsed_metrics,
        "metadata_log_parse": parsed_metrics,
    }
    training_metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment_report = {
        "instance_id": os.environ.get("EC2_INSTANCE_ID")
        or ec2_metadata["instance_id"],
        "instance_type": os.environ.get("EC2_INSTANCE_TYPE")
        or ec2_metadata["instance_type"],
        "ami_id": os.environ.get("EC2_AMI_ID") or ec2_metadata["ami_id"],
        "container_tag": container_tag,
        "container_digest": container_digest,
        "gpu": _gpu_name(),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "adapter_output_path": args.adapter_path,
        "config_path": str(args.config),
        "git_revision": git_revision or None,
        "git_dirty": bool(git_dirty_files),
        "git_dirty_files": git_dirty_files,
        "host_memory_gb": _host_memory_gb(),
        "packages": {
            line.split("=", 1)[0]: line.split("=", 1)[1]
            for line in package_lines
        },
    }
    write_artifact_pair(
        run_info_dir=args.run_info_dir,
        results_dir=args.results_dir,
        name="environment_report.json",
        content=json.dumps(environment_report, indent=2, sort_keys=True) + "\n",
    )

    run_metadata = {
        **environment_report,
        **metrics,
        "run_id": args.run_id,
        "training_log": str(args.training_log),
    }
    generation_metadata = load_existing_json(
        args.results_dir / "generation_metadata.json",
    )
    for key in ("peak_allocated_vram_gb", "peak_reserved_vram_gb"):
        if key in generation_metadata and run_metadata.get(key) is None:
            run_metadata[key] = generation_metadata[key]
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    scores = load_existing_json(args.results_dir / "scores.json")
    dataset_manifest_path = args.run_info_dir / "dataset_checksums.sha256"
    if not dataset_manifest_path.is_file():
        dataset_manifest_path = None
    initial_manifest = build_exp00_run_manifest(
        run_metadata=run_metadata,
        generation_metadata=generation_metadata,
        scores=scores,
        training_config=_load_yaml(args.config),
        results_dir=args.results_dir,
        training_log=args.training_log,
        dataset_manifest_path=dataset_manifest_path,
        run_id=args.run_id,
        status="succeeded",
        task_id=args.task_id,
    )
    artifact_lines = []
    for name, artifact in initial_manifest["artifacts"].items():
        if name in {"checksums", "run_manifest"}:
            continue
        sha256 = artifact.get("sha256")
        path = artifact.get("path")
        if sha256 and path:
            artifact_lines.append(f"{sha256}  {name}:{path}")
    write_artifact_pair(
        run_info_dir=args.run_info_dir,
        results_dir=args.results_dir,
        name="checksums.sha256",
        content="\n".join(sorted(artifact_lines)) + "\n",
    )
    run_manifest = build_exp00_run_manifest(
        run_metadata=run_metadata,
        generation_metadata=generation_metadata,
        scores=scores,
        training_config=_load_yaml(args.config),
        results_dir=args.results_dir,
        training_log=args.training_log,
        dataset_manifest_path=dataset_manifest_path,
        run_id=args.run_id,
        status="succeeded",
        task_id=args.task_id,
    )
    run_manifest["artifacts"]["run_manifest"] = {
        "path": str(args.results_dir / "run_manifest.json"),
        "sha256": None,
    }
    write_artifact_pair(
        run_info_dir=args.run_info_dir,
        results_dir=args.results_dir,
        name="run_manifest.json",
        content=json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
