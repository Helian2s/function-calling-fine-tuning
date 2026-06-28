#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation_compare import write_comparison
from function_calling_ft.generation import read_jsonl
from function_calling_ft.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    validate_run_manifest,
)
from function_calling_ft.split_guard import assert_split_allowed


DEFAULT_MATRIX_CONFIG = Path("configs/exp02_baseline/matrix.yaml")
DEFAULT_COMPARISON_METRICS = (
    "strict_complete_match",
    "schema_equivalent_complete_match",
    "executable_complete_match",
    "tool_call_emitted",
    "no_tool_false_positive",
)
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "datasets",
    "nemo_automodel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Experiment 2 inference feasibility matrix.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--only-run", action="append", dest="only_runs")
    parser.add_argument("--only-dataset", action="append", dest="only_datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-generation-if-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=True),
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


def _path_entry(path: Path | None) -> dict[str, str | None]:
    return {
        "path": str(path) if path is not None else None,
        "sha256": _sha256_file(path) if path is not None and path.is_file() else None,
    }


def _write_checksums(path: Path, files: Iterable[Path]) -> None:
    rows = []
    for file_path in sorted(set(files), key=lambda item: item.as_posix()):
        if not file_path.is_file():
            continue
        try:
            display_path = file_path.relative_to(path.parent)
        except ValueError:
            display_path = file_path
        rows.append(f"{_sha256_file(file_path)}  {display_path}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout + completed.stderr


def _run_logged(command: list[str], *, log_path: Path, cwd: Path = ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.monotonic() - start
        log_file.write(f"\nexit_code={return_code} elapsed_seconds={elapsed:.3f}\n")
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(command)}",
        )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_metadata() -> dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"]).strip() or None
    status = _run_text(["git", "status", "--porcelain"]).splitlines()
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_dirty_files": [line[3:] for line in status if len(line) > 3],
    }


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


def _host_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return round(int(parts[1]) / 1024 / 1024, 3)
    return None


def _gpu_name() -> str | None:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
    ).strip()
    return output.splitlines()[0].strip() if output else None


def _environment_report() -> dict[str, Any]:
    token = _imds_token()
    return {
        "schema_version": "1.0",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_metadata(),
        "hardware": {
            "instance_id": _imds_value("meta-data/instance-id", token),
            "instance_type": _imds_value("meta-data/instance-type", token),
            "ami_id": _imds_value("meta-data/ami-id", token),
            "gpu": _gpu_name(),
            "host_memory_gb": _host_memory_gb(),
        },
    }


def _package_versions_text(packages: Mapping[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))


def _dataset_count(path: Path) -> int:
    return len(read_jsonl(path))


def _selected_names(
    names: Iterable[str],
    selected: list[str] | None,
) -> set[str]:
    available = set(names)
    if not selected:
        return available
    requested = set(selected)
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Unknown selection(s): {unknown}")
    return requested


def _run_configs(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for item in matrix.get("runs", []):
        if not isinstance(item, Mapping):
            raise ValueError("matrix.runs entries must be mappings")
        config_path = Path(str(item["config"]))
        config = _load_yaml(config_path)
        run_id = str(config.get("run_id"))
        if not run_id:
            raise ValueError(f"Run config is missing run_id: {config_path}")
        config["_config_path"] = str(config_path)
        runs[run_id] = config
    return runs


def _dataset_specs(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = matrix.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("matrix.datasets must be a mapping")
    return {
        str(name): dict(spec)
        for name, spec in datasets.items()
        if isinstance(spec, Mapping)
    }


def _validate_config_pair(run_config: Mapping[str, Any]) -> None:
    model = run_config.get("model")
    decoding = run_config.get("decoding")
    if not isinstance(model, Mapping) or not isinstance(decoding, Mapping):
        raise ValueError("Run config must contain model and decoding mappings")
    if model.get("name") != "Qwen/Qwen3-1.7B":
        raise ValueError(f"Unexpected model name: {model.get('name')}")
    if model.get("revision") != model.get("tokenizer_revision"):
        raise ValueError("Model and tokenizer revisions must match for Exp 02")
    if decoding.get("enable_thinking") is not False:
        raise ValueError("Exp 02 primary configs must disable thinking")
    if not decoding.get("do_sample") and any(
        key in decoding for key in ("temperature", "top_p", "top_k")
    ):
        raise ValueError("Sampling parameters require do_sample=true")


def _dry_run_plan(
    *,
    matrix: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, Any]],
    selected_runs: set[str],
    selected_datasets: set[str],
) -> dict[str, Any]:
    plan_runs = []
    for run_id in sorted(selected_runs):
        run_config = runs[run_id]
        _validate_config_pair(run_config)
        for dataset_name in sorted(selected_datasets):
            dataset_spec = datasets[dataset_name]
            dataset_path = Path(str(dataset_spec["path"]))
            decision = assert_split_allowed(
                dataset_path,
                command_name="exp02-dry-run",
            )
            actual_count = _dataset_count(dataset_path) if dataset_path.is_file() else None
            expected_count = dataset_spec.get("expected_records")
            if actual_count is not None and expected_count is not None:
                if int(expected_count) != actual_count:
                    raise ValueError(
                        f"{dataset_path} count {actual_count} does not match "
                        f"expected {expected_count}",
                    )
            plan_runs.append(
                {
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "records": actual_count,
                    "split_name": decision.split_name,
                    "split_lock_status": decision.split_lock_status,
                    "load_in_4bit": run_config["model"].get("load_in_4bit"),
                    "do_sample": run_config["decoding"].get("do_sample"),
                },
            )

    locked_unused = matrix.get("locked_unused_datasets", {})
    return {
        "schema_version": "1.0",
        "experiment_id": matrix.get("experiment_id"),
        "task_id": matrix.get("task_id"),
        "runs": plan_runs,
        "locked_unused_datasets": locked_unused,
    }


def _prediction_ids(path: Path) -> list[str]:
    return [str(record.get("id", "")) for record in read_jsonl(path)]


def _verify_complete_predictions(
    *,
    dataset_path: Path,
    predictions_path: Path,
) -> None:
    dataset_ids = _prediction_ids(dataset_path)
    prediction_ids = _prediction_ids(predictions_path)
    duplicates = sorted(
        record_id for record_id in set(prediction_ids) if prediction_ids.count(record_id) > 1
    )
    missing = sorted(set(dataset_ids) - set(prediction_ids))
    extra = sorted(set(prediction_ids) - set(dataset_ids))
    if duplicates or missing or extra:
        raise ValueError(
            "Prediction ID mismatch: "
            f"duplicates={duplicates[:5]} missing={missing[:5]} extra={extra[:5]}",
        )


def _append_sampling_args(command: list[str], decoding: Mapping[str, Any]) -> None:
    if not decoding.get("do_sample"):
        return
    command.append("--do-sample")
    for key, flag in (
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
    ):
        if key in decoding:
            command.extend([flag, str(decoding[key])])


def _generation_command(
    *,
    run_config: Mapping[str, Any],
    dataset_path: Path,
    run_dir: Path,
    cache_dir: Path | None,
) -> list[str]:
    model = run_config["model"]
    decoding = run_config["decoding"]
    generation = run_config["generation"]
    command = [
        sys.executable,
        "scripts/generate_predictions.py",
        "--dataset",
        str(dataset_path),
        "--output",
        str(run_dir / "predictions.jsonl"),
        "--model-name",
        str(model["name"]),
        "--model-revision",
        str(model["revision"]),
        "--seed",
        str(decoding["seed"]),
        "--max-new-tokens",
        str(decoding["max_new_tokens"]),
        "--batch-size",
        str(generation.get("batch_size", 1)),
        "--torch-dtype",
        str(model.get("torch_dtype", "bfloat16")),
        "--metadata-output",
        str(run_dir / "generation_metadata.json"),
        "--progress-interval",
        str(generation.get("progress_interval", 25)),
        "--progress-file",
        str(run_dir / "progress.json"),
        "--stop-file",
        str(run_dir / "STOP"),
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    if bool(model.get("load_in_4bit")):
        command.append("--load-in-4bit")
    else:
        command.append("--no-load-in-4bit")
    if bool(generation.get("stream_output", True)):
        command.append("--stream-output")
    if bool(generation.get("resume", True)):
        command.append("--resume")
    _append_sampling_args(command, decoding)
    return command


def _render_prompt_command(
    *,
    run_config: Mapping[str, Any],
    dataset_path: Path,
    run_dir: Path,
    cache_dir: Path | None,
) -> list[str]:
    model = run_config["model"]
    command = [
        sys.executable,
        "scripts/render_prompt_hashes.py",
        "--dataset",
        str(dataset_path),
        "--output",
        str(run_dir / "prompt_hashes.jsonl"),
        "--summary-output",
        str(run_dir / "prompt_hash_summary.json"),
        "--model-name",
        str(model["name"]),
        "--model-revision",
        str(model["tokenizer_revision"]),
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    return command


def _evaluate_command(*, dataset_path: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate.py",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(run_dir / "predictions.jsonl"),
        "--output-dir",
        str(run_dir),
    ]


def _summarize_command(*, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/summarize_evaluation_report.py",
        "--output-dir",
        str(run_dir),
    ]


def _artifact_paths(run_dir: Path, log_path: Path) -> dict[str, Path]:
    return {
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_manifest": run_dir / "run_manifest.json",
        "environment": run_dir / "environment_report.json",
        "predictions": run_dir / "predictions.jsonl",
        "per_example_scores": run_dir / "scored_predictions.jsonl",
        "metrics": run_dir / "scores.json",
        "training_memory": run_dir / "training_torch_memory.json",
        "logs": log_path,
        "checksums": run_dir / "checksums.sha256",
        "report": run_dir / "case_report.md",
    }


def _manifest_artifacts(run_dir: Path, log_path: Path) -> dict[str, dict[str, str | None]]:
    return {
        key: _path_entry(path)
        for key, path in _artifact_paths(run_dir, log_path).items()
    }


def _write_run_artifacts(
    *,
    matrix: Mapping[str, Any],
    run_config: Mapping[str, Any],
    dataset_name: str,
    dataset_spec: Mapping[str, Any],
    dataset_path: Path,
    run_dir: Path,
    log_path: Path,
    started_at: float,
) -> None:
    environment = _environment_report()
    generation_metadata = _read_json(run_dir / "generation_metadata.json")
    scores = _read_json(run_dir / "scores.json")
    prompt_summary = _read_json(run_dir / "prompt_hash_summary.json")
    container = matrix.get("container", {})
    if not isinstance(container, Mapping):
        container = {}
    model = run_config["model"]
    decoding = run_config["decoding"]
    generation = run_config["generation"]
    method = run_config["method"]
    elapsed = time.monotonic() - started_at
    memory_report = {
        "schema_version": "1.0",
        "stage": "inference",
        "peak_allocated_vram_gb": generation_metadata.get(
            "peak_allocated_vram_gb",
        ),
        "peak_reserved_vram_gb": generation_metadata.get(
            "peak_reserved_vram_gb",
        ),
        "cuda_memory_error": generation_metadata.get("cuda_memory_error"),
        "training_steps": 0,
    }
    _write_json(run_dir / "training_torch_memory.json", memory_report)
    _write_json(run_dir / "environment_report.json", environment)
    (run_dir / "package_versions.txt").write_text(
        _package_versions_text(environment["packages"]),
        encoding="utf-8",
    )
    (run_dir / "nvidia-smi.txt").write_text(
        _run_text(["nvidia-smi"]),
        encoding="utf-8",
    )
    resolved_config = {
        "matrix": {
            "experiment_id": matrix.get("experiment_id"),
            "task_id": matrix.get("task_id"),
            "container": container,
        },
        "run": {
            key: value
            for key, value in run_config.items()
            if not str(key).startswith("_")
        },
        "dataset": {"name": dataset_name, **dict(dataset_spec)},
    }
    _write_yaml(run_dir / "resolved_config.yaml", resolved_config)
    run_metadata = {
        "schema_version": "1.0",
        "experiment_id": matrix.get("experiment_id"),
        "task_id": matrix.get("task_id"),
        "run_id": run_config["run_id"],
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "prompt_summary": prompt_summary,
        "generation_metadata": generation_metadata,
        "scores": scores,
        "environment": environment,
        "elapsed_seconds": elapsed,
    }
    _write_json(run_dir / "run_metadata.json", run_metadata)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(matrix.get("experiment_id")),
        "task_id": str(matrix.get("task_id")),
        "run_id": f"{run_config['run_id']}-{dataset_name}",
        "status": "succeeded",
        "comparison": {
            "parent_run_id": None,
            "comparison_run_ids": [],
        },
        "source": environment["git"],
        "environment": {
            "container_tag": container.get("tag"),
            "container_digest": container.get("digest"),
            "package_versions": environment["packages"],
        },
        "model": {
            "model_id": model["name"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
        },
        "dataset": {
            "manifests": [_path_entry(dataset_path)],
            "split_name": dataset_spec.get("split_name", dataset_name),
            "split_lock_status": dataset_spec.get(
                "split_lock_status",
                "screening_allowed",
            ),
        },
        "method": {
            "name": method["name"],
            "precision": method["precision"],
            "quantization": method["quantization"],
            "sequence_length": 2048,
            "packing": {"enabled": False},
            "checkpointing": {"enabled": False, "path": None},
        },
        "training": {
            "microbatch_size": 0,
            "gradient_accumulation_steps": 0,
            "supervised_tokens_per_optimizer_step": 0,
            "total_supervised_token_budget": 0,
            "optimizer": None,
            "learning_rate": None,
            "warmup_steps": 0,
            "seed": decoding["seed"],
        },
        "decoding": {
            "do_sample": bool(decoding.get("do_sample")),
            "max_new_tokens": decoding.get("max_new_tokens"),
            "seed": decoding.get("seed"),
            "enable_thinking": bool(decoding.get("enable_thinking")),
            "temperature": decoding.get("temperature"),
            "top_p": decoding.get("top_p"),
            "top_k": decoding.get("top_k"),
        },
        "hardware": {
            "instance_type": environment["hardware"].get("instance_type"),
            "gpu": environment["hardware"].get("gpu"),
            "host_memory_gb": environment["hardware"].get("host_memory_gb"),
            "peak_allocated_vram_gb": generation_metadata.get(
                "peak_allocated_vram_gb",
            ),
            "peak_reserved_vram_gb": generation_metadata.get(
                "peak_reserved_vram_gb",
            ),
            "wall_time_seconds": generation_metadata.get(
                "generation_wall_time_seconds",
            ),
            "throughput": {
                "records_per_second": generation_metadata.get(
                    "records_per_second",
                ),
                "generated_tokens_per_second": generation_metadata.get(
                    "generated_tokens_per_second",
                ),
                "load_time_seconds": generation_metadata.get(
                    "load_time_seconds",
                ),
                "batch_size": generation.get("batch_size"),
            },
            "cost": None,
        },
        "artifacts": _manifest_artifacts(run_dir, log_path),
    }
    validation = validate_run_manifest(manifest)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_checksums(
        run_dir / "checksums.sha256",
        [
            path
            for path in run_dir.glob("*")
            if path.is_file() and path.name != "checksums.sha256"
        ]
        + [log_path],
    )


def _run_one(
    *,
    matrix: Mapping[str, Any],
    run_config: Mapping[str, Any],
    dataset_name: str,
    dataset_spec: Mapping[str, Any],
    results_root: Path,
    logs_root: Path,
    cache_dir: Path | None,
    skip_generation_if_complete: bool,
) -> Path:
    _validate_config_pair(run_config)
    dataset_path = Path(str(dataset_spec["path"]))
    assert_split_allowed(dataset_path, command_name="exp02-matrix")
    expected_records = int(dataset_spec.get("expected_records", 0) or 0)
    if expected_records and _dataset_count(dataset_path) != expected_records:
        raise ValueError(f"Unexpected record count for {dataset_path}")

    run_dir = results_root / str(run_config["run_id"]) / dataset_name
    log_path = logs_root / f"{run_config['run_id']}-{dataset_name}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    _write_yaml(
        run_dir / "resolved_config.yaml",
        {
            "matrix_config": str(DEFAULT_MATRIX_CONFIG),
            "run": dict(run_config),
            "dataset": dict(dataset_spec),
        },
    )
    _run_logged(
        _render_prompt_command(
            run_config=run_config,
            dataset_path=dataset_path,
            run_dir=run_dir,
            cache_dir=cache_dir,
        ),
        log_path=log_path,
    )
    predictions_path = run_dir / "predictions.jsonl"
    generation_complete = (
        predictions_path.is_file()
        and len(_prediction_ids(predictions_path)) == expected_records
    )
    if generation_complete and skip_generation_if_complete:
        print(f"generation_complete={predictions_path}")
    else:
        _run_logged(
            _generation_command(
                run_config=run_config,
                dataset_path=dataset_path,
                run_dir=run_dir,
                cache_dir=cache_dir,
            ),
            log_path=log_path,
        )
    _verify_complete_predictions(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
    )
    _run_logged(_evaluate_command(dataset_path=dataset_path, run_dir=run_dir), log_path=log_path)
    _run_logged(_summarize_command(run_dir=run_dir), log_path=log_path)
    _write_run_artifacts(
        matrix=matrix,
        run_config=run_config,
        dataset_name=dataset_name,
        dataset_spec=dataset_spec,
        dataset_path=dataset_path,
        run_dir=run_dir,
        log_path=log_path,
        started_at=started_at,
    )
    return run_dir


def _prompt_hashes_by_id(path: Path) -> dict[str, tuple[str, str]]:
    records = read_jsonl(path)
    return {
        str(record["id"]): (
            str(record["prompt_sha256"]),
            str(record["input_ids_sha256"]),
        )
        for record in records
    }


def _compare_prompt_hashes(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    output_path: Path,
) -> None:
    baseline = _prompt_hashes_by_id(baseline_dir / "prompt_hashes.jsonl")
    candidate = _prompt_hashes_by_id(candidate_dir / "prompt_hashes.jsonl")
    mismatches = [
        record_id
        for record_id in sorted(set(baseline) | set(candidate))
        if baseline.get(record_id) != candidate.get(record_id)
    ]
    _write_json(
        output_path,
        {
            "schema_version": "1.0",
            "records": len(baseline),
            "candidate_records": len(candidate),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:100],
        },
    )
    if mismatches:
        raise ValueError(f"Prompt hash mismatch: {mismatches[:5]}")


def _write_comparisons(
    *,
    matrix: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    selected_datasets: set[str],
    results_root: Path,
) -> None:
    comparison_root = results_root / "comparisons"
    for comparison in matrix.get("comparisons", []):
        if not isinstance(comparison, Mapping):
            continue
        baseline_run_id = str(comparison["baseline_run_id"])
        candidate_run_id = str(comparison["candidate_run_id"])
        comparison_name = str(comparison["name"])
        for dataset_name in sorted(selected_datasets):
            if dataset_name not in datasets:
                continue
            baseline_dir = results_root / baseline_run_id / dataset_name
            candidate_dir = results_root / candidate_run_id / dataset_name
            output_dir = comparison_root / comparison_name / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            _compare_prompt_hashes(
                baseline_dir=baseline_dir,
                candidate_dir=candidate_dir,
                output_path=output_dir / "prompt_hash_comparison.json",
            )
            write_comparison(
                baseline_scored_path=baseline_dir / "scored_predictions.jsonl",
                candidate_scored_path=candidate_dir / "scored_predictions.jsonl",
                output_dir=output_dir,
                metrics=DEFAULT_COMPARISON_METRICS,
                bootstrap_samples=1000,
                seed=42,
                confidence=0.95,
            )
            _write_checksums(
                output_dir / "checksums.sha256",
                [path for path in output_dir.glob("*") if path.is_file()],
            )


def _case_reason(record: Mapping[str, Any]) -> str:
    return str(record.get("reason_category") or "unknown")


def _write_failure_taxonomy_sample(
    *,
    matrix: Mapping[str, Any],
    selected_runs: set[str],
    selected_datasets: set[str],
    results_root: Path,
) -> None:
    sample_size = int(
        dict(matrix.get("failure_review", {})).get("sample_size", 100),
    )
    candidates: list[dict[str, Any]] = []
    for run_id in sorted(selected_runs):
        for dataset_name in sorted(selected_datasets):
            report_path = results_root / run_id / dataset_name / "case_report.json"
            report = _read_json(report_path)
            for case in report.get("cases", []):
                if isinstance(case, Mapping) and not bool(case.get("passed")):
                    candidates.append(
                        {
                            "run_id": run_id,
                            "dataset_name": dataset_name,
                            "classification_source": "evaluator-diagnostic",
                            **dict(case),
                        },
                    )
    candidates.sort(
        key=lambda item: (
            _case_reason(item),
            str(item.get("run_id")),
            str(item.get("dataset_name")),
            str(item.get("id")),
        ),
    )
    selected = candidates[:sample_size]
    sample_path = results_root / "failure_taxonomy_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    reason_counts: dict[str, int] = {}
    for row in selected:
        reason = _case_reason(row)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    _write_json(
        results_root / "failure_taxonomy_summary.json",
        {
            "schema_version": "1.0",
            "requested_sample_size": sample_size,
            "available_failures": len(candidates),
            "sampled_failures": len(selected),
            "classification_source": "evaluator-diagnostic",
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    )


def _metric_value(scores: Mapping[str, Any], key: str) -> Any:
    value = scores.get(key)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _run_summary(run_dir: Path) -> dict[str, Any]:
    scores = _read_json(run_dir / "scores.json")
    requested = _read_json(run_dir / "requested_metrics.json")
    generation = _read_json(run_dir / "generation_metadata.json")
    return {
        "total_records": scores.get("total_records"),
        "strict_complete_match_rate": scores.get("strict_complete_match_rate"),
        "schema_equivalent_complete_match_rate": scores.get(
            "schema_equivalent_complete_match_rate",
        ),
        "executable_complete_match_rate": scores.get(
            "executable_complete_match_rate",
        ),
        "function_name_precision": scores.get("function_name_precision"),
        "function_name_recall": scores.get("function_name_recall"),
        "complete_call_precision": scores.get("complete_call_precision"),
        "complete_call_recall": scores.get("complete_call_recall"),
        "no_tool_false_positive_rate": _metric_value(
            requested,
            "no_tool_false_positive_rate",
        ),
        "malformed_call_rate": _metric_value(requested, "malformed_call_rate"),
        "records_per_second": generation.get("records_per_second"),
        "generated_tokens_per_second": generation.get(
            "generated_tokens_per_second",
        ),
        "load_time_seconds": generation.get("load_time_seconds"),
        "generation_wall_time_seconds": generation.get(
            "generation_wall_time_seconds",
        ),
        "peak_allocated_vram_gb": generation.get("peak_allocated_vram_gb"),
        "peak_reserved_vram_gb": generation.get("peak_reserved_vram_gb"),
    }


def _write_decision_artifact(
    *,
    matrix: Mapping[str, Any],
    selected_runs: set[str],
    selected_datasets: set[str],
    results_root: Path,
) -> None:
    summaries: dict[str, Any] = {}
    for run_id in sorted(selected_runs):
        summaries[run_id] = {}
        for dataset_name in sorted(selected_datasets):
            summaries[run_id][dataset_name] = _run_summary(
                results_root / run_id / dataset_name,
            )
    bf16_tool = summaries.get("bf16-deterministic", {}).get("tool_dev", {})
    base_exec = bf16_tool.get("executable_complete_match_rate")
    base_malformed = bf16_tool.get("malformed_call_rate")
    base_useful = (
        isinstance(base_exec, int | float)
        and base_exec >= 0.5
        and (not isinstance(base_malformed, int | float) or base_malformed <= 0.02)
    )

    def comparison_metric(
        comparison_name: str,
        dataset_name: str,
        metric_name: str,
    ) -> Mapping[str, Any]:
        path = (
            results_root
            / "comparisons"
            / comparison_name
            / dataset_name
            / "comparison.json"
        )
        if not path.is_file():
            return {}
        data = _read_json(path)
        metric = data.get("metrics", {}).get(metric_name, {})
        return metric if isinstance(metric, Mapping) else {}

    nf4_exec_delta = comparison_metric(
        "bf16-vs-nf4-deterministic",
        "tool_dev",
        "executable_complete_match",
    )
    nf4_ci = nf4_exec_delta.get("paired_bootstrap_ci", {})
    nf4_ci_upper = (
        nf4_ci.get("upper") if isinstance(nf4_ci, Mapping) else None
    )
    nf4_feasible = not (
        isinstance(nf4_ci_upper, int | float) and nf4_ci_upper < -0.02
    )

    sampling_exec_delta = comparison_metric(
        "bf16-deterministic-vs-sampling",
        "tool_dev",
        "executable_complete_match",
    )
    sampling_ci = sampling_exec_delta.get("paired_bootstrap_ci", {})
    sampling_ci_lower = (
        sampling_ci.get("lower") if isinstance(sampling_ci, Mapping) else None
    )
    sampling_ci_upper = (
        sampling_ci.get("upper") if isinstance(sampling_ci, Mapping) else None
    )
    sampling_clear_gain = (
        isinstance(sampling_ci_lower, int | float) and sampling_ci_lower > 0.0
    )
    sampling_clear_loss = (
        isinstance(sampling_ci_upper, int | float) and sampling_ci_upper < 0.0
    )

    base_decision = (
        "continue_qwen3_1_7b_training"
        if base_useful
        else "consider_qwen3_4b_before_training"
    )
    nf4_decision = (
        "acceptable_for_primary_comparison"
        if nf4_feasible
        else "not_acceptable_for_primary_comparison"
    )
    sampling_decision = (
        "sampling_showed_clear_gain"
        if sampling_clear_gain
        else (
            "sampling_showed_clear_loss"
            if sampling_clear_loss
            else "sampling_is_sensitivity_only_no_primary_replacement"
        )
    )
    qwen3_4b_decision = (
        "not_required_before_next_training_step"
        if base_useful
        else "evaluate_before_training"
    )
    decision = {
        "schema_version": "1.0",
        "experiment_id": matrix.get("experiment_id"),
        "task_id": matrix.get("task_id"),
        "status": "complete",
        "runs": summaries,
        "comparisons": matrix.get("comparisons", []),
        "decision": {
            "base_model_learning_region": base_decision,
            "nf4_feasibility": nf4_decision,
            "sampling_policy": sampling_decision,
            "qwen3_4b_evaluation_needed": qwen3_4b_decision,
        },
        "notes": [
            "No training or optimizer steps were run.",
            "Final no-tool records are frozen and were not evaluated.",
            "Failure taxonomy sample uses deterministic evaluator diagnostics.",
        ],
    }
    _write_json(results_root / "decision.json", decision)
    lines = [
        "# Experiment 2 Decision Artifact",
        "",
        "No training or optimizer steps were run.",
        "Final no-tool records were locked and unused.",
        "",
        "## Run Metrics",
    ]
    for run_id, by_dataset in summaries.items():
        lines.append(f"### {run_id}")
        for dataset_name, summary in by_dataset.items():
            lines.append(f"- {dataset_name}:")
            lines.append(
                "  "
                + json.dumps(
                    summary,
                    sort_keys=True,
                    separators=(",", ": "),
                ),
            )
    lines.extend(
        [
            "",
            "## Decision",
            f"- Base model learning region: {base_decision}",
            f"- NF4 feasibility: {nf4_decision}",
            f"- Sampling: {sampling_decision}",
            f"- Qwen3-4B evaluation: {qwen3_4b_decision}",
        ],
    )
    (results_root / "decision.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    _write_checksums(
        results_root / "matrix_checksums.sha256",
        [
            path
            for path in results_root.rglob("*")
            if path.is_file() and path.name != "matrix_checksums.sha256"
        ],
    )


def main() -> None:
    args = parse_args()
    matrix = _load_yaml(args.config)
    runs = _run_configs(matrix)
    datasets = _dataset_specs(matrix)
    selected_runs = _selected_names(runs, args.only_runs)
    selected_datasets = _selected_names(datasets, args.only_datasets)
    output = matrix.get("output", {})
    if not isinstance(output, Mapping):
        output = {}
    results_root = args.results_root or Path(str(output.get("results_root")))
    logs_root = args.logs_root or Path(str(output.get("logs_root")))
    cache_dir = args.cache_dir
    if cache_dir is None and output.get("cache_dir") is not None:
        cache_dir = Path(str(output["cache_dir"]))

    plan = _dry_run_plan(
        matrix=matrix,
        runs=runs,
        datasets=datasets,
        selected_runs=selected_runs,
        selected_datasets=selected_datasets,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if not results_root:
        raise ValueError("results_root is required")
    if not logs_root:
        raise ValueError("logs_root is required")
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    _write_json(results_root / "dry_run_plan.json", plan)
    _write_json(results_root / "matrix_environment_report.json", _environment_report())

    for run_id in sorted(selected_runs):
        for dataset_name in sorted(selected_datasets):
            print(f"exp02_run={run_id} dataset={dataset_name}", flush=True)
            _run_one(
                matrix=matrix,
                run_config=runs[run_id],
                dataset_name=dataset_name,
                dataset_spec=datasets[dataset_name],
                results_root=results_root,
                logs_root=logs_root,
                cache_dir=cache_dir,
                skip_generation_if_complete=args.skip_generation_if_complete,
            )

    _write_comparisons(
        matrix=matrix,
        datasets=datasets,
        selected_datasets=selected_datasets,
        results_root=results_root,
    )
    _write_failure_taxonomy_sample(
        matrix=matrix,
        selected_runs=selected_runs,
        selected_datasets=selected_datasets,
        results_root=results_root,
    )
    _write_decision_artifact(
        matrix=matrix,
        selected_runs=selected_runs,
        selected_datasets=selected_datasets,
        results_root=results_root,
    )
    print(f"results_root={results_root}")


if __name__ == "__main__":
    main()
