from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


RUN_MANIFEST_SCHEMA_VERSION = "1.0"
CANONICAL_ARTIFACT_KEYS = (
    "resolved_config",
    "run_manifest",
    "environment",
    "predictions",
    "per_example_scores",
    "metrics",
    "training_memory",
    "logs",
    "checksums",
    "report",
)
ALLOWED_METHODS = {
    "base",
    "bf16_lora",
    "nf4_qlora",
    "full_parameter_sft",
}
ALLOWED_STATUSES = {
    "planned",
    "running",
    "succeeded",
    "failed",
    "stopped",
    "cancelled",
    "migrated",
}


@dataclass(frozen=True)
class RunManifestValidation:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(
    manifest: Mapping[str, Any],
    section: str,
    errors: list[str],
) -> Mapping[str, Any]:
    value = manifest.get(section)
    if not isinstance(value, Mapping):
        errors.append(f"{section} must be a mapping")
        return {}
    return value


def _require_fields(
    mapping: Mapping[str, Any],
    section: str,
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    for field in fields:
        if field not in mapping:
            errors.append(f"{section}.{field} is required")


def _require_nonempty_str(
    mapping: Mapping[str, Any],
    section: str,
    field: str,
    errors: list[str],
) -> None:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{section}.{field} must be a non-empty string")


def validate_run_manifest(
    manifest: Mapping[str, Any],
) -> RunManifestValidation:
    errors: list[str] = []

    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{RUN_MANIFEST_SCHEMA_VERSION!r}",
        )

    for field in ("experiment_id", "task_id", "run_id", "status"):
        _require_nonempty_str(manifest, "manifest", field, errors)

    status = manifest.get("status")
    if isinstance(status, str) and status not in ALLOWED_STATUSES:
        errors.append(f"status {status!r} is not allowed")

    source = _require_mapping(manifest, "source", errors)
    _require_fields(source, "source", ("git_commit", "git_dirty"), errors)

    environment = _require_mapping(manifest, "environment", errors)
    _require_fields(
        environment,
        "environment",
        ("container_tag", "container_digest", "package_versions"),
        errors,
    )
    if not isinstance(environment.get("package_versions"), Mapping):
        errors.append("environment.package_versions must be a mapping")

    model = _require_mapping(manifest, "model", errors)
    for field in ("model_id", "model_revision", "tokenizer_revision"):
        _require_nonempty_str(model, "model", field, errors)

    dataset = _require_mapping(manifest, "dataset", errors)
    _require_fields(
        dataset,
        "dataset",
        ("manifests", "split_name", "split_lock_status"),
        errors,
    )
    if not isinstance(dataset.get("manifests"), list):
        errors.append("dataset.manifests must be a list")

    method = _require_mapping(manifest, "method", errors)
    _require_fields(
        method,
        "method",
        (
            "name",
            "precision",
            "quantization",
            "sequence_length",
            "packing",
            "checkpointing",
        ),
        errors,
    )
    method_name = method.get("name")
    if isinstance(method_name, str) and method_name not in ALLOWED_METHODS:
        errors.append(f"method.name {method_name!r} is not allowed")
    if not isinstance(method.get("sequence_length"), int):
        errors.append("method.sequence_length must be an integer")

    training = _require_mapping(manifest, "training", errors)
    _require_fields(
        training,
        "training",
        (
            "microbatch_size",
            "gradient_accumulation_steps",
            "supervised_tokens_per_optimizer_step",
            "total_supervised_token_budget",
            "optimizer",
            "learning_rate",
            "warmup_steps",
            "seed",
        ),
        errors,
    )

    decoding = _require_mapping(manifest, "decoding", errors)
    _require_fields(
        decoding,
        "decoding",
        ("do_sample", "max_new_tokens", "seed", "enable_thinking"),
        errors,
    )
    if not isinstance(decoding.get("enable_thinking"), bool):
        errors.append("decoding.enable_thinking must be a boolean")

    hardware = _require_mapping(manifest, "hardware", errors)
    _require_fields(
        hardware,
        "hardware",
        (
            "instance_type",
            "gpu",
            "host_memory_gb",
            "peak_allocated_vram_gb",
            "peak_reserved_vram_gb",
            "wall_time_seconds",
            "throughput",
            "cost",
        ),
        errors,
    )

    artifacts = _require_mapping(manifest, "artifacts", errors)
    for key in CANONICAL_ARTIFACT_KEYS:
        artifact = artifacts.get(key)
        if not isinstance(artifact, Mapping):
            errors.append(f"artifacts.{key} must be a mapping")
            continue
        _require_fields(artifact, f"artifacts.{key}", ("path", "sha256"), errors)

    return RunManifestValidation(tuple(errors))


def parse_package_versions(text: str) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or "=" not in line:
            continue
        name, version = line.split("=", 1)
        packages[name.strip()] = version.strip()
    return packages


def artifact_entry(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def canonical_artifacts(
    *,
    results_dir: Path,
    training_log: Path | None = None,
) -> dict[str, dict[str, str | None]]:
    return {
        "resolved_config": artifact_entry(results_dir / "resolved_config.yaml"),
        "run_manifest": artifact_entry(results_dir / "run_manifest.json"),
        "environment": artifact_entry(results_dir / "environment_report.json"),
        "predictions": artifact_entry(results_dir / "predictions.jsonl"),
        "per_example_scores": artifact_entry(
            results_dir / "scored_predictions.jsonl",
        ),
        "metrics": artifact_entry(results_dir / "scores.json"),
        "training_memory": artifact_entry(
            results_dir / "training_torch_memory.json",
        ),
        "logs": artifact_entry(training_log),
        "checksums": artifact_entry(results_dir / "checksums.sha256"),
        "report": artifact_entry(results_dir / "case_report.md"),
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _parse_container_image_text(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    container: dict[str, str] = {}
    image = parsed.get("image")
    if image:
        container["container_tag"] = image

    repo_digest = parsed.get("repo_digest")
    if repo_digest:
        container["container_repo_digest"] = repo_digest
        if "@sha256:" in repo_digest:
            container["container_digest"] = repo_digest.split("@", 1)[1]
    return container


def _package_versions_from_container_report(
    path: Path | None,
) -> dict[str, str]:
    report = _load_json(path)
    packages = report.get("packages")
    if not isinstance(packages, Mapping):
        return {}

    versions: dict[str, str] = {}
    for name, payload in packages.items():
        if not isinstance(name, str) or not isinstance(payload, Mapping):
            continue
        version = payload.get("version")
        if isinstance(version, str):
            versions[name] = version
    return versions


def _method_name(config: Mapping[str, Any], generation: Mapping[str, Any]) -> str:
    quantization = _as_mapping(config.get("quantization"))
    if quantization.get("load_in_4bit") or generation.get("load_in_4bit"):
        return "nf4_qlora"
    if isinstance(config.get("peft"), Mapping):
        return "bf16_lora"
    return "base"


def _sequence_length(config: Mapping[str, Any]) -> int:
    dataset = _as_mapping(config.get("dataset"))
    value = dataset.get("seq_length", 0)
    return int(value or 0)


def _gradient_accumulation(config: Mapping[str, Any]) -> int | None:
    scheduler = _as_mapping(config.get("step_scheduler"))
    global_batch = int(scheduler.get("global_batch_size", 0) or 0)
    local_batch = int(scheduler.get("local_batch_size", 0) or 0)
    if not global_batch or not local_batch:
        return None
    return max(global_batch // local_batch, 1)


def _dataset_manifests(
    dataset_manifest_path: Path | None,
) -> list[dict[str, str | None]]:
    if dataset_manifest_path is None:
        return []
    return [
        {
            "path": str(dataset_manifest_path),
            "sha256": (
                sha256_file(dataset_manifest_path)
                if dataset_manifest_path.is_file()
                else None
            ),
        },
    ]


def build_exp00_run_manifest(
    *,
    run_metadata: Mapping[str, Any],
    generation_metadata: Mapping[str, Any],
    scores: Mapping[str, Any],
    training_config: Mapping[str, Any],
    results_dir: Path,
    training_log: Path | None,
    dataset_manifest_path: Path | None,
    run_id: str | None = None,
    status: str = "migrated",
    task_id: str = "C9",
) -> dict[str, Any]:
    model = _as_mapping(training_config.get("model"))
    tokenizer = _as_mapping(_as_mapping(training_config.get("tokenizer")))
    common_tokenizer = _as_mapping(
        _as_mapping(training_config.get("model")).get("tokenizer"),
    )
    scheduler = _as_mapping(training_config.get("step_scheduler"))
    optimizer = _as_mapping(training_config.get("optimizer"))
    rng = _as_mapping(training_config.get("rng"))
    quantization = _as_mapping(training_config.get("quantization"))
    packed_sequence = _as_mapping(training_config.get("packed_sequence"))
    checkpoint = _as_mapping(training_config.get("checkpoint"))

    model_id = str(
        run_metadata.get("model_name")
        or generation_metadata.get("model_name")
        or model.get("pretrained_model_name_or_path")
        or model.get("name")
        or "",
    )
    model_revision = str(
        run_metadata.get("model_revision")
        or generation_metadata.get("model_revision")
        or model.get("revision")
        or "",
    )
    tokenizer_revision = str(
        common_tokenizer.get("revision")
        or tokenizer.get("revision")
        or run_metadata.get("tokenizer_revision")
        or model_revision,
    )
    seq_length = _sequence_length(training_config)
    global_batch = int(scheduler.get("global_batch_size", 0) or 0)
    max_steps = int(scheduler.get("max_steps", 0) or 0)
    supervised_tokens = seq_length * global_batch if seq_length else None
    package_versions = {
        **_as_mapping(run_metadata.get("packages")),
        **_as_mapping(run_metadata.get("package_versions")),
    }

    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "experiment_id": "exp-00",
        "task_id": task_id,
        "run_id": run_id
        or str(run_metadata.get("run_id") or _default_run_id(run_metadata)),
        "status": status,
        "comparison": {
            "parent_run_id": run_metadata.get("parent_run_id"),
            "comparison_run_ids": run_metadata.get("comparison_run_ids", []),
        },
        "source": {
            "git_commit": run_metadata.get("git_revision"),
            "git_dirty": run_metadata.get("git_dirty"),
            "git_dirty_files": run_metadata.get("git_dirty_files", []),
        },
        "environment": {
            "container_tag": run_metadata.get("container_tag"),
            "container_digest": run_metadata.get("container_digest"),
            "package_versions": package_versions,
        },
        "model": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
        },
        "dataset": {
            "manifests": _dataset_manifests(dataset_manifest_path),
            "split_name": run_metadata.get("split_name", "smoke-v1-test"),
            "split_lock_status": run_metadata.get(
                "split_lock_status",
                "frozen_smoke",
            ),
        },
        "method": {
            "name": _method_name(training_config, generation_metadata),
            "precision": str(model.get("torch_dtype") or "bfloat16"),
            "quantization": {
                "load_in_4bit": bool(
                    quantization.get("load_in_4bit")
                    or generation_metadata.get("load_in_4bit"),
                ),
                "type": quantization.get("bnb_4bit_quant_type"),
            },
            "sequence_length": seq_length,
            "packing": {
                "enabled": bool(packed_sequence.get("packed_sequence_size", 0)),
                "packed_sequence_size": packed_sequence.get(
                    "packed_sequence_size",
                ),
            },
            "checkpointing": {
                "enabled": bool(checkpoint.get("enabled")),
                "path": checkpoint.get("checkpoint_dir")
                or run_metadata.get("adapter_output_path"),
            },
        },
        "training": {
            "microbatch_size": scheduler.get("local_batch_size"),
            "gradient_accumulation_steps": _gradient_accumulation(
                training_config,
            ),
            "supervised_tokens_per_optimizer_step": supervised_tokens,
            "total_supervised_token_budget": (
                supervised_tokens * max_steps if supervised_tokens else None
            ),
            "optimizer": optimizer.get("_target_"),
            "learning_rate": optimizer.get("lr"),
            "warmup_steps": training_config.get("warmup_steps"),
            "seed": rng.get("seed")
            or generation_metadata.get("seed")
            or run_metadata.get("seed"),
        },
        "decoding": {
            "do_sample": bool(generation_metadata.get("do_sample", False)),
            "max_new_tokens": generation_metadata.get("max_new_tokens"),
            "seed": generation_metadata.get("seed"),
            "enable_thinking": bool(
                generation_metadata.get("enable_thinking", False),
            ),
        },
        "hardware": {
            "instance_type": run_metadata.get("instance_type"),
            "gpu": run_metadata.get("gpu"),
            "host_memory_gb": run_metadata.get("host_memory_gb"),
            "peak_allocated_vram_gb": run_metadata.get(
                "peak_allocated_vram_gb",
            ),
            "peak_reserved_vram_gb": run_metadata.get(
                "peak_reserved_vram_gb",
            ),
            "wall_time_seconds": run_metadata.get("wall_time_seconds"),
            "throughput": {
                "records_per_second": generation_metadata.get(
                    "records_per_second",
                ),
                "tokens_per_second": generation_metadata.get(
                    "generated_tokens_per_second",
                ),
                "total_records": scores.get("total_records"),
            },
            "cost": run_metadata.get("cost"),
        },
        "artifacts": canonical_artifacts(
            results_dir=results_dir,
            training_log=training_log,
        ),
    }
    validation = validate_run_manifest(manifest)
    if not validation.ok:
        raise ValueError(
            "Invalid run manifest: " + "; ".join(validation.errors),
        )
    return manifest


def migrate_smoke_metadata(
    *,
    run_metadata_path: Path,
    output_path: Path,
    generation_metadata_path: Path | None = None,
    scores_path: Path | None = None,
    resolved_config_path: Path | None = None,
    package_versions_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    training_log_path: Path | None = None,
    container_image_path: Path | None = None,
    container_report_path: Path | None = None,
    run_id: str | None = None,
    status: str = "migrated",
    task_id: str = "C9",
) -> dict[str, Any]:
    run_metadata = _load_json(run_metadata_path)
    generation_metadata = _load_json(generation_metadata_path)
    scores = _load_json(scores_path)
    training_config = _load_yaml(resolved_config_path)
    package_text = (
        package_versions_path.read_text(encoding="utf-8")
        if package_versions_path is not None and package_versions_path.is_file()
        else ""
    )
    package_versions = parse_package_versions(package_text)
    package_versions = {
        **_package_versions_from_container_report(container_report_path),
        **package_versions,
    }
    if package_versions:
        run_metadata = {
            **run_metadata,
            "package_versions": package_versions,
        }

    container_metadata = _parse_container_image_text(container_image_path)
    if container_metadata:
        run_metadata = {
            **run_metadata,
            **{
                key: value
                for key, value in container_metadata.items()
                if key in {"container_tag", "container_digest"}
                and not run_metadata.get(key)
            },
            "container_repo_digest": container_metadata.get(
                "container_repo_digest",
            ),
        }

    manifest = build_exp00_run_manifest(
        run_metadata=run_metadata,
        generation_metadata=generation_metadata,
        scores=scores,
        training_config=training_config,
        results_dir=output_path.parent,
        training_log=training_log_path,
        dataset_manifest_path=dataset_manifest_path,
        run_id=run_id,
        status=status,
        task_id=task_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _default_run_id(run_metadata: Mapping[str, Any]) -> str:
    revision = str(run_metadata.get("git_revision") or "unknown")
    short_revision = revision[:12] if revision else "unknown"
    return f"exp-00-{short_revision}-{int(time.time())}"
