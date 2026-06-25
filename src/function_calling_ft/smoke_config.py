from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CANONICAL_CHECKPOINT_ROOT = Path("/workspace/checkpoints")
EXPECTED_MODEL_NAME = "Qwen/Qwen3-1.7B"
EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


@dataclass(frozen=True)
class SmokeConfigValidation:
    path: Path
    checkpoint_dir: Path
    model_name: str
    model_revision: str
    max_steps: int
    global_batch_size: int
    local_batch_size: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Smoke config must be a YAML mapping: {path}")

    return loaded


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_smoke_config(path: Path) -> SmokeConfigValidation:
    errors: list[str] = []

    try:
        config = load_yaml_config(path)
        checkpoint = _as_mapping(config.get("checkpoint"), name="checkpoint")
        model = _as_mapping(config.get("model"), name="model")
        scheduler = _as_mapping(
            config.get("step_scheduler"),
            name="step_scheduler",
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return SmokeConfigValidation(
            path=path,
            checkpoint_dir=Path(""),
            model_name="",
            model_revision="",
            max_steps=0,
            global_batch_size=0,
            local_batch_size=0,
            errors=(str(exc),),
        )

    checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir", "")))
    model_name = str(model.get("pretrained_model_name_or_path", ""))
    model_revision = str(model.get("revision", ""))
    max_steps = int(scheduler.get("max_steps", 0) or 0)
    global_batch_size = int(scheduler.get("global_batch_size", 0) or 0)
    local_batch_size = int(scheduler.get("local_batch_size", 0) or 0)

    if not str(checkpoint_dir).startswith(
        str(CANONICAL_CHECKPOINT_ROOT) + "/",
    ):
        errors.append(
            "checkpoint.checkpoint_dir must be under "
            f"{CANONICAL_CHECKPOINT_ROOT}",
        )

    if model_name != EXPECTED_MODEL_NAME:
        errors.append(
            "model.pretrained_model_name_or_path must remain "
            f"{EXPECTED_MODEL_NAME}",
        )

    if model_revision != EXPECTED_MODEL_REVISION:
        errors.append(
            "model.revision must remain the immutable Experiment 0 revision",
        )

    if max_steps != 30:
        errors.append("step_scheduler.max_steps must be 30")

    if global_batch_size != 4:
        errors.append("step_scheduler.global_batch_size must be 4")

    if local_batch_size != 1:
        errors.append("step_scheduler.local_batch_size must be 1")

    return SmokeConfigValidation(
        path=path,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        model_revision=model_revision,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        errors=tuple(errors),
    )


def validation_to_dict(
    validation: SmokeConfigValidation,
) -> dict[str, Any]:
    return {
        "path": str(validation.path),
        "checkpoint_dir": str(validation.checkpoint_dir),
        "model_name": validation.model_name,
        "model_revision": validation.model_revision,
        "max_steps": validation.max_steps,
        "global_batch_size": validation.global_batch_size,
        "local_batch_size": validation.local_batch_size,
        "ok": validation.ok,
        "errors": list(validation.errors),
    }
