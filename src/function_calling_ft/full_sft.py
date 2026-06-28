from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_MODEL_NAME = "Qwen/Qwen3-1.7B"
EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_TOKENIZER_REVISION = EXPECTED_MODEL_REVISION
EXPECTED_SEQUENCE_LENGTH = 2048
EXPECTED_TRAIN_2K_PATH = "/workspace/data/processed/xlam_splits_v1/train_2k.jsonl"
EXPECTED_TRAIN_10K_PATH = "/workspace/data/processed/xlam_splits_v1/train_10k.jsonl"
EXPECTED_TRAIN_PATH = EXPECTED_TRAIN_2K_PATH
EXPECTED_VALIDATION_PATH = "/workspace/data/processed/xlam_splits_v1/validation.jsonl"
EXPECTED_EXP05A_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-05a")
EXPECTED_EXP05B_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-05b")
EXPECTED_CHECKPOINT_ROOT = EXPECTED_EXP05A_CHECKPOINT_ROOT
EXPECTED_LOCAL_BATCH_SIZE = 1
EXPECTED_GLOBAL_BATCH_SIZE = 4
EXPECTED_LR = 1.0e-5
EXPECTED_WARMUP_RATIO = 0.03
EXPECTED_GRADIENT_CLIP_NORM = 1.0
EXPECTED_MIN_TRAINABLE_RATIO = 0.95


@dataclass(frozen=True)
class FullSftProfile:
    name: str
    train_path: str
    validation_path: str
    checkpoint_root: Path
    policy_key: str


EXP05A_PROFILE = FullSftProfile(
    name="exp05a",
    train_path=EXPECTED_TRAIN_2K_PATH,
    validation_path=EXPECTED_VALIDATION_PATH,
    checkpoint_root=EXPECTED_EXP05A_CHECKPOINT_ROOT,
    policy_key="task09_policy",
)
EXP05B_PROFILE = FullSftProfile(
    name="exp05b",
    train_path=EXPECTED_TRAIN_10K_PATH,
    validation_path=EXPECTED_VALIDATION_PATH,
    checkpoint_root=EXPECTED_EXP05B_CHECKPOINT_ROOT,
    policy_key="task10_policy",
)
FULL_SFT_PROFILES = {
    EXP05A_PROFILE.name: EXP05A_PROFILE,
    EXP05B_PROFILE.name: EXP05B_PROFILE,
}


@dataclass(frozen=True)
class FullSftConfigValidation:
    path: Path
    checkpoint_dir: Path
    model_name: str
    model_revision: str
    train_path: str
    validation_path: str
    sequence_length: int
    max_steps: int
    global_batch_size: int
    local_batch_size: int
    warmup_steps: int | None
    gradient_clip_norm: float | None
    activation_checkpointing: bool
    errors: tuple[str, ...]
    profile: str = EXP05A_PROFILE.name

    @property
    def ok(self) -> bool:
        return not self.errors


def load_yaml_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def write_yaml_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _almost_equal(left: float, right: float, *, tolerance: float = 1.0e-9) -> bool:
    return abs(left - right) <= tolerance


def _checkpoint_under_root(checkpoint_dir: Path, root: Path) -> bool:
    text = checkpoint_dir.as_posix()
    return text == root.as_posix() or text.startswith(root.as_posix() + "/")


def resolve_full_sft_profile(profile: str | FullSftProfile | None) -> FullSftProfile:
    if profile is None:
        return EXP05A_PROFILE
    if isinstance(profile, FullSftProfile):
        return profile
    try:
        return FULL_SFT_PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(FULL_SFT_PROFILES))
        raise ValueError(f"unknown full-SFT profile {profile!r}; valid profiles: {valid}") from exc


def warmup_steps_for_stage(
    max_steps: int,
    *,
    warmup_ratio: float = EXPECTED_WARMUP_RATIO,
) -> int:
    if max_steps <= 0:
        return 0
    if max_steps == 1:
        return 0
    return max(1, math.ceil(max_steps * warmup_ratio))


def _gradient_clip_norm(
    config: Mapping[str, Any],
    *,
    policy_key: str,
) -> float | None:
    optimizer = _as_mapping(config.get("optimizer"))
    policy = _as_mapping(config.get(policy_key))
    fallback_policy = _as_mapping(config.get("task09_policy"))
    sources = [policy]
    if policy_key != "task09_policy":
        sources.append(fallback_policy)
    sources.append(optimizer)
    for source in sources:
        for key in (
            "gradient_clip_val",
            "grad_clip",
            "clip_grad",
            "max_grad_norm",
            "gradient_clip_norm",
        ):
            value = source.get(key)
            if value is not None:
                return float(value)
    return None


def validate_full_sft_config(
    path: Path,
    *,
    profile: str | FullSftProfile | None = None,
    allow_alternate_validation_path: bool = False,
    checkpoint_root_override: Path | None = None,
) -> FullSftConfigValidation:
    errors: list[str] = []
    resolved_profile = resolve_full_sft_profile(profile)
    try:
        config = load_yaml_config(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return FullSftConfigValidation(
            path=path,
            checkpoint_dir=Path(""),
            model_name="",
            model_revision="",
            train_path="",
            validation_path="",
            sequence_length=0,
            max_steps=0,
            global_batch_size=0,
            local_batch_size=0,
            warmup_steps=None,
            gradient_clip_norm=None,
            activation_checkpointing=False,
            errors=(str(exc),),
            profile=resolved_profile.name,
        )

    scheduler = _as_mapping(config.get("step_scheduler"))
    model = _as_mapping(config.get("model"))
    checkpoint = _as_mapping(config.get("checkpoint"))
    dataset = _as_mapping(config.get("dataset"))
    validation_dataset = _as_mapping(config.get("validation_dataset"))
    packed_sequence = _as_mapping(config.get("packed_sequence"))
    optimizer = _as_mapping(config.get("optimizer"))
    lr_scheduler = _as_mapping(config.get("lr_scheduler"))
    distributed = _as_mapping(config.get("distributed"))
    legacy_activation_checkpointing = _as_mapping(config.get("activation_checkpointing"))

    checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir", "")))
    model_name = str(model.get("pretrained_model_name_or_path", ""))
    model_revision = str(model.get("revision", ""))
    train_path = str(dataset.get("path_or_dataset_id", ""))
    validation_path = str(validation_dataset.get("path_or_dataset_id", ""))
    sequence_length = int(dataset.get("seq_length", 0) or 0)
    validation_sequence_length = int(validation_dataset.get("seq_length", 0) or 0)
    max_steps = int(scheduler.get("max_steps", 0) or 0)
    global_batch_size = int(scheduler.get("global_batch_size", 0) or 0)
    local_batch_size = int(scheduler.get("local_batch_size", 0) or 0)
    warmup_steps_raw = lr_scheduler.get("lr_warmup_steps")
    warmup_steps = int(warmup_steps_raw) if warmup_steps_raw is not None else None
    gradient_clip_norm = _gradient_clip_norm(
        config,
        policy_key=resolved_profile.policy_key,
    )
    activation_checkpointing_enabled = bool(
        distributed.get(
            "activation_checkpointing",
            legacy_activation_checkpointing.get("enabled", False),
        ),
    )

    if model_name != EXPECTED_MODEL_NAME:
        errors.append(f"model must be {EXPECTED_MODEL_NAME}")
    if model_revision != EXPECTED_MODEL_REVISION:
        errors.append("model revision must remain the pinned Qwen3-1.7B revision")
    if str(model.get("torch_dtype")) not in {"bfloat16", "bf16", "torch.bfloat16"}:
        errors.append("model.torch_dtype must be bfloat16/bf16")
    if bool(model.get("force_hf", False)):
        errors.append("full SFT must not force the HF QLoRA loading path")
    if "peft" in config:
        errors.append("full SFT config must not include a peft block")
    if "quantization" in config:
        errors.append("full SFT config must not include a quantization block")
    if train_path != resolved_profile.train_path:
        errors.append(f"dataset.path_or_dataset_id must be {resolved_profile.train_path}")
    if (
        validation_path != resolved_profile.validation_path
        and not allow_alternate_validation_path
    ):
        errors.append(
            "validation_dataset.path_or_dataset_id must be "
            f"{resolved_profile.validation_path}",
        )
    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        errors.append(f"dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}")
    if validation_sequence_length != EXPECTED_SEQUENCE_LENGTH:
        errors.append(
            f"validation_dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}",
        )
    if int(packed_sequence.get("packed_sequence_size", -1) or 0) != 0:
        errors.append("packed_sequence.packed_sequence_size must be 0")
    if str(optimizer.get("_target_")) != "torch.optim.AdamW":
        errors.append("optimizer._target_ must remain torch.optim.AdamW")
    if not _almost_equal(float(optimizer.get("lr", 0.0) or 0.0), EXPECTED_LR):
        errors.append(f"optimizer.lr must be {EXPECTED_LR}")
    if not _almost_equal(float(optimizer.get("weight_decay", 0.0) or 0.0), 0.0):
        errors.append("optimizer.weight_decay must remain 0.0 for the pilot")
    if gradient_clip_norm is None:
        errors.append("gradient clipping norm must be configured")
    elif not _almost_equal(gradient_clip_norm, EXPECTED_GRADIENT_CLIP_NORM):
        errors.append(
            f"gradient clipping norm must be {EXPECTED_GRADIENT_CLIP_NORM}",
        )
    if str(lr_scheduler.get("lr_decay_style")) != "cosine":
        errors.append("lr_scheduler.lr_decay_style must be cosine")
    if warmup_steps is None:
        errors.append("lr_scheduler.lr_warmup_steps must be set")
    if max_steps <= 0:
        errors.append("step_scheduler.max_steps must be positive")
    if global_batch_size <= 0:
        errors.append("step_scheduler.global_batch_size must be positive")
    if local_batch_size != EXPECTED_LOCAL_BATCH_SIZE:
        errors.append(f"step_scheduler.local_batch_size must be {EXPECTED_LOCAL_BATCH_SIZE}")
    if global_batch_size != EXPECTED_GLOBAL_BATCH_SIZE:
        errors.append(
            f"step_scheduler.global_batch_size must be {EXPECTED_GLOBAL_BATCH_SIZE}",
        )
    if global_batch_size and local_batch_size and global_batch_size % local_batch_size:
        errors.append("global_batch_size must be divisible by local_batch_size")
    checkpoint_root = checkpoint_root_override or resolved_profile.checkpoint_root
    if not _checkpoint_under_root(checkpoint_dir, checkpoint_root):
        errors.append(
            "checkpoint.checkpoint_dir must be under "
            f"{checkpoint_root}",
        )

    return FullSftConfigValidation(
        path=path,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        model_revision=model_revision,
        train_path=train_path,
        validation_path=validation_path,
        sequence_length=sequence_length,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
        activation_checkpointing=activation_checkpointing_enabled,
        errors=tuple(errors),
        profile=resolved_profile.name,
    )


def validation_to_dict(validation: FullSftConfigValidation) -> dict[str, Any]:
    return {
        "path": str(validation.path),
        "checkpoint_dir": str(validation.checkpoint_dir),
        "model_name": validation.model_name,
        "model_revision": validation.model_revision,
        "train_path": validation.train_path,
        "validation_path": validation.validation_path,
        "sequence_length": validation.sequence_length,
        "max_steps": validation.max_steps,
        "global_batch_size": validation.global_batch_size,
        "local_batch_size": validation.local_batch_size,
        "warmup_steps": validation.warmup_steps,
        "gradient_clip_norm": validation.gradient_clip_norm,
        "activation_checkpointing": validation.activation_checkpointing,
        "method": "full_parameter_sft",
        "profile": validation.profile,
        "ok": validation.ok,
        "errors": list(validation.errors),
    }


def clone_full_sft_config_for_stage(
    config: Mapping[str, Any],
    *,
    checkpoint_dir: str,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    profile: str | FullSftProfile | None = None,
    validation_path: str | None = None,
    checkpoint_enabled: bool = True,
    activation_checkpointing_enabled: bool | None = None,
    policy_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_profile = resolve_full_sft_profile(profile)
    cloned = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in dict(config).items()
    }
    scheduler = dict(_as_mapping(cloned.get("step_scheduler")))
    scheduler.update(
        {
            "global_batch_size": EXPECTED_GLOBAL_BATCH_SIZE,
            "local_batch_size": EXPECTED_LOCAL_BATCH_SIZE,
            "max_steps": max_steps,
            "ckpt_every_steps": ckpt_every_steps,
            "val_every_steps": val_every_steps,
        },
    )
    cloned["step_scheduler"] = scheduler

    checkpoint = dict(_as_mapping(cloned.get("checkpoint")))
    checkpoint["enabled"] = checkpoint_enabled
    checkpoint["checkpoint_dir"] = checkpoint_dir
    cloned["checkpoint"] = checkpoint

    if validation_path is not None:
        validation_dataset = dict(_as_mapping(cloned.get("validation_dataset")))
        validation_dataset["path_or_dataset_id"] = validation_path
        cloned["validation_dataset"] = validation_dataset

    lr_scheduler = dict(_as_mapping(cloned.get("lr_scheduler")))
    lr_scheduler["lr_warmup_steps"] = warmup_steps_for_stage(max_steps)
    cloned["lr_scheduler"] = lr_scheduler

    if activation_checkpointing_enabled is not None:
        distributed = dict(_as_mapping(cloned.get("distributed")))
        distributed["activation_checkpointing"] = activation_checkpointing_enabled
        cloned["distributed"] = distributed
        cloned.pop("activation_checkpointing", None)

    policy = dict(_as_mapping(cloned.get(resolved_profile.policy_key)))
    policy["gradient_clip_norm"] = EXPECTED_GRADIENT_CLIP_NORM
    policy.setdefault(
        "gradient_clip_source",
        "nemo_automodel_0.2.0rc0_train_ft_hardcoded",
    )
    if policy_updates:
        policy.update(dict(policy_updates))
    cloned[resolved_profile.policy_key] = policy

    cloned.pop("peft", None)
    cloned.pop("quantization", None)
    return cloned
