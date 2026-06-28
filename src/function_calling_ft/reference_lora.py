from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_MODEL_NAME = "Qwen/Qwen3-1.7B"
EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_TOKENIZER_REVISION = EXPECTED_MODEL_REVISION
EXPECTED_TARGET_MODULES = ("*q_proj", "*k_proj", "*v_proj", "*o_proj")
EXPECTED_MLP_TARGET_MODULES = ("*gate_proj", "*up_proj", "*down_proj")
EXPECTED_ATTENTION_MLP_TARGET_MODULES = (
    *EXPECTED_TARGET_MODULES,
    *EXPECTED_MLP_TARGET_MODULES,
)
FORBIDDEN_TARGET_SUFFIXES = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    "embed_tokens",
)
FORBIDDEN_NON_ADAPTER_SUFFIXES = (
    "lm_head",
    "embed_tokens",
    "norm",
    "input_layernorm",
    "post_attention_layernorm",
)
EXPECTED_SEQUENCE_LENGTH = 2048
EXPECTED_LORA_RANK = 8
EXPECTED_LORA_ALPHA = 16
EXPECTED_LORA_DROPOUT = 0.05
EXPECTED_LR = 1.0e-4
EXPECTED_WARMUP_RATIO = 0.03
EXPECTED_TRAIN_PATH = "/workspace/data/processed/xlam_splits_v1/train_10k.jsonl"
EXPECTED_TRAIN_2K_PATH = "/workspace/data/processed/xlam_splits_v1/train_2k.jsonl"
EXPECTED_TRAIN_10K_PATH = EXPECTED_TRAIN_PATH
EXPECTED_TRAIN_FULL_PATH = "/workspace/data/processed/xlam_splits_v1/train_full.jsonl"
EXPECTED_VALIDATION_PATH = "/workspace/data/processed/xlam_splits_v1/validation.jsonl"
CANONICAL_EXP03_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-03")
CANONICAL_EXP04_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-04")
CANONICAL_EXP06_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-06")
CANONICAL_EXP07_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-07")
CANONICAL_EXP08_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-08")
CANONICAL_EXP09_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-09a")
EXPECTED_NF4_QUANTIZATION = {
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_quant_storage": "bfloat16",
}


@dataclass(frozen=True)
class LoraRankProfile:
    rank: int
    alpha: int
    checkpoint_root: Path = CANONICAL_EXP06_CHECKPOINT_ROOT

    @property
    def name(self) -> str:
        return f"rank{self.rank}_alpha{self.alpha}"


EXP06_RANK_PROFILES: tuple[LoraRankProfile, ...] = (
    LoraRankProfile(rank=4, alpha=8),
    LoraRankProfile(rank=8, alpha=16),
    LoraRankProfile(rank=16, alpha=32),
)
EXP06_RANK_PROFILE_BY_RANK = {
    profile.rank: profile for profile in EXP06_RANK_PROFILES
}


@dataclass(frozen=True)
class LoraTargetProfile:
    name: str
    target_modules: tuple[str, ...]
    rank: int = 4
    alpha: int = 8
    checkpoint_root: Path = CANONICAL_EXP07_CHECKPOINT_ROOT

    @property
    def stage_name(self) -> str:
        return self.name.replace("_", "-")


EXP07_TARGET_PROFILES: tuple[LoraTargetProfile, ...] = (
    LoraTargetProfile(name="attention", target_modules=EXPECTED_TARGET_MODULES),
    LoraTargetProfile(
        name="attention_mlp",
        target_modules=EXPECTED_ATTENTION_MLP_TARGET_MODULES,
    ),
)
EXP07_TARGET_PROFILE_BY_NAME = {
    profile.name: profile for profile in EXP07_TARGET_PROFILES
}


@dataclass(frozen=True)
class LoraSampleProfile:
    name: str
    train_path: str
    manifest_path: str
    checkpoint_root: Path = CANONICAL_EXP08_CHECKPOINT_ROOT
    rank: int = 4
    alpha: int = 8
    target_modules: tuple[str, ...] = EXPECTED_TARGET_MODULES

    @property
    def stage_name(self) -> str:
        return self.name.replace("_", "-")


EXP08_SAMPLE_PROFILES: tuple[LoraSampleProfile, ...] = (
    LoraSampleProfile(
        name="train_2k",
        train_path=EXPECTED_TRAIN_2K_PATH,
        manifest_path="/workspace/data/processed/xlam_splits_v1/manifests/train_2k_manifest.jsonl",
    ),
    LoraSampleProfile(
        name="train_10k",
        train_path=EXPECTED_TRAIN_10K_PATH,
        manifest_path="/workspace/data/processed/xlam_splits_v1/manifests/train_10k_manifest.jsonl",
    ),
    LoraSampleProfile(
        name="train_full",
        train_path=EXPECTED_TRAIN_FULL_PATH,
        manifest_path="/workspace/data/processed/xlam_splits_v1/manifests/train_manifest.jsonl",
    ),
)
EXP08_SAMPLE_PROFILE_BY_NAME = {
    profile.name: profile for profile in EXP08_SAMPLE_PROFILES
}


@dataclass(frozen=True)
class LossMaskAblationProfile:
    name: str
    answer_only_loss_mask: bool
    checkpoint_root: Path = CANONICAL_EXP09_CHECKPOINT_ROOT
    rank: int = EXPECTED_LORA_RANK
    alpha: int = EXPECTED_LORA_ALPHA
    target_modules: tuple[str, ...] = EXPECTED_TARGET_MODULES

    @property
    def stage_name(self) -> str:
        return self.name.replace("_", "-")


EXP09_LOSS_MASK_PROFILES: tuple[LossMaskAblationProfile, ...] = (
    LossMaskAblationProfile(
        name="assistant_only_short",
        answer_only_loss_mask=True,
    ),
    LossMaskAblationProfile(
        name="full_sequence_short",
        answer_only_loss_mask=False,
    ),
)
EXP09_LOSS_MASK_PROFILE_BY_NAME = {
    profile.name: profile for profile in EXP09_LOSS_MASK_PROFILES
}


@dataclass(frozen=True)
class ReferenceLoraConfigValidation:
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
    target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    errors: tuple[str, ...]
    method: str = "bf16_lora"
    quantization: Mapping[str, Any] | None = None

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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _almost_equal(left: float, right: float, *, tolerance: float = 1.0e-9) -> bool:
    return abs(left - right) <= tolerance


def _checkpoint_under_root(checkpoint_dir: Path, root: Path) -> bool:
    text = checkpoint_dir.as_posix()
    return text == root.as_posix() or text.startswith(
        root.as_posix() + "/",
    )


def _validate_reference_adapter_config(
    path: Path,
    *,
    method: str,
    checkpoint_root: Path,
    require_nf4_quantization: bool,
    expected_lora_rank: int = EXPECTED_LORA_RANK,
    expected_lora_alpha: int = EXPECTED_LORA_ALPHA,
    expected_target_modules: tuple[str, ...] = EXPECTED_TARGET_MODULES,
    expected_train_path: str = EXPECTED_TRAIN_PATH,
    forbidden_target_suffixes: tuple[str, ...] = FORBIDDEN_TARGET_SUFFIXES,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    errors: list[str] = []

    try:
        config = load_yaml_config(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ReferenceLoraConfigValidation(
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
            target_modules=(),
            lora_rank=0,
            lora_alpha=0,
            errors=(str(exc),),
            method=method,
            quantization=None,
        )

    scheduler = _as_mapping(config.get("step_scheduler"))
    model = _as_mapping(config.get("model"))
    checkpoint = _as_mapping(config.get("checkpoint"))
    peft = _as_mapping(config.get("peft"))
    dataset = _as_mapping(config.get("dataset"))
    validation_dataset = _as_mapping(config.get("validation_dataset"))
    packed_sequence = _as_mapping(config.get("packed_sequence"))
    optimizer = _as_mapping(config.get("optimizer"))
    lr_scheduler = _as_mapping(config.get("lr_scheduler"))
    quantization = _as_mapping(config.get("quantization"))

    checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir", "")))
    model_name = str(model.get("pretrained_model_name_or_path", ""))
    model_revision = str(model.get("revision", ""))
    train_path = str(dataset.get("path_or_dataset_id", ""))
    validation_path = str(validation_dataset.get("path_or_dataset_id", ""))
    sequence_length = int(dataset.get("seq_length", 0) or 0)
    validation_sequence_length = int(
        validation_dataset.get("seq_length", 0) or 0,
    )
    max_steps = int(scheduler.get("max_steps", 0) or 0)
    global_batch_size = int(scheduler.get("global_batch_size", 0) or 0)
    local_batch_size = int(scheduler.get("local_batch_size", 0) or 0)
    warmup_steps_raw = lr_scheduler.get("lr_warmup_steps")
    warmup_steps = int(warmup_steps_raw) if warmup_steps_raw is not None else None
    target_modules = _string_tuple(peft.get("target_modules"))
    lora_rank = int(peft.get("dim", 0) or 0)
    lora_alpha = int(peft.get("alpha", 0) or 0)

    if model_name != EXPECTED_MODEL_NAME:
        errors.append(f"model must be {EXPECTED_MODEL_NAME}")
    if model_revision != EXPECTED_MODEL_REVISION:
        errors.append("model revision must remain the pinned Qwen3-1.7B revision")
    if str(model.get("torch_dtype")) not in {"bfloat16", "bf16", "torch.bfloat16"}:
        errors.append("model.torch_dtype must be bfloat16/bf16")
    if require_nf4_quantization and bool(model.get("force_hf")) is not True:
        errors.append("NF4 QLoRA config must set model.force_hf=true in this container")
    if not require_nf4_quantization and "force_hf" in model:
        errors.append("reference BF16 LoRA config must not set model.force_hf")
    if train_path != expected_train_path:
        errors.append(f"dataset.path_or_dataset_id must be {expected_train_path}")
    if (
        validation_path != EXPECTED_VALIDATION_PATH
        and not allow_alternate_validation_path
    ):
        errors.append(
            "validation_dataset.path_or_dataset_id must be "
            f"{EXPECTED_VALIDATION_PATH}",
        )
    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        errors.append(f"dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}")
    if validation_sequence_length != EXPECTED_SEQUENCE_LENGTH:
        errors.append(
            f"validation_dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}",
        )
    if bool(peft.get("match_all_linear", False)):
        errors.append("peft.match_all_linear must be false for controlled LoRA")
    if set(target_modules) != set(expected_target_modules):
        errors.append(
            "peft.target_modules must match the expected projection patterns "
            f"{expected_target_modules}",
        )
    if lora_rank != expected_lora_rank:
        errors.append(f"peft.dim must be {expected_lora_rank}")
    if lora_alpha != expected_lora_alpha:
        errors.append(f"peft.alpha must be {expected_lora_alpha}")
    if not _almost_equal(float(peft.get("dropout", -1.0)), EXPECTED_LORA_DROPOUT):
        errors.append(f"peft.dropout must be {EXPECTED_LORA_DROPOUT}")
    if not set(forbidden_target_suffixes).isdisjoint(
        {item.lstrip("*") for item in target_modules},
    ):
        errors.append("peft.target_modules includes forbidden module suffixes")
    if not _checkpoint_under_root(checkpoint_dir, checkpoint_root):
        errors.append(f"checkpoint.checkpoint_dir must be under {checkpoint_root}")
    if require_nf4_quantization:
        for key, expected_value in EXPECTED_NF4_QUANTIZATION.items():
            actual_value = quantization.get(key)
            if actual_value != expected_value:
                errors.append(f"quantization.{key} must be {expected_value!r}")
    elif bool(quantization.get("load_in_4bit", False)):
        errors.append("reference BF16 LoRA config must not enable 4-bit loading")
    if int(packed_sequence.get("packed_sequence_size", -1) or 0) != 0:
        errors.append("packed_sequence.packed_sequence_size must be 0")
    if str(optimizer.get("_target_")) != "torch.optim.AdamW":
        errors.append("optimizer._target_ must remain torch.optim.AdamW")
    if not _almost_equal(float(optimizer.get("lr", 0.0) or 0.0), EXPECTED_LR):
        errors.append(f"optimizer.lr must be {EXPECTED_LR}")
    if str(lr_scheduler.get("lr_decay_style")) != "cosine":
        errors.append("lr_scheduler.lr_decay_style must be cosine")
    if warmup_steps is None:
        errors.append("lr_scheduler.lr_warmup_steps must be set")
    if max_steps <= 0:
        errors.append("step_scheduler.max_steps must be positive")
    if global_batch_size <= 0:
        errors.append("step_scheduler.global_batch_size must be positive")
    if local_batch_size <= 0:
        errors.append("step_scheduler.local_batch_size must be positive")
    if global_batch_size and local_batch_size and global_batch_size % local_batch_size:
        errors.append("global_batch_size must be divisible by local_batch_size")

    return ReferenceLoraConfigValidation(
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
        target_modules=target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        errors=tuple(errors),
        method=method,
        quantization=dict(quantization) if quantization else None,
    )


def validate_reference_lora_config(
    path: Path,
    *,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    return _validate_reference_adapter_config(
        path,
        method="bf16_lora",
        checkpoint_root=CANONICAL_EXP03_CHECKPOINT_ROOT,
        require_nf4_quantization=False,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )


def validate_reference_qlora_config(
    path: Path,
    *,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    return _validate_reference_adapter_config(
        path,
        method="nf4_qlora",
        checkpoint_root=CANONICAL_EXP04_CHECKPOINT_ROOT,
        require_nf4_quantization=True,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )


def validate_lora_rank_config(
    path: Path,
    *,
    rank: int,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    profile = EXP06_RANK_PROFILE_BY_RANK.get(rank)
    if profile is None:
        raise ValueError(f"Unsupported LoRA rank for Exp 06: {rank}")
    return _validate_reference_adapter_config(
        path,
        method=f"bf16_lora_{profile.name}",
        checkpoint_root=profile.checkpoint_root,
        require_nf4_quantization=False,
        expected_lora_rank=profile.rank,
        expected_lora_alpha=profile.alpha,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )


def validate_lora_target_config(
    path: Path,
    *,
    target_profile: str,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    profile = EXP07_TARGET_PROFILE_BY_NAME.get(target_profile)
    if profile is None:
        raise ValueError(f"Unsupported LoRA target profile for Exp 07: {target_profile}")
    return _validate_reference_adapter_config(
        path,
        method=f"bf16_lora_r{profile.rank}_alpha{profile.alpha}_{profile.name}",
        checkpoint_root=profile.checkpoint_root,
        require_nf4_quantization=False,
        expected_lora_rank=profile.rank,
        expected_lora_alpha=profile.alpha,
        expected_target_modules=profile.target_modules,
        forbidden_target_suffixes=FORBIDDEN_NON_ADAPTER_SUFFIXES,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )


def validate_lora_sample_efficiency_config(
    path: Path,
    *,
    sample_profile: str,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    profile = EXP08_SAMPLE_PROFILE_BY_NAME.get(sample_profile)
    if profile is None:
        raise ValueError(f"Unsupported LoRA sample profile for Exp 08: {sample_profile}")
    return _validate_reference_adapter_config(
        path,
        method=f"bf16_lora_r{profile.rank}_alpha{profile.alpha}_{profile.name}",
        checkpoint_root=profile.checkpoint_root,
        require_nf4_quantization=False,
        expected_lora_rank=profile.rank,
        expected_lora_alpha=profile.alpha,
        expected_target_modules=profile.target_modules,
        expected_train_path=profile.train_path,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )


def validate_loss_mask_ablation_config(
    path: Path,
    *,
    loss_mask_profile: str,
    allow_alternate_validation_path: bool = False,
) -> ReferenceLoraConfigValidation:
    profile = EXP09_LOSS_MASK_PROFILE_BY_NAME.get(loss_mask_profile)
    if profile is None:
        raise ValueError(
            f"Unsupported loss-mask ablation profile for Exp 09A: {loss_mask_profile}",
        )
    validation = _validate_reference_adapter_config(
        path,
        method=f"bf16_lora_r{profile.rank}_alpha{profile.alpha}_{profile.name}",
        checkpoint_root=profile.checkpoint_root,
        require_nf4_quantization=False,
        expected_lora_rank=profile.rank,
        expected_lora_alpha=profile.alpha,
        expected_target_modules=profile.target_modules,
        allow_alternate_validation_path=allow_alternate_validation_path,
    )
    if not validation.ok:
        return validation

    config = load_yaml_config(path)
    dataset = _as_mapping(config.get("dataset"))
    validation_dataset = _as_mapping(config.get("validation_dataset"))
    errors = list(validation.errors)
    expected_policy = (
        "assistant_only" if profile.answer_only_loss_mask else "full_sequence"
    )
    expected_target = "function_calling_ft.automodel_datasets.ToolCallChatDataset"
    if str(dataset.get("_target_")) != expected_target:
        errors.append(f"dataset._target_ must be {expected_target}")
    if str(validation_dataset.get("_target_")) != expected_target:
        errors.append(f"validation_dataset._target_ must be {expected_target}")
    if dataset.get("loss_mask_policy") != expected_policy:
        errors.append(
            "dataset.loss_mask_policy must be "
            f"{expected_policy!r}",
        )
    if validation_dataset.get("loss_mask_policy") != expected_policy:
        errors.append(
            "validation_dataset.loss_mask_policy must be "
            f"{expected_policy!r}",
        )
    if dataset.get("enable_thinking") is not False:
        errors.append("dataset.enable_thinking must be false")
    if validation_dataset.get("enable_thinking") is not False:
        errors.append("validation_dataset.enable_thinking must be false")

    return ReferenceLoraConfigValidation(
        path=validation.path,
        checkpoint_dir=validation.checkpoint_dir,
        model_name=validation.model_name,
        model_revision=validation.model_revision,
        train_path=validation.train_path,
        validation_path=validation.validation_path,
        sequence_length=validation.sequence_length,
        max_steps=validation.max_steps,
        global_batch_size=validation.global_batch_size,
        local_batch_size=validation.local_batch_size,
        warmup_steps=validation.warmup_steps,
        target_modules=validation.target_modules,
        lora_rank=validation.lora_rank,
        lora_alpha=validation.lora_alpha,
        errors=tuple(errors),
        method=validation.method,
        quantization=validation.quantization,
    )


def validation_to_dict(
    validation: ReferenceLoraConfigValidation,
) -> dict[str, Any]:
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
        "target_modules": list(validation.target_modules),
        "lora_rank": validation.lora_rank,
        "lora_alpha": validation.lora_alpha,
        "method": validation.method,
        "quantization": dict(validation.quantization or {}),
        "ok": validation.ok,
        "errors": list(validation.errors),
    }


def warmup_steps_for_stage(max_steps: int, *, warmup_ratio: float = EXPECTED_WARMUP_RATIO) -> int:
    if max_steps <= 0:
        return 0
    return max(1, math.ceil(max_steps * warmup_ratio))


def clone_training_config_for_stage(
    config: Mapping[str, Any],
    *,
    checkpoint_dir: str,
    global_batch_size: int,
    local_batch_size: int,
    max_steps: int,
    ckpt_every_steps: int,
    val_every_steps: int,
    validation_path: str | None = None,
    checkpoint_enabled: bool = True,
) -> dict[str, Any]:
    cloned = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in dict(config).items()
    }
    scheduler = dict(_as_mapping(cloned.get("step_scheduler")))
    scheduler.update(
        {
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
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
    return cloned


def module_name_matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def summarize_lora_target_matches(
    module_names: list[str],
    *,
    target_patterns: tuple[str, ...] = EXPECTED_TARGET_MODULES,
    forbidden_suffixes: tuple[str, ...] = FORBIDDEN_TARGET_SUFFIXES,
) -> dict[str, Any]:
    matched = sorted(
        name for name in module_names if module_name_matches_any(name, target_patterns)
    )
    expected_suffixes = tuple(pattern.lstrip("*") for pattern in target_patterns)
    by_suffix = {
        suffix: [name for name in matched if name.endswith(suffix)]
        for suffix in expected_suffixes
    }
    forbidden = sorted(
        name for name in matched if any(name.endswith(suffix) for suffix in forbidden_suffixes)
    )
    missing_suffixes = sorted(suffix for suffix, names in by_suffix.items() if not names)
    unexpected_suffixes = sorted(
        name
        for name in matched
        if not any(name.endswith(suffix) for suffix in by_suffix)
    )
    return {
        "target_patterns": list(target_patterns),
        "matched_count": len(matched),
        "matched_modules": matched,
        "counts_by_suffix": {suffix: len(names) for suffix, names in by_suffix.items()},
        "forbidden_matches": forbidden,
        "missing_suffixes": missing_suffixes,
        "unexpected_suffix_matches": unexpected_suffixes,
        "ok": bool(matched)
        and not forbidden
        and not missing_suffixes
        and not unexpected_suffixes,
    }
