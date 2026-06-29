from __future__ import annotations

import copy
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from function_calling_ft.reference_lora import (
    EXPECTED_LORA_DROPOUT,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SEQUENCE_LENGTH,
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAIN_10K_PATH,
    EXPECTED_VALIDATION_PATH,
    load_yaml_config,
)


CANONICAL_EXP09C_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-09c")
EXPECTED_EXP09C_RANK = 4
EXPECTED_EXP09C_ALPHA = 8
EXPECTED_EXP09C_LR = 1.0e-4
EXPECTED_EXP09C_DATASET_TARGET = "function_calling_ft.automodel_datasets.ToolCallChatDataset"


@dataclass(frozen=True)
class ActivationCheckpointingProfile:
    name: str
    activation_checkpointing: bool
    local_batch_size: int = 4
    global_batch_size: int = 4

    @property
    def stage_name(self) -> str:
        return self.name.replace("_", "-")


EXP09C_PROFILES: tuple[ActivationCheckpointingProfile, ...] = (
    ActivationCheckpointingProfile(name="lora_off", activation_checkpointing=False),
    ActivationCheckpointingProfile(name="lora_on", activation_checkpointing=True),
    ActivationCheckpointingProfile(
        name="lora_on_microbatch8",
        activation_checkpointing=True,
        local_batch_size=8,
        global_batch_size=8,
    ),
)
EXP09C_PROFILE_BY_NAME = {profile.name: profile for profile in EXP09C_PROFILES}


@dataclass(frozen=True)
class ActivationCheckpointingConfigValidation:
    path: Path
    profile: str
    checkpoint_dir: Path
    model_name: str
    model_revision: str
    train_path: str
    validation_path: str
    sequence_length: int
    warmup_steps: int | None
    target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    activation_checkpointing: bool | None
    local_batch_size: int
    global_batch_size: int
    max_steps: int
    errors: tuple[str, ...]
    method: str = "bf16_lora_activation_checkpointing"
    quantization: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _checkpoint_under_root(checkpoint_dir: Path, root: Path) -> bool:
    text = checkpoint_dir.as_posix()
    return text == root.as_posix() or text.startswith(root.as_posix() + "/")


def activation_checkpointing_enabled(config: Mapping[str, Any]) -> bool | None:
    distributed = _as_mapping(config.get("distributed"))
    if "activation_checkpointing" in distributed:
        return bool(distributed["activation_checkpointing"])
    legacy = _as_mapping(config.get("activation_checkpointing"))
    if "enabled" in legacy:
        return bool(legacy["enabled"])
    return None


def validate_activation_checkpointing_config(
    path: Path,
    *,
    profile_name: str,
    checkpoint_root: Path = CANONICAL_EXP09C_CHECKPOINT_ROOT,
) -> ActivationCheckpointingConfigValidation:
    profile = EXP09C_PROFILE_BY_NAME.get(profile_name)
    if profile is None:
        raise ValueError(f"Unsupported Exp09C profile: {profile_name}")

    errors: list[str] = []
    try:
        config = load_yaml_config(path)
    except Exception as exc:
        return ActivationCheckpointingConfigValidation(
            path=path,
            profile=profile_name,
            checkpoint_dir=Path(""),
            model_name="",
            model_revision="",
            train_path="",
            validation_path="",
            sequence_length=0,
            warmup_steps=None,
            target_modules=(),
            lora_rank=0,
            lora_alpha=0,
            activation_checkpointing=None,
            local_batch_size=0,
            global_batch_size=0,
            max_steps=0,
            errors=(str(exc),),
        )

    scheduler = _as_mapping(config.get("step_scheduler"))
    model = _as_mapping(config.get("model"))
    checkpoint = _as_mapping(config.get("checkpoint"))
    peft = _as_mapping(config.get("peft"))
    distributed = _as_mapping(config.get("distributed"))
    dataset = _as_mapping(config.get("dataset"))
    validation_dataset = _as_mapping(config.get("validation_dataset"))
    packed_sequence = _as_mapping(config.get("packed_sequence"))
    optimizer = _as_mapping(config.get("optimizer"))
    lr_scheduler = _as_mapping(config.get("lr_scheduler"))

    checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir", "")))
    model_name = str(model.get("pretrained_model_name_or_path", ""))
    model_revision = str(model.get("revision", ""))
    activation_checkpointing = activation_checkpointing_enabled(config)
    local_batch_size = int(scheduler.get("local_batch_size", 0) or 0)
    global_batch_size = int(scheduler.get("global_batch_size", 0) or 0)
    max_steps = int(scheduler.get("max_steps", 0) or 0)
    target_modules = tuple(_string_list(peft.get("target_modules")))
    lora_rank = int(peft.get("dim", 0) or 0)
    lora_alpha = int(peft.get("alpha", 0) or 0)
    train_path = str(dataset.get("path_or_dataset_id", ""))
    validation_path = str(validation_dataset.get("path_or_dataset_id", ""))
    sequence_length = int(dataset.get("seq_length", 0) or 0)
    warmup_raw = lr_scheduler.get("warmup_steps")
    warmup_steps = int(warmup_raw) if isinstance(warmup_raw, int | str) and str(warmup_raw).isdigit() else None

    if model_name != EXPECTED_MODEL_NAME:
        errors.append(f"model must be {EXPECTED_MODEL_NAME}")
    if model_revision != EXPECTED_MODEL_REVISION:
        errors.append("model revision must remain the pinned Qwen3-1.7B revision")
    if str(model.get("torch_dtype")) not in {"bfloat16", "bf16", "torch.bfloat16"}:
        errors.append("model.torch_dtype must be bfloat16/bf16")
    if bool(model.get("force_hf", False)):
        errors.append("Exp09C BF16 LoRA benchmark must not force HF quantized path")
    if _as_mapping(config.get("quantization")).get("load_in_4bit"):
        errors.append("Exp09C BF16 LoRA benchmark must not enable 4-bit loading")
    if dataset.get("_target_") != EXPECTED_EXP09C_DATASET_TARGET:
        errors.append(f"dataset._target_ must be {EXPECTED_EXP09C_DATASET_TARGET}")
    if validation_dataset.get("_target_") != EXPECTED_EXP09C_DATASET_TARGET:
        errors.append(
            f"validation_dataset._target_ must be {EXPECTED_EXP09C_DATASET_TARGET}",
        )
    if dataset.get("loss_mask_policy") != "assistant_only":
        errors.append("dataset.loss_mask_policy must be assistant_only")
    if validation_dataset.get("loss_mask_policy") != "assistant_only":
        errors.append("validation_dataset.loss_mask_policy must be assistant_only")
    if dataset.get("enable_thinking") is not False:
        errors.append("dataset.enable_thinking must be false")
    if validation_dataset.get("enable_thinking") is not False:
        errors.append("validation_dataset.enable_thinking must be false")
    if train_path != EXPECTED_TRAIN_10K_PATH:
        errors.append(f"dataset.path_or_dataset_id must be {EXPECTED_TRAIN_10K_PATH}")
    if validation_path != EXPECTED_VALIDATION_PATH:
        errors.append(
            f"validation_dataset.path_or_dataset_id must be {EXPECTED_VALIDATION_PATH}",
        )
    if sequence_length != EXPECTED_SEQUENCE_LENGTH:
        errors.append(f"dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}")
    if int(validation_dataset.get("seq_length", 0) or 0) != EXPECTED_SEQUENCE_LENGTH:
        errors.append(f"validation_dataset.seq_length must be {EXPECTED_SEQUENCE_LENGTH}")
    if set(target_modules) != set(EXPECTED_TARGET_MODULES):
        errors.append("peft.target_modules must be attention projections only")
    if lora_rank != EXPECTED_EXP09C_RANK:
        errors.append(f"peft.dim must be {EXPECTED_EXP09C_RANK}")
    if lora_alpha != EXPECTED_EXP09C_ALPHA:
        errors.append(f"peft.alpha must be {EXPECTED_EXP09C_ALPHA}")
    if float(peft.get("dropout", -1.0)) != EXPECTED_LORA_DROPOUT:
        errors.append(f"peft.dropout must be {EXPECTED_LORA_DROPOUT}")
    if bool(peft.get("match_all_linear", False)):
        errors.append("peft.match_all_linear must be false")
    if distributed.get("_target_") != "nemo_automodel.components.distributed.fsdp2.FSDP2Manager":
        errors.append("distributed._target_ must remain FSDP2Manager")
    if activation_checkpointing != profile.activation_checkpointing:
        errors.append(
            "distributed.activation_checkpointing must be "
            f"{profile.activation_checkpointing}",
        )
    if int(packed_sequence.get("packed_sequence_size", -1) or 0) != 0:
        errors.append("packed_sequence.packed_sequence_size must be 0")
    if str(optimizer.get("_target_")) != "torch.optim.AdamW":
        errors.append("optimizer._target_ must remain torch.optim.AdamW")
    if not math.isclose(
        float(optimizer.get("lr", 0.0) or 0.0),
        EXPECTED_EXP09C_LR,
        abs_tol=1.0e-12,
    ):
        errors.append(f"optimizer.lr must be {EXPECTED_EXP09C_LR}")
    if str(lr_scheduler.get("lr_decay_style")) != "cosine":
        errors.append("lr_scheduler.lr_decay_style must be cosine")
    if not _checkpoint_under_root(checkpoint_dir, checkpoint_root):
        errors.append(f"checkpoint.checkpoint_dir must be under {checkpoint_root}")
    if local_batch_size != profile.local_batch_size:
        errors.append(f"step_scheduler.local_batch_size must be {profile.local_batch_size}")
    if global_batch_size != profile.global_batch_size:
        errors.append(
            f"step_scheduler.global_batch_size must be {profile.global_batch_size}",
        )
    if global_batch_size and local_batch_size and global_batch_size % local_batch_size:
        errors.append("global_batch_size must be divisible by local_batch_size")
    if max_steps <= 0:
        errors.append("step_scheduler.max_steps must be positive")

    return ActivationCheckpointingConfigValidation(
        path=path,
        profile=profile_name,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        model_revision=model_revision,
        train_path=train_path,
        validation_path=validation_path,
        sequence_length=sequence_length,
        warmup_steps=warmup_steps,
        target_modules=target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        activation_checkpointing=activation_checkpointing,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        max_steps=max_steps,
        errors=tuple(errors),
    )


def validation_to_dict(
    validation: ActivationCheckpointingConfigValidation,
) -> dict[str, Any]:
    return {
        "path": str(validation.path),
        "profile": validation.profile,
        "checkpoint_dir": str(validation.checkpoint_dir),
        "model_name": validation.model_name,
        "model_revision": validation.model_revision,
        "train_path": validation.train_path,
        "validation_path": validation.validation_path,
        "sequence_length": validation.sequence_length,
        "warmup_steps": validation.warmup_steps,
        "target_modules": list(validation.target_modules),
        "lora_rank": validation.lora_rank,
        "lora_alpha": validation.lora_alpha,
        "method": validation.method,
        "quantization": dict(validation.quantization or {}),
        "activation_checkpointing": validation.activation_checkpointing,
        "local_batch_size": validation.local_batch_size,
        "global_batch_size": validation.global_batch_size,
        "max_steps": validation.max_steps,
        "ok": validation.ok,
        "errors": list(validation.errors),
    }


IGNORED_DIFF_KEYS = {
    "experiment_id",
    "task_id",
    "run_id",
    "title",
    "checkpoint",
    "task16_policy",
}


def controlled_config_view(
    config: Mapping[str, Any],
    *,
    ignore_activation_checkpointing: bool,
) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(config))
    for key in IGNORED_DIFF_KEYS:
        cloned.pop(key, None)
    distributed = dict(_as_mapping(cloned.get("distributed")))
    if ignore_activation_checkpointing:
        distributed.pop("activation_checkpointing", None)
    cloned["distributed"] = distributed
    return cloned


def compare_primary_configs(off_config: Mapping[str, Any], on_config: Mapping[str, Any]) -> dict[str, Any]:
    off_view = controlled_config_view(off_config, ignore_activation_checkpointing=True)
    on_view = controlled_config_view(on_config, ignore_activation_checkpointing=True)
    mismatched_keys = sorted(
        key
        for key in set(off_view) | set(on_view)
        if off_view.get(key) != on_view.get(key)
    )
    return {
        "schema_version": "1.0",
        "only_activation_checkpointing_and_output_paths_differ": not mismatched_keys,
        "mismatched_top_level_keys": mismatched_keys,
        "off_activation_checkpointing": activation_checkpointing_enabled(off_config),
        "on_activation_checkpointing": activation_checkpointing_enabled(on_config),
    }


def summarize_step_times(metrics: Mapping[str, Any]) -> dict[str, float | int | None]:
    raw = metrics.get("step_times_seconds")
    if not isinstance(raw, list):
        return {
            "step_time_count": None,
            "step_time_mean_seconds": None,
            "step_time_p50_seconds": None,
            "step_time_p90_seconds": None,
            "step_time_stddev_seconds": None,
        }
    values = [float(value) for value in raw if isinstance(value, int | float)]
    if not values:
        return {
            "step_time_count": 0,
            "step_time_mean_seconds": None,
            "step_time_p50_seconds": None,
            "step_time_p90_seconds": None,
            "step_time_stddev_seconds": None,
        }
    return {
        "step_time_count": len(values),
        "step_time_mean_seconds": statistics.fmean(values),
        "step_time_p50_seconds": statistics.median(values),
        "step_time_p90_seconds": sorted(values)[min(len(values) - 1, int(len(values) * 0.9))],
        "step_time_stddev_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def compact_training_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "return_code": metrics.get("return_code"),
        "aborted": metrics.get("aborted"),
        "abort_reason": metrics.get("abort_reason"),
        "duration_seconds": metrics.get("duration_seconds"),
        "global_step": metrics.get("global_step"),
        "initial_loss": metrics.get("initial_loss"),
        "final_loss": metrics.get("final_loss"),
        "losses_are_finite": metrics.get("losses_are_finite"),
        "trainable_parameter_count": metrics.get("trainable_parameter_count"),
        "frozen_parameter_count": metrics.get("frozen_parameter_count"),
        "trainable_parameter_ratio": metrics.get("trainable_parameter_ratio"),
        "average_gpu_utilization_pct": metrics.get("average_gpu_utilization_pct"),
        "peak_allocated_vram_gb": metrics.get("peak_allocated_vram_gb"),
        "peak_reserved_vram_gb": metrics.get("peak_reserved_vram_gb"),
        "tokens_per_second_mean": metrics.get("tokens_per_second_mean"),
        "supervised_tokens_per_second_mean": metrics.get(
            "supervised_tokens_per_second_mean",
        ),
        "checkpoint_exists_after": metrics.get("checkpoint_exists_after"),
        **summarize_step_times(metrics),
    }


def pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return ((new - old) / old) * 100.0


def pct_saved(old: float | None, new: float | None) -> float | None:
    delta = pct_delta(new, old)
    if delta is None:
        return None
    return -delta


def gb_saved(off_value: float | None, on_value: float | None) -> float | None:
    if off_value is None or on_value is None:
        return None
    return off_value - on_value


def build_tradeoff_summary(
    *,
    off_metrics: Mapping[str, Any],
    on_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    off = compact_training_metrics(off_metrics)
    on = compact_training_metrics(on_metrics)
    off_allocated = _number_or_none(off.get("peak_allocated_vram_gb"))
    on_allocated = _number_or_none(on.get("peak_allocated_vram_gb"))
    off_reserved = _number_or_none(off.get("peak_reserved_vram_gb"))
    on_reserved = _number_or_none(on.get("peak_reserved_vram_gb"))
    off_step = _number_or_none(off.get("step_time_mean_seconds"))
    on_step = _number_or_none(on.get("step_time_mean_seconds"))
    off_tps = _number_or_none(off.get("tokens_per_second_mean"))
    on_tps = _number_or_none(on.get("tokens_per_second_mean"))
    return {
        "schema_version": "1.0",
        "off": off,
        "on": on,
        "allocated_vram_saved_gb": gb_saved(off_allocated, on_allocated),
        "reserved_vram_saved_gb": gb_saved(off_reserved, on_reserved),
        "allocated_vram_saved_pct": pct_saved(off_allocated, on_allocated),
        "reserved_vram_saved_pct": pct_saved(off_reserved, on_reserved),
        "step_time_slowdown_pct": pct_delta(on_step, off_step),
        "tokens_per_second_delta_pct": pct_delta(on_tps, off_tps),
    }


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def policy_from_tradeoff(tradeoff: Mapping[str, Any]) -> dict[str, Any]:
    reserved_saved_pct = _number_or_none(tradeoff.get("reserved_vram_saved_pct"))
    allocated_saved_pct = _number_or_none(tradeoff.get("allocated_vram_saved_pct"))
    step_slowdown_pct = _number_or_none(tradeoff.get("step_time_slowdown_pct"))
    on_metrics = _as_mapping(tradeoff.get("on"))
    if on_metrics.get("return_code") != 0 or on_metrics.get("losses_are_finite") is False:
        return {
            "schema_version": "1.0",
            "peft_l40s_default": "off",
            "peft_l4_default": "off",
            "full_sft_l40s_default": "use_prior_full_sft_memory_evidence_or_preflight",
            "full_sft_l4_default": "not_recommended_without_separate_memory_preflight",
            "primary_reason": (
                "activation-checkpointing profile failed under the pinned "
                "BF16 LoRA path"
            ),
            "packing_interaction_measured": False,
        }
    memory_saved_pct = max(
        value
        for value in (reserved_saved_pct, allocated_saved_pct, 0.0)
        if value is not None
    )
    slowdown = step_slowdown_pct if step_slowdown_pct is not None else 0.0
    if memory_saved_pct < 10.0 and slowdown > 5.0:
        peft_default = "off"
        reason = "memory saving below 10% with measurable slowdown"
    elif memory_saved_pct >= 20.0 and slowdown <= 20.0:
        peft_default = "conditional_on_memory_pressure"
        reason = "large memory saving with acceptable recomputation cost"
    elif memory_saved_pct >= 10.0 and slowdown <= 15.0:
        peft_default = "conditional_on_memory_pressure"
        reason = "moderate memory saving with acceptable recomputation cost"
    else:
        peft_default = "off"
        reason = "trade-off does not justify enabling by default"
    return {
        "schema_version": "1.0",
        "peft_l40s_default": peft_default,
        "peft_l4_default": "conditional_memory_preflight_required",
        "full_sft_l40s_default": "use_prior_full_sft_memory_evidence_or_preflight",
        "full_sft_l4_default": "not_recommended_without_separate_memory_preflight",
        "primary_reason": reason,
        "packing_interaction_measured": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
