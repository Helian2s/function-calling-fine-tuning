from __future__ import annotations

from pathlib import Path

from function_calling_ft.activation_checkpointing import (
    build_tradeoff_summary,
    compare_primary_configs,
    load_yaml_config,
    policy_from_tradeoff,
    validate_activation_checkpointing_config,
)


CONFIG_ROOT = Path("configs/exp09c_activation_checkpointing")


def test_exp09c_primary_configs_validate() -> None:
    off = validate_activation_checkpointing_config(
        CONFIG_ROOT / "lora_off.yaml",
        profile_name="lora_off",
    )
    on = validate_activation_checkpointing_config(
        CONFIG_ROOT / "lora_on.yaml",
        profile_name="lora_on",
    )

    assert off.ok, off.errors
    assert on.ok, on.errors
    assert off.activation_checkpointing is False
    assert on.activation_checkpointing is True
    assert off.local_batch_size == on.local_batch_size == 4
    assert off.global_batch_size == on.global_batch_size == 4


def test_exp09c_secondary_config_is_explicitly_separate() -> None:
    validation = validate_activation_checkpointing_config(
        CONFIG_ROOT / "lora_on_microbatch8.yaml",
        profile_name="lora_on_microbatch8",
    )

    assert validation.ok, validation.errors
    assert validation.activation_checkpointing is True
    assert validation.local_batch_size == 8
    assert validation.global_batch_size == 8


def test_exp09c_primary_config_diff_is_only_checkpointing_and_outputs() -> None:
    off = load_yaml_config(CONFIG_ROOT / "lora_off.yaml")
    on = load_yaml_config(CONFIG_ROOT / "lora_on.yaml")

    diff = compare_primary_configs(off, on)

    assert diff["only_activation_checkpointing_and_output_paths_differ"] is True
    assert diff["mismatched_top_level_keys"] == []
    assert diff["off_activation_checkpointing"] is False
    assert diff["on_activation_checkpointing"] is True


def test_exp09c_tradeoff_policy_prefers_off_for_low_memory_saving() -> None:
    tradeoff = build_tradeoff_summary(
        off_metrics={
            "peak_allocated_vram_gb": 20.0,
            "peak_reserved_vram_gb": 30.0,
            "step_times_seconds": [1.0, 1.0, 1.0],
            "tokens_per_second_mean": 100.0,
        },
        on_metrics={
            "peak_allocated_vram_gb": 19.5,
            "peak_reserved_vram_gb": 29.5,
            "step_times_seconds": [1.1, 1.1, 1.1],
            "tokens_per_second_mean": 90.0,
        },
    )
    policy = policy_from_tradeoff(tradeoff)

    assert policy["peft_l40s_default"] == "off"


def test_exp09c_tradeoff_policy_prefers_off_when_checkpointing_fails() -> None:
    tradeoff = build_tradeoff_summary(
        off_metrics={
            "return_code": 0,
            "losses_are_finite": True,
            "peak_allocated_vram_gb": 30.0,
            "peak_reserved_vram_gb": 36.0,
        },
        on_metrics={
            "return_code": 1,
            "losses_are_finite": False,
            "peak_allocated_vram_gb": 8.0,
            "peak_reserved_vram_gb": 9.0,
        },
    )
    policy = policy_from_tradeoff(tradeoff)

    assert policy["peft_l40s_default"] == "off"
    assert policy["peft_l4_default"] == "off"
