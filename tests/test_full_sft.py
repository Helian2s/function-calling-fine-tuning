from __future__ import annotations

from pathlib import Path

from function_calling_ft.full_sft import (
    EXPECTED_GLOBAL_BATCH_SIZE,
    EXPECTED_GRADIENT_CLIP_NORM,
    EXPECTED_LOCAL_BATCH_SIZE,
    EXPECTED_TRAIN_10K_PATH,
    clone_full_sft_config_for_stage,
    load_yaml_config,
    validate_full_sft_config,
    warmup_steps_for_stage,
    write_yaml_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp05a_full_sft/full_sft_pilot.yaml"
EXP05B_CONFIG_PATH = ROOT / "configs/exp05b_full_sft/full_sft_10k.yaml"


def test_full_sft_config_is_not_peft_or_quantized() -> None:
    config = load_yaml_config(CONFIG_PATH)
    validation = validate_full_sft_config(CONFIG_PATH)

    assert validation.ok, validation.errors
    assert "peft" not in config
    assert "quantization" not in config
    assert validation.train_path.endswith("/train_2k.jsonl")
    assert validation.validation_path.endswith("/validation.jsonl")
    assert validation.sequence_length == 2048
    assert validation.local_batch_size == EXPECTED_LOCAL_BATCH_SIZE
    assert validation.global_batch_size == EXPECTED_GLOBAL_BATCH_SIZE
    assert validation.gradient_clip_norm == EXPECTED_GRADIENT_CLIP_NORM
    assert validation.activation_checkpointing is False
    assert "gradient_clip_val" not in config["step_scheduler"]
    assert config["distributed"]["activation_checkpointing"] is False


def test_exp05b_full_sft_uses_10k_without_peft_or_quantization() -> None:
    config = load_yaml_config(EXP05B_CONFIG_PATH)
    validation = validate_full_sft_config(EXP05B_CONFIG_PATH, profile="exp05b")

    assert validation.ok, validation.errors
    assert validation.profile == "exp05b"
    assert "peft" not in config
    assert "quantization" not in config
    assert validation.train_path == EXPECTED_TRAIN_10K_PATH
    assert validation.validation_path.endswith("/validation.jsonl")
    assert validation.sequence_length == 2048
    assert validation.max_steps == 2501
    assert validation.warmup_steps == 76
    assert validation.local_batch_size == EXPECTED_LOCAL_BATCH_SIZE
    assert validation.global_batch_size == EXPECTED_GLOBAL_BATCH_SIZE
    assert validation.gradient_clip_norm == EXPECTED_GRADIENT_CLIP_NORM
    assert validation.activation_checkpointing is False
    assert validation.checkpoint_dir.as_posix().startswith("/workspace/checkpoints/exp-05b/")
    assert config["task10_policy"]["no_peft"] is True
    assert config["task10_policy"]["no_quantization"] is True


def test_full_sft_rejects_peft_block(tmp_path: Path) -> None:
    config = load_yaml_config(CONFIG_PATH)
    config["peft"] = {"_target_": "nemo_automodel.components._peft.lora.PeftConfig"}
    path = tmp_path / "bad_peft.yaml"
    write_yaml_config(path, config)

    validation = validate_full_sft_config(path)

    assert not validation.ok
    assert any("peft" in error for error in validation.errors)


def test_full_sft_rejects_quantization_block(tmp_path: Path) -> None:
    config = load_yaml_config(CONFIG_PATH)
    config["quantization"] = {"load_in_4bit": True}
    path = tmp_path / "bad_quantization.yaml"
    write_yaml_config(path, config)

    validation = validate_full_sft_config(path)

    assert not validation.ok
    assert any("quantization" in error for error in validation.errors)


def test_full_sft_stage_clone_preserves_full_sft_contract(tmp_path: Path) -> None:
    config = load_yaml_config(CONFIG_PATH)
    staged = clone_full_sft_config_for_stage(
        config,
        checkpoint_dir="/workspace/checkpoints/exp-05a/test/pilot",
        max_steps=101,
        ckpt_every_steps=101,
        val_every_steps=50,
        validation_path="/workspace/data/processed/xlam_splits_v1/validation.jsonl",
        checkpoint_enabled=True,
        activation_checkpointing_enabled=True,
    )
    path = tmp_path / "staged.yaml"
    write_yaml_config(path, staged)

    validation = validate_full_sft_config(path)

    assert validation.ok, validation.errors
    assert validation.max_steps == 101
    assert validation.warmup_steps == 4
    assert validation.activation_checkpointing is True
    assert "gradient_clip_val" not in staged["step_scheduler"]
    assert staged["distributed"]["activation_checkpointing"] is True
    assert "activation_checkpointing" not in staged
    assert "peft" not in staged
    assert "quantization" not in staged


def test_exp05b_stage_clone_preserves_10k_profile(tmp_path: Path) -> None:
    config = load_yaml_config(EXP05B_CONFIG_PATH)
    staged = clone_full_sft_config_for_stage(
        config,
        checkpoint_dir="/workspace/checkpoints/exp-05b/test/full-epoch",
        max_steps=2501,
        ckpt_every_steps=834,
        val_every_steps=834,
        profile="exp05b",
        checkpoint_enabled=True,
        activation_checkpointing_enabled=False,
        policy_updates={"checkpoint_interval_steps": 834},
    )
    path = tmp_path / "staged_exp05b.yaml"
    write_yaml_config(path, staged)

    validation = validate_full_sft_config(path, profile="exp05b")

    assert validation.ok, validation.errors
    assert validation.train_path == EXPECTED_TRAIN_10K_PATH
    assert validation.max_steps == 2501
    assert validation.warmup_steps == 76
    assert validation.activation_checkpointing is False
    assert staged["task10_policy"]["gradient_clip_norm"] == EXPECTED_GRADIENT_CLIP_NORM
    assert staged["task10_policy"]["checkpoint_interval_steps"] == 834
    assert "task09_policy" not in staged
    assert "peft" not in staged
    assert "quantization" not in staged


def test_full_sft_warmup_uses_three_percent_ceiling() -> None:
    assert warmup_steps_for_stage(100) == 3
    assert warmup_steps_for_stage(101) == 4
    assert warmup_steps_for_stage(5) == 1
    assert warmup_steps_for_stage(1) == 0
