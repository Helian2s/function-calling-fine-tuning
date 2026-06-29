from __future__ import annotations

from pathlib import Path

from function_calling_ft.reference_lora import (
    EXPECTED_ATTENTION_MLP_TARGET_MODULES,
    EXPECTED_TARGET_MODULES,
    EXPECTED_NF4_QUANTIZATION,
    EXP06_RANK_PROFILES,
    EXP07_TARGET_PROFILES,
    EXP08_SAMPLE_PROFILES,
    EXP09_LOSS_MASK_PROFILES,
    clone_training_config_for_stage,
    load_yaml_config,
    summarize_lora_target_matches,
    validate_loss_mask_ablation_config,
    validate_lora_rank_config,
    validate_lora_sample_efficiency_config,
    validate_lora_target_config,
    validate_reference_lora_config,
    validate_reference_qlora_config,
    warmup_steps_for_stage,
)
from scripts.run_exp06_lora_rank import _select_rank
from scripts.run_exp07_target_modules import _select_target_profile
from scripts.run_exp08_sample_efficiency import (
    _select_dataset_size,
    verify_nested_split_paths,
)
from scripts.run_exp03_reference_lora import _training_command
from scripts.run_training_with_monitor import child_environment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/exp03_reference_lora/lora_r8_attention.yaml"
QLORA_CONFIG_PATH = ROOT / "configs/exp04_lora_vs_qlora/qlora.yaml"
EXP06_CONFIG_ROOT = ROOT / "configs/exp06_lora_rank"
EXP07_CONFIG_ROOT = ROOT / "configs/exp07_target_modules"
EXP08_CONFIG_ROOT = ROOT / "configs/exp08_sample_efficiency"
EXP09_CONFIG_ROOT = ROOT / "configs/exp09_loss_masking"
SPLITS_ROOT = ROOT / "data/processed/xlam_splits_v1"


def test_reference_lora_config_is_rank8_attention_only() -> None:
    validation = validate_reference_lora_config(CONFIG_PATH)

    assert validation.ok, validation.errors
    assert validation.sequence_length == 2048
    assert validation.target_modules == EXPECTED_TARGET_MODULES
    assert validation.train_path.endswith("/train_10k.jsonl")
    assert validation.validation_path.endswith("/validation.jsonl")
    assert validation.method == "bf16_lora"
    assert validation.quantization is None
    assert validation.lora_rank == 8
    assert validation.lora_alpha == 16


def test_reference_qlora_config_is_nf4_matched_rank8_attention_only() -> None:
    validation = validate_reference_qlora_config(QLORA_CONFIG_PATH)

    assert validation.ok, validation.errors
    assert validation.method == "nf4_qlora"
    assert validation.sequence_length == 2048
    assert validation.target_modules == EXPECTED_TARGET_MODULES
    assert validation.train_path.endswith("/train_10k.jsonl")
    assert validation.validation_path.endswith("/validation.jsonl")
    assert validation.quantization == EXPECTED_NF4_QUANTIZATION
    assert validation.lora_rank == 8
    assert validation.lora_alpha == 16


def test_exp06_lora_rank_configs_are_valid() -> None:
    for profile in EXP06_RANK_PROFILES:
        config_path = (
            EXP06_CONFIG_ROOT
            / f"rank{profile.rank}_alpha{profile.alpha}.yaml"
        )
        validation = validate_lora_rank_config(config_path, rank=profile.rank)

        assert validation.ok, validation.errors
        assert validation.method == f"bf16_lora_{profile.name}"
        assert validation.lora_rank == profile.rank
        assert validation.lora_alpha == profile.alpha
        assert validation.target_modules == EXPECTED_TARGET_MODULES
        assert validation.train_path.endswith("/train_10k.jsonl")
        assert validation.validation_path.endswith("/validation.jsonl")


def test_exp07_target_module_configs_are_valid() -> None:
    for profile in EXP07_TARGET_PROFILES:
        config_path = EXP07_CONFIG_ROOT / f"{profile.name}.yaml"
        validation = validate_lora_target_config(
            config_path,
            target_profile=profile.name,
        )

        assert validation.ok, validation.errors
        assert validation.method == f"bf16_lora_r4_alpha8_{profile.name}"
        assert validation.lora_rank == 4
        assert validation.lora_alpha == 8
        assert validation.target_modules == profile.target_modules
        assert validation.train_path.endswith("/train_10k.jsonl")
        assert validation.validation_path.endswith("/validation.jsonl")


def test_exp08_sample_efficiency_configs_are_valid() -> None:
    for profile in EXP08_SAMPLE_PROFILES:
        config_path = EXP08_CONFIG_ROOT / f"{profile.name}.yaml"
        validation = validate_lora_sample_efficiency_config(
            config_path,
            sample_profile=profile.name,
        )

        assert validation.ok, validation.errors
        assert validation.method == f"bf16_lora_r4_alpha8_{profile.name}"
        assert validation.lora_rank == 4
        assert validation.lora_alpha == 8
        assert validation.target_modules == EXPECTED_TARGET_MODULES
        assert validation.train_path == profile.train_path
        assert validation.validation_path.endswith("/validation.jsonl")


def test_exp09_loss_mask_ablation_configs_are_valid() -> None:
    for profile in EXP09_LOSS_MASK_PROFILES:
        config_path = EXP09_CONFIG_ROOT / f"{profile.name}.yaml"
        validation = validate_loss_mask_ablation_config(
            config_path,
            loss_mask_profile=profile.name,
        )
        config = load_yaml_config(config_path)

        assert validation.ok, validation.errors
        assert validation.method == f"bf16_lora_r8_alpha16_{profile.name}"
        assert validation.lora_rank == 8
        assert validation.lora_alpha == 16
        assert validation.target_modules == EXPECTED_TARGET_MODULES
        assert validation.train_path.endswith("/train_10k.jsonl")
        assert validation.validation_path.endswith("/validation.jsonl")
        expected_policy = (
            "assistant_only" if profile.answer_only_loss_mask else "full_sequence"
        )
        assert config["dataset"]["loss_mask_policy"] == expected_policy
        assert (
            config["validation_dataset"]["loss_mask_policy"]
            == expected_policy
        )


def test_exp06_rank_configs_only_change_capacity_and_identity() -> None:
    def normalized(path: Path) -> dict[str, object]:
        config = load_yaml_config(path)
        for key in ["experiment_id", "run_id", "title"]:
            config.pop(key, None)
        config["checkpoint"] = dict(config["checkpoint"])
        config["checkpoint"].pop("checkpoint_dir", None)
        config["peft"] = dict(config["peft"])
        config["peft"].pop("dim", None)
        config["peft"].pop("alpha", None)
        return config

    baseline = normalized(EXP06_CONFIG_ROOT / "rank8_alpha16.yaml")

    assert normalized(EXP06_CONFIG_ROOT / "rank4_alpha8.yaml") == baseline
    assert normalized(EXP06_CONFIG_ROOT / "rank16_alpha32.yaml") == baseline


def test_exp07_configs_only_change_target_placement_and_identity() -> None:
    def normalized(path: Path) -> dict[str, object]:
        config = load_yaml_config(path)
        for key in ["run_id", "title"]:
            config.pop(key, None)
        config["checkpoint"] = dict(config["checkpoint"])
        config["checkpoint"].pop("checkpoint_dir", None)
        config["peft"] = dict(config["peft"])
        config["peft"].pop("target_modules", None)
        config.pop("task12_policy", None)
        return config

    baseline = normalized(EXP07_CONFIG_ROOT / "attention.yaml")

    assert normalized(EXP07_CONFIG_ROOT / "attention_mlp.yaml") == baseline


def test_exp08_configs_only_change_dataset_size_and_identity() -> None:
    def normalized(path: Path) -> dict[str, object]:
        config = load_yaml_config(path)
        for key in ["run_id", "title"]:
            config.pop(key, None)
        config["checkpoint"] = dict(config["checkpoint"])
        config["checkpoint"].pop("checkpoint_dir", None)
        config["dataset"] = dict(config["dataset"])
        config["dataset"].pop("path_or_dataset_id", None)
        config["step_scheduler"] = dict(config["step_scheduler"])
        for key in ["max_steps", "ckpt_every_steps", "val_every_steps"]:
            config["step_scheduler"].pop(key, None)
        config["lr_scheduler"] = dict(config["lr_scheduler"])
        config["lr_scheduler"].pop("lr_warmup_steps", None)
        config.pop("task13_policy", None)
        return config

    baseline = normalized(EXP08_CONFIG_ROOT / "train_10k.yaml")

    assert normalized(EXP08_CONFIG_ROOT / "train_2k.yaml") == baseline
    assert normalized(EXP08_CONFIG_ROOT / "train_full.yaml") == baseline


def test_exp09_configs_only_change_loss_mask_policy_and_identity() -> None:
    def normalized(path: Path) -> dict[str, object]:
        config = load_yaml_config(path)
        for key in ["run_id", "title"]:
            config.pop(key, None)
        config["checkpoint"] = dict(config["checkpoint"])
        config["checkpoint"].pop("checkpoint_dir", None)
        config["dataset"] = dict(config["dataset"])
        config["dataset"].pop("loss_mask_policy", None)
        config["validation_dataset"] = dict(config["validation_dataset"])
        config["validation_dataset"].pop("loss_mask_policy", None)
        config.pop("task14_policy", None)
        return config

    baseline = normalized(EXP09_CONFIG_ROOT / "assistant_only_short.yaml")

    assert normalized(EXP09_CONFIG_ROOT / "full_sequence_short.yaml") == baseline


def test_exp08_frozen_subsets_are_nested() -> None:
    report = verify_nested_split_paths(
        train_2k=SPLITS_ROOT / "train_2k.jsonl",
        train_10k=SPLITS_ROOT / "train_10k.jsonl",
        train_full=SPLITS_ROOT / "train_full.jsonl",
    )

    assert report["ok"], report
    assert report["record_counts"] == {
        "train_2k": 2019,
        "train_10k": 10003,
        "train_full": 39526,
    }
    assert report["duplicate_counts"] == {
        "train_2k": 0,
        "train_10k": 0,
        "train_full": 0,
    }


def test_exp06_selection_chooses_smallest_rank_within_tie_threshold() -> None:
    rows = [
        {
            "dataset": "validation",
            "rank": 4,
            "alpha": 8,
            "executable_complete_match_rate": 0.731,
            "complete_call_f1": 0.804,
        },
        {
            "dataset": "validation",
            "rank": 8,
            "alpha": 16,
            "executable_complete_match_rate": 0.735,
            "complete_call_f1": 0.810,
        },
        {
            "dataset": "validation",
            "rank": 16,
            "alpha": 32,
            "executable_complete_match_rate": 0.736,
            "complete_call_f1": 0.811,
        },
    ]

    decision = _select_rank(rows)

    assert decision["status"] == "selected"
    assert decision["selected_rank"] == 4


def test_exp07_selection_keeps_attention_without_meaningful_broader_gain() -> None:
    rows = [
        {
            "dataset": "validation",
            "target_profile": "attention",
            "executable_complete_match_rate": 0.788,
            "complete_call_f1": 0.834,
        },
        {
            "dataset": "validation",
            "target_profile": "attention_mlp",
            "executable_complete_match_rate": 0.793,
            "complete_call_f1": 0.837,
        },
        {
            "dataset": "no_tool_dev",
            "target_profile": "attention",
            "no_tool_false_positive_rate": 0.68,
        },
        {
            "dataset": "no_tool_dev",
            "target_profile": "attention_mlp",
            "no_tool_false_positive_rate": 0.69,
        },
    ]

    decision = _select_target_profile(rows)

    assert decision["status"] == "selected"
    assert decision["selected_target_profile"] == "attention"


def test_exp07_selection_accepts_meaningful_broader_gain() -> None:
    rows = [
        {
            "dataset": "validation",
            "target_profile": "attention",
            "executable_complete_match_rate": 0.788,
            "complete_call_f1": 0.834,
        },
        {
            "dataset": "validation",
            "target_profile": "attention_mlp",
            "executable_complete_match_rate": 0.802,
            "complete_call_f1": 0.842,
        },
        {
            "dataset": "no_tool_dev",
            "target_profile": "attention",
            "no_tool_false_positive_rate": 0.68,
        },
        {
            "dataset": "no_tool_dev",
            "target_profile": "attention_mlp",
            "no_tool_false_positive_rate": 0.70,
        },
    ]

    decision = _select_target_profile(rows)

    assert decision["status"] == "selected"
    assert decision["selected_target_profile"] == "attention_mlp"


def test_exp08_selection_chooses_smallest_dataset_within_thresholds() -> None:
    rows = [
        {
            "dataset": "validation",
            "sample_profile": "train_2k",
            "train_records": 2019,
            "supervised_target_tokens": 100,
            "executable_complete_match_rate": 0.782,
            "complete_call_f1": 0.830,
        },
        {
            "dataset": "validation",
            "sample_profile": "train_10k",
            "train_records": 10003,
            "supervised_target_tokens": 500,
            "executable_complete_match_rate": 0.789,
            "complete_call_f1": 0.834,
        },
        {
            "dataset": "validation",
            "sample_profile": "train_full",
            "train_records": 39526,
            "supervised_target_tokens": 2000,
            "executable_complete_match_rate": 0.791,
            "complete_call_f1": 0.835,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_2k",
            "no_tool_false_positive_rate": 0.70,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_10k",
            "no_tool_false_positive_rate": 0.68,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_full",
            "no_tool_false_positive_rate": 0.69,
        },
    ]

    decision = _select_dataset_size(
        rows=rows,
        hard_bucket_guardrails={
            "ok_by_profile": {
                "train_2k": True,
                "train_10k": True,
                "train_full": True,
            },
        },
    )

    assert decision["status"] == "selected"
    assert decision["selected_sample_profile"] == "train_2k"


def test_exp08_selection_respects_no_tool_guardrail() -> None:
    rows = [
        {
            "dataset": "validation",
            "sample_profile": "train_2k",
            "train_records": 2019,
            "executable_complete_match_rate": 0.790,
            "complete_call_f1": 0.834,
        },
        {
            "dataset": "validation",
            "sample_profile": "train_10k",
            "train_records": 10003,
            "executable_complete_match_rate": 0.789,
            "complete_call_f1": 0.834,
        },
        {
            "dataset": "validation",
            "sample_profile": "train_full",
            "train_records": 39526,
            "executable_complete_match_rate": 0.791,
            "complete_call_f1": 0.835,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_2k",
            "no_tool_false_positive_rate": 0.90,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_10k",
            "no_tool_false_positive_rate": 0.68,
        },
        {
            "dataset": "no_tool_dev",
            "sample_profile": "train_full",
            "no_tool_false_positive_rate": 0.69,
        },
    ]

    decision = _select_dataset_size(
        rows=rows,
        hard_bucket_guardrails={
            "ok_by_profile": {
                "train_2k": True,
                "train_10k": True,
                "train_full": True,
            },
        },
    )

    assert decision["status"] == "selected"
    assert decision["selected_sample_profile"] == "train_10k"


def test_qlora_matches_reference_lora_except_quantization_and_identity() -> None:
    lora = load_yaml_config(CONFIG_PATH)
    qlora = load_yaml_config(QLORA_CONFIG_PATH)

    for key in [
        "dist_env",
        "rng",
        "compile",
        "peft",
        "distributed",
        "loss_fn",
        "dataset",
        "packed_sequence",
        "dataloader",
        "validation_dataset",
        "validation_dataloader",
        "optimizer",
        "lr_scheduler",
    ]:
        assert qlora[key] == lora[key]

    qlora_model = dict(qlora["model"])
    assert qlora_model.pop("force_hf") is True
    assert qlora_model == lora["model"]
    assert qlora["quantization"] == EXPECTED_NF4_QUANTIZATION
    assert qlora["experiment_id"] == "exp-04"
    assert qlora["run_id"] != lora["run_id"]


def test_reference_lora_stage_overrides_batch_steps_and_warmup(
    tmp_path: Path,
) -> None:
    config = load_yaml_config(CONFIG_PATH)
    staged = clone_training_config_for_stage(
        config,
        checkpoint_dir="/workspace/checkpoints/exp-03/test/pilot",
        global_batch_size=8,
        local_batch_size=4,
        max_steps=101,
        ckpt_every_steps=101,
        val_every_steps=50,
    )

    scheduler = staged["step_scheduler"]
    assert scheduler["global_batch_size"] == 8
    assert scheduler["local_batch_size"] == 4
    assert scheduler["max_steps"] == 101
    assert staged["lr_scheduler"]["lr_warmup_steps"] == 4

    staged_path = tmp_path / "staged.yaml"
    from function_calling_ft.reference_lora import write_yaml_config

    write_yaml_config(staged_path, staged)
    validation = validate_reference_lora_config(staged_path)
    assert validation.ok, validation.errors


def test_reference_qlora_stage_overrides_batch_steps_and_warmup(
    tmp_path: Path,
) -> None:
    config = load_yaml_config(QLORA_CONFIG_PATH)
    staged = clone_training_config_for_stage(
        config,
        checkpoint_dir="/workspace/checkpoints/exp-04/test/pilot",
        global_batch_size=8,
        local_batch_size=4,
        max_steps=101,
        ckpt_every_steps=101,
        val_every_steps=50,
    )

    scheduler = staged["step_scheduler"]
    assert scheduler["global_batch_size"] == 8
    assert scheduler["local_batch_size"] == 4
    assert scheduler["max_steps"] == 101
    assert staged["lr_scheduler"]["lr_warmup_steps"] == 4
    assert staged["quantization"] == EXPECTED_NF4_QUANTIZATION

    staged_path = tmp_path / "staged_qlora.yaml"
    from function_calling_ft.reference_lora import write_yaml_config

    write_yaml_config(staged_path, staged)
    validation = validate_reference_qlora_config(staged_path)
    assert validation.ok, validation.errors


def test_warmup_steps_uses_three_percent_ceiling() -> None:
    assert warmup_steps_for_stage(100) == 3
    assert warmup_steps_for_stage(101) == 4
    assert warmup_steps_for_stage(1) == 1


def test_target_matching_accepts_attention_and_rejects_mlp() -> None:
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ]

    summary = summarize_lora_target_matches(names)

    assert summary["ok"] is True
    assert summary["matched_count"] == 4
    assert summary["forbidden_matches"] == []
    assert summary["counts_by_suffix"] == {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
    }


def test_target_matching_accepts_attention_plus_mlp() -> None:
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ]

    summary = summarize_lora_target_matches(
        names,
        target_patterns=EXPECTED_ATTENTION_MLP_TARGET_MODULES,
        forbidden_suffixes=("lm_head", "embed_tokens"),
    )

    assert summary["ok"] is True
    assert summary["matched_count"] == 7
    assert summary["forbidden_matches"] == []
    assert summary["counts_by_suffix"] == {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 1,
    }


def test_memory_trace_env_is_injected(tmp_path: Path) -> None:
    memory_output = tmp_path / "memory.json"
    trace_output = tmp_path / "trace.json"

    env = child_environment(memory_output, trace_output)

    assert env["FCFT_TORCH_MEMORY_OUTPUT"] == str(memory_output)
    assert env["FCFT_TORCH_MEMORY_TRACE_OUTPUT"] == str(trace_output)


def test_training_command_can_stop_after_observed_step(tmp_path: Path) -> None:
    command = _training_command(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint",
        log_path=tmp_path / "train.log",
        metrics_path=tmp_path / "metrics.json",
        gpu_log_path=tmp_path / "gpu.csv",
        torch_memory_path=tmp_path / "memory.json",
        torch_trace_path=None,
        qlora_patch_report_path=None,
        automodel_bin="automodel",
        stop_after_step=299,
    )

    assert "--stop-after-step" in command
    assert command[command.index("--stop-after-step") + 1] == "299"


def test_qlora_peft_state_dict_patch_env_is_explicit(tmp_path: Path) -> None:
    report_output = tmp_path / "qlora_patch.json"

    env = child_environment(None, None, report_output)

    assert env["FCFT_PATCH_QLORA_PEFT_STATE_DICT"] == "1"
    assert env["FCFT_QLORA_PEFT_STATE_DICT_PATCH_REPORT"] == str(report_output)
