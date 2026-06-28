#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.reference_lora import (  # noqa: E402
    EXPECTED_LORA_RANK,
    FORBIDDEN_NON_ADAPTER_SUFFIXES,
    FORBIDDEN_TARGET_SUFFIXES,
    load_yaml_config,
    summarize_lora_target_matches,
    validate_lora_rank_config,
    validate_lora_sample_efficiency_config,
    validate_lora_target_config,
    validate_loss_mask_ablation_config,
    validate_reference_lora_config,
    validate_reference_qlora_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect actual model modules matched by the Exp 03 LoRA config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp03_reference_lora/lora_r8_attention.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--method",
        choices=("lora", "qlora"),
        default="lora",
        help="Validation profile to apply before inspecting target modules.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="Validate an Exp 06 BF16 LoRA rank profile instead of Exp 03 rank 8.",
    )
    parser.add_argument(
        "--target-profile",
        choices=("attention", "attention_mlp"),
        help="Validate an Exp 07 BF16 LoRA target-module profile.",
    )
    parser.add_argument(
        "--sample-profile",
        choices=("train_2k", "train_10k", "train_full"),
        help="Validate an Exp 08 BF16 LoRA sample-efficiency profile.",
    )
    parser.add_argument(
        "--loss-mask-profile",
        choices=("assistant_only_short", "full_sequence_short"),
        help="Validate an Exp 09A loss-mask ablation profile.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face files.",
    )
    return parser.parse_args()


def _load_model_empty(config_path: Path, cache_dir: Path | None, local_files_only: bool) -> Any:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = load_yaml_config(config_path)
    model_cfg = config["model"]
    model_name = str(model_cfg["pretrained_model_name_or_path"])
    revision = str(model_cfg["revision"])
    hf_config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    try:
        from accelerate import init_empty_weights
    except ImportError:
        return AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=torch.bfloat16,
        )

    with init_empty_weights():
        return AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=torch.bfloat16,
        )


def _linear_module_details(model: Any) -> list[dict[str, Any]]:
    import torch

    details: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        details.append(
            {
                "name": name,
                "class_name": type(module).__name__,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "bias": module.bias is not None,
            },
        )
    return details


def _estimated_lora_params(
    *,
    matched_modules: list[str],
    details: list[dict[str, Any]],
    rank: int = EXPECTED_LORA_RANK,
) -> int:
    by_name = {str(item["name"]): item for item in details}
    total = 0
    for name in matched_modules:
        item = by_name[name]
        total += rank * (int(item["in_features"]) + int(item["out_features"]))
    return total


def main() -> None:
    args = parse_args()
    if args.sample_profile is not None:
        validation = validate_lora_sample_efficiency_config(
            args.config,
            sample_profile=args.sample_profile,
        )
    elif args.target_profile is not None:
        validation = validate_lora_target_config(
            args.config,
            target_profile=args.target_profile,
        )
    elif args.loss_mask_profile is not None:
        validation = validate_loss_mask_ablation_config(
            args.config,
            loss_mask_profile=args.loss_mask_profile,
        )
    elif args.rank is not None:
        validation = validate_lora_rank_config(args.config, rank=args.rank)
    else:
        validator = (
            validate_reference_qlora_config
            if args.method == "qlora"
            else validate_reference_lora_config
        )
        validation = validator(args.config)
    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    model = _load_model_empty(args.config, args.cache_dir, args.local_files_only)
    details = _linear_module_details(model)
    module_names = [str(item["name"]) for item in details]
    forbidden_suffixes = (
        FORBIDDEN_NON_ADAPTER_SUFFIXES
        if args.target_profile == "attention_mlp"
        else FORBIDDEN_TARGET_SUFFIXES
    )
    summary = summarize_lora_target_matches(
        module_names,
        target_patterns=validation.target_modules,
        forbidden_suffixes=forbidden_suffixes,
    )
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    estimated_trainable = _estimated_lora_params(
        matched_modules=list(summary["matched_modules"]),
        details=details,
        rank=validation.lora_rank,
    )

    payload = {
        "schema_version": "1.0",
        "config": str(args.config),
        "method": validation.method,
        "quantization": dict(validation.quantization or {}),
        "model_name": validation.model_name,
        "model_revision": validation.model_revision,
        "target_match_summary": summary,
        "linear_modules": details,
        "total_parameter_count": total_params,
        "estimated_lora_trainable_parameter_count": estimated_trainable,
        "estimated_frozen_parameter_count": total_params,
        "estimated_trainable_parameter_ratio": (
            estimated_trainable / total_params if total_params else None
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"lora_target_inspection={args.output}")
    print(f"matched_modules={summary['matched_count']}")
    print(f"estimated_lora_trainable_parameters={estimated_trainable}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
