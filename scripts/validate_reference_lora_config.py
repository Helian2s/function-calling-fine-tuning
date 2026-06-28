#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.reference_lora import (  # noqa: E402
    validate_reference_lora_config,
    validate_reference_qlora_config,
    validation_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Experiment 3 reference LoRA config.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--method",
        choices=("lora", "qlora"),
        default="lora",
        help="Validation profile to apply.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validator = (
        validate_reference_qlora_config
        if args.method == "qlora"
        else validate_reference_lora_config
    )
    validation = validator(args.config)
    payload = validation_to_dict(validation)

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"config: {validation.path}")
        print(f"checkpoint_dir: {validation.checkpoint_dir}")
        print(f"model_name: {validation.model_name}")
        print(f"model_revision: {validation.model_revision}")
        print(f"train_path: {validation.train_path}")
        print(f"validation_path: {validation.validation_path}")
        print(f"sequence_length: {validation.sequence_length}")
        print(f"max_steps: {validation.max_steps}")
        print(f"global_batch_size: {validation.global_batch_size}")
        print(f"local_batch_size: {validation.local_batch_size}")
        print(f"warmup_steps: {validation.warmup_steps}")
        print(f"target_modules: {', '.join(validation.target_modules)}")
        print(f"method: {validation.method}")
        if validation.quantization:
            print(f"quantization: {json.dumps(validation.quantization, sort_keys=True)}")

    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
