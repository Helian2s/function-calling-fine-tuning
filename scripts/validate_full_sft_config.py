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

from function_calling_ft.full_sft import (  # noqa: E402
    FULL_SFT_PROFILES,
    validate_full_sft_config,
    validation_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Experiment 5A full-parameter SFT config.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(FULL_SFT_PROFILES),
        default="exp05a",
        help="Full-SFT experiment profile to validate against.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_full_sft_config(args.config, profile=args.profile)
    payload = validation_to_dict(validation)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"config: {validation.path}")
        print("method: full_parameter_sft")
        print(f"profile: {validation.profile}")
        print(f"checkpoint_dir: {validation.checkpoint_dir}")
        print(f"train_path: {validation.train_path}")
        print(f"validation_path: {validation.validation_path}")
        print(f"sequence_length: {validation.sequence_length}")
        print(f"global_batch_size: {validation.global_batch_size}")
        print(f"local_batch_size: {validation.local_batch_size}")
        print(f"gradient_clip_norm: {validation.gradient_clip_norm}")
        print(f"activation_checkpointing: {validation.activation_checkpointing}")
        print(f"ok: {validation.ok}")
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    if not validation.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
