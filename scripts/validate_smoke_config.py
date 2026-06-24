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

from function_calling_ft.smoke_config import (
    validate_smoke_config,
    validation_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Experiment 0 smoke training config invariants.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write machine-readable validation details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_smoke_config(args.config)

    if args.json:
        print(json.dumps(validation_to_dict(validation), sort_keys=True))
    else:
        print(f"config: {validation.path}")
        print(f"checkpoint_dir: {validation.checkpoint_dir}")
        print(f"model_name: {validation.model_name}")
        print(f"model_revision: {validation.model_revision}")
        print(f"max_steps: {validation.max_steps}")

    if not validation.ok:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
