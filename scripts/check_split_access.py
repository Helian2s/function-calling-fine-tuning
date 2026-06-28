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

from function_calling_ft.split_guard import (
    SplitAccessError,
    assert_split_allowed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a dataset split may be used for screening.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Permit explicitly locked final-evaluation splits.",
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        help=(
            "Frozen final-evaluation config required with "
            "--final-evaluation on locked final splits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        decision = assert_split_allowed(
            args.dataset,
            final_evaluation=args.final_evaluation,
            final_config=args.final_config,
            command_name="split access check",
        )
    except SplitAccessError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "allowed": decision.allowed,
                "requires_final_evaluation": (
                    decision.requires_final_evaluation
                ),
                "split_lock_status": decision.split_lock_status,
                "split_name": decision.split_name,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
