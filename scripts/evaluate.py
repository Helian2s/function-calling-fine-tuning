#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation import evaluate_predictions
from function_calling_ft.split_guard import assert_split_allowed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and score function-call prediction JSONL.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--order-matters",
        action="store_true",
        help="Require predicted calls to match expected order.",
    )
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
    assert_split_allowed(
        args.dataset,
        final_evaluation=args.final_evaluation,
        final_config=args.final_config,
        command_name="evaluation",
    )
    outputs = evaluate_predictions(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        order_matters=args.order_matters,
    )
    print(f"scored_predictions={outputs.scored_predictions_path}")
    print(f"parse_failures={outputs.parse_failures_path}")
    print(f"scores={outputs.scores_path}")


if __name__ == "__main__":
    main()
