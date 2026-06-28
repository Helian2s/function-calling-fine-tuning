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

from function_calling_ft.evaluation_compare import (
    DEFAULT_METRICS,
    write_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two scored evaluation JSONL files.",
    )
    parser.add_argument("--baseline-scored", required=True, type=Path)
    parser.add_argument("--candidate-scored", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help=(
            "Metric to compare. May be repeated. Defaults to strict, "
            "schema-equivalent, and executable complete record matches."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_comparison(
        baseline_scored_path=args.baseline_scored,
        candidate_scored_path=args.candidate_scored,
        output_dir=args.output_dir,
        metrics=tuple(args.metrics) if args.metrics else DEFAULT_METRICS,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        confidence=args.confidence,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
