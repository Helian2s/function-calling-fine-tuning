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

from function_calling_ft.leakage import run_leakage_audit


DEFAULT_GROUP_METADATA = Path(
    "data/processed/xlam_curated_v1/group_metadata.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/processed/xlam_curated_v1/manifests/"
    "leakage_audit_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when a leakage-control split group appears in multiple "
            "non-ignored splits."
        )
    )
    parser.add_argument(
        "--group-metadata",
        type=Path,
        default=DEFAULT_GROUP_METADATA,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Write the report but do not fail on detected overlap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_leakage_audit(
        group_metadata_path=args.group_metadata,
        output_path=args.output,
        fail_on_overlap=not args.allow_overlap,
    )
    print(f"Leakage audit status: {report['status']}")
    print(f"Cross-split groups:   {report['cross_split_group_count']}")
    print(f"Report:               {args.output}")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
