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

from function_calling_ft.evaluation_report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write requested evaluation metrics and case report.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_report(args.output_dir)
    summary = {
        "total_cases": report["total_cases"],
        "passed": report["passed"],
        "failed": report["failed"],
        "pass_rate": report["pass_rate"],
        "failure_reason_counts": report["failure_reason_counts"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"requested_metrics={args.output_dir / 'requested_metrics.json'}")
    print(f"case_report_json={args.output_dir / 'case_report.json'}")
    print(f"case_report_md={args.output_dir / 'case_report.md'}")


if __name__ == "__main__":
    main()
