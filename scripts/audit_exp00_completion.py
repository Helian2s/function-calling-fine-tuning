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

from function_calling_ft.exp00_completion import (
    build_completion_report,
    write_completion_markdown,
)


def _load_optional_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and freeze Experiment 0 completion evidence.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/smoke/normalized/test.jsonl"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/workspace/results/exp-00"),
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=Path("/workspace/results/exp-00/baseline"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/workspace/logs/exp-00"),
    )
    parser.add_argument(
        "--run-info-dir",
        type=Path,
        default=Path("/workspace/run-info"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/workspace/checkpoints/exp-00/smoke-lora"),
    )
    parser.add_argument(
        "--template-report",
        type=Path,
        default=Path("data/manifests/smoke_v1_template_report.json"),
    )
    parser.add_argument(
        "--loss-mask-report",
        type=Path,
        default=Path("data/manifests/smoke_v1_loss_mask_report.json"),
    )
    parser.add_argument("--s3-inventory-json", type=Path)
    parser.add_argument("--instance-state")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/exp00_completion/exp00_completion.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/exp00_completion/exp00_completion.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_completion_report(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        baseline_results_dir=args.baseline_results_dir,
        logs_dir=args.logs_dir,
        run_info_dir=args.run_info_dir,
        adapter_dir=args.adapter_dir,
        template_report_path=args.template_report,
        loss_mask_report_path=args.loss_mask_report,
        s3_inventory=_load_optional_json(args.s3_inventory_json),
        instance_state=args.instance_state,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_completion_markdown(report, args.output_md)
    print(f"exp00_completion_json={args.output_json}")
    print(f"exp00_completion_md={args.output_md}")
    print(f"overall_status={report['overall_status']}")
    print(f"may_proceed={report['may_proceed_to_later_experiments']}")


if __name__ == "__main__":
    main()
