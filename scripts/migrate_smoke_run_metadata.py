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

from function_calling_ft.run_manifest import migrate_smoke_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy Experiment 0 metadata to run_manifest.json.",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=Path("/workspace/results/exp-00/run_metadata.json"),
    )
    parser.add_argument(
        "--generation-metadata",
        type=Path,
        default=Path("/workspace/results/exp-00/generation_metadata.json"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("/workspace/results/exp-00/scores.json"),
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path("/workspace/results/exp-00/resolved_config.yaml"),
    )
    parser.add_argument(
        "--package-versions",
        type=Path,
        default=Path("/workspace/results/exp-00/package_versions.txt"),
    )
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--training-log",
        type=Path,
        default=Path("/workspace/logs/exp-00/training.log"),
    )
    parser.add_argument("--container-image", type=Path)
    parser.add_argument("--container-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/results/exp-00/run_manifest.json"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--status", default="migrated")
    parser.add_argument("--task-id", default="C9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = migrate_smoke_metadata(
        run_metadata_path=args.run_metadata,
        generation_metadata_path=args.generation_metadata,
        scores_path=args.scores,
        resolved_config_path=args.resolved_config,
        package_versions_path=args.package_versions,
        dataset_manifest_path=args.dataset_manifest,
        training_log_path=args.training_log,
        container_image_path=args.container_image,
        container_report_path=args.container_report,
        output_path=args.output,
        run_id=args.run_id,
        status=args.status,
        task_id=args.task_id,
    )
    print(f"run_manifest={args.output}")
    print(json.dumps({"run_id": manifest["run_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
