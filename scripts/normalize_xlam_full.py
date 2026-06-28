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

from function_calling_ft.full_normalization import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    normalize_full_dataset,
)


DEFAULT_RAW_PATH = Path("data/raw/xlam/xlam_function_calling_60k.json")
DEFAULT_SOURCE_MANIFEST_PATH = Path("data/manifests/xlam_source.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/xlam_full_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the full Salesforce/xLAM function-calling dataset "
            "with deterministic canonical serialization and quarantine."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Normalize only the first N source records for local checks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
    )
    parser.add_argument(
        "--context-token-limit",
        type=int,
        default=DEFAULT_CONTEXT_TOKEN_LIMIT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw_path.is_file():
        raise FileNotFoundError(
            f"Raw xLAM file not found: {args.raw_path}. "
            "Run scripts/download_xlam.py through the approved workflow."
        )

    report = normalize_full_dataset(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        source_manifest_path=args.source_manifest,
        repo_root=ROOT,
        limit=args.limit,
        chunk_size=args.chunk_size,
        context_token_limit=args.context_token_limit,
    )

    processing = report["processing"]
    outputs = report["outputs"]

    print("Full xLAM normalization completed.")
    print(f"Input records:       {processing['input_records']}")
    print(f"Accepted records:    {processing['accepted_records']}")
    print(f"Quarantined records: {processing['quarantined_records']}")
    print(f"Reconciled:          {processing['reconciled']}")
    print(f"Normalized:          {outputs['normalized']['path']}")
    print(f"Quarantine:          {outputs['quarantine']['path']}")
    print(f"Report:              {outputs['report_path']}")
    print(f"Checksums:           {outputs['checksums_path']}")

    if not processing["reconciled"]:
        raise SystemExit(
            "Input count does not equal accepted + quarantined records."
        )

    print(
        json.dumps(
            {
                "accepted": processing["accepted_records"],
                "quarantined": processing["quarantined_records"],
                "normalized_sha256": outputs["normalized"]["sha256"],
                "quarantine_sha256": outputs["quarantine"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
