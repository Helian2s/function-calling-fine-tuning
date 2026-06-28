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

from function_calling_ft.curation import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    compare_curation_hashes,
    curate_normalized_dataset,
    verify_stable_under_shuffle,
)


DEFAULT_CURATOR_IMAGE = "nvcr.io/nvidia/nemo-curator:25.09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach deterministic grouping metadata, exact duplicate maps, "
            "and fuzzy review candidates to the normalized xLAM dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.82,
    )
    parser.add_argument(
        "--fuzzy-review-sample-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--curator-image",
        default=DEFAULT_CURATOR_IMAGE,
    )
    parser.add_argument(
        "--verify-shuffle-stability",
        action="store_true",
        help=(
            "Run a deterministic shuffled-input rerun and compare stable "
            "curation hashes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Normalized dataset not found: {args.input}. "
            "Run make normalize-xlam-full first."
        )

    report = curate_normalized_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        repo_root=ROOT,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_review_sample_size=args.fuzzy_review_sample_size,
        curator_image=args.curator_image,
    )

    stability_report = None
    if args.verify_shuffle_stability:
        verify_stable_under_shuffle(
            input_path=args.input,
            output_dir=args.output_dir / "_stability",
            repo_root=ROOT,
            curator_image=args.curator_image,
        )
        stability_report = compare_curation_hashes(
            args.output_dir,
            args.output_dir / "_stability" / "shuffled_output",
        )
        if not stability_report["stable"]:
            raise SystemExit(
                "Curation outputs are not stable under shuffled input."
            )

    exact = report["exact_deduplication"]
    grouping = report["grouping"]
    fuzzy = report["fuzzy_candidates"]

    print("xLAM curation metadata completed.")
    print(f"Input records:        {exact['input_records']}")
    print(f"Retained records:     {exact['retained_records']}")
    print(f"Duplicate groups:     {exact['duplicate_groups']}")
    print(f"Duplicate records:    {exact['duplicate_records']}")
    print(f"Split groups:         {grouping['split_group_count']}")
    print(f"Largest split group:  {grouping['largest_split_group_size']}")
    print(f"Fuzzy pairs:          {fuzzy['candidate_pairs']}")
    print(f"Report:               {args.output_dir / 'manifests' / 'curation_report.json'}")
    print(f"Checksums:            {args.output_dir / 'checksums.sha256'}")
    if stability_report is not None:
        print("Shuffle stability:    pass")

    print(
        json.dumps(
            {
                "retained_records": exact["retained_records"],
                "duplicate_groups": exact["duplicate_groups"],
                "duplicate_records": exact["duplicate_records"],
                "split_group_count": grouping["split_group_count"],
                "fuzzy_candidate_pairs": fuzzy["candidate_pairs"],
                "shuffle_stable": (
                    stability_report["stable"]
                    if stability_report is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
