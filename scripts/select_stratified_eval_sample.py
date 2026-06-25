#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC_DIR = ROOT / "src"

for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import select_smoke_sample
from function_calling_ft.normalization import (
    NormalizationError,
    normalize_xlam_row,
)


DEFAULT_OUTPUT_DIR = Path("data/eval/stratified_1000")
DEFAULT_SMOKE_SELECTION_MANIFEST = Path(
    "data/manifests/smoke_v1_selection.json",
)
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42
EVAL_DATASET_ID = "stratified_1000"

Candidate = dict[str, Any]
StratumKey = tuple[str, str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and normalize a deterministic stratified xLAM "
            "evaluation subset without modifying the frozen smoke split."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--smoke-selection-manifest",
        type=Path,
        default=DEFAULT_SMOKE_SELECTION_MANIFEST,
    )
    parser.add_argument(
        "--include-smoke-records",
        action="store_true",
        help="Allow overlap with the frozen 200-record smoke split.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def available_tool_bucket(count: int) -> str:
    if count == 1:
        return "one_available_tool"

    if count <= 4:
        return "two_to_four_available_tools"

    return "five_or_more_available_tools"


def call_diversity(candidate: Candidate) -> str:
    if int(candidate["expected_call_count"]) == 1:
        return "single_expected_call"

    if bool(candidate["multiple_distinct_tools"]):
        return "multiple_distinct_tools"

    if bool(candidate["repeated_same_tool"]):
        return "repeated_same_tool"

    return "multi_call_single_tool"


def stratum_key(candidate: Candidate) -> StratumKey:
    return (
        str(candidate["generator"]),
        str(candidate["call_bucket"]),
        available_tool_bucket(int(candidate["available_tool_count"])),
        "complex_parameters"
        if bool(candidate["has_complex_parameters"])
        else "simple_parameters",
        call_diversity(candidate),
    )


def stratum_name(key: StratumKey) -> str:
    return "/".join(key)


def load_excluded_smoke_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()

    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])

    if not isinstance(records, list):
        raise ValueError(f"Smoke selection manifest has invalid records: {path}")

    excluded: set[int] = set()

    for record in records:
        if isinstance(record, dict) and "id" in record:
            excluded.add(int(record["id"]))

    return excluded


def filter_unique_candidates(
    candidates: list[Candidate],
    *,
    excluded_ids: set[int],
) -> list[Candidate]:
    filtered: list[Candidate] = []
    seen_ids: set[int] = set()
    seen_fingerprints: set[str] = set()

    for candidate in candidates:
        source_id = int(candidate["id"])
        fingerprint = str(candidate["fingerprint"])

        if source_id in excluded_ids:
            continue

        if source_id in seen_ids or fingerprint in seen_fingerprints:
            continue

        seen_ids.add(source_id)
        seen_fingerprints.add(fingerprint)
        filtered.append(candidate)

    return filtered


def select_stratified_records(
    candidates: list[Candidate],
    *,
    sample_size: int,
    seed: int,
    excluded_ids: set[int] | None = None,
) -> list[Candidate]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    eligible = filter_unique_candidates(
        candidates,
        excluded_ids=excluded_ids or set(),
    )

    if len(eligible) < sample_size:
        raise RuntimeError(
            f"Only {len(eligible)} eligible candidates are available; "
            f"cannot select {sample_size} records."
        )

    rng = random.Random(seed)
    pools: dict[StratumKey, list[Candidate]] = defaultdict(list)

    for candidate in eligible:
        pools[stratum_key(candidate)].append(candidate)

    for pool in pools.values():
        rng.shuffle(pool)

    strata = sorted(pools)
    rng.shuffle(strata)

    selected: list[Candidate] = []
    used_ids: set[int] = set()
    used_fingerprints: set[str] = set()

    while len(selected) < sample_size:
        progressed = False

        for key in tuple(strata):
            pool = pools[key]

            while pool:
                candidate = pool.pop()
                source_id = int(candidate["id"])
                fingerprint = str(candidate["fingerprint"])

                if source_id in used_ids or fingerprint in used_fingerprints:
                    continue

                selected.append(candidate)
                used_ids.add(source_id)
                used_fingerprints.add(fingerprint)
                progressed = True
                break

            if len(selected) == sample_size:
                break

        if not progressed:
            break

    if len(selected) != sample_size:
        raise RuntimeError(
            f"Selected {len(selected)} records; expected {sample_size}."
        )

    return selected


def normalize_selected_records(
    selected: list[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for position, candidate in enumerate(selected, start=1):
        try:
            record = normalize_xlam_row(
                candidate["raw_row"],
                split="test",
            )
        except (NormalizationError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "position": position,
                    "source_id": candidate.get("id"),
                    "error": str(exc),
                }
            )
            continue

        normalized.append(record)

    return normalized, errors


def counter_for(
    records: list[Candidate],
    field: str,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(record[field]) for record in records).items(),
        )
    )


def create_summary(
    *,
    selected: list[Candidate],
    normalized: list[dict[str, Any]],
    seed: int,
    sample_size: int,
    excluded_smoke_ids: set[int],
    candidate_count: int,
    rejected: Counter[str],
) -> dict[str, Any]:
    stratum_distribution = Counter(
        stratum_name(stratum_key(record)) for record in selected
    )
    available_tool_distribution = Counter(
        int(record["available_tool_count"]) for record in selected
    )
    expected_call_distribution = Counter(
        int(record["expected_call_count"]) for record in selected
    )
    parameter_type_distribution: Counter[str] = Counter()

    for record in selected:
        parameter_type_distribution.update(record["parameter_types"])

    return {
        "dataset_id": EVAL_DATASET_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "requested_sample_size": sample_size,
        "selected_records": len(selected),
        "normalized_records": len(normalized),
        "raw_candidate_count": candidate_count,
        "excluded_smoke_record_count": len(excluded_smoke_ids),
        "normalization_rejected_candidate_counts": dict(
            sorted(rejected.items())
        ),
        "stratification_dimensions": [
            "generator",
            "expected_call_count_bucket",
            "available_tool_count_bucket",
            "parameter_complexity",
            "call_diversity",
        ],
        "generator_distribution": counter_for(selected, "generator"),
        "call_bucket_distribution": counter_for(selected, "call_bucket"),
        "available_tool_count_distribution": dict(
            sorted(available_tool_distribution.items())
        ),
        "expected_call_count_distribution": dict(
            sorted(expected_call_distribution.items())
        ),
        "stratum_distribution": dict(
            sorted(stratum_distribution.items())
        ),
        "examples_with_complex_parameters": sum(
            bool(record["has_complex_parameters"]) for record in selected
        ),
        "multi_call_examples_using_multiple_tools": sum(
            bool(record["multiple_distinct_tools"]) for record in selected
        ),
        "multi_call_examples_repeating_same_tool": sum(
            bool(record["repeated_same_tool"]) for record in selected
        ),
        "parameter_types": dict(parameter_type_distribution.most_common()),
    }


def create_selection_manifest(
    *,
    selected: list[Candidate],
    seed: int,
    sample_size: int,
    excluded_smoke_ids: set[int],
) -> dict[str, Any]:
    source_metadata = select_smoke_sample.load_source_metadata()
    records: list[dict[str, Any]] = []

    for position, record in enumerate(selected, start=1):
        key = stratum_key(record)
        records.append(
            {
                "position": position,
                "id": int(record["id"]),
                "row_index": int(record["row_index"]),
                "generator": record["generator"],
                "call_bucket": record["call_bucket"],
                "available_tool_count": int(
                    record["available_tool_count"]
                ),
                "available_tool_bucket": key[2],
                "expected_call_count": int(
                    record["expected_call_count"]
                ),
                "distinct_expected_tool_count": int(
                    record["distinct_expected_tool_count"]
                ),
                "call_diversity": key[4],
                "repeated_same_tool": bool(record["repeated_same_tool"]),
                "multiple_distinct_tools": bool(
                    record["multiple_distinct_tools"]
                ),
                "parameter_types": record["parameter_types"],
                "has_complex_parameters": bool(
                    record["has_complex_parameters"]
                ),
                "stratum": stratum_name(key),
            }
        )

    return {
        "manifest_version": "eval-stratified-1000-v1",
        "dataset_id": EVAL_DATASET_ID,
        "dataset_repository_id": source_metadata.get(
            "repository_id",
            "Salesforce/xlam-function-calling-60k",
        ),
        "dataset_revision": source_metadata.get("revision"),
        "selection_seed": seed,
        "requested_sample_size": sample_size,
        "excluded_smoke_record_count": len(excluded_smoke_ids),
        "selection_strategy": (
            "Round-robin sampling across generator, expected-call-count "
            "bucket, available-tool-count bucket, parameter complexity, "
            "and call-diversity strata after deterministic per-stratum "
            "shuffling."
        ),
        "records": records,
    }


def write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "checksums.sha256"
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != checksum_path.name
    ]

    with checksum_path.open("w", encoding="utf-8") as file:
        for path in files:
            relative = path.relative_to(output_dir)
            file.write(f"{sha256(path)}  {relative.as_posix()}\n")

    return checksum_path


def main() -> None:
    args = parse_args()

    dataset = select_smoke_sample.load_raw_dataset()
    candidates, rejected = select_smoke_sample.collect_candidates(dataset)
    excluded_smoke_ids = (
        set()
        if args.include_smoke_records
        else load_excluded_smoke_ids(args.smoke_selection_manifest)
    )
    selected = select_stratified_records(
        candidates,
        sample_size=args.sample_size,
        seed=args.seed,
        excluded_ids=excluded_smoke_ids,
    )
    normalized, errors = normalize_selected_records(selected)

    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    manifest_dir = output_dir / "manifests"

    if errors:
        write_json(
            manifest_dir / "normalization_errors.json",
            {
                "dataset_id": EVAL_DATASET_ID,
                "error_count": len(errors),
                "errors": errors,
            },
        )
        raise SystemExit(
            f"Normalization failed for {len(errors)} selected record(s)."
        )

    write_jsonl(
        raw_dir / "test.jsonl",
        [record["raw_row"] for record in selected],
    )
    write_jsonl(normalized_dir / "test.jsonl", normalized)

    summary = create_summary(
        selected=selected,
        normalized=normalized,
        seed=args.seed,
        sample_size=args.sample_size,
        excluded_smoke_ids=excluded_smoke_ids,
        candidate_count=len(candidates),
        rejected=rejected,
    )
    selection_manifest = create_selection_manifest(
        selected=selected,
        seed=args.seed,
        sample_size=args.sample_size,
        excluded_smoke_ids=excluded_smoke_ids,
    )
    normalization_report = {
        "dataset_id": EVAL_DATASET_ID,
        "input_path": str(raw_dir / "test.jsonl"),
        "output_path": str(normalized_dir / "test.jsonl"),
        "input_records": len(selected),
        "normalized_records": len(normalized),
        "errors": [],
    }

    write_json(manifest_dir / "summary.json", summary)
    write_json(manifest_dir / "selection.json", selection_manifest)
    write_json(manifest_dir / "normalization_report.json", normalization_report)
    checksum_path = write_checksums(output_dir)

    print(f"Dataset rows scanned: {len(dataset)}")
    print(f"Valid candidates: {len(candidates)}")
    print(f"Excluded smoke records: {len(excluded_smoke_ids)}")
    print(f"Selected records: {len(selected)}")
    print(f"Normalized records: {len(normalized)}")
    print(f"Output directory: {output_dir}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()
