#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised in container
        raise RuntimeError("Reading parquet requires pandas.") from exc

    frame = pd.read_parquet(path)
    return [
        {
            str(key): value
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        return read_json(path)
    if path.suffix == ".csv":
        return read_csv(path)
    if path.suffix == ".parquet":
        return read_parquet(path)
    return []


def candidate_output_files(output_dir: Path) -> list[Path]:
    suffixes = {".jsonl", ".json", ".csv", ".parquet"}
    return [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def value_as_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def id_from_record(record: dict[str, Any]) -> str | None:
    for key in ("id", "example_id", "document_id", "doc_id"):
        if key in record:
            return value_as_string(record[key])
    return None


def hash_from_record(record: dict[str, Any]) -> str | None:
    for key in (
        "_hashes",
        "hash",
        "hashes",
        "md5",
        "digest",
        "exact_duplicate_hash",
    ):
        if key in record:
            return value_as_string(record[key])
    return None


def load_curator_records(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in candidate_output_files(output_dir):
        try:
            file_records = read_records(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
            continue
        for record in file_records:
            record["_source_file"] = str(path)
        records.extend(file_records)
    return records, errors


def md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()  # noqa: S324


def expected_groups(curator_input_path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in read_jsonl(curator_input_path):
        record_id = str(record["id"])
        text = str(record["text"])
        groups[md5_text(text)].append(record_id)
    return {
        key: sorted(value)
        for key, value in groups.items()
        if len(value) > 1
    }


def load_duplicate_map(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def expected_removed_ids(duplicate_map: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for record in duplicate_map:
        for value in record.get("duplicate_example_ids", []):
            ids.add(str(value))
    return ids


def expected_all_duplicate_ids(duplicate_map: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for record in duplicate_map:
        if int(record.get("group_size", 0)) <= 1:
            continue
        for value in record.get("all_example_ids", []):
            ids.add(str(value))
    return ids


def group_records_by_hash(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        record_id = id_from_record(record)
        record_hash = hash_from_record(record)
        if record_id is None or record_hash is None:
            continue
        groups[record_hash].append(record_id)
    return {
        key: sorted(set(value))
        for key, value in groups.items()
        if len(set(value)) > 1
    }


def removal_group_shape_match(
    expected_md5_groups: dict[str, list[str]],
    curator_ids: set[str],
) -> bool:
    expected_removed_count = sum(
        len(group_ids) - 1
        for group_ids in expected_md5_groups.values()
    )
    expected_all_ids = {
        record_id
        for group_ids in expected_md5_groups.values()
        for record_id in group_ids
    }
    if len(curator_ids) != expected_removed_count:
        return False
    if not curator_ids.issubset(expected_all_ids):
        return False
    return all(
        len(set(group_ids) & curator_ids) == len(group_ids) - 1
        for group_ids in expected_md5_groups.values()
    )


def compare(
    *,
    curator_output_dir: Path,
    curator_input_path: Path,
    independent_duplicate_map_path: Path,
) -> dict[str, Any]:
    duplicate_map = load_duplicate_map(independent_duplicate_map_path)
    expected_md5_groups = expected_groups(curator_input_path)
    expected_all_ids = expected_all_duplicate_ids(duplicate_map)
    expected_removed = expected_removed_ids(duplicate_map)
    curator_records, read_errors = load_curator_records(curator_output_dir)
    curator_ids = {
        record_id
        for record in curator_records
        if (record_id := id_from_record(record)) is not None
    }
    curator_hash_groups = group_records_by_hash(curator_records)
    expected_group_sets = {
        tuple(value) for value in expected_md5_groups.values()
    }
    curator_group_sets = {
        tuple(value) for value in curator_hash_groups.values()
    }

    id_mode: str | None = None
    id_match = False
    if curator_ids == expected_all_ids:
        id_mode = "all_duplicate_group_ids"
        id_match = True
    elif curator_ids == expected_removed:
        id_mode = "removed_duplicate_ids"
        id_match = True

    hash_group_match = (
        bool(curator_group_sets)
        and curator_group_sets == expected_group_sets
    )
    removal_shape_match = removal_group_shape_match(
        expected_md5_groups,
        curator_ids,
    )
    status = (
        "pass"
        if id_match or hash_group_match or removal_shape_match
        else "fail"
    )

    return {
        "schema_version": "1.0",
        "status": status,
        "curator_output_dir": str(curator_output_dir),
        "curator_input_path": str(curator_input_path),
        "independent_duplicate_map_path": str(independent_duplicate_map_path),
        "independent": {
            "duplicate_groups": len(expected_md5_groups),
            "all_duplicate_ids": len(expected_all_ids),
            "removed_duplicate_ids": len(expected_removed),
        },
        "curator": {
            "records_read": len(curator_records),
            "ids_read": len(curator_ids),
            "hash_groups": len(curator_hash_groups),
            "read_errors": read_errors,
            "output_files": [
                str(path) for path in candidate_output_files(curator_output_dir)
            ],
        },
        "comparison": {
            "id_match": id_match,
            "id_mode": id_mode,
            "hash_group_match": hash_group_match,
            "removal_group_shape_match": removal_shape_match,
            "missing_expected_ids": sorted(expected_all_ids - curator_ids)[:100],
            "unexpected_curator_ids": sorted(curator_ids - expected_all_ids)[:100],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare scoped NeMo Curator exact-dedup output with the "
            "independent exact-hash duplicate map."
        )
    )
    parser.add_argument(
        "--curator-output-dir",
        type=Path,
        default=Path("data/processed/xlam_curated_v1/curator_exact/output"),
    )
    parser.add_argument(
        "--curator-input",
        type=Path,
        default=Path(
            "data/processed/xlam_curated_v1/curator_input/"
            "exact_dedup_input.jsonl"
        ),
    )
    parser.add_argument(
        "--independent-duplicate-map",
        type=Path,
        default=Path("data/processed/xlam_curated_v1/duplicate_map.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/xlam_curated_v1/manifests/"
            "curator_comparison_report.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(
        curator_output_dir=args.curator_output_dir,
        curator_input_path=args.curator_input,
        independent_duplicate_map_path=args.independent_duplicate_map,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Curator comparison status: {report['status']}")
    print(f"Report: {args.output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
