from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/smoke/raw")
NORMALIZED_DIR = Path("data/smoke/normalized")
REPORT_PATH = Path("data/manifests/smoke_v1_split_verification.json")
EXPECTED_SPLIT_SIZES = {
    "train": 120,
    "validation": 40,
    "test": 40,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))

    return records


def _record_id(record: dict[str, Any]) -> str:
    return str(record["id"])


def _normalized_source_id(record: dict[str, Any]) -> str:
    return str(record["metadata"]["source_id"])


def verify_split_directory(
    directory: Path,
    expected_sizes: dict[str, int],
    *,
    record_identity: Any = _record_id,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "directory": str(directory),
        "splits": {},
        "total_records": 0,
    }
    ids_by_split: dict[str, set[str]] = {}

    for split, expected_count in expected_sizes.items():
        path = directory / f"{split}.jsonl"

        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")

        records = _read_jsonl(path)
        record_ids = [str(record_identity(record)) for record in records]
        unique_ids = set(record_ids)

        if len(records) != expected_count:
            raise ValueError(
                f"{path} has {len(records)} records; expected {expected_count}."
            )

        if len(unique_ids) != len(record_ids):
            raise ValueError(f"{path} contains duplicate ids.")

        ids_by_split[split] = unique_ids
        report["splits"][split] = {
            "path": str(path),
            "count": len(records),
            "unique_ids": len(unique_ids),
        }
        report["total_records"] += len(records)

    all_seen: set[str] = set()

    for split in expected_sizes:
        overlap = all_seen & ids_by_split[split]

        if overlap:
            raise ValueError(
                f"{directory} has ids shared across splits; first overlap: "
                f"{sorted(overlap)[0]!r}."
            )

        all_seen.update(ids_by_split[split])

    report["unique_ids_total"] = len(all_seen)
    report["ids_by_split"] = {
        split: sorted(ids)
        for split, ids in ids_by_split.items()
    }
    return report


def ensure_matching_split_ids(
    raw_report: dict[str, Any],
    normalized_report: dict[str, Any],
) -> None:
    for split in EXPECTED_SPLIT_SIZES:
        raw_ids = raw_report["ids_by_split"][split]
        normalized_ids = normalized_report["ids_by_split"][split]

        if raw_ids != normalized_ids:
            raise ValueError(
                f"Raw and normalized ids differ for split {split}."
            )


def build_report(
    raw_report: dict[str, Any],
    normalized_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verification_schema_version": "1.0",
        "expected_split_sizes": EXPECTED_SPLIT_SIZES,
        "raw": {
            "directory": raw_report["directory"],
            "total_records": raw_report["total_records"],
            "split_counts": {
                split: raw_report["splits"][split]["count"]
                for split in EXPECTED_SPLIT_SIZES
            },
        },
        "normalized": {
            "directory": normalized_report["directory"],
            "total_records": normalized_report["total_records"],
            "split_counts": {
                split: normalized_report["splits"][split]["count"]
                for split in EXPECTED_SPLIT_SIZES
            },
        },
        "raw_and_normalized_ids_match": True,
        "total_examples": raw_report["total_records"],
    }


def main() -> None:
    raw_report = verify_split_directory(
        RAW_DIR,
        EXPECTED_SPLIT_SIZES,
        record_identity=_record_id,
    )
    normalized_report = verify_split_directory(
        NORMALIZED_DIR,
        EXPECTED_SPLIT_SIZES,
        record_identity=_normalized_source_id,
    )
    ensure_matching_split_ids(raw_report, normalized_report)
    report = build_report(raw_report, normalized_report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Smoke split verification completed successfully.")
    print(f"Train:      {report['raw']['split_counts']['train']}")
    print(f"Validation: {report['raw']['split_counts']['validation']}")
    print(f"Test:       {report['raw']['split_counts']['test']}")
    print(f"Total:      {report['total_examples']}")
    print(f"Report:     {REPORT_PATH}")


if __name__ == "__main__":
    main()
