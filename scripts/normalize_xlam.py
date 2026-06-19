from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.normalization import (
    NormalizationError,
    normalize_xlam_row,
)


INPUT_DIR = Path("data/smoke/raw")
OUTPUT_DIR = Path("data/smoke/normalized")
REPORT_PATH = Path(
    "data/manifests/smoke_v1_normalization_report.json"
)

SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def read_and_normalize_split(
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_path = INPUT_DIR / f"{split}.jsonl"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input split not found: {input_path}"
        )

    normalized_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "error": f"Invalid JSONL line: {exc}",
                    }
                )
                continue

            try:
                normalized = normalize_xlam_row(
                    raw_row,
                    split=split,
                )
            except (NormalizationError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "source_id": raw_row.get("id"),
                        "error": str(exc),
                    }
                )
                continue

            normalized_records.append(normalized)

    return normalized_records, errors


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
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


def main() -> None:
    normalized_by_split: dict[str, list[dict[str, Any]]] = {}
    all_errors: list[dict[str, Any]] = []

    for split in SPLITS:
        normalized, errors = read_and_normalize_split(split)
        normalized_by_split[split] = normalized
        all_errors.extend(errors)

    report: dict[str, Any] = {
        "normalization_schema_version": "1.0",
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "splits": {},
        "total_input_records": 0,
        "total_normalized_records": 0,
        "total_errors": len(all_errors),
        "errors": all_errors,
    }

    for split in SPLITS:
        input_path = INPUT_DIR / f"{split}.jsonl"

        input_count = sum(
            1
            for line in input_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )

        output_count = len(normalized_by_split[split])

        report["splits"][split] = {
            "input_path": str(input_path),
            "input_sha256": sha256(input_path),
            "input_records": input_count,
            "normalized_records": output_count,
        }

        report["total_input_records"] += input_count
        report["total_normalized_records"] += output_count

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fail before creating final normalized files.
    if all_errors:
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

        print(
            f"Normalization failed with {len(all_errors)} error(s)."
        )
        print(f"Inspect: {REPORT_PATH}")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        output_path = OUTPUT_DIR / f"{split}.jsonl"

        write_jsonl(
            output_path,
            normalized_by_split[split],
        )

        report["splits"][split]["output_path"] = str(output_path)
        report["splits"][split]["output_sha256"] = sha256(
            output_path
        )

    tool_count_distribution: Counter[int] = Counter()
    call_count_distribution: Counter[int] = Counter()

    for records in normalized_by_split.values():
        for record in records:
            metadata = record["metadata"]

            tool_count_distribution[
                metadata["available_tool_count"]
            ] += 1

            call_count_distribution[
                metadata["expected_call_count"]
            ] += 1

    report["available_tool_count_distribution"] = dict(
        sorted(tool_count_distribution.items())
    )

    report["expected_call_count_distribution"] = dict(
        sorted(call_count_distribution.items())
    )

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

    print("Normalization completed successfully.")
    print(
        f"Train: {len(normalized_by_split['train'])}"
    )
    print(
        f"Validation: {len(normalized_by_split['validation'])}"
    )
    print(
        f"Test: {len(normalized_by_split['test'])}"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
