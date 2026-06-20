from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_PATH = Path("data/manifests/smoke_v1_checksums.json")

REQUIRED_GROUPS: dict[str, list[Path]] = {
    "raw_splits": [
        Path("data/smoke/raw/train.jsonl"),
        Path("data/smoke/raw/validation.jsonl"),
        Path("data/smoke/raw/test.jsonl"),
    ],
    "normalized_splits": [
        Path("data/smoke/normalized/train.jsonl"),
        Path("data/smoke/normalized/validation.jsonl"),
        Path("data/smoke/normalized/test.jsonl"),
    ],
    "manifests": [
        Path("data/manifests/smoke_v1_selection.json"),
        Path("data/manifests/smoke_v1_summary.json"),
        Path("data/manifests/smoke_v1_normalization_report.json"),
        Path("data/manifests/smoke_v1_validation_report.json"),
        Path("data/manifests/smoke_v1_split_verification.json"),
        Path("data/manifests/smoke_v1_template_report.json"),
        Path("data/manifests/smoke_v1_loss_mask_report.json"),
    ],
}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def file_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file for checksum calculation: {path}")

    line_count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "line_count": line_count,
        "sha256": calculate_sha256(path),
    }


def build_report(groups: dict[str, list[Path]]) -> dict[str, Any]:
    report_groups: dict[str, list[dict[str, Any]]] = {}

    for name, paths in groups.items():
        report_groups[name] = [file_report(path) for path in paths]

    return {
        "checksum_schema_version": "1.0",
        "groups": report_groups,
    }


def main() -> None:
    report = build_report(REQUIRED_GROUPS)
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

    print("Calculated smoke artifact checksums.")
    for group_name, entries in report["groups"].items():
        print(f"{group_name}: {len(entries)} file(s)")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
