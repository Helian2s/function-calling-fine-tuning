from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_SPLITS_PATH = ROOT / "scripts" / "verify_smoke_splits.py"
CHECKSUMS_PATH = ROOT / "scripts" / "calculate_checksums.py"


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_smoke_splits = load_script_module(
    VERIFY_SPLITS_PATH,
    "verify_smoke_splits_for_tests",
)
calculate_checksums = load_script_module(
    CHECKSUMS_PATH,
    "calculate_checksums_for_tests",
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def _record(record_id: str) -> dict[str, object]:
    return {"id": record_id, "messages": [], "tools": []}


def test_verify_split_directory_accepts_expected_layout(tmp_path: Path) -> None:
    split_dir = tmp_path / "smoke"
    expected_sizes = {"train": 2, "validation": 1, "test": 1}

    _write_jsonl(
        split_dir / "train.jsonl",
        [_record("a"), _record("b")],
    )
    _write_jsonl(
        split_dir / "validation.jsonl",
        [_record("c")],
    )
    _write_jsonl(
        split_dir / "test.jsonl",
        [_record("d")],
    )

    report = verify_smoke_splits.verify_split_directory(
        split_dir,
        expected_sizes,
    )

    assert report["total_records"] == 4
    assert report["unique_ids_total"] == 4
    assert report["splits"]["train"]["count"] == 2


def test_verify_split_directory_rejects_wrong_count(tmp_path: Path) -> None:
    split_dir = tmp_path / "smoke"
    expected_sizes = {"train": 2, "validation": 1, "test": 1}

    _write_jsonl(
        split_dir / "train.jsonl",
        [_record("a")],
    )
    _write_jsonl(
        split_dir / "validation.jsonl",
        [_record("b")],
    )
    _write_jsonl(
        split_dir / "test.jsonl",
        [_record("c")],
    )

    try:
        verify_smoke_splits.verify_split_directory(
            split_dir,
            expected_sizes,
        )
    except ValueError as exc:
        assert "expected 2" in str(exc)
    else:
        raise AssertionError("Expected split verification to fail")


def test_ensure_matching_split_ids_rejects_mismatched_layout() -> None:
    raw_report = {
        "ids_by_split": {
            "train": ["a", "b"],
            "validation": ["c"],
            "test": ["d"],
        }
    }
    normalized_report = {
        "ids_by_split": {
            "train": ["a", "b"],
            "validation": ["x"],
            "test": ["d"],
        }
    }

    try:
        verify_smoke_splits.ensure_matching_split_ids(
            raw_report,
            normalized_report,
        )
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("Expected raw/normalized id comparison to fail")


def test_file_report_includes_sha256_and_line_count(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")

    report = calculate_checksums.file_report(path)

    assert report["line_count"] == 2
    assert report["bytes"] == path.stat().st_size
    assert report["sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_build_report_groups_files(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    report = calculate_checksums.build_report(
        {"sample": [first, second]}
    )

    assert report["checksum_schema_version"] == "1.0"
    assert [entry["path"] for entry in report["groups"]["sample"]] == [
        str(first),
        str(second),
    ]
