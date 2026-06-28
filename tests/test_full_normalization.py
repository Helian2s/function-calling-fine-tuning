import json
from pathlib import Path

import pytest

from function_calling_ft.full_normalization import (
    canonical_json_dumps,
    iter_json_array,
    load_source_manifest,
    normalize_full_dataset,
    normalize_full_xlam_row,
    sha256_file,
)
from function_calling_ft.normalization import normalize_xlam_row


ROOT = Path(__file__).resolve().parents[1]


def make_raw_row(row_id: int = 123) -> dict:
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "city": {
                    "type": "str",
                    "description": "City name",
                    "required": True,
                },
                "days": {
                    "type": "int",
                    "description": "Forecast length",
                    "required": False,
                },
            },
        }
    ]

    answers = [
        {
            "name": "get_weather",
            "arguments": {
                "city": "Boston",
                "days": 3,
            },
        }
    ]

    return {
        "id": row_id,
        "query": "Give me a three-day forecast for Boston.",
        "tools": json.dumps(tools),
        "answers": json.dumps(answers),
    }


def source_manifest(raw_path: Path) -> dict:
    return {
        "repository_id": "Salesforce/xlam-function-calling-60k",
        "repository_type": "dataset",
        "dataset_config": "default",
        "dataset_split": "train",
        "filename": raw_path.name,
        "revision": "test-revision",
        "license": "cc-by-4.0",
        "access": "gated",
        "local_path": str(raw_path),
        "size_bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "raw_data_committed_to_git": False,
    }


def test_iter_json_array_reads_chunked_array(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(
        '[{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]',
        encoding="utf-8",
    )

    assert list(iter_json_array(path, chunk_size=5)) == [
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
    ]


def test_canonical_json_dumps_is_stable() -> None:
    assert (
        canonical_json_dumps({"b": 1, "a": {"d": 4, "c": 3}})
        == '{"a":{"c":3,"d":4},"b":1}'
    )


def test_valid_full_record_adds_stable_example_id(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "source.json"
    raw_path.write_text("[]", encoding="utf-8")
    source = source_manifest(raw_path)

    result = normalize_full_xlam_row(
        make_raw_row(123),
        source_row_index=7,
        source=source,
    )

    assert result.quarantine is None
    assert result.accepted is not None
    assert result.accepted["id"] == "xlam-123"
    assert result.accepted["example_id"] == "xlam-123"
    assert result.accepted["messages"][1]["tool_calls"][0][
        "function"
    ]["arguments"] == {"city": "Boston", "days": 3}
    assert result.accepted["metadata"]["source_row_index"] == 7
    assert (
        result.accepted["metadata"]["source_revision"]
        == "test-revision"
    )


def test_openai_wire_format_argument_string_is_strictly_decoded(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "source.json"
    raw_path.write_text("[]", encoding="utf-8")
    source = source_manifest(raw_path)
    row = make_raw_row(124)
    answers = json.loads(row["answers"])
    answers[0]["arguments"] = '{"city":"Denver","days":2}'
    row["answers"] = json.dumps(answers)

    result = normalize_full_xlam_row(
        row,
        source_row_index=0,
        source=source,
    )

    assert result.accepted is not None
    arguments = result.accepted["messages"][1]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert arguments == {"city": "Denver", "days": 2}


@pytest.mark.parametrize(
    ("row_mutation", "expected_reason"),
    [
        (
            lambda row: row.update(
                {
                    "answers": json.dumps(
                        [
                            {
                                "name": "get_weather",
                                "arguments": "{bad json",
                            }
                        ]
                    )
                }
            ),
            "invalid_argument_object",
        ),
        (
            lambda row: row.update(
                {
                    "answers": json.dumps(
                        [
                            {
                                "name": "get_weather",
                                "arguments": {"days": 3},
                            }
                        ]
                    )
                }
            ),
            "missing_required_arguments",
        ),
        (
            lambda row: row.update(
                {
                    "answers": json.dumps(
                        [
                            {
                                "name": "get_weather",
                                "arguments": {
                                    "city": "Boston",
                                    "days": "three",
                                },
                            }
                        ]
                    )
                }
            ),
            "incompatible_argument_type",
        ),
    ],
)
def test_invalid_full_records_are_quarantined(
    tmp_path: Path,
    row_mutation,
    expected_reason: str,
) -> None:
    raw_path = tmp_path / "source.json"
    raw_path.write_text("[]", encoding="utf-8")
    source = source_manifest(raw_path)
    row = make_raw_row(125)
    row_mutation(row)

    result = normalize_full_xlam_row(
        row,
        source_row_index=1,
        source=source,
    )

    assert result.accepted is None
    assert result.quarantine is not None
    assert expected_reason in result.quarantine["reason_codes"]
    assert result.quarantine["example_id"] == "xlam-125"
    assert result.quarantine["source"]["source_row_index"] == 1


def test_duplicate_tool_names_are_quarantined(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "source.json"
    raw_path.write_text("[]", encoding="utf-8")
    source = source_manifest(raw_path)
    row = make_raw_row(126)
    tools = json.loads(row["tools"])
    tools.append(tools[0])
    row["tools"] = json.dumps(tools)

    result = normalize_full_xlam_row(
        row,
        source_row_index=2,
        source=source,
    )

    assert result.quarantine is not None
    assert "duplicate_tool_names" in result.quarantine["reason_codes"]


def test_normalize_full_dataset_is_idempotent(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "source.json"
    valid = make_raw_row(127)
    invalid = make_raw_row(128)
    invalid_answers = json.loads(invalid["answers"])
    invalid_answers[0]["arguments"] = {"days": 5}
    invalid["answers"] = json.dumps(invalid_answers)
    raw_path.write_text(
        json.dumps([valid, invalid]),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(source_manifest(raw_path), sort_keys=True),
        encoding="utf-8",
    )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_report = normalize_full_dataset(
        raw_path=raw_path,
        output_dir=first_dir,
        source_manifest_path=manifest_path,
        repo_root=ROOT,
    )
    second_report = normalize_full_dataset(
        raw_path=raw_path,
        output_dir=second_dir,
        source_manifest_path=manifest_path,
        repo_root=ROOT,
    )

    assert first_report["processing"]["input_records"] == 2
    assert first_report["processing"]["accepted_records"] == 1
    assert first_report["processing"]["quarantined_records"] == 1
    assert first_report["processing"]["reconciled"] is True
    assert (
        first_report["outputs"]["normalized"]["sha256"]
        == second_report["outputs"]["normalized"]["sha256"]
    )
    assert (
        first_report["outputs"]["quarantine"]["sha256"]
        == second_report["outputs"]["quarantine"]["sha256"]
    )
    assert (first_dir / "checksums.sha256").is_file()
    assert (
        first_report["distributions"]["quarantine_reasons"]
        == {"missing_required_arguments": 1}
    )


def test_load_source_manifest_records_raw_hash(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "source.json"
    raw_path.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "missing.json"

    source = load_source_manifest(manifest_path, raw_path)

    assert source["repository_id"] == "Salesforce/xlam-function-calling-60k"
    assert source["dataset_config"] == "default"
    assert source["dataset_split"] == "train"
    assert source["sha256"] == sha256_file(raw_path)


def test_full_pipeline_preserves_smoke_normalization_semantics() -> None:
    raw_dir = ROOT / "data/smoke/raw"
    normalized_dir = ROOT / "data/smoke/normalized"

    if not raw_dir.is_dir() or not normalized_dir.is_dir():
        pytest.skip("Generated smoke dataset is not available locally.")

    for split in ("train", "validation", "test"):
        raw_path = raw_dir / f"{split}.jsonl"
        normalized_path = normalized_dir / f"{split}.jsonl"

        if not raw_path.is_file() or not normalized_path.is_file():
            pytest.skip("Generated smoke dataset is incomplete locally.")

        with raw_path.open(encoding="utf-8") as raw_file:
            raw_records = [
                json.loads(line)
                for line in raw_file
                if line.strip()
            ]

        with normalized_path.open(encoding="utf-8") as normalized_file:
            frozen_records = [
                json.loads(line)
                for line in normalized_file
                if line.strip()
            ]

        assert len(raw_records) == len(frozen_records)

        for raw_record, frozen_record in zip(
            raw_records,
            frozen_records,
            strict=True,
        ):
            renormalized = normalize_xlam_row(
                raw_record,
                split=split,
            )

            assert renormalized["id"] == frozen_record["id"]
            assert renormalized["tools"] == frozen_record["tools"]
            assert renormalized["messages"] == frozen_record["messages"]
            assert (
                renormalized["metadata"]["source_id"]
                == frozen_record["metadata"]["source_id"]
            )
