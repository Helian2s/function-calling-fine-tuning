import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from function_calling_ft.normalization import NormalizationError


SELECT_SMOKE_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_smoke_sample.py"
)
SELECT_SMOKE_SAMPLE_SPEC = importlib.util.spec_from_file_location(
    "select_smoke_sample",
    SELECT_SMOKE_SAMPLE_PATH,
)
assert SELECT_SMOKE_SAMPLE_SPEC is not None
assert SELECT_SMOKE_SAMPLE_SPEC.loader is not None
select_smoke_sample = importlib.util.module_from_spec(
    SELECT_SMOKE_SAMPLE_SPEC
)
SELECT_SMOKE_SAMPLE_SPEC.loader.exec_module(select_smoke_sample)

GENERATOR_BOUNDARY = select_smoke_sample.GENERATOR_BOUNDARY
PRIMARY_STRATIFICATION = (
    select_smoke_sample.PRIMARY_STRATIFICATION
)
SPLIT_QUOTAS = select_smoke_sample.SPLIT_QUOTAS
build_source_metadata = select_smoke_sample.build_source_metadata
classify_normalization_rejection = (
    select_smoke_sample.classify_normalization_rejection
)
collect_candidates = select_smoke_sample.collect_candidates
create_manifest = select_smoke_sample.create_manifest
load_source_metadata = select_smoke_sample.load_source_metadata
select_records = select_smoke_sample.select_records


class StubDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_raw_row(
    row_id: int = 123,
) -> dict[str, object]:
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


def test_duplicate_tool_names_rejected_at_candidate_collection() -> None:
    row = make_raw_row()
    tools = json.loads(row["tools"])
    duplicate = copy.deepcopy(tools[0])
    tools.append(duplicate)
    row["tools"] = json.dumps(tools)

    candidates, rejected = collect_candidates(StubDataset([row]))

    assert candidates == []
    assert rejected["normalization_duplicate_tool_names"] == 1


def test_callable_parameters_rejected_at_candidate_collection() -> None:
    row = make_raw_row()
    tools = json.loads(row["tools"])
    tools[0]["parameters"]["city"]["type"] = (
        "Callable[[str], int]"
    )
    row["tools"] = json.dumps(tools)

    candidates, rejected = collect_candidates(StubDataset([row]))

    assert candidates == []
    assert (
        rejected[
            "normalization_unsupported_callable_parameters"
        ]
        == 1
    )


def test_valid_candidate_remains_eligible() -> None:
    candidates, rejected = collect_candidates(
        StubDataset([make_raw_row()])
    )

    assert len(candidates) == 1
    assert candidates[0]["id"] == 123
    assert rejected == Counter()


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (
            NormalizationError(
                "Row 1: duplicate tool names are present."
            ),
            "normalization_duplicate_tool_names",
        ),
        (
            NormalizationError(
                "Unsupported generic parameter type: "
                "'Callable'."
            ),
            (
                "normalization_unsupported_callable_parameters"
            ),
        ),
        (
            NormalizationError(
                "Unsupported parameter type name: "
                "'PathLike'."
            ),
            "normalization_unsupported_parameter_types",
        ),
        (
            NormalizationError(
                "Invalid parameter type expression: 'List['."
            ),
            (
                "normalization_invalid_parameter_type_expressions"
            ),
        ),
        (
            NormalizationError(
                "Row 1: answer 1 references unavailable tool "
                "'missing_tool'."
            ),
            "normalization_unavailable_answer_tools",
        ),
        (
            NormalizationError("Row 1: tool has no valid name."),
            "normalization_other_errors",
        ),
    ],
)
def test_normalization_rejection_categorization(
    error: NormalizationError,
    expected_category: str,
) -> None:
    assert classify_normalization_rejection(error) == (
        expected_category
    )


def make_selection_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    next_ids = {
        "deepseek": 1,
        "mixtral": GENERATOR_BOUNDARY + 1,
    }
    required_per_generator = {
        bucket: sum(
            split_quotas[bucket]
            for split_quotas in SPLIT_QUOTAS.values()
        )
        // 2
        for bucket in (
            "single_call",
            "two_calls",
            "three_or_more_calls",
        )
    }

    for generator in ("deepseek", "mixtral"):
        for call_bucket, count in (
            required_per_generator.items()
        ):
            for index in range(count):
                row_id = next_ids[generator]
                candidates.append(
                    {
                        "id": row_id,
                        "generator": generator,
                        "call_bucket": call_bucket,
                        "fingerprint": (
                            f"{generator}-{call_bucket}-{index}"
                        ),
                    }
                )
                next_ids[generator] += 1

    return candidates


def selected_ids_by_split(
    selected: dict[str, list[dict[str, object]]],
) -> dict[str, list[int]]:
    return {
        split: [int(record["id"]) for record in records]
        for split, records in selected.items()
    }


def test_selected_split_ids_are_unique_and_disjoint() -> None:
    selected = select_records(make_selection_candidates())
    split_ids = selected_ids_by_split(selected)

    assert {split: len(ids) for split, ids in split_ids.items()} == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }

    all_ids = (
        split_ids["train"]
        + split_ids["validation"]
        + split_ids["test"]
    )

    assert len(all_ids) == len(set(all_ids)) == 200
    assert set(split_ids["train"]).isdisjoint(
        split_ids["validation"]
    )
    assert set(split_ids["train"]).isdisjoint(split_ids["test"])
    assert set(split_ids["validation"]).isdisjoint(
        split_ids["test"]
    )

    for split, records in selected.items():
        assert Counter(
            record["call_bucket"] for record in records
        ) == Counter(SPLIT_QUOTAS[split])
        assert Counter(
            record["generator"] for record in records
        ) == Counter(
            {
                "deepseek": len(records) // 2,
                "mixtral": len(records) // 2,
            }
        )


def test_selection_is_deterministic() -> None:
    first = select_records(make_selection_candidates())
    second = select_records(make_selection_candidates())

    assert selected_ids_by_split(first) == selected_ids_by_split(
        second
    )


def test_build_source_metadata_recovers_local_revision_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "data/raw/xlam"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "xlam_function_calling_60k.json"
    raw_path.write_text("[]\n", encoding="utf-8")

    metadata_path = (
        raw_dir
        / ".cache/huggingface/download/"
        / "xlam_function_calling_60k.json.metadata"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "\n".join(
            [
                "revision-from-metadata",
                "etag-placeholder",
                "0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        select_smoke_sample,
        "RAW_PATH",
        raw_path,
    )
    monkeypatch.setattr(
        select_smoke_sample,
        "DOWNLOAD_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        select_smoke_sample,
        "HF_REPO_CACHE_DIR",
        tmp_path / "hf-cache",
    )

    metadata = build_source_metadata()

    assert metadata["repository_id"] == (
        "Salesforce/xlam-function-calling-60k"
    )
    assert metadata["revision"] == "revision-from-metadata"
    assert metadata["sha256"] == select_smoke_sample.sha256(
        raw_path
    )


def test_load_source_metadata_persists_recovered_local_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "data/raw/xlam"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "xlam_function_calling_60k.json"
    raw_path.write_text("[]\n", encoding="utf-8")

    metadata_path = (
        raw_dir
        / ".cache/huggingface/download/"
        / "xlam_function_calling_60k.json.metadata"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "revision-from-metadata\n",
        encoding="utf-8",
    )

    source_manifest_path = tmp_path / "xlam_source.json"

    monkeypatch.setattr(
        select_smoke_sample,
        "RAW_PATH",
        raw_path,
    )
    monkeypatch.setattr(
        select_smoke_sample,
        "DOWNLOAD_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        select_smoke_sample,
        "HF_REPO_CACHE_DIR",
        tmp_path / "hf-cache",
    )
    monkeypatch.setattr(
        select_smoke_sample,
        "SOURCE_MANIFEST_PATH",
        source_manifest_path,
    )

    metadata = load_source_metadata()

    assert source_manifest_path.is_file()
    assert metadata["revision"] == "revision-from-metadata"

    persisted = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    assert persisted == metadata


def test_public_selection_manifest_omits_fingerprint_and_adds_primary_stratification() -> None:
    selected = {
        "train": [
            {
                "id": 123,
                "row_index": 7,
                "generator": "deepseek",
                "call_bucket": "single_call",
                "available_tool_count": 1,
                "expected_call_count": 1,
                "distinct_expected_tool_count": 1,
                "repeated_same_tool": False,
                "multiple_distinct_tools": False,
                "parameter_types": ["str"],
                "has_complex_parameters": False,
                "fingerprint": "internal-only",
            }
        ],
        "validation": [],
        "test": [],
    }

    manifest = create_manifest(selected, Counter())

    assert (
        manifest["recommended_primary_stratification"]
        == PRIMARY_STRATIFICATION
    )
    assert "fingerprint" not in manifest["records"][0]
    assert manifest["records"][0]["id"] == 123
