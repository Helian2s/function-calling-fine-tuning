from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SELECT_EVAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_stratified_eval_sample.py"
)
SELECT_EVAL_SPEC = importlib.util.spec_from_file_location(
    "select_stratified_eval_sample",
    SELECT_EVAL_PATH,
)
assert SELECT_EVAL_SPEC is not None
assert SELECT_EVAL_SPEC.loader is not None
select_eval = importlib.util.module_from_spec(SELECT_EVAL_SPEC)
SELECT_EVAL_SPEC.loader.exec_module(select_eval)


def make_candidate(
    *,
    source_id: int,
    generator: str,
    call_bucket: str,
    available_tool_count: int,
    has_complex_parameters: bool,
) -> dict[str, object]:
    expected_call_count = {
        "single_call": 1,
        "two_calls": 2,
        "three_or_more_calls": 3,
    }[call_bucket]
    multiple_distinct_tools = call_bucket == "three_or_more_calls"
    repeated_same_tool = call_bucket == "two_calls"

    return {
        "id": source_id,
        "row_index": source_id - 1,
        "generator": generator,
        "call_bucket": call_bucket,
        "available_tool_count": available_tool_count,
        "expected_call_count": expected_call_count,
        "distinct_expected_tool_count": (
            2 if multiple_distinct_tools else 1
        ),
        "repeated_same_tool": repeated_same_tool,
        "multiple_distinct_tools": multiple_distinct_tools,
        "parameter_types": ["list"]
        if has_complex_parameters
        else ["str"],
        "has_complex_parameters": has_complex_parameters,
        "fingerprint": f"fp-{source_id}",
        "raw_row": {
            "id": source_id,
            "query": f"Query {source_id}",
            "tools": "[]",
            "answers": "[]",
        },
    }


def make_candidate_grid(records_per_stratum: int = 3) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    source_id = 1

    for generator in ("deepseek", "mixtral"):
        for call_bucket in (
            "single_call",
            "two_calls",
            "three_or_more_calls",
        ):
            for available_tool_count in (1, 2, 5):
                for has_complex_parameters in (False, True):
                    for _ in range(records_per_stratum):
                        candidates.append(
                            make_candidate(
                                source_id=source_id,
                                generator=generator,
                                call_bucket=call_bucket,
                                available_tool_count=available_tool_count,
                                has_complex_parameters=(
                                    has_complex_parameters
                                ),
                            )
                        )
                        source_id += 1

    return candidates


def test_stratified_selection_covers_every_available_stratum() -> None:
    candidates = make_candidate_grid(records_per_stratum=3)

    selected = select_eval.select_stratified_records(
        candidates,
        sample_size=72,
        seed=42,
    )

    assert len(selected) == 72
    assert len({record["id"] for record in selected}) == 72

    selected_strata = {
        select_eval.stratum_key(record) for record in selected
    }
    all_strata = {
        select_eval.stratum_key(record) for record in candidates
    }

    assert selected_strata == all_strata


def test_stratified_selection_is_deterministic() -> None:
    candidates = make_candidate_grid(records_per_stratum=4)

    first = select_eval.select_stratified_records(
        candidates,
        sample_size=50,
        seed=7,
    )
    second = select_eval.select_stratified_records(
        candidates,
        sample_size=50,
        seed=7,
    )

    assert [record["id"] for record in first] == [
        record["id"] for record in second
    ]


def test_stratified_selection_excludes_smoke_ids_and_duplicate_fingerprints() -> None:
    candidates = make_candidate_grid(records_per_stratum=4)
    candidates[2]["fingerprint"] = candidates[1]["fingerprint"]

    selected = select_eval.select_stratified_records(
        candidates,
        sample_size=20,
        seed=42,
        excluded_ids={1},
    )
    selected_ids = {int(record["id"]) for record in selected}
    selected_fingerprints = [
        str(record["fingerprint"]) for record in selected
    ]

    assert 1 not in selected_ids
    assert len(selected_fingerprints) == len(set(selected_fingerprints))


def test_stratified_selection_fails_when_sample_is_too_large() -> None:
    with pytest.raises(RuntimeError, match="cannot select"):
        select_eval.select_stratified_records(
            make_candidate_grid(records_per_stratum=1),
            sample_size=10_000,
            seed=42,
        )


def test_load_excluded_smoke_ids_reads_selection_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {"id": 11},
                    {"id": "12"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert select_eval.load_excluded_smoke_ids(manifest_path) == {
        11,
        12,
    }


def test_normalize_selected_records_uses_test_split() -> None:
    tools = [
        {
            "name": "search",
            "description": "Search for documents.",
            "parameters": {
                "query": {
                    "type": "str",
                    "description": "Search text",
                    "required": True,
                }
            },
        }
    ]
    answers = [{"name": "search", "arguments": {"query": "abc"}}]
    candidate = make_candidate(
        source_id=123,
        generator="deepseek",
        call_bucket="single_call",
        available_tool_count=1,
        has_complex_parameters=False,
    )
    candidate["raw_row"] = {
        "id": 123,
        "query": "Find abc.",
        "tools": json.dumps(tools),
        "answers": json.dumps(answers),
    }

    normalized, errors = select_eval.normalize_selected_records([candidate])

    assert errors == []
    assert normalized[0]["id"] == "xlam-123"
    assert normalized[0]["metadata"]["split"] == "test"


def test_write_checksums_records_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "normalized").mkdir()
    (tmp_path / "normalized" / "test.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )

    checksum_path = select_eval.write_checksums(tmp_path)

    text = checksum_path.read_text(encoding="utf-8")
    assert "normalized/test.jsonl" in text
    assert "checksums.sha256" not in text
