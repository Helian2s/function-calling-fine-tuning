from __future__ import annotations

import json
from pathlib import Path

import pytest

from function_calling_ft.evaluation import (
    evaluate_predictions,
    score_prediction_records,
)


def _record(record_id: str, calls: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": record_id,
        "tools": [],
        "messages": [
            {"role": "user", "content": f"request {record_id}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for index, call in enumerate(calls, start=1)
                ],
            },
        ],
        "metadata": {"source_id": int(record_id.rsplit("-", 1)[1])},
    }


def _prediction(record_id: str, raw_generation: str) -> dict[str, object]:
    return {
        "id": record_id,
        "raw_generation": raw_generation,
        "generation_error": None,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def test_scores_correct_prediction() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [
        _prediction("xlam-1", '{"name":"weather","arguments":{"city":"Denver"}}'),
    ]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["score"]["complete_call_match"] is True
    assert failures == []
    assert summary["complete_match_count"] == 1
    assert summary["complete_match_rate"] == 1.0


def test_malformed_prediction_is_preserved_as_failure() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [_prediction("xlam-1", "not json")]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["parse"]["valid_structure"] is False
    assert failures[0]["id"] == "xlam-1"
    assert summary["parse_failure_count"] == 1


def test_missing_prediction_is_scored_as_empty_output() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]

    scored, failures, summary = score_prediction_records(dataset, [])

    assert scored[0]["missing_prediction"] is True
    assert scored[0]["parse"]["errors"] == ["Model output is empty."]
    assert failures[0]["id"] == "xlam-1"
    assert summary["missing_predictions"] == 1


def test_duplicate_prediction_ids_are_rejected() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [
        _prediction("xlam-1", '{"name":"weather","arguments":{"city":"Denver"}}'),
        _prediction("xlam-1", '{"name":"weather","arguments":{"city":"Denver"}}'),
    ]

    with pytest.raises(ValueError, match="Duplicate prediction id"):
        score_prediction_records(dataset, predictions)


def test_unknown_prediction_ids_are_rejected() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]

    with pytest.raises(ValueError, match="absent from dataset"):
        score_prediction_records(dataset, [_prediction("xlam-2", "{}")])


def test_parallel_calls_score_order_insensitively_by_default() -> None:
    dataset = [
        _record(
            "xlam-1",
            [
                {"name": "weather", "arguments": {"city": "Denver"}},
                {"name": "time", "arguments": {"city": "Denver"}},
            ],
        )
    ]
    predictions = [
        _prediction(
            "xlam-1",
            (
                '['
                '{"name":"time","arguments":{"city":"Denver"}},'
                '{"name":"weather","arguments":{"city":"Denver"}}'
                ']'
            ),
        )
    ]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["score"]["complete_call_match"] is True
    assert failures == []
    assert summary["complete_match_count"] == 1


def test_all_40_records_are_processed_with_empty_predictions_file(
    tmp_path: Path,
) -> None:
    dataset = [
        _record(
            f"xlam-{index}",
            [{"name": "weather", "arguments": {"city": f"city-{index}"}}],
        )
        for index in range(40)
    ]
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "scores"
    _write_jsonl(dataset_path, dataset)
    predictions_path.write_text("", encoding="utf-8")

    outputs = evaluate_predictions(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
    )

    scored_lines = outputs.scored_predictions_path.read_text(
        encoding="utf-8",
    ).splitlines()
    failure_lines = outputs.parse_failures_path.read_text(
        encoding="utf-8",
    ).splitlines()
    summary = json.loads(outputs.scores_path.read_text(encoding="utf-8"))

    assert len(scored_lines) == 40
    assert len(failure_lines) == 40
    assert summary["total_records"] == 40
    assert summary["predictions_present"] == 0
    assert summary["missing_predictions"] == 40
