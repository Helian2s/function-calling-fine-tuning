from __future__ import annotations

import json
from pathlib import Path

import pytest

from function_calling_ft.evaluation import (
    METRIC_SCHEMA_VERSION,
    evaluate_predictions,
    score_prediction_records,
)


def _record(
    record_id: str,
    calls: list[dict[str, object]],
    *,
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "tools": tools or [],
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
    assert summary["strict_complete_match_count"] == 1
    assert summary["schema_equivalent_complete_match_count"] == 1
    assert summary["executable_complete_match_count"] == 1
    assert summary["tool_call_emitted_count"] == 1
    assert summary["parseable_given_emission_count"] == 1
    assert summary["metric_schema_version"] == METRIC_SCHEMA_VERSION


def test_openai_wire_format_arguments_string_is_scored() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [
        {
            "id": "xlam-1",
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Denver"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        }
    ]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert failures == []
    assert scored[0]["prediction_format"] == "openai_message_tool_calls"
    assert scored[0]["headline_scores"]["strict_complete_match"] is True
    assert summary["tool_call_emitted_count"] == 1


def test_no_tool_record_without_gold_is_objectively_classified() -> None:
    dataset = [_record("xlam-1", [])]
    predictions = [_prediction("xlam-1", "The current answer is 42.")]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert failures == []
    assert scored[0]["no_tool_score"]["status"] == "unsupported_without_gold"
    assert scored[0]["no_tool_score"]["correct_direct_answer"] is None
    assert summary["no_tool_record_count"] == 1
    assert summary["no_tool_unsupported_gold_count"] == 1
    assert summary["parse_failure_count"] == 0


def test_no_tool_record_with_gold_direct_answer_can_score_exact_match() -> None:
    record = _record("xlam-1", [])
    record["expected_response"] = {
        "type": "direct_answer",
        "content": "The current answer is 42.",
    }
    predictions = [_prediction("xlam-1", "The current answer is 42.")]

    scored, _failures, summary = score_prediction_records([record], predictions)

    assert scored[0]["no_tool_score"]["status"] == "correct_direct_answer"
    assert scored[0]["no_tool_score"]["correct_direct_answer"] is True
    assert summary["no_tool_correct_direct_answer_count"] == 1


def test_string_source_id_is_preserved() -> None:
    record = _record("xlam-1", [])
    record["metadata"] = {"source_id": "human-authored-001"}
    record["expected_response"] = {
        "type": "direct_answer",
        "content": "The current answer is 42.",
    }
    predictions = [_prediction("xlam-1", "The current answer is 42.")]

    scored, _failures, _summary = score_prediction_records([record], predictions)

    assert scored[0]["source_id"] == "human-authored-001"


def test_grouped_metrics_use_curation_and_split_metadata() -> None:
    record = _record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])
    record["curation_metadata"] = {
        "call_category": "single",
        "primary_tool_family": "weather",
        "primary_api_category": "weather",
        "expected_call_count": 1,
        "tool_count": 1,
    }
    record["split_metadata"] = {
        "primary_split": "validation",
        "split_lock_status": "screening_allowed",
        "token_counts": {"full_tokens": 700},
    }
    predictions = [
        _prediction("xlam-1", '{"name":"weather","arguments":{"city":"Denver"}}'),
    ]

    scored, _failures, summary = score_prediction_records([record], predictions)

    assert scored[0]["groups"]["primary_tool_family"] == "weather"
    family_metrics = summary["metrics_by_group"]["primary_tool_family"]["weather"]
    assert family_metrics["executable_complete_match_rate"] == 1.0
    assert "0513-1024" in summary["metrics_by_group"]["length_bucket"]


def test_malformed_prediction_is_preserved_as_failure() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [_prediction("xlam-1", "not json")]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["parse"]["valid_structure"] is False
    assert failures[0]["id"] == "xlam-1"
    assert summary["parse_failure_count"] == 1
    assert summary["no_tool_call_emitted_count"] == 1
    assert summary["malformed_tool_call_count"] == 0
    assert scored[0]["emission"]["prose_only_response"] is True


def test_malformed_tool_call_is_distinct_from_omission() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]
    predictions = [_prediction("xlam-1", "<tool_call>{bad</tool_call>")]

    scored, failures, summary = score_prediction_records(dataset, predictions)

    assert failures[0]["id"] == "xlam-1"
    assert scored[0]["emission"]["tool_call_emitted"] is True
    assert scored[0]["emission"]["malformed_tool_call"] is True
    assert scored[0]["emission"]["no_tool_call_emitted"] is False
    assert summary["tool_call_emitted_count"] == 1
    assert summary["malformed_tool_call_count"] == 1
    assert summary["no_tool_call_emitted_count"] == 0


def test_missing_prediction_is_scored_as_empty_output() -> None:
    dataset = [_record("xlam-1", [{"name": "weather", "arguments": {"city": "Denver"}}])]

    scored, failures, summary = score_prediction_records(dataset, [])

    assert scored[0]["missing_prediction"] is True
    assert scored[0]["parse"]["errors"] == ["Model output is empty."]
    assert failures[0]["id"] == "xlam-1"
    assert summary["missing_predictions"] == 1
    assert summary["no_tool_call_emitted_count"] == 1
    assert scored[0]["emission"]["prose_only_response"] is False


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


def test_call_level_metrics_preserve_partial_parallel_success() -> None:
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
            '{"name":"weather","arguments":{"city":"Denver"}}',
        )
    ]

    scored, _failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["score"]["complete_call_match"] is False
    assert scored[0]["call_metrics"]["expected_call_count"] == 2
    assert scored[0]["call_metrics"]["predicted_call_count"] == 1
    assert scored[0]["call_metrics"]["matched_call_count"] == 1
    assert scored[0]["call_metrics"]["missing_call_count"] == 1
    assert scored[0]["call_metrics"]["extra_call_count"] == 0
    assert scored[0]["call_metrics"]["function_name_precision"] == 1.0
    assert scored[0]["call_metrics"]["function_name_recall"] == 0.5
    assert summary["matched_call_count"] == 1
    assert summary["missing_call_count"] == 1


def test_schema_equivalent_match_allows_declared_optional_defaults() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "product_reviews",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asin": {"type": "string"},
                        "page": {"type": "integer", "default": "1"},
                        "images_only": {
                            "type": "boolean",
                            "default": "false",
                        },
                    },
                },
            },
        }
    ]
    dataset = [
        _record(
            "xlam-1",
            [
                {
                    "name": "product_reviews",
                    "arguments": {"asin": "B08PPDJWC8"},
                }
            ],
            tools=tools,
        )
    ]
    predictions = [
        _prediction(
            "xlam-1",
            (
                '{"name":"product_reviews","arguments":'
                '{"asin":"B08PPDJWC8","page":1,"images_only":false}}'
            ),
        )
    ]

    scored, _failures, summary = score_prediction_records(dataset, predictions)

    assert scored[0]["headline_scores"]["strict_complete_match"] is False
    assert (
        scored[0]["headline_scores"]["schema_equivalent_complete_match"]
        is True
    )
    assert scored[0]["headline_scores"]["executable_complete_match"] is True
    assert summary["strict_complete_match_count"] == 0
    assert summary["schema_equivalent_complete_match_count"] == 1
    assert summary["executable_complete_match_count"] == 1


def test_schema_equivalent_match_rejects_type_errors() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_all_advisories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "info": {"type": "boolean", "default": "true"},
                    },
                },
            },
        }
    ]
    dataset = [
        _record(
            "xlam-1",
            [
                {
                    "name": "get_all_advisories",
                    "arguments": {"info": True},
                }
            ],
            tools=tools,
        )
    ]
    predictions = [
        _prediction(
            "xlam-1",
            '{"name":"get_all_advisories","arguments":{"info":"true"}}',
        )
    ]

    scored, _failures, summary = score_prediction_records(dataset, predictions)

    match = scored[0]["call_matches"][0]
    assert scored[0]["headline_scores"]["schema_equivalent_complete_match"] is False
    assert match["required_arguments_present"] is True
    assert match["argument_type_accuracy"] == 0.0
    assert match["argument_value_accuracy"] == 0.0
    assert match["schema_validation_success"] is False
    assert summary["schema_validation_success_count"] == 0


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
    assert outputs.requested_metrics_path is not None
    assert outputs.requested_metrics_path.is_file()
    assert outputs.failure_sample_path is not None
    assert outputs.failure_sample_path.is_file()
    assert outputs.summary_markdown_path is not None
    assert outputs.summary_markdown_path.is_file()
    assert outputs.checksums_path is not None
    assert outputs.checksums_path.is_file()
    assert "scores.json" in outputs.checksums_path.read_text(encoding="utf-8")


def test_exp00_qwen3_1_7b_base_regression() -> None:
    dataset = json.loads(
        "["
        + ",".join(
            line
            for line in Path("data/smoke/normalized/test.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        + "]"
    )
    predictions = json.loads(
        "["
        + ",".join(
            line
            for line in Path(
                "tests/fixtures/exp00/qwen3_1_7b_base_predictions.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        + "]"
    )

    _scored, _failures, summary = score_prediction_records(
        dataset,
        predictions,
    )

    assert summary["total_records"] == 40
    assert summary["strict_complete_match_count"] == 22
    assert summary["strict_complete_match_rate"] == 0.55
    assert summary["schema_equivalent_complete_match_count"] == 24
    assert summary["executable_complete_match_count"] == 24
    assert summary["expected_call_count"] == 65
    assert summary["predicted_call_count"] == 49
    assert summary["matched_call_count"] == 49
    assert summary["missing_call_count"] == 16
    assert summary["extra_call_count"] == 0
    assert summary["malformed_tool_call_count"] == 0
    assert summary["function_name_precision"] == 1.0
