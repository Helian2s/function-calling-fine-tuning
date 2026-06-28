from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from function_calling_ft.exp00_completion import (
    REQUIRED_FINAL_RESULT_FILES,
    build_completion_report,
    verify_scores_from_predictions,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _dataset_record() -> dict[str, Any]:
    return {
        "id": "xlam-1",
        "tools": [],
        "messages": [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Denver"},
                        },
                    }
                ],
            },
        ],
        "metadata": {"source_id": 1},
    }


def _prediction() -> dict[str, Any]:
    return {
        "id": "xlam-1",
        "raw_generation": '{"name":"weather","arguments":{"city":"Denver"}}',
        "generation_error": None,
    }


def test_verify_scores_recomputes_predictions_without_overwriting(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.json"
    _write_jsonl(dataset, [_dataset_record()])
    _write_jsonl(predictions, [_prediction()])
    _write_json(
        scores,
        {
            "total_records": 1,
            "predictions_present": 1,
            "missing_predictions": 0,
            "valid_structure_count": 1,
            "parse_failure_count": 0,
            "extra_prose_count": 0,
            "correct_function_name_count": 1,
            "correct_argument_names_count": 1,
            "correct_argument_values_count": 1,
            "complete_match_count": 1,
            "tool_call_emitted_count": 1,
            "no_tool_call_emitted_count": 0,
            "malformed_tool_call_count": 0,
            "parseable_given_emission_count": 1,
            "prose_only_response_count": 0,
            "extra_prose_with_tool_call_count": 0,
            "strict_complete_match_count": 1,
            "schema_equivalent_complete_match_count": 1,
            "executable_complete_match_count": 1,
            "expected_call_count": 1,
            "predicted_call_count": 1,
            "matched_call_count": 1,
            "missing_call_count": 0,
            "extra_call_count": 0,
            "strict_complete_call_count": 1,
            "schema_equivalent_complete_call_count": 1,
            "executable_complete_call_count": 1,
            "required_arguments_present_count": 1,
            "missing_required_argument_count": 0,
            "undeclared_argument_count": 0,
            "enum_validity_count": 1,
            "schema_validation_success_count": 1,
            "parseable_given_emission_rate": 1.0,
            "valid_structure_rate": 1.0,
            "complete_match_rate": 1.0,
            "strict_complete_match_rate": 1.0,
            "schema_equivalent_complete_match_rate": 1.0,
            "executable_complete_match_rate": 1.0,
            "function_name_precision": 1.0,
            "function_name_recall": 1.0,
            "function_name_f1": 1.0,
            "complete_call_precision": 1.0,
            "complete_call_recall": 1.0,
            "complete_call_f1": 1.0,
            "schema_validation_success_rate": 1.0,
            "average_argument_name_accuracy": 1.0,
            "average_argument_type_accuracy": 1.0,
            "average_argument_value_accuracy": 1.0,
            "order_matters": False,
        },
    )

    result = verify_scores_from_predictions(
        dataset_path=dataset,
        predictions_path=predictions,
        scores_path=scores,
    )

    assert result["status"] == "pass"
    assert result["prediction_sha256"]


def test_verify_scores_marks_mismatched_scores_invalid(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.json"
    _write_jsonl(dataset, [_dataset_record()])
    _write_jsonl(predictions, [_prediction()])
    _write_json(scores, {"total_records": 999})

    result = verify_scores_from_predictions(
        dataset_path=dataset,
        predictions_path=predictions,
        scores_path=scores,
    )

    assert result["status"] == "fail"
    assert "total_records" in result["mismatches"]


def test_completion_report_marks_missing_artifacts_incomplete(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    results = tmp_path / "results"
    baseline = results / "baseline"
    logs = tmp_path / "logs"
    run_info = tmp_path / "run-info"
    adapter = tmp_path / "adapter"
    template = tmp_path / "template.json"
    loss_mask = tmp_path / "loss_mask.json"
    _write_jsonl(dataset, [_dataset_record()])
    _write_json(
        template,
        {
            "examples_rendered": 5,
            "thinking_mode_enabled": False,
            "total_failures": 0,
            "model_name": "Qwen/Qwen3-1.7B",
            "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        },
    )
    _write_json(
        loss_mask,
        {
            "thinking_mode_enabled": False,
            "smoke_examples": [
                {
                    "included_token_count": 1,
                    "spans": [
                        {
                            "region": "assistant_tool_call",
                            "include_in_loss": True,
                        }
                    ],
                }
            ],
        },
    )

    report = build_completion_report(
        dataset_path=dataset,
        results_dir=results,
        baseline_results_dir=baseline,
        logs_dir=logs,
        run_info_dir=run_info,
        adapter_dir=adapter,
        template_report_path=template,
        loss_mask_report_path=loss_mask,
    )

    statuses = {stage["name"]: stage["status"] for stage in report["stages"]}
    assert report["overall_status"] == "incomplete"
    assert statuses["native_template_rendering"] == "pass"
    assert statuses["loss_mask"] == "pass"
    assert statuses["canonical_artifact_bundle"] == "missing"
    assert set(
        report["stages"][10]["evidence"]["missing"],
    ) == set(REQUIRED_FINAL_RESULT_FILES)
