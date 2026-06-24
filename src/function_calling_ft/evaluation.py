from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from function_calling_ft.dataset import extract_expected_tool_calls
from function_calling_ft.parser import ParseResult, ToolCall, parse_tool_calls
from function_calling_ft.scorer import CallSetScore, score_calls


SCORED_PREDICTIONS_FILENAME = "scored_predictions.jsonl"
PARSE_FAILURES_FILENAME = "parse_failures.jsonl"
SCORES_FILENAME = "scores.json"


@dataclass(frozen=True)
class EvaluationOutputs:
    scored_predictions_path: Path
    parse_failures_path: Path
    scores_path: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON",
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object",
                )
            records.append(record)

    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True),
            )
            file.write("\n")


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "name": call.name,
        "arguments": call.arguments,
    }


def _parse_result_to_dict(result: ParseResult) -> dict[str, Any]:
    return {
        "valid_structure": result.valid_structure,
        "errors": list(result.errors),
        "had_extra_prose": result.had_extra_prose,
        "calls": [_call_to_dict(call) for call in result.calls],
    }


def _score_to_dict(score: CallSetScore) -> dict[str, Any]:
    return {
        "valid_structure": score.valid_structure,
        "correct_function_name": score.correct_function_name,
        "correct_argument_names": score.correct_argument_names,
        "correct_argument_values": score.correct_argument_values,
        "complete_call_match": score.complete_call_match,
        "predicted_count": score.predicted_count,
        "expected_count": score.expected_count,
        "parse_errors": list(score.parse_errors),
        "order_matters": score.order_matters,
    }


def _source_id(record: dict[str, Any]) -> int | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_id")
    return int(value) if value is not None else None


def _prediction_id(prediction: dict[str, Any]) -> str:
    value = prediction.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Each prediction must contain a non-empty string id")
    return value


def index_predictions(
    predictions: Iterable[dict[str, Any]],
    *,
    dataset_ids: set[str],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        prediction_id = _prediction_id(prediction)

        if prediction_id in indexed:
            raise ValueError(f"Duplicate prediction id: {prediction_id}")

        if prediction_id not in dataset_ids:
            raise ValueError(
                f"Prediction id is absent from dataset: {prediction_id}",
            )

        indexed[prediction_id] = prediction

    return indexed


def score_prediction_records(
    dataset_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    *,
    order_matters: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_ids = {str(record["id"]) for record in dataset_records}
    predictions_by_id = index_predictions(
        prediction_records,
        dataset_ids=dataset_ids,
    )
    scored_records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []

    counts = {
        "valid_structure_count": 0,
        "parse_failure_count": 0,
        "extra_prose_count": 0,
        "correct_function_name_count": 0,
        "correct_argument_names_count": 0,
        "correct_argument_values_count": 0,
        "complete_match_count": 0,
    }

    for dataset_record in dataset_records:
        record_id = str(dataset_record["id"])
        prediction = predictions_by_id.get(record_id)
        missing_prediction = prediction is None
        raw_generation = ""
        generation_error = None

        if prediction is not None:
            raw_value = prediction.get("raw_generation", "")
            raw_generation = raw_value if isinstance(raw_value, str) else ""
            error_value = prediction.get("generation_error")
            generation_error = (
                str(error_value) if error_value is not None else None
            )

        parse_result = parse_tool_calls(raw_generation)
        expected_calls = extract_expected_tool_calls(dataset_record)
        score = score_calls(
            parse_result,
            list(expected_calls),
            order_matters=order_matters,
        )

        if score.valid_structure:
            counts["valid_structure_count"] += 1
        else:
            counts["parse_failure_count"] += 1

        if parse_result.had_extra_prose:
            counts["extra_prose_count"] += 1

        if score.correct_function_name:
            counts["correct_function_name_count"] += 1

        if score.correct_argument_names:
            counts["correct_argument_names_count"] += 1

        if score.correct_argument_values:
            counts["correct_argument_values_count"] += 1

        if score.complete_call_match:
            counts["complete_match_count"] += 1

        scored_record = {
            "id": record_id,
            "source_id": _source_id(dataset_record),
            "missing_prediction": missing_prediction,
            "generation_error": generation_error,
            "raw_generation": raw_generation,
            "expected_calls": list(expected_calls),
            "parse": _parse_result_to_dict(parse_result),
            "score": _score_to_dict(score),
        }
        scored_records.append(scored_record)

        if (
            missing_prediction
            or generation_error is not None
            or not parse_result.valid_structure
            or parse_result.errors
        ):
            parse_failures.append(scored_record)

    total_records = len(dataset_records)
    predictions_present = len(predictions_by_id)
    missing_predictions = total_records - predictions_present

    summary = {
        "total_records": total_records,
        "predictions_present": predictions_present,
        "missing_predictions": missing_predictions,
        **counts,
        "valid_structure_rate": (
            counts["valid_structure_count"] / total_records
            if total_records
            else 0.0
        ),
        "complete_match_rate": (
            counts["complete_match_count"] / total_records
            if total_records
            else 0.0
        ),
        "order_matters": order_matters,
    }

    return scored_records, parse_failures, summary


def evaluate_predictions(
    *,
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
    order_matters: bool = False,
) -> EvaluationOutputs:
    dataset_records = read_jsonl(dataset_path)
    prediction_records = (
        read_jsonl(predictions_path)
        if predictions_path.exists()
        else []
    )
    scored_records, parse_failures, summary = score_prediction_records(
        dataset_records,
        prediction_records,
        order_matters=order_matters,
    )

    scored_path = output_dir / SCORED_PREDICTIONS_FILENAME
    failures_path = output_dir / PARSE_FAILURES_FILENAME
    scores_path = output_dir / SCORES_FILENAME

    write_jsonl(scored_path, scored_records)
    write_jsonl(failures_path, parse_failures)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return EvaluationOutputs(
        scored_predictions_path=scored_path,
        parse_failures_path=failures_path,
        scores_path=scores_path,
    )
