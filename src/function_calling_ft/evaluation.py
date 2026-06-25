from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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


@dataclass(frozen=True)
class EmissionClassification:
    tool_call_emitted: bool
    no_tool_call_emitted: bool
    malformed_tool_call: bool
    parseable_given_emission: bool
    prose_only_response: bool
    extra_prose_with_tool_call: bool


@dataclass(frozen=True)
class CallComparison:
    expected_index: int
    predicted_index: int
    expected_call: ToolCall
    predicted_call: ToolCall
    function_name_match: bool
    strict_complete_match: bool
    schema_equivalent_complete_match: bool
    executable_complete_match: bool
    required_arguments_present: bool
    missing_required_arguments: tuple[str, ...]
    undeclared_arguments: tuple[str, ...]
    argument_name_accuracy: float
    argument_type_accuracy: float
    argument_value_accuracy: float
    enum_validity: bool
    schema_validation_success: bool


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


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _tool_call_blocks(text: str) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    cursor = 0

    while True:
        start = text.find(start_tag, cursor)
        if start == -1:
            return blocks

        end = text.find(end_tag, start + len(start_tag))
        if end == -1:
            blocks.append((start, len(text)))
            return blocks

        blocks.append((start, end + len(end_tag)))
        cursor = end + len(end_tag)


def _has_prose_outside_tool_blocks(text: str) -> bool:
    blocks = _tool_call_blocks(text)
    if not blocks:
        return False

    cursor = 0
    outside: list[str] = []
    for start, end in blocks:
        outside.append(text[cursor:start])
        cursor = end
    outside.append(text[cursor:])

    return bool("".join(outside).strip())


def classify_emission(
    *,
    raw_generation: str,
    parse_result: ParseResult,
) -> EmissionClassification:
    has_tool_tag = bool(_tool_call_blocks(raw_generation))
    tool_call_emitted = has_tool_tag or bool(parse_result.calls)
    no_tool_call_emitted = not tool_call_emitted
    malformed_tool_call = tool_call_emitted and not parse_result.valid_structure
    parseable_given_emission = (
        tool_call_emitted and parse_result.valid_structure
    )
    prose_only_response = no_tool_call_emitted and bool(
        raw_generation.strip(),
    )

    if has_tool_tag:
        extra_prose_with_tool_call = _has_prose_outside_tool_blocks(
            raw_generation,
        )
    else:
        extra_prose_with_tool_call = (
            tool_call_emitted and parse_result.had_extra_prose
        )

    return EmissionClassification(
        tool_call_emitted=tool_call_emitted,
        no_tool_call_emitted=no_tool_call_emitted,
        malformed_tool_call=malformed_tool_call,
        parseable_given_emission=parseable_given_emission,
        prose_only_response=prose_only_response,
        extra_prose_with_tool_call=extra_prose_with_tool_call,
    )


def _emission_to_dict(
    classification: EmissionClassification,
) -> dict[str, bool]:
    return {
        "tool_call_emitted": classification.tool_call_emitted,
        "no_tool_call_emitted": classification.no_tool_call_emitted,
        "malformed_tool_call": classification.malformed_tool_call,
        "parseable_given_emission": (
            classification.parseable_given_emission
        ),
        "prose_only_response": classification.prose_only_response,
        "extra_prose_with_tool_call": (
            classification.extra_prose_with_tool_call
        ),
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


def _tool_schemas_by_name(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}

    for tool in record.get("tools", []):
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            schemas[name] = parameters

    return schemas


def _schema_properties(schema: dict[str, Any] | None) -> dict[str, Any]:
    if schema is None:
        return {}

    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _schema_required_names(
    schema: dict[str, Any] | None,
) -> set[str]:
    if schema is None:
        return set()

    required = schema.get("required")
    if not isinstance(required, list):
        return set()

    return {item for item in required if isinstance(item, str)}


def _schema_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    return schema_type if isinstance(schema_type, str) else None


def _matches_schema_type(value: Any, schema: dict[str, Any]) -> bool:
    schema_type = _schema_type(schema)

    if schema_type is None:
        return True

    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
        )
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)

    return True


_NO_DEFAULT = object()


def _typed_schema_default(schema: dict[str, Any]) -> Any:
    if "default" not in schema:
        return _NO_DEFAULT

    default = schema["default"]
    schema_type = _schema_type(schema)

    if schema_type == "boolean" and isinstance(default, str):
        lowered = default.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if schema_type == "integer" and isinstance(default, str):
        try:
            return int(default)
        except ValueError:
            return default
    if schema_type == "number" and isinstance(default, str):
        try:
            number = float(default)
        except ValueError:
            return default
        return int(number) if number.is_integer() else number

    return default


def _canonicalize_default_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    properties = _schema_properties(schema)
    canonical = dict(arguments)

    for name in list(canonical):
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            continue

        default = _typed_schema_default(property_schema)
        if default is _NO_DEFAULT:
            continue

        value = canonical[name]
        if (
            value == default
            and _matches_schema_type(value, property_schema)
        ):
            del canonical[name]

    return canonical


def _same_json_type(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool)
    if isinstance(left, int) and isinstance(right, int):
        return True
    if isinstance(left, float) and isinstance(right, float | int):
        return not isinstance(right, bool)
    if isinstance(left, int) and isinstance(right, float):
        return True
    return type(left) is type(right)


def _enum_validity(
    predicted_arguments: dict[str, Any],
    properties: dict[str, Any],
) -> bool:
    for name, value in predicted_arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            continue
        enum_values = property_schema.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            return False

    return True


def _argument_diagnostics(
    *,
    predicted_call: ToolCall,
    expected_call: ToolCall,
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    predicted_arguments = (
        predicted_call.arguments
        if isinstance(predicted_call.arguments, dict)
        else {}
    )
    expected_arguments = (
        expected_call.arguments
        if isinstance(expected_call.arguments, dict)
        else {}
    )
    properties = _schema_properties(schema)
    expected_names = set(expected_arguments)
    predicted_names = set(predicted_arguments)
    required_names = expected_names | _schema_required_names(schema)
    missing_required = tuple(sorted(required_names - predicted_names))

    if properties:
        undeclared_arguments = tuple(
            sorted(predicted_names - set(properties)),
        )
    else:
        undeclared_arguments = ()

    all_names = expected_names | predicted_names
    argument_name_accuracy = (
        _safe_rate(len(expected_names & predicted_names), len(all_names))
        if all_names
        else 1.0
    )

    if expected_names:
        type_matches = sum(
            int(
                name in predicted_arguments
                and _same_json_type(
                    predicted_arguments[name],
                    expected_arguments[name],
                )
            )
            for name in expected_names
        )
        value_matches = sum(
            int(
                name in predicted_arguments
                and predicted_arguments[name] == expected_arguments[name]
            )
            for name in expected_names
        )
        argument_type_accuracy = _safe_rate(type_matches, len(expected_names))
        argument_value_accuracy = _safe_rate(
            value_matches,
            len(expected_names),
        )
    else:
        argument_type_accuracy = 1.0 if not predicted_arguments else 0.0
        argument_value_accuracy = 1.0 if not predicted_arguments else 0.0

    predicted_schema_types_valid = all(
        not isinstance(properties.get(name), dict)
        or _matches_schema_type(
            value,
            properties[name],
        )
        for name, value in predicted_arguments.items()
    )
    enum_validity = _enum_validity(predicted_arguments, properties)
    schema_validation_success = (
        not missing_required
        and not undeclared_arguments
        and predicted_schema_types_valid
        and enum_validity
    )

    return {
        "required_arguments_present": not missing_required,
        "missing_required_arguments": missing_required,
        "undeclared_arguments": undeclared_arguments,
        "argument_name_accuracy": argument_name_accuracy,
        "argument_type_accuracy": argument_type_accuracy,
        "argument_value_accuracy": argument_value_accuracy,
        "enum_validity": enum_validity,
        "schema_validation_success": schema_validation_success,
    }


def _compare_calls(
    *,
    expected_index: int,
    predicted_index: int,
    expected_call: ToolCall,
    predicted_call: ToolCall,
    schema: dict[str, Any] | None,
) -> CallComparison:
    function_name_match = (
        expected_call.name is not None
        and predicted_call.name is not None
        and expected_call.name == predicted_call.name
    )
    expected_arguments = (
        expected_call.arguments
        if isinstance(expected_call.arguments, dict)
        else {}
    )
    predicted_arguments = (
        predicted_call.arguments
        if isinstance(predicted_call.arguments, dict)
        else {}
    )
    strict_complete_match = (
        function_name_match
        and predicted_arguments == expected_arguments
    )
    canonical_expected = _canonicalize_default_arguments(
        expected_arguments,
        schema,
    )
    canonical_predicted = _canonicalize_default_arguments(
        predicted_arguments,
        schema,
    )
    schema_equivalent_complete_match = (
        function_name_match
        and canonical_predicted == canonical_expected
    )
    diagnostics = _argument_diagnostics(
        predicted_call=predicted_call,
        expected_call=expected_call,
        schema=schema,
    )
    executable_complete_match = (
        schema_equivalent_complete_match
        and bool(diagnostics["schema_validation_success"])
    )

    return CallComparison(
        expected_index=expected_index,
        predicted_index=predicted_index,
        expected_call=expected_call,
        predicted_call=predicted_call,
        function_name_match=function_name_match,
        strict_complete_match=strict_complete_match,
        schema_equivalent_complete_match=schema_equivalent_complete_match,
        executable_complete_match=executable_complete_match,
        required_arguments_present=bool(
            diagnostics["required_arguments_present"],
        ),
        missing_required_arguments=diagnostics[
            "missing_required_arguments"
        ],
        undeclared_arguments=diagnostics["undeclared_arguments"],
        argument_name_accuracy=float(
            diagnostics["argument_name_accuracy"],
        ),
        argument_type_accuracy=float(
            diagnostics["argument_type_accuracy"],
        ),
        argument_value_accuracy=float(
            diagnostics["argument_value_accuracy"],
        ),
        enum_validity=bool(diagnostics["enum_validity"]),
        schema_validation_success=bool(
            diagnostics["schema_validation_success"],
        ),
    )


def _call_comparison_weight(comparison: CallComparison) -> int:
    if not comparison.function_name_match:
        return 0

    return (
        int(comparison.strict_complete_match) * 1_000_000
        + int(comparison.schema_equivalent_complete_match) * 100_000
        + 10_000
        + int(comparison.argument_value_accuracy * 1_000)
        + int(comparison.argument_name_accuracy * 100)
    )


def _match_call_comparisons(
    *,
    predicted_calls: tuple[ToolCall, ...],
    expected_calls: tuple[ToolCall, ...],
    schemas_by_name: dict[str, dict[str, Any]],
    order_matters: bool,
) -> tuple[CallComparison, ...]:
    comparison_matrix = [
        [
            _compare_calls(
                expected_index=expected_index,
                predicted_index=predicted_index,
                expected_call=expected_call,
                predicted_call=predicted_call,
                schema=(
                    schemas_by_name.get(expected_call.name)
                    if expected_call.name is not None
                    else None
                ),
            )
            for predicted_index, predicted_call in enumerate(predicted_calls)
        ]
        for expected_index, expected_call in enumerate(expected_calls)
    ]

    if order_matters:
        ordered_matches: list[CallComparison] = []
        for index in range(min(len(expected_calls), len(predicted_calls))):
            comparison = comparison_matrix[index][index]
            if comparison.function_name_match:
                ordered_matches.append(comparison)
        return tuple(ordered_matches)

    @lru_cache(maxsize=None)
    def best_assignment(
        expected_index: int,
        used_mask: int,
    ) -> tuple[int, tuple[int | None, ...]]:
        if expected_index == len(expected_calls):
            return 0, ()

        best_weight, best_indices = best_assignment(
            expected_index + 1,
            used_mask,
        )
        best_indices = (None,) + best_indices

        for predicted_index in range(len(predicted_calls)):
            if used_mask & (1 << predicted_index):
                continue

            comparison = comparison_matrix[expected_index][predicted_index]
            comparison_weight = _call_comparison_weight(comparison)
            if comparison_weight == 0:
                continue

            remaining_weight, remaining_indices = best_assignment(
                expected_index + 1,
                used_mask | (1 << predicted_index),
            )
            total_weight = comparison_weight + remaining_weight

            if total_weight > best_weight:
                best_weight = total_weight
                best_indices = (predicted_index,) + remaining_indices

        return best_weight, best_indices

    _, assignment = best_assignment(0, 0)
    matches: list[CallComparison] = []
    for expected_index, predicted_index in enumerate(assignment):
        if predicted_index is not None:
            matches.append(comparison_matrix[expected_index][predicted_index])

    return tuple(matches)


def _comparison_to_dict(
    comparison: CallComparison,
) -> dict[str, Any]:
    return {
        "expected_index": comparison.expected_index,
        "predicted_index": comparison.predicted_index,
        "expected_call": _call_to_dict(comparison.expected_call),
        "predicted_call": _call_to_dict(comparison.predicted_call),
        "function_name_match": comparison.function_name_match,
        "strict_complete_match": comparison.strict_complete_match,
        "schema_equivalent_complete_match": (
            comparison.schema_equivalent_complete_match
        ),
        "executable_complete_match": comparison.executable_complete_match,
        "required_arguments_present": (
            comparison.required_arguments_present
        ),
        "missing_required_arguments": list(
            comparison.missing_required_arguments,
        ),
        "undeclared_arguments": list(comparison.undeclared_arguments),
        "argument_name_accuracy": comparison.argument_name_accuracy,
        "argument_type_accuracy": comparison.argument_type_accuracy,
        "argument_value_accuracy": comparison.argument_value_accuracy,
        "enum_validity": comparison.enum_validity,
        "schema_validation_success": comparison.schema_validation_success,
    }


def _call_level_metrics(
    *,
    predicted_calls: tuple[ToolCall, ...],
    expected_calls: tuple[ToolCall, ...],
    matches: tuple[CallComparison, ...],
) -> dict[str, Any]:
    expected_call_count = len(expected_calls)
    predicted_call_count = len(predicted_calls)
    matched_call_count = len(matches)
    strict_complete_count = sum(
        int(match.strict_complete_match) for match in matches
    )
    schema_equivalent_complete_count = sum(
        int(match.schema_equivalent_complete_match) for match in matches
    )
    executable_complete_count = sum(
        int(match.executable_complete_match) for match in matches
    )

    function_name_precision = _safe_rate(
        matched_call_count,
        predicted_call_count,
    )
    function_name_recall = _safe_rate(
        matched_call_count,
        expected_call_count,
    )
    complete_call_precision = _safe_rate(
        strict_complete_count,
        predicted_call_count,
    )
    complete_call_recall = _safe_rate(
        strict_complete_count,
        expected_call_count,
    )

    return {
        "expected_call_count": expected_call_count,
        "predicted_call_count": predicted_call_count,
        "matched_call_count": matched_call_count,
        "missing_call_count": expected_call_count - matched_call_count,
        "extra_call_count": predicted_call_count - matched_call_count,
        "strict_complete_call_count": strict_complete_count,
        "schema_equivalent_complete_call_count": (
            schema_equivalent_complete_count
        ),
        "executable_complete_call_count": executable_complete_count,
        "function_name_precision": function_name_precision,
        "function_name_recall": function_name_recall,
        "function_name_f1": _f1(
            function_name_precision,
            function_name_recall,
        ),
        "complete_call_precision": complete_call_precision,
        "complete_call_recall": complete_call_recall,
        "complete_call_f1": _f1(
            complete_call_precision,
            complete_call_recall,
        ),
    }


def _headline_scores(
    *,
    expected_call_count: int,
    predicted_call_count: int,
    matches: tuple[CallComparison, ...],
) -> dict[str, bool]:
    counts_match = (
        expected_call_count == predicted_call_count == len(matches)
        and expected_call_count > 0
    )

    return {
        "strict_complete_match": (
            counts_match
            and all(match.strict_complete_match for match in matches)
        ),
        "schema_equivalent_complete_match": (
            counts_match
            and all(
                match.schema_equivalent_complete_match
                for match in matches
            )
        ),
        "executable_complete_match": (
            counts_match
            and all(match.executable_complete_match for match in matches)
        ),
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
        "tool_call_emitted_count": 0,
        "no_tool_call_emitted_count": 0,
        "malformed_tool_call_count": 0,
        "parseable_given_emission_count": 0,
        "prose_only_response_count": 0,
        "extra_prose_with_tool_call_count": 0,
        "strict_complete_match_count": 0,
        "schema_equivalent_complete_match_count": 0,
        "executable_complete_match_count": 0,
        "expected_call_count": 0,
        "predicted_call_count": 0,
        "matched_call_count": 0,
        "missing_call_count": 0,
        "extra_call_count": 0,
        "strict_complete_call_count": 0,
        "schema_equivalent_complete_call_count": 0,
        "executable_complete_call_count": 0,
        "required_arguments_present_count": 0,
        "missing_required_argument_count": 0,
        "undeclared_argument_count": 0,
        "enum_validity_count": 0,
        "schema_validation_success_count": 0,
    }
    argument_name_accuracy_sum = 0.0
    argument_type_accuracy_sum = 0.0
    argument_value_accuracy_sum = 0.0
    argument_comparison_count = 0

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
        emission = classify_emission(
            raw_generation=raw_generation,
            parse_result=parse_result,
        )
        expected_parse_result = parse_tool_calls(list(expected_calls))
        schemas_by_name = _tool_schemas_by_name(dataset_record)
        call_matches = _match_call_comparisons(
            predicted_calls=parse_result.calls,
            expected_calls=expected_parse_result.calls,
            schemas_by_name=schemas_by_name,
            order_matters=order_matters,
        )
        call_metrics = _call_level_metrics(
            predicted_calls=parse_result.calls,
            expected_calls=expected_parse_result.calls,
            matches=call_matches,
        )
        headline_scores = _headline_scores(
            expected_call_count=call_metrics["expected_call_count"],
            predicted_call_count=call_metrics["predicted_call_count"],
            matches=call_matches,
        )

        if score.valid_structure:
            counts["valid_structure_count"] += 1
        else:
            counts["parse_failure_count"] += 1

        if emission.extra_prose_with_tool_call:
            counts["extra_prose_count"] += 1

        if score.correct_function_name:
            counts["correct_function_name_count"] += 1

        if score.correct_argument_names:
            counts["correct_argument_names_count"] += 1

        if score.correct_argument_values:
            counts["correct_argument_values_count"] += 1

        if score.complete_call_match:
            counts["complete_match_count"] += 1

        for key, value in _emission_to_dict(emission).items():
            if value:
                counts[f"{key}_count"] += 1

        for key, value in headline_scores.items():
            if value:
                counts[f"{key}_count"] += 1

        for key in (
            "expected_call_count",
            "predicted_call_count",
            "matched_call_count",
            "missing_call_count",
            "extra_call_count",
            "strict_complete_call_count",
            "schema_equivalent_complete_call_count",
            "executable_complete_call_count",
        ):
            counts[key] += int(call_metrics[key])

        for match in call_matches:
            counts["required_arguments_present_count"] += int(
                match.required_arguments_present,
            )
            counts["missing_required_argument_count"] += len(
                match.missing_required_arguments,
            )
            counts["undeclared_argument_count"] += len(
                match.undeclared_arguments,
            )
            counts["enum_validity_count"] += int(match.enum_validity)
            counts["schema_validation_success_count"] += int(
                match.schema_validation_success,
            )
            argument_name_accuracy_sum += match.argument_name_accuracy
            argument_type_accuracy_sum += match.argument_type_accuracy
            argument_value_accuracy_sum += match.argument_value_accuracy
            argument_comparison_count += 1

        scored_record = {
            "id": record_id,
            "source_id": _source_id(dataset_record),
            "missing_prediction": missing_prediction,
            "generation_error": generation_error,
            "raw_generation": raw_generation,
            "expected_calls": list(expected_calls),
            "parse": _parse_result_to_dict(parse_result),
            "emission": _emission_to_dict(emission),
            "score": _score_to_dict(score),
            "headline_scores": headline_scores,
            "call_metrics": call_metrics,
            "call_matches": [
                _comparison_to_dict(match) for match in call_matches
            ],
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
    function_name_precision = _safe_rate(
        counts["matched_call_count"],
        counts["predicted_call_count"],
    )
    function_name_recall = _safe_rate(
        counts["matched_call_count"],
        counts["expected_call_count"],
    )
    complete_call_precision = _safe_rate(
        counts["strict_complete_call_count"],
        counts["predicted_call_count"],
    )
    complete_call_recall = _safe_rate(
        counts["strict_complete_call_count"],
        counts["expected_call_count"],
    )

    summary = {
        "total_records": total_records,
        "predictions_present": predictions_present,
        "missing_predictions": missing_predictions,
        **counts,
        "parseable_given_emission_rate": _safe_rate(
            counts["parseable_given_emission_count"],
            counts["tool_call_emitted_count"],
        ),
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
        "strict_complete_match_rate": _safe_rate(
            counts["strict_complete_match_count"],
            total_records,
        ),
        "schema_equivalent_complete_match_rate": _safe_rate(
            counts["schema_equivalent_complete_match_count"],
            total_records,
        ),
        "executable_complete_match_rate": _safe_rate(
            counts["executable_complete_match_count"],
            total_records,
        ),
        "function_name_precision": function_name_precision,
        "function_name_recall": function_name_recall,
        "function_name_f1": _f1(
            function_name_precision,
            function_name_recall,
        ),
        "complete_call_precision": complete_call_precision,
        "complete_call_recall": complete_call_recall,
        "complete_call_f1": _f1(
            complete_call_precision,
            complete_call_recall,
        ),
        "schema_validation_success_rate": _safe_rate(
            counts["schema_validation_success_count"],
            counts["matched_call_count"],
        ),
        "average_argument_name_accuracy": _safe_rate(
            argument_name_accuracy_sum,
            argument_comparison_count,
        ),
        "average_argument_type_accuracy": _safe_rate(
            argument_type_accuracy_sum,
            argument_comparison_count,
        ),
        "average_argument_value_accuracy": _safe_rate(
            argument_value_accuracy_sum,
            argument_comparison_count,
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
