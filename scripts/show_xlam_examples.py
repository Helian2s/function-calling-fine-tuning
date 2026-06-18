from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


RAW_DATASET_PATH = Path(
    "data/raw/xlam/xlam_function_calling_60k.json"
)
OUTPUT_PATH = Path(
    "data/raw/xlam/examples_for_analysis.json"
)

RANDOM_SEED = 42


def decode_json_field(
    value: Any,
    *,
    field_name: str,
    row_id: Any,
) -> Any:
    """Decode tools or answers while accepting already-decoded values."""
    if isinstance(value, (list, dict)):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"Row {row_id}: {field_name} must be a JSON string, "
            f"list, or dict; received {type(value).__name__}"
        )

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Row {row_id}: {field_name} contains invalid JSON. "
            f"Preview: {value[:200]!r}"
        ) from exc


def load_raw_dataset() -> Dataset:
    if not RAW_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATASET_PATH}\n"
            "Run scripts/download_xlam.py first."
        )

    return load_dataset(
        "json",
        data_files=str(RAW_DATASET_PATH),
        split="train",
    )


def parse_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    row_id = row.get("id", row_index)
    query = row.get("query")

    if not isinstance(query, str):
        raise TypeError(
            f"Row {row_id}: query must be a string; "
            f"received {type(query).__name__}"
        )

    tools = decode_json_field(
        row.get("tools"),
        field_name="tools",
        row_id=row_id,
    )

    answers = decode_json_field(
        row.get("answers"),
        field_name="answers",
        row_id=row_id,
    )

    if not isinstance(tools, list):
        raise TypeError(
            f"Row {row_id}: decoded tools must be a list."
        )

    if not isinstance(answers, list):
        raise TypeError(
            f"Row {row_id}: decoded answers must be a list."
        )

    return {
        "row_index": row_index,
        "id": row_id,
        "query": query,
        "tools": tools,
        "answers": answers,
    }


def extract_parameter_types(tools: list[Any]) -> list[str]:
    parameter_types: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            continue

        for specification in parameters.values():
            if not isinstance(specification, dict):
                continue

            parameter_type = specification.get("type")
            if parameter_type is not None:
                parameter_types.add(str(parameter_type))

    return sorted(parameter_types)


def has_complex_parameter(tools: list[Any]) -> bool:
    complex_type_terms = {
        "array",
        "dict",
        "dictionary",
        "list",
        "object",
    }

    for parameter_type in extract_parameter_types(tools):
        normalized = parameter_type.lower()

        if any(term in normalized for term in complex_type_terms):
            return True

    return False


def derive_analysis(record: dict[str, Any]) -> dict[str, Any]:
    tools = record["tools"]
    answers = record["answers"]

    available_tool_names = {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool, dict) and tool.get("name") is not None
    }

    answer_tool_names = [
        str(answer.get("name"))
        for answer in answers
        if isinstance(answer, dict) and answer.get("name") is not None
    ]

    unknown_answer_tools = sorted(
        set(answer_tool_names) - available_tool_names
    )

    answer_argument_keys: list[list[str]] = []

    for answer in answers:
        arguments = (
            answer.get("arguments", {})
            if isinstance(answer, dict)
            else {}
        )

        if isinstance(arguments, dict):
            answer_argument_keys.append(sorted(arguments))
        else:
            answer_argument_keys.append([])

    return {
        "available_tool_count": len(tools),
        "expected_call_count": len(answers),
        "available_tool_names": sorted(available_tool_names),
        "expected_tool_names": answer_tool_names,
        "unknown_answer_tools": unknown_answer_tools,
        "parameter_types": extract_parameter_types(tools),
        "answer_argument_keys": answer_argument_keys,
        "multi_call_requires_manual_dependency_review": (
            len(answers) > 1
        ),
    }


Predicate = Callable[[dict[str, Any]], bool]


def main() -> None:
    dataset = load_raw_dataset()

    indices = list(range(len(dataset)))
    random.Random(RANDOM_SEED).shuffle(indices)

    targets: list[tuple[str, Predicate]] = [
        (
            "single_call",
            lambda record: len(record["answers"]) == 1,
        ),
        (
            "exactly_two_calls",
            lambda record: len(record["answers"]) == 2,
        ),
        (
            "three_or_more_calls",
            lambda record: len(record["answers"]) >= 3,
        ),
        (
            "many_available_tools",
            lambda record: len(record["tools"]) >= 5,
        ),
        (
            "complex_parameter_type",
            lambda record: has_complex_parameter(record["tools"]),
        ),
    ]

    selected_records: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    for category, predicate in targets:
        matching_record: dict[str, Any] | None = None

        for index in indices:
            if index in selected_indices:
                continue

            record = parse_row(dataset[index], index)

            if predicate(record):
                matching_record = record
                selected_indices.add(index)
                break

        if matching_record is None:
            print(f"Warning: no example found for category {category!r}")
            continue

        selected_records.append(
            {
                "selection_category": category,
                **matching_record,
                "derived_analysis": derive_analysis(matching_record),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            selected_records,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for record in selected_records:
        print("=" * 100)
        print(f"CATEGORY: {record['selection_category']}")
        print(f"ROW INDEX: {record['row_index']}")
        print(f"DATASET ID: {record['id']}")
        print("-" * 100)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print()

    print(f"Selected {len(selected_records)} records.")
    print(f"Full output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()