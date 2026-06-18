from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


RAW_PATH = Path("data/raw/xlam/xlam_function_calling_60k.json")
REPORT_PATH = Path("data/manifests/xlam_inspection_summary.json")
PREVIEW_PATH = Path("data/raw/xlam/inspection_100_pretty.txt")

SEED = 42
GENERATOR_BOUNDARY = 33_659
SAMPLES_PER_GENERATOR = 50


def decode_json_field(
    value: Any,
    field_name: str,
    row_id: Any,
) -> Any:
    """Decode a JSON-serialized field.

    Already-decoded values are returned unchanged.
    """
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Row {row_id}: field {field_name!r} is invalid JSON. "
            f"Preview: {value[:200]!r}"
        ) from exc


def load_raw_dataset() -> Dataset:
    """Load the locally downloaded raw xLAM JSON file."""
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}. "
            "Run scripts/download_xlam.py first."
        )

    return load_dataset(
        "json",
        data_files=str(RAW_PATH),
        split="train",
    )


def select_indices(dataset_size: int) -> list[int]:
    """Select 50 deterministic records from each generator partition."""
    if dataset_size <= GENERATOR_BOUNDARY:
        raise ValueError(
            f"Dataset has only {dataset_size} rows; "
            f"expected more than {GENERATOR_BOUNDARY}."
        )

    rng = random.Random(SEED)

    first_partition = rng.sample(
        range(0, GENERATOR_BOUNDARY),
        SAMPLES_PER_GENERATOR,
    )
    second_partition = rng.sample(
        range(GENERATOR_BOUNDARY, dataset_size),
        SAMPLES_PER_GENERATOR,
    )

    return sorted(first_partition + second_partition)


def main() -> None:
    dataset = load_raw_dataset()
    indices = select_indices(len(dataset))

    tool_counts: Counter[int] = Counter()
    answer_counts: Counter[int] = Counter()
    parameter_types: Counter[str] = Counter()
    tool_namespaces: Counter[str] = Counter()

    invalid_fields: Counter[str] = Counter()
    missing_top_level_fields: Counter[str] = Counter()

    preview_records: list[dict[str, Any]] = []

    required_fields = {"id", "query", "tools", "answers"}

    # This loop must be inside main().
    for index in indices:
        row = dataset[index]

        missing_fields = required_fields - set(row)

        if missing_fields:
            for field in missing_fields:
                missing_top_level_fields[field] += 1
            continue

        row_id = row.get("id", index)
        query = row.get("query")

        # query is ordinary natural-language text, not serialized JSON.
        if not isinstance(query, str):
            invalid_fields["query_not_string"] += 1
            continue

        if not query.strip():
            invalid_fields["query_empty"] += 1
            continue

        try:
            tools = decode_json_field(
                row.get("tools"),
                "tools",
                row_id,
            )
        except ValueError:
            invalid_fields["tools_invalid_json"] += 1
            continue

        try:
            answers = decode_json_field(
                row.get("answers"),
                "answers",
                row_id,
            )
        except ValueError:
            invalid_fields["answers_invalid_json"] += 1
            continue

        if not isinstance(tools, list):
            invalid_fields["tools_not_list"] += 1
            continue

        if not isinstance(answers, list):
            invalid_fields["answers_not_list"] += 1
            continue

        tool_counts[len(tools)] += 1
        answer_counts[len(answers)] += 1

        for tool in tools:
            if not isinstance(tool, dict):
                invalid_fields["tool_not_object"] += 1
                continue

            name = tool.get("name", "")

            if isinstance(name, str) and name:
                namespace = (
                    name.split(".", maxsplit=1)[0]
                    if "." in name
                    else name
                )
                tool_namespaces[namespace] += 1

            parameters = tool.get("parameters", {})

            if not isinstance(parameters, dict):
                continue

            for specification in parameters.values():
                if not isinstance(specification, dict):
                    continue

                parameter_type = str(
                    specification.get("type", "<missing>")
                )
                parameter_types[parameter_type] += 1

        preview_records.append(
            {
                "row_index": index,
                "id": row_id,
                "generator": (
                    "deepseek"
                    if int(row_id) <= 33_658
                    else "mixtral"
                ),
                "query": query,
                "tools": tools,
                "answers": answers,
            }
        )

    # This section must be after the loop, but still inside main().
    summary = {
        "dataset_rows": len(dataset),
        "columns": dataset.column_names,
        "features": {
            name: str(feature)
            for name, feature in dataset.features.items()
        },
        "inspection_seed": SEED,
        "selected_rows": len(indices),
        "successfully_parsed_rows": len(preview_records),
        "failed_rows": len(indices) - len(preview_records),
        "sample_indices": indices,
        "available_tool_count_distribution": dict(
            sorted(tool_counts.items())
        ),
        "expected_call_count_distribution": dict(
            sorted(answer_counts.items())
        ),
        "parameter_types": dict(
            parameter_types.most_common()
        ),
        "most_common_tool_namespaces": dict(
            tool_namespaces.most_common(30)
        ),
        "invalid_fields": dict(invalid_fields),
        "missing_top_level_fields": dict(
            missing_top_level_fields
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PREVIEW_PATH.open("w", encoding="utf-8") as file:
        for record_number, record in enumerate(
            preview_records,
            start=1,
        ):
            file.write("=" * 100 + "\n")
            file.write(f"INSPECTION RECORD {record_number}\n")
            file.write("=" * 100 + "\n")
            file.write(
                json.dumps(
                    record,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=False,
                )
            )
            file.write("\n\n")

    print(f"Rows in dataset: {len(dataset)}")
    print(f"Selected rows: {len(indices)}")
    print(f"Successfully parsed: {len(preview_records)}")
    print(f"Failed rows: {len(indices) - len(preview_records)}")
    print(f"Summary written to: {REPORT_PATH}")
    print(f"Manual preview written to: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
