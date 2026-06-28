import copy
import json
from pathlib import Path

import pytest

from function_calling_ft.curation import (
    audit_group_leakage,
    call_category,
    compare_curation_hashes,
    compute_curation_metadata,
    curate_normalized_dataset,
    deduplicated_records,
    duplicate_map_records,
    expected_calls,
    fuzzy_candidate_records,
    normalized_user_request,
    request_shingles,
    schema_shape_fingerprint,
    tool_set_signature_hash,
    tools,
    verify_stable_under_shuffle,
)
from function_calling_ft.leakage import run_leakage_audit


ROOT = Path(__file__).resolve().parents[1]


def make_record(
    source_id: int,
    *,
    query: str = "Get the weather for Boston and Denver.",
    calls: list[dict] | None = None,
) -> dict:
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_temperature",
                "description": "Convert temperature units.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["value", "unit"],
                },
            },
        },
    ]
    tool_calls = calls or [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Boston"},
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Denver"},
            },
        },
    ]
    return {
        "schema_version": "1.0",
        "id": f"xlam-{source_id}",
        "example_id": f"xlam-{source_id}",
        "tools": tool_definitions,
        "messages": [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        ],
        "metadata": {
            "source_id": source_id,
            "source_row_index": source_id,
            "split": "full",
            "available_tool_count": len(tool_definitions),
            "expected_call_count": len(tool_calls),
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def test_tool_set_signature_is_stable_under_tool_order() -> None:
    record = make_record(1)
    reversed_record = copy.deepcopy(record)
    reversed_record["tools"] = list(reversed(record["tools"]))

    assert tool_set_signature_hash(tools(record)) == tool_set_signature_hash(
        tools(reversed_record)
    )


def test_schema_shape_fingerprint_ignores_function_names() -> None:
    record = make_record(1)
    renamed = copy.deepcopy(record)
    renamed["tools"][0]["function"]["name"] = "fetch_forecast"
    renamed["tools"][1]["function"]["name"] = "temperature_convert"

    assert schema_shape_fingerprint(tools(record)) == schema_shape_fingerprint(
        tools(renamed)
    )


def test_call_category_distinguishes_multiple_and_parallel() -> None:
    repeated = make_record(1)
    parallel = make_record(
        2,
        calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Boston"},
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "convert_temperature",
                    "arguments": {"value": 20, "unit": "F"},
                },
            },
        ],
    )

    assert call_category(expected_calls(repeated)) == "multiple"
    assert call_category(expected_calls(parallel)) == "parallel"


def test_duplicate_mapping_retains_smallest_source_id() -> None:
    first = make_record(2)
    second = make_record(1)
    records = [first, second]
    metadatas = [
        compute_curation_metadata(record) for record in records
    ]

    duplicate_maps = duplicate_map_records(metadatas)
    deduped = deduplicated_records(records, metadatas)

    assert len(duplicate_maps) == 1
    assert duplicate_maps[0]["retained_example_id"] == "xlam-1"
    assert duplicate_maps[0]["duplicate_example_ids"] == ["xlam-2"]
    assert len(deduped) == 1
    assert deduped[0]["example_id"] == "xlam-1"


def test_group_ids_are_stable_under_record_order() -> None:
    records = [make_record(3), make_record(1), make_record(2)]
    forward = {
        metadata["example_id"]: metadata["split_group_id"]
        for metadata in (
            compute_curation_metadata(record) for record in records
        )
    }
    backward = {
        metadata["example_id"]: metadata["split_group_id"]
        for metadata in (
            compute_curation_metadata(record)
            for record in reversed(records)
        )
    }

    assert forward == backward


def test_fuzzy_candidates_are_review_only() -> None:
    left = make_record(
        1,
        query="Please find the weather forecast for Boston tomorrow.",
    )
    right = make_record(
        2,
        query="Please find the weather forecast for Boston tomorrow.",
    )
    metadatas = [
        compute_curation_metadata(left),
        compute_curation_metadata(right),
    ]
    shingles = {
        metadata["example_id"]: request_shingles(
            normalized_user_request(record)
        )
        for metadata, record in zip(
            metadatas,
            [left, right],
            strict=True,
        )
    }

    candidates = fuzzy_candidate_records(
        metadatas,
        shingles,
        threshold=0.1,
        num_hashes=12,
        bands=3,
    )

    assert candidates
    assert candidates[0]["disposition"] == "review_only"


def test_leakage_audit_fails_on_cross_split_group_overlap() -> None:
    train = compute_curation_metadata(make_record(1))
    validation = copy.deepcopy(train)
    validation["example_id"] = "xlam-2"
    train["split"] = "train"
    validation["split"] = "validation"

    report = audit_group_leakage([train, validation])

    assert report["status"] == "fail"
    assert report["cross_split_group_count"] == 1


def test_run_leakage_audit_raises_on_overlap(tmp_path: Path) -> None:
    train = compute_curation_metadata(make_record(1))
    validation = copy.deepcopy(train)
    validation["example_id"] = "xlam-2"
    train["split"] = "train"
    validation["split"] = "validation"
    path = tmp_path / "group_metadata.jsonl"
    write_jsonl(path, [train, validation])

    with pytest.raises(RuntimeError, match="Leakage audit failed"):
        run_leakage_audit(
            group_metadata_path=path,
            output_path=tmp_path / "report.json",
        )


def test_curate_normalized_dataset_outputs_no_silent_removal(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "normalized.jsonl"
    records = [make_record(2), make_record(1), make_record(3, query="Other")]
    write_jsonl(input_path, records)

    report = curate_normalized_dataset(
        input_path=input_path,
        output_dir=tmp_path / "curated",
        repo_root=ROOT,
        fuzzy_threshold=0.1,
    )

    exact = report["exact_deduplication"]
    assert exact["input_records"] == 3
    assert exact["retained_records"] == 2
    assert exact["duplicate_records"] == 1
    assert exact["no_silent_removal"] is True
    assert (tmp_path / "curated" / "duplicate_map.jsonl").is_file()
    assert (
        tmp_path
        / "curated"
        / "curator_input"
        / "exact_dedup_input.jsonl"
    ).is_file()


def test_curate_normalized_dataset_merges_curator_completion(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "normalized.jsonl"
    output_dir = tmp_path / "curated"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True)
    write_jsonl(input_path, [make_record(1), make_record(2)])
    (manifests_dir / "curator_comparison_report.json").write_text(
        json.dumps({"status": "pass"}) + "\n",
        encoding="utf-8",
    )
    (manifests_dir / "curator_ec2_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "run_id": "task03-curator-test",
                "curator_version": "1.0.0rc0.dev0",
                "curator_image_digest": "sha256:test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests_dir / "curator_ec2_s3_uri.txt").write_text(
        "s3://bucket/prefix\n",
        encoding="utf-8",
    )

    report = curate_normalized_dataset(
        input_path=input_path,
        output_dir=output_dir,
        repo_root=ROOT,
    )

    curator = report["curator"]
    assert curator["status"] == "pass"
    assert curator["comparison_required"] is False
    assert curator["ec2_report"]["run_id"] == "task03-curator-test"
    assert curator["s3_uri"] == "s3://bucket/prefix"


def test_shuffle_stability_outputs_same_hashes(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.jsonl"
    records = [make_record(4), make_record(2), make_record(1)]
    write_jsonl(input_path, records)
    output_dir = tmp_path / "curated"

    curate_normalized_dataset(
        input_path=input_path,
        output_dir=output_dir,
        repo_root=ROOT,
    )
    verify_stable_under_shuffle(
        input_path=input_path,
        output_dir=output_dir / "_stability",
        repo_root=ROOT,
    )

    comparison = compare_curation_hashes(
        output_dir,
        output_dir / "_stability" / "shuffled_output",
    )

    assert comparison["stable"] is True
