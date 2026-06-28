from __future__ import annotations

import json
from pathlib import Path

from function_calling_ft.split_freeze import (
    EXCLUDED_SPLIT,
    TokenStats,
    build_frozen_splits,
    examples_for_split,
    validate_frozen_splits,
    write_split_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def make_record(
    index: int,
    *,
    group_id: str,
    family: str,
    category: str = "math",
    call_category: str = "single",
    tool_count: int = 1,
    call_count: int = 1,
) -> dict:
    return {
        "schema_version": "1.0",
        "id": f"xlam-{index}",
        "example_id": f"xlam-{index}",
        "messages": [
            {"role": "user", "content": f"Question {index}?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": f"{family}_tool",
                            "arguments": {"value": index},
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": f"{family}_tool",
                    "description": "Synthetic tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "metadata": {
            "split": "full",
            "source_id": index,
            "source_revision": "revision",
            "source_file_sha256": "sha",
            "available_tool_count": tool_count,
            "expected_call_count": call_count,
        },
        "curation_metadata": {
            "example_id": f"xlam-{index}",
            "source_id": index,
            "split_group_id": group_id,
            "primary_tool_family": family,
            "primary_api_category": category,
            "call_category": call_category,
            "tool_count": tool_count,
            "expected_call_count": call_count,
            "exact_duplicate_hash": f"hash-{index}",
            "tool_families": [family],
        },
    }


def token_stats(records: list[dict], *, overlength_id: str | None = None) -> dict[str, TokenStats]:
    stats = {}
    for record in records:
        record_id = str(record["id"])
        full_tokens = 2300 if record_id == overlength_id else 500
        target_tokens = 40
        stats[record_id] = TokenStats(
            example_id=record_id,
            full_tokens=full_tokens,
            prompt_schema_tokens=full_tokens - target_tokens,
            supervised_target_tokens=target_tokens,
            truncation_risk_2048=full_tokens > 2048,
            truncation_risk_4096=full_tokens > 4096,
        )
    return stats


def small_config() -> dict:
    return {
        "seed": 7,
        "splits": {
            "validation_target_records": 2,
            "internal_test_target_records": 2,
            "challenge_target_records": 2,
            "train_10k_target_records": 2,
            "train_2k_target_records": 1,
            "dev_eval_target_records": 1,
            "max_overshoot_records": 0,
        },
        "sequence_length": {
            "preferred_max_length": 2048,
            "fallback_max_length": 4096,
            "preferred_coverage_threshold": 0.8,
        },
    }


def test_split_freeze_assigns_every_record_once_and_keeps_groups_disjoint() -> None:
    records = [
        make_record(index, group_id=f"group-{index}", family=f"family-{index}")
        for index in range(10)
    ]

    result = build_frozen_splits(records, token_stats(records), small_config())
    validation = validate_frozen_splits(result)

    assert validation["status"] == "pass"
    assert set(result.primary_assignments) == {
        f"xlam-{index}" for index in range(10)
    }
    split_groups: dict[str, set[str]] = {}
    for split_name in (
        "train",
        "validation",
        "internal_test_locked",
        "reserved_challenge_locked",
    ):
        split_groups[split_name] = {
            example.split_group_id
            for example in examples_for_split(result, split_name)
        }
    assert len(set().union(*split_groups.values())) == sum(
        len(value) for value in split_groups.values()
    )


def test_split_freeze_nested_subsets() -> None:
    records = [
        make_record(index, group_id=f"group-{index}", family=f"family-{index}")
        for index in range(12)
    ]
    result = build_frozen_splits(records, token_stats(records), small_config())

    train = {example.example_id for example in examples_for_split(result, "train")}
    train_10k = {
        example.example_id for example in examples_for_split(result, "train_10k")
    }
    train_2k = {
        example.example_id for example in examples_for_split(result, "train_2k")
    }
    validation = {
        example.example_id for example in examples_for_split(result, "validation")
    }
    dev_eval = {
        example.example_id for example in examples_for_split(result, "dev_eval_1k")
    }

    assert train_2k <= train_10k <= train
    assert dev_eval <= validation


def test_split_freeze_excludes_overlength_group() -> None:
    records = [
        make_record(0, group_id="group-long", family="long"),
        make_record(1, group_id="group-ok", family="ok"),
        make_record(2, group_id="group-ok-2", family="ok2"),
        make_record(3, group_id="group-ok-3", family="ok3"),
        make_record(4, group_id="group-ok-4", family="ok4"),
    ]
    stats = token_stats(records, overlength_id="xlam-0")
    config = small_config()
    config["sequence_length"]["preferred_coverage_threshold"] = 0.5

    result = build_frozen_splits(records, stats, config)

    assert result.selected_max_sequence_length == 2048
    assert result.primary_assignments["xlam-0"] == EXCLUDED_SPLIT
    assert "group-long" in result.overlength_group_ids


def test_split_freeze_artifact_hashes_are_deterministic(tmp_path: Path) -> None:
    records = [
        make_record(index, group_id=f"group-{index}", family=f"family-{index}")
        for index in range(10)
    ]
    result = build_frozen_splits(records, token_stats(records), small_config())
    curation_report = tmp_path / "curation.json"
    normalization_report = tmp_path / "normalization.json"
    config_path = tmp_path / "config.yaml"
    curation_report.write_text(
        json.dumps(
            {
                "exact_deduplication": {
                    "retained_records": 10,
                    "duplicate_groups": 0,
                    "duplicate_records": 0,
                },
                "curator": {"status": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    normalization_report.write_text(
        json.dumps(
            {
                "dataset": {
                    "revision": "revision",
                    "license": "cc-by-4.0",
                    "access": "test",
                },
                "processing": {
                    "input_records": 10,
                    "accepted_records": 10,
                    "quarantined_records": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text("seed: 7\n", encoding="utf-8")

    output_dir = tmp_path / "splits"
    write_split_artifacts(
        result=result,
        output_dir=output_dir,
        repo_root=ROOT,
        curation_report_path=curation_report,
        normalization_report_path=normalization_report,
        config_path=config_path,
    )
    first_checksums = (output_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    )
    write_split_artifacts(
        result=result,
        output_dir=output_dir,
        repo_root=ROOT,
        curation_report_path=curation_report,
        normalization_report_path=normalization_report,
        config_path=config_path,
    )
    second_checksums = (output_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    )

    assert second_checksums == first_checksums
