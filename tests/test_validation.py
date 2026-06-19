import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

from function_calling_ft.validation import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    validate_raw_example,
)


ROOT = Path(__file__).resolve().parents[1]
SELECT_SMOKE_SAMPLE_PATH = ROOT / "scripts" / "select_smoke_sample.py"
VALIDATE_EXAMPLES_PATH = ROOT / "scripts" / "validate_examples.py"


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_smoke_sample = load_script_module(
    SELECT_SMOKE_SAMPLE_PATH,
    "select_smoke_sample_for_validation_tests",
)
validate_examples = load_script_module(
    VALIDATE_EXAMPLES_PATH,
    "validate_examples_for_validation_tests",
)


def make_raw_row(row_id: int = 123) -> dict:
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "city": {
                    "type": "str",
                    "description": "City name",
                    "required": True,
                },
                "days": {
                    "type": "int",
                    "description": "Forecast length",
                    "required": False,
                },
            },
        }
    ]

    answers = [
        {
            "name": "get_weather",
            "arguments": {
                "city": "Boston",
                "days": 3,
            },
        }
    ]

    return {
        "id": row_id,
        "query": "Give me a three-day forecast for Boston.",
        "tools": json.dumps(tools),
        "answers": json.dumps(answers),
    }


def parse_candidate(row: dict) -> dict:
    return select_smoke_sample.parse_candidate(row, 0)


def test_validate_raw_example_accepts_valid_row() -> None:
    result = validate_raw_example(
        make_raw_row(),
        split="train",
    )

    assert result.is_valid is True
    assert result.estimated_tokens is not None


def test_validate_raw_example_rejects_missing_required_argument() -> None:
    row = make_raw_row()
    row["answers"] = json.dumps(
        [
            {
                "name": "get_weather",
                "arguments": {"days": 3},
            }
        ]
    )

    result = validate_raw_example(row, split="train")

    assert result.is_valid is False
    assert {
        issue.category for issue in result.issues
    } == {"missing_required_arguments"}


def test_validate_raw_example_rejects_incompatible_argument_type() -> None:
    row = make_raw_row()
    row["answers"] = json.dumps(
        [
            {
                "name": "get_weather",
                "arguments": {
                    "city": 123,
                    "days": "three",
                },
            }
        ]
    )

    result = validate_raw_example(row, split="train")

    assert result.is_valid is False
    assert "incompatible_argument_type" in {
        issue.category for issue in result.issues
    }


def test_validate_raw_example_rejects_invalid_tool_schema() -> None:
    row = make_raw_row()
    tools = json.loads(row["tools"])
    del tools[0]["parameters"]["city"]["type"]
    row["tools"] = json.dumps(tools)

    result = validate_raw_example(row, split="train")

    assert result.is_valid is False
    assert {
        issue.category for issue in result.issues
    } == {"invalid_tool_schema"}


def test_validate_raw_example_rejects_unparseable_expected_calls() -> None:
    row = make_raw_row()
    row["answers"] = "not-json"

    result = validate_raw_example(row, split="train")

    assert result.is_valid is False
    assert {
        issue.category for issue in result.issues
    } == {"unparseable_expected_calls"}


def test_validate_raw_example_rejects_invalid_argument_object() -> None:
    row = make_raw_row()
    row["answers"] = json.dumps(
        [
            {
                "name": "get_weather",
                "arguments": '["Boston"]',
            }
        ]
    )

    result = validate_raw_example(row, split="train")

    assert result.is_valid is False
    assert {
        issue.category for issue in result.issues
    } == {"invalid_argument_object"}


def test_validate_raw_example_rejects_context_length_overflow() -> None:
    row = make_raw_row()
    row["query"] = "weather " * 10_000

    result = validate_raw_example(
        row,
        split="train",
        context_token_limit=128,
    )

    assert result.is_valid is False
    assert "context_length_exceeded" in {
        issue.category for issue in result.issues
    }
    assert result.estimated_tokens is not None
    assert result.estimated_tokens > 128


def test_evaluate_selected_records_flags_cross_split_duplicates() -> None:
    train_candidate = parse_candidate(make_raw_row(100))
    test_candidate = copy.deepcopy(train_candidate)
    test_candidate["id"] = 200

    selected = {
        "train": [train_candidate],
        "validation": [],
        "test": [test_candidate],
    }

    entries = validate_examples.evaluate_selected_records(
        selected,
        context_token_limit=DEFAULT_CONTEXT_TOKEN_LIMIT,
    )

    duplicate_entry = next(
        entry for entry in entries if entry["split"] == "test"
    )
    assert "duplicate_cross_split" in {
        issue.category for issue in duplicate_entry["issues"]
    }


def test_find_replacement_uses_same_cell_and_next_valid_candidate() -> None:
    invalid_row = make_raw_row(101)
    invalid_row["answers"] = json.dumps(
        [
            {
                "name": "get_weather",
                "arguments": {"days": 3},
            }
        ]
    )
    invalid_candidate = parse_candidate(invalid_row)

    replacement_candidate = parse_candidate(make_raw_row(102))
    replacement_candidate["fingerprint"] = "replacement-fingerprint"

    pools = {
        ("deepseek", "single_call"): [
            invalid_candidate,
            replacement_candidate,
        ]
    }

    rejected_entry = {
        "split": "train",
        "index": 0,
        "candidate": invalid_candidate,
        "issues": tuple(),
        "estimated_tokens": None,
    }

    replacement, result = validate_examples.find_replacement(
        split="train",
        rejected_entry=rejected_entry,
        pools=pools,
        used_ids={invalid_candidate["id"]},
        used_fingerprints={invalid_candidate["fingerprint"]},
        blocked_ids={invalid_candidate["id"]},
        blocked_fingerprints={invalid_candidate["fingerprint"]},
        replacement_candidate_rejections=Counter(),
        context_token_limit=DEFAULT_CONTEXT_TOKEN_LIMIT,
    )

    assert replacement["id"] == replacement_candidate["id"]
    assert replacement["generator"] == invalid_candidate["generator"]
    assert replacement["call_bucket"] == invalid_candidate["call_bucket"]
    assert result.is_valid is True


def test_build_validation_report_keeps_count_and_records_separate() -> None:
    candidate = parse_candidate(make_raw_row(301))
    rejected_entry = {
        "split": "train",
        "index": 0,
        "candidate": candidate,
        "issues": (
            validate_examples.ValidationIssue(
                category="missing_required_arguments",
                message="city is required.",
            ),
        ),
        "estimated_tokens": 123,
    }

    report = validate_examples.build_validation_report(
        examples_selected=200,
        valid_examples=199,
        rejected_entries=[rejected_entry],
        replacements=[],
        final_selected={
            "train": [candidate],
            "validation": [],
            "test": [],
        },
        selected_rejections=Counter(
            {"missing_required_arguments": 1}
        ),
        replacement_candidate_rejections=Counter(),
        context_token_limit=DEFAULT_CONTEXT_TOKEN_LIMIT,
    )

    assert report["rejected_examples"] == 1
    assert len(report["rejected_records"]) == 1
