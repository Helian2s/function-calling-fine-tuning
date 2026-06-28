from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from function_calling_ft.prompt_audit import (
    prompt_audit_record,
    summarize_prompt_audit,
)
from function_calling_ft.split_guard import SplitAccessError, assert_split_allowed


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> Any:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_no_tool_relevance_set = _load_script(
    "build_no_tool_relevance_set",
    "scripts/build_no_tool_relevance_set.py",
)
run_exp02_matrix = _load_script(
    "run_exp02_matrix",
    "scripts/run_exp02_matrix.py",
)


class AuditTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, rendered: str) -> None:
        self.rendered = rendered

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        del conversation, tools, add_generation_prompt, kwargs
        if tokenize:
            return {"input_ids": [ord(char) for char in self.rendered]}
        return self.rendered

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def test_no_tool_relevance_records_are_balanced_and_locked() -> None:
    dev = build_no_tool_relevance_set.build_records("dev")
    final = build_no_tool_relevance_set.build_records("final")

    assert len(dev) == 100
    assert len(final) == 100
    assert {
        record["split_metadata"]["split_lock_status"] for record in dev
    } == {"screening_allowed"}
    assert {
        record["split_metadata"]["split_lock_status"] for record in final
    } == {"locked_final_no_tool"}
    assert {
        category: sum(
            int(record["relevance_metadata"]["category"] == category)
            for record in dev
        )
        for category in build_no_tool_relevance_set.CATEGORIES
    } == {
        "no_available_tool_can_satisfy": 25,
        "available_tools_irrelevant": 25,
        "missing_required_information": 25,
        "direct_answer_without_tool": 25,
    }


def test_no_tool_final_split_is_locked() -> None:
    with pytest.raises(SplitAccessError, match="--final-evaluation"):
        assert_split_allowed(
            "data/eval/no_tool_relevance_v1/final_locked.jsonl",
            command_name="generation",
        )

    decision = assert_split_allowed("data/eval/no_tool_relevance_v1/dev.jsonl")
    assert decision.split_name == "no-tool-relevance-v1-dev"
    assert decision.split_lock_status == "screening_allowed"


def test_prompt_audit_detects_hidden_expected_response() -> None:
    record = {
        "id": "no-tool-1",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "tools": [],
        "expected_response": {
            "type": "direct_answer",
            "content": "the exact hidden answer phrase",
        },
    }
    audit = prompt_audit_record(
        tokenizer=AuditTokenizer(
            "user asked math; the exact hidden answer phrase",
        ),
        record=record,
    )
    summary = summarize_prompt_audit([audit])

    assert audit["hidden_expected_response"] is True
    assert summary["hidden_expected_response_count"] == 1


def test_prompt_audit_allows_expected_text_already_in_user_query() -> None:
    record = {
        "id": "no-tool-2",
        "messages": [
            {"role": "user", "content": "Rewrite 'the cat sleeps' in uppercase."}
        ],
        "tools": [],
        "expected_response": {
            "type": "direct_answer",
            "content": "THE CAT SLEEPS",
        },
    }
    audit = prompt_audit_record(
        tokenizer=AuditTokenizer("Rewrite 'the cat sleeps' in uppercase."),
        record=record,
    )

    assert audit["hidden_expected_response"] is False


def test_exp02_dry_run_plan_validates_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "one",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
            }
        ],
    )
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        """
schema_version: "1.0"
experiment_id: exp-02
run_id: unit-deterministic
model:
  name: Qwen/Qwen3-1.7B
  revision: rev
  tokenizer_revision: rev
  torch_dtype: bfloat16
  load_in_4bit: false
decoding:
  do_sample: false
  max_new_tokens: 32
  seed: 42
  enable_thinking: false
generation:
  batch_size: 2
method:
  name: base
  precision: bfloat16
  quantization: none
""",
        encoding="utf-8",
    )
    matrix = {
        "experiment_id": "exp-02",
        "task_id": "task-06",
        "runs": [{"config": str(run_config)}],
        "datasets": {
            "unit": {
                "path": str(dataset),
                "split_name": "unit",
                "split_lock_status": "screening_allowed",
                "expected_records": 1,
            }
        },
    }

    runs = run_exp02_matrix._run_configs(matrix)
    datasets = run_exp02_matrix._dataset_specs(matrix)
    plan = run_exp02_matrix._dry_run_plan(
        matrix=matrix,
        runs=runs,
        datasets=datasets,
        selected_runs={"unit-deterministic"},
        selected_datasets={"unit"},
    )

    assert plan["runs"][0]["records"] == 1
    assert plan["runs"][0]["do_sample"] is False
