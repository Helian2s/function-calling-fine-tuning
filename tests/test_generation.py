from __future__ import annotations

import json
from pathlib import Path

import pytest

from function_calling_ft import generation
from function_calling_ft.generation import (
    build_generation_prompt,
    build_inference_messages,
    generate_prediction_records,
    validate_adapter_base_model,
    validate_adapter_path,
)


def _record() -> dict[str, object]:
    return {
        "id": "xlam-1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Weather in Denver?"},
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


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        self.calls.append(
            {
                "conversation": conversation,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        rendered = json.dumps(
            {
                "messages": conversation,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            },
            sort_keys=True,
        )
        if tokenize:
            return {"input_ids": [ord(char) for char in rendered]}
        return rendered

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class FakeModel:
    pass


def test_build_inference_messages_removes_target_assistant_answer() -> None:
    messages = build_inference_messages(_record())

    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_generation_prompt_preserves_tools_and_disables_thinking() -> None:
    tokenizer = FakeTokenizer()

    prompt = build_generation_prompt(tokenizer, _record())

    assert prompt.prompt_token_count > 0
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["enable_thinking"] is False
    assert tokenizer.calls[0]["tools"] == _record()["tools"]
    assert len(tokenizer.calls[0]["conversation"]) == 1


def test_generate_prediction_records_preserves_per_record_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()

    def fake_generate_one(**kwargs):
        prompt = kwargs["prompt"]
        if prompt.prompt_token_count:
            return '{"name":"weather","arguments":{"city":"Denver"}}', 12
        raise AssertionError("unreachable")

    monkeypatch.setattr(generation, "_generate_one", fake_generate_one)

    predictions = generate_prediction_records(
        records=[_record()],
        tokenizer=tokenizer,
        model=FakeModel(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
    )

    assert predictions[0]["raw_generation"].startswith('{"name"')
    assert predictions[0]["generation_error"] is None
    assert predictions[0]["generated_token_count"] == 12


def test_validate_adapter_path_requires_config_and_weights(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="adapter_config"):
        validate_adapter_path(adapter_dir)

    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="adapter_model"):
        validate_adapter_path(adapter_dir)

    (adapter_dir / "adapter_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )

    assert validate_adapter_path(adapter_dir) == adapter_dir


def test_validate_adapter_base_model_rejects_wrong_base_model(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-8B",
            },
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        validate_adapter_base_model(adapter_dir, "Qwen/Qwen3-1.7B")


def test_validate_adapter_base_model_accepts_matching_base_model(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
            },
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )

    assert (
        validate_adapter_base_model(adapter_dir, "Qwen/Qwen3-1.7B")
        == adapter_dir
    )
