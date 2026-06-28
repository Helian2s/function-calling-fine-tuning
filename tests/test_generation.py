from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pytest

from function_calling_ft import generation
from function_calling_ft.generation import (
    DecodingConfig,
    build_generation_prompt,
    build_inference_messages,
    generate_prediction_records,
    validate_adapter_base_model,
    validate_adapter_path,
)


GENERATE_PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_predictions.py"
)
GENERATE_PREDICTIONS_SPEC = importlib.util.spec_from_file_location(
    "generate_predictions",
    GENERATE_PREDICTIONS_PATH,
)
assert GENERATE_PREDICTIONS_SPEC is not None
assert GENERATE_PREDICTIONS_SPEC.loader is not None
generate_predictions = importlib.util.module_from_spec(
    GENERATE_PREDICTIONS_SPEC,
)
GENERATE_PREDICTIONS_SPEC.loader.exec_module(generate_predictions)


def _record() -> dict[str, Any]:
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


def _record_with_content(
    record_id: str,
    content: str,
    source_id: int,
) -> dict[str, Any]:
    record = _record()
    record["id"] = record_id
    record["messages"][0]["content"] = content
    record["metadata"]["source_id"] = source_id
    return record


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
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

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
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


def test_generate_prediction_records_batches_by_prompt_length_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    batch_prompt_lengths: list[list[int]] = []

    def fake_generate_batch(**kwargs):
        prompts = kwargs["prompts"]
        batch_prompt_lengths.append(
            [prompt.prompt_token_count for prompt in prompts],
        )
        return [
            (f"prompt_tokens={prompt.prompt_token_count}", 1)
            for prompt in prompts
        ]

    monkeypatch.setattr(generation, "_generate_batch", fake_generate_batch)

    records = [
        _record_with_content("xlam-1", "short", 1),
        _record_with_content("xlam-2", "long " * 50, 2),
        _record_with_content("xlam-3", "medium " * 10, 3),
    ]

    predictions = generate_prediction_records(
        records=records,
        tokenizer=tokenizer,
        model=FakeModel(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
        batch_size=2,
    )

    assert [prediction["id"] for prediction in predictions] == [
        "xlam-1",
        "xlam-2",
        "xlam-3",
    ]
    prompt_lengths = [
        prediction["prompt_token_count"] for prediction in predictions
    ]
    sorted_prompt_lengths = sorted(prompt_lengths, reverse=True)
    assert batch_prompt_lengths == [
        sorted_prompt_lengths[:2],
        sorted_prompt_lengths[2:],
    ]
    assert [
        prediction["raw_generation"] for prediction in predictions
    ] == [
        f"prompt_tokens={length}" for length in prompt_lengths
    ]


def test_streaming_predictions_flush_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictions.jsonl"
    records = [{"id": "one"}, {"id": "two"}]

    def fake_iter_prediction_record_batches(**kwargs):
        yield [
            {
                "id": record["id"],
                "raw_generation": f"{record['id']}-generation",
            }
            for record in kwargs["records"]
        ]

    monkeypatch.setattr(
        generate_predictions,
        "iter_prediction_record_batches",
        fake_iter_prediction_record_batches,
    )

    written, skipped, stopped, stats = generate_predictions._write_streaming_predictions(
        output=output,
        records=records,
        tokenizer=object(),
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
        decoding=DecodingConfig(),
        batch_size=2,
        device=None,
        resume=False,
        progress_interval=1,
        progress_file=None,
        stop_file=None,
    )

    assert (written, skipped) == (2, 0)
    assert stopped is False
    assert stats["generation_errors"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    written, skipped, stopped, stats = generate_predictions._write_streaming_predictions(
        output=output,
        records=records,
        tokenizer=object(),
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
        decoding=DecodingConfig(),
        batch_size=2,
        device=None,
        resume=True,
        progress_interval=1,
        progress_file=None,
        stop_file=None,
    )

    assert (written, skipped) == (0, 2)
    assert stopped is False
    assert stats["generated_tokens"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_streaming_resume_generates_only_missing_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps({"id": "one", "raw_generation": "existing"})
        + "\n",
        encoding="utf-8",
    )
    records = [{"id": "one"}, {"id": "two"}, {"id": "three"}]
    generated_ids: list[str] = []

    def fake_iter_prediction_record_batches(**kwargs):
        generated_ids.extend(record["id"] for record in kwargs["records"])
        yield [
            {
                "id": record["id"],
                "raw_generation": f"{record['id']}-generation",
            }
            for record in kwargs["records"]
        ]

    monkeypatch.setattr(
        generate_predictions,
        "iter_prediction_record_batches",
        fake_iter_prediction_record_batches,
    )

    written, skipped, stopped, _stats = generate_predictions._write_streaming_predictions(
        output=output,
        records=records,
        tokenizer=object(),
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
        decoding=DecodingConfig(),
        batch_size=16,
        device=None,
        resume=True,
        progress_interval=0,
        progress_file=None,
        stop_file=None,
    )

    assert (written, skipped) == (2, 1)
    assert stopped is False
    assert generated_ids == ["two", "three"]
    output_records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in output_records] == [
        "one",
        "two",
        "three",
    ]


def test_streaming_progress_file_and_stop_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictions.jsonl"
    progress_file = tmp_path / "progress.json"
    stop_file = tmp_path / "STOP"
    records = [{"id": "one"}, {"id": "two"}, {"id": "three"}]
    calls = 0

    def fake_iter_prediction_record_batches(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        yield [{"id": "one", "raw_generation": "first"}]
        calls += 1
        yield [{"id": "two", "raw_generation": "second"}]

    def fake_write_progress_file(path: Path, payload: dict[str, object]):
        path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stop_file.write_text("stop\n", encoding="utf-8")

    monkeypatch.setattr(
        generate_predictions,
        "iter_prediction_record_batches",
        fake_iter_prediction_record_batches,
    )
    monkeypatch.setattr(
        generate_predictions,
        "_write_progress_file",
        fake_write_progress_file,
    )

    written, skipped, stopped, _stats = generate_predictions._write_streaming_predictions(
        output=output,
        records=records,
        tokenizer=object(),
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        model_revision="revision",
        adapter_path=None,
        seed=42,
        max_new_tokens=10,
        decoding=DecodingConfig(),
        batch_size=16,
        device=None,
        resume=False,
        progress_interval=1,
        progress_file=progress_file,
        stop_file=stop_file,
    )

    assert (written, skipped, stopped) == (1, 0, True)
    assert calls == 1
    output_records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in output_records] == ["one"]
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["processed"] == 1
    assert progress["stop_requested"] is True


def test_sampling_decoding_kwargs_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    seen_decoding: list[DecodingConfig] = []

    def fake_generate_one(**kwargs):
        seen_decoding.append(kwargs["decoding"])
        return "sampled", 1

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
        decoding=DecodingConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        ),
    )

    assert predictions[0]["raw_generation"] == "sampled"
    assert seen_decoding == [
        DecodingConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        ),
    ]


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


def test_validate_adapter_path_prefers_latest_checkpoint_model(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "smoke-lora"
    latest_model = checkpoint_root / "LATEST" / "model"
    lowest_val_model = checkpoint_root / "LOWEST_VAL" / "model"
    older_model = checkpoint_root / "epoch_0_step_1" / "model"
    latest_model.mkdir(parents=True)
    lowest_val_model.mkdir(parents=True)
    older_model.mkdir(parents=True)

    for directory in (latest_model, lowest_val_model, older_model):
        (directory / "adapter_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (directory / "adapter_model.safetensors").write_text(
            "weights",
            encoding="utf-8",
        )

    assert validate_adapter_path(checkpoint_root) == latest_model


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
