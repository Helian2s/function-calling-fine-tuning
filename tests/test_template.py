from __future__ import annotations

import json

from function_calling_ft.dataset import (
    render_template_example,
    rendered_example_to_report,
    select_representative_examples,
)


def _make_tool(
    name: str,
    *,
    properties: dict[str, dict] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


def _make_record(
    record_id: str,
    *,
    split: str,
    source_id: int,
    query: str,
    tools: list[dict],
    calls: list[dict],
) -> dict:
    return {
        "id": record_id,
        "schema_version": "1.0",
        "tools": tools,
        "messages": [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for index, call in enumerate(calls, start=1)
                ],
            },
        ],
        "metadata": {
            "split": split,
            "source_id": source_id,
            "available_tool_count": len(tools),
            "expected_call_count": len(calls),
        },
    }


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

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
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        text = self._render_text(
            conversation,
            tools=tools or [],
            enable_thinking=enable_thinking,
        )

        if tokenize:
            return {"input_ids": [ord(char) for char in text]}

        return text

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def _render_text(
        self,
        conversation,
        *,
        tools,
        enable_thinking: bool,
    ) -> str:
        assistant = conversation[1]
        tool_blocks = []

        for tool_call in assistant["tool_calls"]:
            payload = {
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"],
            }
            tool_blocks.append(
                "<tool_call>\n"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n</tool_call>"
            )

        thinking_block = (
            "<think>internal reasoning</think>\n"
            if enable_thinking
            else ""
        )

        return (
            "<tools>\n"
            + json.dumps(
                tools,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n</tools>\n"
            + f"<user>{conversation[0]['content']}</user>\n"
            + "<assistant>\n"
            + thinking_block
            + "\n".join(tool_blocks)
            + "\n</assistant>"
        )


class BrokenPythonDictTokenizer(FakeQwenTokenizer):
    def _render_text(
        self,
        conversation,
        *,
        tools,
        enable_thinking: bool,
    ) -> str:
        del tools, enable_thinking
        assistant = conversation[1]
        payload = {
            "name": assistant["tool_calls"][0]["function"]["name"],
            "arguments": assistant["tool_calls"][0]["function"][
                "arguments"
            ],
        }
        return (
            "<tools>\n"
            "[{'name': 'weather'}]\n"
            "</tools>\n"
            + f"<user>{conversation[0]['content']}</user>\n"
            + "<assistant>\n"
            + "<tool_call>\n"
            + str(payload)
            + "\n</tool_call>\n"
            + "</assistant>"
        )


class VariadicThinkingTokenizer(FakeQwenTokenizer):
    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        return super().apply_chat_template(
            conversation,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=kwargs.get("enable_thinking", False),
        )


class EmptyThinkTokenizer(FakeQwenTokenizer):
    def _render_text(
        self,
        conversation,
        *,
        tools,
        enable_thinking: bool,
    ) -> str:
        del enable_thinking
        assistant = conversation[1]
        payload = {
            "name": assistant["tool_calls"][0]["function"]["name"],
            "arguments": assistant["tool_calls"][0]["function"][
                "arguments"
            ],
        }
        return (
            "<tools>\n"
            + json.dumps(
                tools,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n</tools>\n"
            + f"<user>{conversation[0]['content']}</user>\n"
            + "<assistant>\n"
            + "<think>\n\n</think>\n"
            + "<tool_call>\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n</tool_call>\n"
            + "</assistant>"
        )


def test_select_representative_examples_is_deterministic():
    one_call = _make_record(
        "xlam-1",
        split="train",
        source_id=1,
        query="Find weather",
        tools=[_make_tool("weather")],
        calls=[{"name": "weather", "arguments": {"city": "Denver"}}],
    )
    two_distinct = _make_record(
        "xlam-2",
        split="train",
        source_id=2,
        query="Weather and forecast",
        tools=[
            _make_tool("weather"),
            _make_tool("forecast"),
        ],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}},
            {
                "name": "forecast",
                "arguments": {"city": "Denver", "days": 3},
            },
        ],
    )
    three_plus = _make_record(
        "xlam-3",
        split="train",
        source_id=3,
        query="Batch lookup",
        tools=[_make_tool("lookup")],
        calls=[
            {"name": "lookup", "arguments": {"id": 1}},
            {"name": "lookup", "arguments": {"id": 2}},
            {"name": "lookup", "arguments": {"id": 3}},
        ],
    )
    repeated = _make_record(
        "xlam-4",
        split="train",
        source_id=4,
        query="Two lookups",
        tools=[_make_tool("lookup")],
        calls=[
            {"name": "lookup", "arguments": {"id": 10}},
            {"name": "lookup", "arguments": {"id": 11}},
        ],
    )
    five_plus = _make_record(
        "xlam-5",
        split="train",
        source_id=5,
        query="Complex request",
        tools=[
            _make_tool(
                "complex",
                properties={
                    "filters": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            ),
            _make_tool("tool_b"),
            _make_tool("tool_c"),
            _make_tool("tool_d"),
            _make_tool("tool_e"),
        ],
        calls=[
            {
                "name": "complex",
                "arguments": {"filters": ["recent", "top"]},
            }
        ],
    )

    records = [five_plus, repeated, three_plus, two_distinct, one_call]

    first = select_representative_examples(records, count=5)
    second = select_representative_examples(records, count=5)

    assert [example.record_id for example in first] == [
        "xlam-1",
        "xlam-2",
        "xlam-3",
        "xlam-4",
        "xlam-5",
    ]
    assert [example.record_id for example in second] == [
        "xlam-1",
        "xlam-2",
        "xlam-3",
        "xlam-4",
        "xlam-5",
    ]


def test_render_template_example_passes_qwen_style_checks():
    tokenizer = FakeQwenTokenizer()
    record = _make_record(
        "xlam-200",
        split="validation",
        source_id=200,
        query="Weather and forecast",
        tools=[
            _make_tool("weather"),
            _make_tool("forecast"),
            _make_tool("alerts"),
        ],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}},
            {
                "name": "forecast",
                "arguments": {"city": "Denver", "days": 3},
            },
        ],
    )
    example = select_representative_examples([record], count=1)[0]

    rendered = render_template_example(
        tokenizer,
        example,
        enable_thinking=False,
    )
    report = rendered_example_to_report(rendered)

    assert rendered.checks.failures == ()
    assert rendered.checks.tool_section_present
    assert rendered.checks.function_names_unchanged
    assert rendered.checks.arguments_as_json_objects
    assert rendered.checks.tool_call_delimiters_present
    assert rendered.checks.no_python_dict_syntax
    assert rendered.checks.multiple_calls_serialized_correctly
    assert rendered.checks.thinking_mode_disabled
    assert rendered.token_count == len(rendered.decoded_text)
    assert report["checks"]["failures"] == []
    assert report["rendered_calls"] == [
        {"name": "weather", "arguments": {"city": "Denver"}},
        {
            "name": "forecast",
            "arguments": {"city": "Denver", "days": 3},
        },
    ]
    assert tokenizer.calls[0]["enable_thinking"] is False
    assert tokenizer.calls[1]["enable_thinking"] is False


def test_render_template_example_detects_python_dict_syntax():
    tokenizer = BrokenPythonDictTokenizer()
    record = _make_record(
        "xlam-201",
        split="test",
        source_id=201,
        query="Weather",
        tools=[_make_tool("weather")],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}}
        ],
    )
    example = select_representative_examples([record], count=1)[0]

    rendered = render_template_example(tokenizer, example)

    assert not rendered.checks.no_python_dict_syntax
    assert not rendered.checks.arguments_as_json_objects
    assert not rendered.checks.function_names_unchanged
    assert (
        "rendered output contains Python-style single-quoted keys"
        in rendered.checks.failures
    )


def test_render_template_example_can_enable_thinking():
    tokenizer = FakeQwenTokenizer()
    record = _make_record(
        "xlam-202",
        split="train",
        source_id=202,
        query="Weather",
        tools=[_make_tool("weather")],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}}
        ],
    )
    example = select_representative_examples([record], count=1)[0]

    rendered = render_template_example(
        tokenizer,
        example,
        enable_thinking=True,
    )

    assert rendered.thinking_mode_supported
    assert rendered.thinking_mode_requested
    assert rendered.checks.thinking_mode_disabled
    assert tokenizer.calls[0]["enable_thinking"] is True
    assert tokenizer.calls[1]["enable_thinking"] is True


def test_render_template_example_passes_enable_thinking_through_kwargs():
    tokenizer = VariadicThinkingTokenizer()
    record = _make_record(
        "xlam-203",
        split="train",
        source_id=203,
        query="Weather",
        tools=[_make_tool("weather")],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}}
        ],
    )
    example = select_representative_examples([record], count=1)[0]

    rendered = render_template_example(
        tokenizer,
        example,
        enable_thinking=False,
    )

    assert rendered.thinking_mode_supported
    assert not rendered.thinking_mode_requested
    assert tokenizer.calls[0]["enable_thinking"] is False
    assert tokenizer.calls[1]["enable_thinking"] is False


def test_render_template_example_accepts_empty_think_tags_when_disabled():
    tokenizer = EmptyThinkTokenizer()
    record = _make_record(
        "xlam-204",
        split="train",
        source_id=204,
        query="Weather",
        tools=[_make_tool("weather")],
        calls=[
            {"name": "weather", "arguments": {"city": "Denver"}}
        ],
    )
    example = select_representative_examples([record], count=1)[0]

    rendered = render_template_example(
        tokenizer,
        example,
        enable_thinking=False,
    )

    assert rendered.checks.thinking_mode_disabled
    assert (
        "thinking markers were present while thinking mode was disabled"
        not in rendered.checks.failures
    )
