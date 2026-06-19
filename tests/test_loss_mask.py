from __future__ import annotations

import json
import re

from function_calling_ft.loss_mask import (
    IGNORE_INDEX,
    build_expected_loss_mask,
    format_loss_mask_diagnostic,
)


class FakeLossMaskTokenizer:
    TOKEN_PATTERN = re.compile(
        r"<\|im_start\|>|<\|im_end\|>|<tools>|</tools>|"
        r"<tool_call>|</tool_call>|<tool_response>|</tool_response>|"
        r"<think>|</think>|\n| +|[^\s<]+|<"
    )

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self._token_to_id = {self.pad_token: self.pad_token_id}
        self._id_to_token = {self.pad_token_id: self.pad_token}
        self._next_id = 10

    def _token_id(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._token_to_id[token]

    def apply_chat_template(
        self,
        conversation,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del add_generation_prompt
        text = self._render(
            conversation,
            tools=tools or [],
            enable_thinking=enable_thinking,
        )

        if tokenize:
            encoding = self(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            return {"input_ids": encoding["input_ids"]}

        return text

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        tokens = []
        offsets = []

        for match in self.TOKEN_PATTERN.finditer(text):
            token = match.group(0)
            tokens.append(self._token_id(token))
            offsets.append(match.span())

        result = {"input_ids": tokens}

        if return_offsets_mapping:
            result["offset_mapping"] = offsets

        return result

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
    ) -> str:
        del skip_special_tokens
        return "".join(self._id_to_token[token_id] for token_id in token_ids)

    def _render(
        self,
        conversation,
        *,
        tools,
        enable_thinking: bool,
    ) -> str:
        parts = [
            "<|im_start|>system\n",
            "# Tools\n\n",
            "You may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n",
            "<tools>\n",
        ]

        for tool in tools:
            parts.append(
                json.dumps(tool, ensure_ascii=False, sort_keys=True)
            )
            parts.append("\n")

        parts.extend(
            [
                "</tools>\n\n",
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
                "<tool_call>\n",
                '{"name": <function-name>, "arguments": <args-json-object>}\n',
                "</tool_call><|im_end|>\n",
            ]
        )

        for message in conversation:
            if message["role"] == "user":
                parts.append("<|im_start|>user\n")
                parts.append(message["content"])
                parts.append("<|im_end|>\n")
                continue

            if message["role"] == "assistant":
                parts.append("<|im_start|>assistant\n")

                if not enable_thinking:
                    parts.append("<think>\n\n</think>\n\n")

                content = message.get("content", "")
                if content:
                    parts.append(content)

                for index, tool_call in enumerate(
                    message.get("tool_calls", [])
                ):
                    if content or index > 0:
                        parts.append("\n")
                    function = tool_call["function"]
                    payload = {
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                    parts.append("<tool_call>\n")
                    parts.append(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    parts.append("\n</tool_call>")

                parts.append("<|im_end|>\n")
                continue

            if message["role"] == "tool":
                parts.append("<|im_start|>user\n")
                parts.append("<tool_response>\n")
                parts.append(message["content"])
                parts.append("\n</tool_response><|im_end|>\n")
                continue

            raise ValueError(f"Unsupported role: {message['role']}")

        return "".join(parts)


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_loss_mask_marks_only_assistant_tool_call_tokens_for_tool_call_turn():
    tokenizer = FakeLossMaskTokenizer()
    messages = [
        {"role": "user", "content": "Weather in Denver?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Denver"},
                    },
                }
            ],
        },
    ]

    result = build_expected_loss_mask(
        tokenizer,
        messages,
        tools=_tools(),
    )

    assert all(
        token.label == IGNORE_INDEX
        for token in result.tokens
        if token.region
        in {
            "system_prompt",
            "tool_definitions",
            "user_request",
            "assistant_scaffolding",
            "assistant_thinking",
        }
    )
    assert any(
        token.token_text == "<tool_call>"
        and token.region == "assistant_tool_call"
        and token.label == token.token_id
        for token in result.tokens
    )
    assert any(
        "get_weather" in token.token_text
        and token.region == "assistant_tool_call"
        and token.label == token.token_id
        for token in result.tokens
    )
    assert any(
        "arguments" in token.token_text
        and token.region == "assistant_tool_call"
        and token.label == token.token_id
        for token in result.tokens
    )
    assert any(
        "city" in token.token_text
        and token.region == "assistant_tool_call"
        and token.label == token.token_id
        for token in result.tokens
    )
    assert any(
        "Denver" in token.token_text
        and token.region == "assistant_tool_call"
        and token.label == token.token_id
        for token in result.tokens
    )


def test_loss_mask_excludes_tool_results_and_final_answer():
    tokenizer = FakeLossMaskTokenizer()
    messages = [
        {"role": "user", "content": "Weather in Denver?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Denver"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": '{"temperature":72}',
        },
        {
            "role": "assistant",
            "content": "The weather is 72 F.",
        },
    ]

    result = build_expected_loss_mask(
        tokenizer,
        messages,
        tools=_tools(),
    )

    assert all(
        token.label == IGNORE_INDEX
        for token in result.tokens
        if token.region == "tool_execution_result"
    )
    assert any(
        token.region == "assistant_final_answer"
        for token in result.tokens
    )
    assert all(
        token.label == IGNORE_INDEX
        for token in result.tokens
        if token.region == "assistant_final_answer"
    )


def test_loss_mask_marks_padding_as_ignored():
    tokenizer = FakeLossMaskTokenizer()
    messages = [
        {"role": "user", "content": "Weather in Denver?"},
        {
            "role": "assistant",
            "content": "Done.",
        },
    ]

    unpadded = build_expected_loss_mask(
        tokenizer,
        messages,
        tools=_tools(),
    )
    padded = build_expected_loss_mask(
        tokenizer,
        messages,
        tools=_tools(),
        pad_to_length=len(unpadded.input_ids) + 3,
    )

    assert len(padded.input_ids) == len(unpadded.input_ids) + 3
    assert all(
        token.region == "padding" and token.label == IGNORE_INDEX
        for token in padded.tokens[-3:]
    )


def test_loss_mask_diagnostic_prints_token_and_label_columns():
    tokenizer = FakeLossMaskTokenizer()
    messages = [
        {"role": "user", "content": "Weather in Denver?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Denver"},
                    },
                }
            ],
        },
    ]

    result = build_expected_loss_mask(
        tokenizer,
        messages,
        tools=_tools(),
    )
    diagnostic = format_loss_mask_diagnostic(
        result,
        max_rows=200,
    )

    assert "TOKEN" in diagnostic
    assert "LABEL" in diagnostic
    assert "<tool_call>" in diagnostic
