from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, Sequence

from function_calling_ft.generation import (
    GenerationTokenizer,
    build_inference_messages,
    normalize_token_ids,
)


PROMPT_AUDIT_SCHEMA_VERSION = "1.0"


class PromptAuditTokenizer(GenerationTokenizer, Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        ...


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ids(input_ids: Sequence[int]) -> str:
    payload = json.dumps(list(input_ids), separators=(",", ":"))
    return _sha256_text(payload)


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _expected_response_text(record: dict[str, Any]) -> str | None:
    expected = record.get("expected_response")
    if isinstance(expected, dict):
        value = expected.get("content")
        if isinstance(value, str):
            return value
    return None


def _hidden_expected_response(
    *,
    expected_text: str | None,
    rendered_text: str,
    messages: list[dict[str, Any]],
) -> bool:
    if expected_text is None:
        return False

    normalized_expected = " ".join(expected_text.casefold().split())
    if len(normalized_expected) < 12:
        return False

    user_text = " ".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )
    normalized_user_text = " ".join(user_text.casefold().split())
    if normalized_expected in normalized_user_text:
        return False

    normalized_rendered = " ".join(rendered_text.casefold().split())
    return normalized_expected in normalized_rendered


def prompt_audit_record(
    *,
    tokenizer: PromptAuditTokenizer,
    record: dict[str, Any],
    enable_thinking: bool = False,
) -> dict[str, Any]:
    messages = build_inference_messages(record)
    tools_value = record.get("tools", [])
    tools = tools_value if isinstance(tools_value, list) else []
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    rendered_text = str(rendered)
    tokenized = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    input_ids = normalize_token_ids(tokenized)
    expected_text = _expected_response_text(record)
    hidden_expected_response = _hidden_expected_response(
        expected_text=expected_text,
        rendered_text=rendered_text,
        messages=messages,
    )
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    return {
        "prompt_audit_schema_version": PROMPT_AUDIT_SCHEMA_VERSION,
        "id": str(record.get("id")),
        "prompt_sha256": _sha256_text(rendered_text),
        "input_ids_sha256": _sha256_ids(input_ids),
        "prompt_token_count": len(input_ids),
        "message_count": len(messages),
        "tool_count": len(tools),
        "tool_names": _tool_names(tools),
        "enable_thinking": enable_thinking,
        "assistant_messages_in_prompt": len(assistant_messages),
        "hidden_expected_response": hidden_expected_response,
        "has_target_tool_calls_in_prompt": any(
            message.get("tool_calls") is not None
            for message in assistant_messages
        ),
    }


def summarize_prompt_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_hashes = {str(record["prompt_sha256"]) for record in records}
    return {
        "prompt_audit_schema_version": PROMPT_AUDIT_SCHEMA_VERSION,
        "records": len(records),
        "unique_prompt_hashes": len(prompt_hashes),
        "hidden_expected_response_count": sum(
            int(bool(record.get("hidden_expected_response")))
            for record in records
        ),
        "target_tool_call_leak_count": sum(
            int(bool(record.get("has_target_tool_calls_in_prompt")))
            for record in records
        ),
        "assistant_messages_in_prompt_count": sum(
            int(int(record.get("assistant_messages_in_prompt", 0) or 0) > 0)
            for record in records
        ),
    }
