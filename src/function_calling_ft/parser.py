from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any


CALL_LIST_KEYS = (
    "tool_calls",
    "calls",
    "parallel_calls",
    "parallel_tool_calls",
    "function_calls",
)


@dataclass(frozen=True)
class ToolCall:
    name: str | None
    arguments: dict[str, Any] | None


@dataclass(frozen=True)
class ParseResult:
    calls: tuple[ToolCall, ...]
    valid_structure: bool
    errors: tuple[str, ...]
    had_extra_prose: bool = False


def _unique_errors(errors: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []

    for error in errors:
        if error not in seen:
            seen.add(error)
            ordered.append(error)

    return tuple(ordered)


def _extract_json_fragments(
    text: str,
) -> list[tuple[int, int, Any]]:
    decoder = JSONDecoder()
    fragments: list[tuple[int, int, Any]] = []
    index = 0

    while index < len(text):
        if text[index] not in "[{":
            index += 1
            continue

        try:
            payload, end = decoder.raw_decode(text, index)
        except JSONDecodeError:
            index += 1
            continue

        fragments.append((index, end, payload))
        index = end

    return fragments


def _has_extra_prose(
    text: str,
    fragments: list[tuple[int, int, Any]],
) -> bool:
    cursor = 0

    for start, end, _ in fragments:
        if text[cursor:start].strip():
            return True
        cursor = end

    return bool(text[cursor:].strip())


def _parse_arguments(
    value: Any,
    *,
    path: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except JSONDecodeError:
            errors.append(
                f"{path}.arguments contains invalid JSON."
            )
            return None, errors

    if not isinstance(value, dict):
        errors.append(
            f"{path}.arguments must be a JSON object."
        )
        return None, errors

    return value, errors


def _normalize_single_call(
    payload: Any,
    *,
    path: str,
) -> tuple[ToolCall | None, list[str]]:
    if not isinstance(payload, dict):
        return None, [f"{path} must be an object."]

    if (
        payload.get("type") == "function"
        and isinstance(payload.get("function"), dict)
    ):
        payload = payload["function"]
        path = f"{path}.function"
    elif isinstance(payload.get("function"), dict):
        payload = payload["function"]
        path = f"{path}.function"

    errors: list[str] = []
    name = payload.get("name")

    if not isinstance(name, str) or not name.strip():
        errors.append(f"{path}.name must be a non-empty string.")
        normalized_name = None
    else:
        normalized_name = name.strip()

    if "arguments" not in payload:
        errors.append(f"{path}.arguments is missing.")
        normalized_arguments = None
    else:
        normalized_arguments, argument_errors = _parse_arguments(
            payload.get("arguments"),
            path=path,
        )
        errors.extend(argument_errors)

    return (
        ToolCall(
            name=normalized_name,
            arguments=normalized_arguments,
        ),
        errors,
    )


def _normalize_call_collection(
    payload: Any,
    *,
    path: str,
) -> tuple[list[ToolCall], list[str]]:
    calls: list[ToolCall] = []
    errors: list[str] = []

    if isinstance(payload, ToolCall):
        return [payload], []

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            item_calls, item_errors = _normalize_call_collection(
                item,
                path=f"{path}[{index}]",
            )
            calls.extend(item_calls)
            errors.extend(item_errors)
        return calls, errors

    if isinstance(payload, dict):
        for key in CALL_LIST_KEYS:
            if key in payload:
                return _normalize_call_collection(
                    payload[key],
                    path=f"{path}.{key}",
                )

        call, call_errors = _normalize_single_call(
            payload,
            path=path,
        )

        if call is not None:
            calls.append(call)

        errors.extend(call_errors)
        return calls, errors

    return [], [f"{path} must be a JSON object or list."]


def parse_tool_calls(
    value: str | dict[str, Any] | list[Any] | ToolCall | None,
) -> ParseResult:
    if isinstance(value, str):
        text = value.strip()

        if not text:
            return ParseResult(
                calls=(),
                valid_structure=False,
                errors=("Model output is empty.",),
            )

        fragments = _extract_json_fragments(text)

        if not fragments:
            return ParseResult(
                calls=(),
                valid_structure=False,
                errors=(
                    "No JSON object or array found in model output.",
                ),
            )

        calls: list[ToolCall] = []
        errors: list[str] = []

        for index, (_, _, payload) in enumerate(fragments):
            fragment_calls, fragment_errors = (
                _normalize_call_collection(
                    payload,
                    path=f"fragment[{index}]",
                )
            )
            calls.extend(fragment_calls)
            errors.extend(fragment_errors)

        return ParseResult(
            calls=tuple(calls),
            valid_structure=bool(calls) and not errors,
            errors=_unique_errors(errors),
            had_extra_prose=_has_extra_prose(text, fragments),
        )

    calls, errors = _normalize_call_collection(value, path="root")
    return ParseResult(
        calls=tuple(calls),
        valid_structure=bool(calls) and not errors,
        errors=_unique_errors(errors),
    )
