from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from function_calling_ft.normalization import (
    NormalizationError,
    normalize_xlam_row,
)


DEFAULT_CONTEXT_TOKEN_LIMIT = 8_192
CHARS_PER_TOKEN_ESTIMATE = 4
SPLIT_PRIORITY = ("train", "validation", "test")


@dataclass(frozen=True)
class ValidationIssue:
    category: str
    message: str


@dataclass(frozen=True)
class ExampleValidationResult:
    normalized: dict[str, Any] | None
    issues: tuple[ValidationIssue, ...]
    estimated_tokens: int | None

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _issue(
    category: str,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(category=category, message=message)


def classify_normalization_validation_issue(
    error: Exception,
) -> ValidationIssue:
    message = str(error)
    lowered = message.lower()

    if "references unavailable tool" in lowered:
        return _issue(
            "unavailable_expected_tool",
            message,
        )

    if (
        "arguments for call" in lowered
        and "invalid json" in lowered
    ):
        return _issue(
            "invalid_argument_object",
            message,
        )

    if (
        "arguments for call" in lowered
        and "must be an object" in lowered
    ):
        return _issue(
            "invalid_argument_object",
            message,
        )

    if "answers" in lowered or "answer " in lowered:
        return _issue(
            "unparseable_expected_calls",
            message,
        )

    return _issue(
        "invalid_tool_schema",
        message,
    )


def render_validation_payload(
    normalized: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "tools": normalized["tools"],
            "messages": normalized["messages"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def estimate_rendered_tokens(
    normalized: dict[str, Any],
) -> int:
    payload = render_validation_payload(normalized)
    return max(
        1,
        math.ceil(len(payload) / CHARS_PER_TOKEN_ESTIMATE),
    )


def _type_name(value: Any) -> str:
    return type(value).__name__


def _matches_type(
    schema_type: str,
    value: Any,
) -> bool:
    if schema_type == "string":
        return isinstance(value, str)

    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(
            value,
            bool,
        )

    if schema_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if schema_type == "boolean":
        return isinstance(value, bool)

    if schema_type == "array":
        return isinstance(value, list)

    if schema_type == "object":
        return isinstance(value, dict)

    if schema_type == "null":
        return value is None

    return True


def _validate_schema_fragment(
    schema: Any,
    *,
    path: str,
) -> list[ValidationIssue]:
    if not isinstance(schema, dict):
        return [
            _issue(
                "invalid_tool_schema",
                f"{path} must be an object schema.",
            )
        ]

    issues: list[ValidationIssue] = []
    allowed_types = {
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "null",
    }

    if "anyOf" in schema:
        any_of = schema["anyOf"]

        if not isinstance(any_of, list) or not any_of:
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"{path}.anyOf must be a non-empty list.",
                )
            )
            return issues

        for index, child in enumerate(any_of):
            issues.extend(
                _validate_schema_fragment(
                    child,
                    path=f"{path}.anyOf[{index}]",
                )
            )

        return issues

    schema_type = schema.get("type")

    if schema_type is None:
        issues.append(
            _issue(
                "invalid_tool_schema",
                f"{path} does not define a type.",
            )
        )
        return issues

    if not isinstance(schema_type, str) or schema_type not in allowed_types:
        issues.append(
            _issue(
                "invalid_tool_schema",
                f"{path} has unsupported type {schema_type!r}.",
            )
        )
        return issues

    if schema_type == "object":
        properties = schema.get("properties", {})

        if not isinstance(properties, dict):
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"{path}.properties must be an object.",
                )
            )
        else:
            for name, child_schema in properties.items():
                issues.extend(
                    _validate_schema_fragment(
                        child_schema,
                        path=f"{path}.properties.{name}",
                    )
                )

        required = schema.get("required")
        if required is not None:
            if not isinstance(required, list) or not all(
                isinstance(item, str) for item in required
            ):
                issues.append(
                    _issue(
                        "invalid_tool_schema",
                        f"{path}.required must be a list of strings.",
                    )
                )

        additional_properties = schema.get("additionalProperties")
        if isinstance(additional_properties, dict):
            issues.extend(
                _validate_schema_fragment(
                    additional_properties,
                    path=f"{path}.additionalProperties",
                )
            )
        elif (
            additional_properties is not None
            and not isinstance(additional_properties, bool)
        ):
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"{path}.additionalProperties must be a schema or boolean.",
                )
            )

    if schema_type == "array":
        items = schema.get("items")
        prefix_items = schema.get("prefixItems")

        if items is not None:
            issues.extend(
                _validate_schema_fragment(
                    items,
                    path=f"{path}.items",
                )
            )

        if prefix_items is not None:
            if not isinstance(prefix_items, list):
                issues.append(
                    _issue(
                        "invalid_tool_schema",
                        f"{path}.prefixItems must be a list.",
                    )
                )
            else:
                for index, child_schema in enumerate(prefix_items):
                    issues.extend(
                        _validate_schema_fragment(
                            child_schema,
                            path=(
                                f"{path}.prefixItems[{index}]"
                            ),
                        )
                    )

    return issues


def _validate_value_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if "anyOf" in schema:
        variants = schema["anyOf"]
        if any(
            not _validate_value_against_schema(
                value,
                variant,
                path=path,
            )
            for variant in variants
            if isinstance(variant, dict)
        ):
            return []

        issues.append(
            _issue(
                "incompatible_argument_type",
                (
                    f"{path} has incompatible type "
                    f"{_type_name(value)}."
                ),
            )
        )
        return issues

    schema_type = schema.get("type")

    if isinstance(schema_type, str) and not _matches_type(
        schema_type,
        value,
    ):
        issues.append(
            _issue(
                "incompatible_argument_type",
                (
                    f"{path} expects {schema_type} but received "
                    f"{_type_name(value)}."
                ),
            )
        )
        return issues

    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for key in required:
            if key not in value:
                issues.append(
                    _issue(
                        "missing_required_arguments",
                        f"{path}.{key} is required.",
                    )
                )

        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    issues.extend(
                        _validate_value_against_schema(
                            value[key],
                            child_schema,
                            path=f"{path}.{key}",
                        )
                    )

        additional_properties = schema.get("additionalProperties")
        if isinstance(additional_properties, dict):
            defined_keys = (
                set(properties)
                if isinstance(properties, dict)
                else set()
            )

            for key, child_value in value.items():
                if key not in defined_keys:
                    issues.extend(
                        _validate_value_against_schema(
                            child_value,
                            additional_properties,
                            path=f"{path}.{key}",
                        )
                    )

    if schema_type == "array" and isinstance(value, list):
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            for index, child_schema in enumerate(prefix_items):
                if index >= len(value):
                    break
                if isinstance(child_schema, dict):
                    issues.extend(
                        _validate_value_against_schema(
                            value[index],
                            child_schema,
                            path=f"{path}[{index}]",
                        )
                    )

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            start_index = len(prefix_items) if isinstance(
                prefix_items,
                list,
            ) else 0
            for index in range(start_index, len(value)):
                issues.extend(
                    _validate_value_against_schema(
                        value[index],
                        item_schema,
                        path=f"{path}[{index}]",
                    )
                )

    return issues


def validate_normalized_example(
    normalized: dict[str, Any],
    *,
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> ExampleValidationResult:
    issues: list[ValidationIssue] = []
    tool_schemas: dict[str, dict[str, Any]] = {}

    tools = normalized.get("tools")
    if not isinstance(tools, list):
        return ExampleValidationResult(
            normalized=normalized,
            issues=(
                _issue(
                    "invalid_tool_schema",
                    "Normalized tools must be a list.",
                ),
            ),
            estimated_tokens=None,
        )

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"tools[{index}] must be an object.",
                )
            )
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"tools[{index}].function must be an object.",
                )
            )
            continue

        name = function.get("name")
        parameters = function.get("parameters")

        if not isinstance(name, str) or not name.strip():
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"tools[{index}] has no valid function name.",
                )
            )
            continue

        if name in tool_schemas:
            issues.append(
                _issue(
                    "invalid_tool_schema",
                    f"Duplicate tool name {name!r} is present.",
                )
            )
            continue

        schema_issues = _validate_schema_fragment(
            parameters,
            path=f"tools[{index}].function.parameters",
        )
        issues.extend(schema_issues)

        if not schema_issues and isinstance(parameters, dict):
            tool_schemas[name] = parameters

    assistant = None
    messages = normalized.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        assistant = messages[1]

    tool_calls = (
        assistant.get("tool_calls")
        if isinstance(assistant, dict)
        else None
    )

    if not isinstance(tool_calls, list):
        issues.append(
            _issue(
                "unparseable_expected_calls",
                "Assistant tool_calls must be a list.",
            )
        )
    else:
        for index, tool_call in enumerate(tool_calls, start=1):
            function = (
                tool_call.get("function")
                if isinstance(tool_call, dict)
                else None
            )

            if not isinstance(function, dict):
                issues.append(
                    _issue(
                        "unparseable_expected_calls",
                        f"tool_call {index} has no valid function payload.",
                    )
                )
                continue

            name = function.get("name")
            arguments = function.get("arguments")

            if not isinstance(name, str) or not name.strip():
                issues.append(
                    _issue(
                        "unparseable_expected_calls",
                        f"tool_call {index} has no valid function name.",
                    )
                )
                continue

            if name not in tool_schemas:
                issues.append(
                    _issue(
                        "unavailable_expected_tool",
                        (
                            f"tool_call {index} references unavailable tool "
                            f"{name!r}."
                        ),
                    )
                )
                continue

            if not isinstance(arguments, dict):
                issues.append(
                    _issue(
                        "invalid_argument_object",
                        (
                            f"tool_call {index} arguments must be an object; "
                            f"received {_type_name(arguments)}."
                        ),
                    )
                )
                continue

            issues.extend(
                _validate_value_against_schema(
                    arguments,
                    tool_schemas[name],
                    path=f"{name}.arguments",
                )
            )

    estimated_tokens = estimate_rendered_tokens(normalized)
    if estimated_tokens > context_token_limit:
        issues.append(
            _issue(
                "context_length_exceeded",
                (
                    f"Estimated rendered sequence length "
                    f"{estimated_tokens} tokens exceeds target context "
                    f"limit {context_token_limit}."
                ),
            )
        )

    unique_issues: list[ValidationIssue] = []
    seen = set()
    for issue in issues:
        key = (issue.category, issue.message)
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    return ExampleValidationResult(
        normalized=normalized,
        issues=tuple(unique_issues),
        estimated_tokens=estimated_tokens,
    )


def validate_raw_example(
    row: dict[str, Any],
    *,
    split: str,
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> ExampleValidationResult:
    try:
        normalized = normalize_xlam_row(row, split=split)
    except (NormalizationError, TypeError, ValueError) as error:
        return ExampleValidationResult(
            normalized=None,
            issues=(
                classify_normalization_validation_issue(error),
            ),
            estimated_tokens=None,
        )

    return validate_normalized_example(
        normalized,
        context_token_limit=context_token_limit,
    )
