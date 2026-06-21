from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from function_calling_ft.normalization import normalize_xlam_row
from function_calling_ft.parser import ParseResult, parse_tool_calls
from function_calling_ft.scorer import score_calls


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_RAW_DIR = Path("data/smoke/raw")
DEFAULT_NORMALIZED_DIR = Path("data/smoke/normalized")
DEFAULT_TEMPLATE_CACHE_DIR = Path(".cache/huggingface")
SPLIT_ORDER = ("train", "validation", "test")

TOOL_SECTION_TAGS = (
    ("<tools>", "</tools>"),
    ("<|tools_start|>", "<|tools_end|>"),
)

TOOL_CALL_TAGS = (
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call_start|>", "<|tool_call_end|>"),
)

THINKING_MARKERS = (
    "<think>",
    "</think>",
    "<|thinking|>",
    "<|think_start|>",
    "<|think_end|>",
)

SINGLE_QUOTED_KEY_PATTERN = re.compile(
    r"([{,]\s*)'[^']+'\s*:"
)
ARGUMENT_OBJECT_PATTERN = re.compile(
    r'"arguments"\s*:\s*\{'
)


class TemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str | Sequence[int] | Sequence[Sequence[int]]:
        ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        ...


@dataclass(frozen=True)
class RepresentativeExample:
    record: dict[str, Any]
    feature_tags: tuple[str, ...]

    @property
    def split(self) -> str:
        return str(self.record["metadata"]["split"])

    @property
    def record_id(self) -> str:
        return str(self.record["id"])

    @property
    def source_id(self) -> int:
        return int(self.record["metadata"]["source_id"])


@dataclass(frozen=True)
class TemplateChecks:
    tool_section_present: bool
    tool_definitions_before_user_message: bool
    function_names_unchanged: bool
    arguments_as_json_objects: bool
    tool_call_delimiters_present: bool
    no_python_dict_syntax: bool
    multiple_calls_serialized_correctly: bool
    thinking_mode_disabled: bool
    decoded_matches_rendered: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RenderedTemplateExample:
    example: RepresentativeExample
    rendered_text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    tool_section_tags: tuple[str, str] | None
    tool_call_delimiters: tuple[str, ...]
    expected_calls: tuple[dict[str, Any], ...]
    rendered_calls: tuple[dict[str, Any], ...]
    thinking_mode_requested: bool
    thinking_mode_supported: bool
    checks: TemplateChecks

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))

    return records


def _sort_record_key(record: dict[str, Any]) -> tuple[int, int, str]:
    split = str(record["metadata"]["split"])
    return (
        SPLIT_ORDER.index(split),
        int(record["metadata"]["source_id"]),
        str(record["id"]),
    )


def load_smoke_records(
    *,
    normalized_dir: Path = DEFAULT_NORMALIZED_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[str, list[dict[str, Any]]]:
    normalized_paths = [
        normalized_dir / f"{split}.jsonl" for split in SPLIT_ORDER
    ]

    if all(path.is_file() for path in normalized_paths):
        records: list[dict[str, Any]] = []

        for path in normalized_paths:
            records.extend(_read_jsonl(path))

        return "normalized", sorted(records, key=_sort_record_key)

    raw_paths = [raw_dir / f"{split}.jsonl" for split in SPLIT_ORDER]

    if not all(path.is_file() for path in raw_paths):
        raise FileNotFoundError(
            "Smoke dataset files not found. Expected normalized files in "
            f"{normalized_dir} or raw files in {raw_dir}."
        )

    records = []

    for split, path in zip(SPLIT_ORDER, raw_paths, strict=True):
        for row in _read_jsonl(path):
            records.append(normalize_xlam_row(row, split=split))

    return "raw_normalized_on_the_fly", sorted(
        records,
        key=_sort_record_key,
    )


def extract_expected_tool_calls(
    record: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []

    for message in record.get("messages", []):
        for tool_call in message.get("tool_calls", []):
            function = tool_call["function"]
            calls.append(
                {
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )

    return tuple(calls)


def _called_tool_names(record: dict[str, Any]) -> list[str]:
    return [
        call["name"]
        for call in extract_expected_tool_calls(record)
    ]


def _has_repeated_tool_calls(record: dict[str, Any]) -> bool:
    call_names = _called_tool_names(record)
    return len(call_names) != len(set(call_names))


def _has_multiple_called_tools(record: dict[str, Any]) -> bool:
    return len(set(_called_tool_names(record))) >= 2


def _has_complex_parameter_types(record: dict[str, Any]) -> bool:
    def walk(schema: Any) -> bool:
        if isinstance(schema, dict):
            schema_type = schema.get("type")

            if (
                isinstance(schema_type, str)
                and schema_type in {"array", "object"}
            ) or "anyOf" in schema:
                return True

            return any(walk(value) for value in schema.values())

        if isinstance(schema, list):
            return any(walk(value) for value in schema)

        return False

    for tool in record.get("tools", []):
        properties = (
            tool.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )

        if any(walk(schema) for schema in properties.values()):
            return True

    return False


def feature_tags_for_record(
    record: dict[str, Any],
) -> tuple[str, ...]:
    tags: list[str] = []
    call_count = int(record["metadata"]["expected_call_count"])
    available_tool_count = int(
        record["metadata"]["available_tool_count"]
    )

    if call_count == 1:
        tags.append("single_call")
    elif call_count == 2:
        tags.append("two_calls")
    else:
        tags.append("three_plus_calls")

    if _has_multiple_called_tools(record):
        tags.append("multiple_called_tools")

    if _has_repeated_tool_calls(record):
        tags.append("repeated_tool_calls")

    if available_tool_count >= 5:
        tags.append("five_plus_available_tools")

    if _has_complex_parameter_types(record):
        tags.append("complex_parameter_types")

    return tuple(tags)


def select_representative_examples(
    records: Sequence[dict[str, Any]],
    *,
    count: int = 5,
) -> list[RepresentativeExample]:
    sorted_records = sorted(records, key=_sort_record_key)
    selected: list[RepresentativeExample] = []
    used_ids: set[str] = set()

    selectors: tuple[tuple[str, Any], ...] = (
        (
            "single_call",
            lambda record: int(
                record["metadata"]["expected_call_count"]
            )
            == 1,
        ),
        (
            "two_calls_distinct_tools",
            lambda record: int(
                record["metadata"]["expected_call_count"]
            )
            == 2
            and _has_multiple_called_tools(record),
        ),
        (
            "three_plus_calls",
            lambda record: int(
                record["metadata"]["expected_call_count"]
            )
            >= 3,
        ),
        (
            "repeated_tool_calls",
            _has_repeated_tool_calls,
        ),
        (
            "five_plus_available_tools",
            lambda record: int(
                record["metadata"]["available_tool_count"]
            )
            >= 5
            and _has_complex_parameter_types(record),
        ),
    )

    for selector_name, predicate in selectors[:count]:
        for record in sorted_records:
            record_id = str(record["id"])

            if record_id in used_ids or not predicate(record):
                continue

            feature_tags = feature_tags_for_record(record)

            if selector_name not in feature_tags:
                feature_tags = (selector_name,) + feature_tags

            selected.append(
                RepresentativeExample(
                    record=record,
                    feature_tags=feature_tags,
                )
            )
            used_ids.add(record_id)
            break

    if len(selected) < count:
        for record in sorted_records:
            record_id = str(record["id"])

            if record_id in used_ids:
                continue

            selected.append(
                RepresentativeExample(
                    record=record,
                    feature_tags=feature_tags_for_record(record),
                )
            )
            used_ids.add(record_id)

            if len(selected) == count:
                break

    return selected


def _tokenizer_kwargs(
    tokenizer: TemplateTokenizer,
    *,
    enable_thinking: bool,
) -> tuple[dict[str, Any], bool]:
    signature = inspect.signature(
        tokenizer.apply_chat_template
    )
    kwargs: dict[str, Any] = {}
    thinking_mode_supported = (
        "enable_thinking" in signature.parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )

    if thinking_mode_supported:
        kwargs["enable_thinking"] = enable_thinking

    return kwargs, thinking_mode_supported


def _normalize_token_ids(
    token_ids: Any,
) -> tuple[int, ...]:
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    elif hasattr(token_ids, "get") and callable(token_ids.get):
        maybe_input_ids = token_ids.get("input_ids")
        if maybe_input_ids is not None:
            token_ids = maybe_input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if (
        isinstance(token_ids, list)
        and token_ids
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]

    return tuple(int(token_id) for token_id in token_ids)


def _extract_tagged_section(
    text: str,
    tag_pairs: Sequence[tuple[str, str]],
) -> tuple[str | None, tuple[str, str] | None]:
    best_section: str | None = None
    best_tags: tuple[str, str] | None = None

    for start_tag, end_tag in tag_pairs:
        cursor = 0

        while True:
            start_index = text.find(start_tag, cursor)

            if start_index < 0:
                break

            end_index = text.find(
                end_tag,
                start_index + len(start_tag),
            )

            if end_index < 0:
                break

            section = text[
                start_index + len(start_tag) : end_index
            ].strip()

            if best_section is None or len(section) > len(
                best_section
            ):
                best_section = section
                best_tags = (start_tag, end_tag)

            cursor = end_index + len(end_tag)

    return best_section, best_tags


def detect_tool_call_delimiters(
    text: str,
) -> tuple[str, ...]:
    found: list[str] = []

    for start_tag, end_tag in TOOL_SECTION_TAGS + TOOL_CALL_TAGS:
        if start_tag in text:
            found.append(start_tag)
        if end_tag in text:
            found.append(end_tag)

    return tuple(found)


def _thinking_content_present(text: str) -> bool:
    if "<think>" in text and "</think>" in text:
        cursor = 0
        saw_think_section = False

        while True:
            start_index = text.find("<think>", cursor)

            if start_index < 0:
                break

            end_index = text.find(
                "</think>",
                start_index + len("<think>"),
            )

            if end_index < 0:
                break

            saw_think_section = True
            content = text[
                start_index + len("<think>") : end_index
            ]

            if content.strip():
                return True

            cursor = end_index + len("</think>")

        if saw_think_section:
            return False

    return any(
        marker in text
        for marker in THINKING_MARKERS
        if marker not in {"<think>", "</think>"}
    )


def _parse_payload_text(
    payload_text: str,
) -> ParseResult:
    return parse_tool_calls(payload_text)


def _extract_rendered_tool_calls(
    text: str,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    for start_tag, end_tag in TOOL_CALL_TAGS:
        payloads: list[str] = []
        cursor = 0

        while True:
            start_index = text.find(start_tag, cursor)

            if start_index < 0:
                break

            start_index += len(start_tag)
            end_index = text.find(end_tag, start_index)

            if end_index < 0:
                break

            payloads.append(text[start_index:end_index].strip())
            cursor = end_index + len(end_tag)

        if payloads:
            extracted_calls: list[dict[str, Any]] = []

            for payload in payloads:
                result = _parse_payload_text(payload)
                for call in result.calls:
                    extracted_calls.append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    )

            return tuple(payloads), tuple(extracted_calls)

    result = parse_tool_calls(text)
    parsed_calls: tuple[dict[str, Any], ...] = tuple(
        {
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in result.calls
    )
    return (text,), parsed_calls


def render_template_example(
    tokenizer: TemplateTokenizer,
    example: RepresentativeExample,
    *,
    enable_thinking: bool = False,
) -> RenderedTemplateExample:
    tokenizer_kwargs, thinking_mode_supported = (
        _tokenizer_kwargs(
            tokenizer,
            enable_thinking=enable_thinking,
        )
    )
    messages = example.record["messages"]
    tools = example.record["tools"]

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **tokenizer_kwargs,
    )

    if not isinstance(rendered_text, str):
        raise TypeError(
            "Tokenizer returned a non-string chat template when "
            "tokenize=False."
        )

    token_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **tokenizer_kwargs,
        )
    )
    decoded_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
    )
    expected_calls = extract_expected_tool_calls(example.record)
    payload_texts, rendered_calls = _extract_rendered_tool_calls(
        decoded_text
    )
    tool_section_text, tool_section_tags = _extract_tagged_section(
        decoded_text,
        TOOL_SECTION_TAGS,
    )
    user_content = str(messages[0]["content"])
    user_index = decoded_text.find(user_content)

    if tool_section_text is not None:
        tool_names_in_section = all(
            tool["function"]["name"] in tool_section_text
            for tool in tools
        )
        assert tool_section_tags is not None
        section_end_index = decoded_text.find(
            tool_section_tags[1]
        ) + len(tool_section_tags[1])
        tool_definitions_before_user_message = (
            user_index < 0 or section_end_index <= user_index
        )
    else:
        first_call_index = len(decoded_text)

        for start_tag, _ in TOOL_CALL_TAGS:
            tag_index = decoded_text.find(start_tag)
            if 0 <= tag_index < first_call_index:
                first_call_index = tag_index

        if first_call_index == len(decoded_text):
            first_call_index = user_index if user_index >= 0 else len(
                decoded_text
            )

        tool_names_in_section = all(
            decoded_text.find(tool["function"]["name"]) >= 0
            and decoded_text.find(tool["function"]["name"])
            < first_call_index
            for tool in tools
        )
        tool_definitions_before_user_message = (
            user_index < 0
            or all(
                decoded_text.find(tool["function"]["name"])
                < user_index
                for tool in tools
            )
        )

    rendered_score = score_calls(
        list(rendered_calls),
        list(expected_calls),
        order_matters=False,
    )
    argument_object_matches = sum(
        len(ARGUMENT_OBJECT_PATTERN.findall(payload))
        for payload in payload_texts
    )
    single_quoted_keys_detected = bool(
        SINGLE_QUOTED_KEY_PATTERN.search(decoded_text)
    )
    tool_call_delimiters = detect_tool_call_delimiters(
        decoded_text
    )
    thinking_content_present = _thinking_content_present(
        decoded_text
    )

    checks = TemplateChecks(
        tool_section_present=tool_names_in_section,
        tool_definitions_before_user_message=(
            tool_definitions_before_user_message
        ),
        function_names_unchanged=(
            rendered_score.correct_function_name
            and len(rendered_calls) == len(expected_calls)
        ),
        arguments_as_json_objects=(
            argument_object_matches >= len(expected_calls)
            and all(
                isinstance(call["arguments"], dict)
                for call in rendered_calls
            )
        ),
        tool_call_delimiters_present=bool(tool_call_delimiters),
        no_python_dict_syntax=not single_quoted_keys_detected,
        multiple_calls_serialized_correctly=(
            len(expected_calls) <= 1
            or (
                rendered_score.valid_structure
                and rendered_score.complete_call_match
            )
        ),
        thinking_mode_disabled=(
            enable_thinking or not thinking_content_present
        ),
        decoded_matches_rendered=(decoded_text == rendered_text),
        failures=(),
    )

    failures: list[str] = []

    if not checks.tool_section_present:
        failures.append("tool definitions not found in the tool section")
    if not checks.tool_definitions_before_user_message:
        failures.append(
            "tool definitions were not rendered before the user message"
        )
    if not checks.function_names_unchanged:
        failures.append("function names changed in rendered tool calls")
    if not checks.arguments_as_json_objects:
        failures.append(
            "tool-call arguments were not rendered as JSON objects"
        )
    if not checks.tool_call_delimiters_present:
        failures.append("tool-call delimiters were not present")
    if not checks.no_python_dict_syntax:
        failures.append(
            "rendered output contains Python-style single-quoted keys"
        )
    if not checks.multiple_calls_serialized_correctly:
        failures.append("multiple tool calls did not round-trip cleanly")
    if not checks.thinking_mode_disabled:
        failures.append(
            "thinking markers were present while thinking mode was disabled"
        )

    checks = TemplateChecks(
        tool_section_present=checks.tool_section_present,
        tool_definitions_before_user_message=(
            checks.tool_definitions_before_user_message
        ),
        function_names_unchanged=checks.function_names_unchanged,
        arguments_as_json_objects=checks.arguments_as_json_objects,
        tool_call_delimiters_present=(
            checks.tool_call_delimiters_present
        ),
        no_python_dict_syntax=checks.no_python_dict_syntax,
        multiple_calls_serialized_correctly=(
            checks.multiple_calls_serialized_correctly
        ),
        thinking_mode_disabled=checks.thinking_mode_disabled,
        decoded_matches_rendered=checks.decoded_matches_rendered,
        failures=tuple(failures),
    )

    return RenderedTemplateExample(
        example=example,
        rendered_text=rendered_text,
        decoded_text=decoded_text,
        token_ids=token_ids,
        tool_section_tags=tool_section_tags,
        tool_call_delimiters=tool_call_delimiters,
        expected_calls=expected_calls,
        rendered_calls=rendered_calls,
        thinking_mode_requested=enable_thinking,
        thinking_mode_supported=thinking_mode_supported,
        checks=checks,
    )


def rendered_example_to_report(
    rendered_example: RenderedTemplateExample,
) -> dict[str, Any]:
    return {
        "id": rendered_example.example.record_id,
        "split": rendered_example.example.split,
        "source_id": rendered_example.example.source_id,
        "feature_tags": list(
            rendered_example.example.feature_tags
        ),
        "available_tool_count": int(
            rendered_example.example.record["metadata"][
                "available_tool_count"
            ]
        ),
        "expected_call_count": int(
            rendered_example.example.record["metadata"][
                "expected_call_count"
            ]
        ),
        "token_count": rendered_example.token_count,
        "thinking_mode_requested": (
            rendered_example.thinking_mode_requested
        ),
        "thinking_mode_supported": (
            rendered_example.thinking_mode_supported
        ),
        "tool_section_tags": rendered_example.tool_section_tags,
        "tool_call_delimiters": list(
            rendered_example.tool_call_delimiters
        ),
        "expected_calls": list(rendered_example.expected_calls),
        "rendered_calls": list(rendered_example.rendered_calls),
        "checks": {
            "tool_section_present": (
                rendered_example.checks.tool_section_present
            ),
            "tool_definitions_before_user_message": (
                rendered_example.checks.tool_definitions_before_user_message
            ),
            "function_names_unchanged": (
                rendered_example.checks.function_names_unchanged
            ),
            "arguments_as_json_objects": (
                rendered_example.checks.arguments_as_json_objects
            ),
            "tool_call_delimiters_present": (
                rendered_example.checks.tool_call_delimiters_present
            ),
            "no_python_dict_syntax": (
                rendered_example.checks.no_python_dict_syntax
            ),
            "multiple_calls_serialized_correctly": (
                rendered_example.checks.multiple_calls_serialized_correctly
            ),
            "thinking_mode_disabled": (
                rendered_example.checks.thinking_mode_disabled
            ),
            "decoded_matches_rendered": (
                rendered_example.checks.decoded_matches_rendered
            ),
            "failures": list(rendered_example.checks.failures),
        },
        "rendered_text": rendered_example.rendered_text,
        "decoded_text": rendered_example.decoded_text,
    }
