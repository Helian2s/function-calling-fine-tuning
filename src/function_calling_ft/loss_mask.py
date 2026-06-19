from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Sequence


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOLS_START = "<tools>"
TOOLS_END = "</tools>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"
THINK_START = "<think>"
THINK_END = "</think>"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class RenderedBlock:
    role: str
    block_start: int
    header_end: int
    content_start: int
    content_end: int
    block_end: int
    trailing_end: int


@dataclass(frozen=True)
class LossMaskSpan:
    start: int
    end: int
    region: str
    include_in_loss: bool


@dataclass(frozen=True)
class LossMaskToken:
    index: int
    token_id: int
    token_text: str
    label: int
    region: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class LossMaskResult:
    rendered_text: str
    decoded_text: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    tokens: tuple[LossMaskToken, ...]
    spans: tuple[LossMaskSpan, ...]
    pad_token_id: int | None

    @property
    def included_token_count(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)

    @property
    def ignored_token_count(self) -> int:
        return sum(label == IGNORE_INDEX for label in self.labels)


def _tokenizer_kwargs(
    tokenizer: Any,
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    signature = inspect.signature(
        tokenizer.apply_chat_template
    )
    supports_enable_thinking = (
        "enable_thinking" in signature.parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )

    if supports_enable_thinking:
        return {"enable_thinking": enable_thinking}

    return {}


def _normalize_input_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "get") and callable(value.get):
        maybe_input_ids = value.get("input_ids")
        if maybe_input_ids is not None:
            value = maybe_input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
    ):
        value = value[0]

    return tuple(int(token_id) for token_id in value)


def _parse_rendered_blocks(
    text: str,
) -> tuple[RenderedBlock, ...]:
    blocks: list[RenderedBlock] = []
    cursor = 0

    while True:
        block_start = text.find(IM_START, cursor)

        if block_start < 0:
            break

        role_start = block_start + len(IM_START)
        header_end = text.find("\n", role_start)

        if header_end < 0:
            raise ValueError("Malformed rendered chat block header.")

        role = text[role_start:header_end]
        content_start = header_end + 1
        content_end = text.find(IM_END, content_start)

        if content_end < 0:
            raise ValueError("Rendered chat block is missing <|im_end|>.")

        block_end = content_end + len(IM_END)
        trailing_end = block_end

        if trailing_end < len(text) and text[trailing_end] == "\n":
            trailing_end += 1

        blocks.append(
            RenderedBlock(
                role=role,
                block_start=block_start,
                header_end=header_end,
                content_start=content_start,
                content_end=content_end,
                block_end=block_end,
                trailing_end=trailing_end,
            )
        )
        cursor = trailing_end

    return tuple(blocks)


def _add_span(
    spans: list[LossMaskSpan],
    start: int,
    end: int,
    *,
    region: str,
    include_in_loss: bool,
) -> None:
    if end <= start:
        return

    spans.append(
        LossMaskSpan(
            start=start,
            end=end,
            region=region,
            include_in_loss=include_in_loss,
        )
    )


def _merge_adjacent_spans(
    text: str,
    spans: Sequence[LossMaskSpan],
) -> tuple[LossMaskSpan, ...]:
    if not spans:
        return ()

    ordered = sorted(
        spans,
        key=lambda span: (span.start, span.end),
    )
    merged: list[LossMaskSpan] = [ordered[0]]

    for span in ordered[1:]:
        last = merged[-1]

        if (
            last.region == span.region
            and last.include_in_loss == span.include_in_loss
            and text[last.end : span.start].strip() == ""
        ):
            merged[-1] = LossMaskSpan(
                start=last.start,
                end=span.end,
                region=last.region,
                include_in_loss=last.include_in_loss,
            )
            continue

        merged.append(span)

    return tuple(merged)


def _assistant_content_spans(
    text: str,
    block: RenderedBlock,
) -> tuple[LossMaskSpan, ...]:
    spans: list[LossMaskSpan] = []
    cursor = block.content_start

    while cursor < block.content_end:
        tag_positions: list[tuple[int, str, str, str]] = []

        for start_tag, end_tag, region in (
            (THINK_START, THINK_END, "assistant_thinking"),
            (
                TOOL_CALL_START,
                TOOL_CALL_END,
                "assistant_tool_call",
            ),
            (
                TOOL_RESPONSE_START,
                TOOL_RESPONSE_END,
                "tool_execution_result",
            ),
        ):
            start_index = text.find(start_tag, cursor, block.content_end)
            if start_index >= 0:
                tag_positions.append(
                    (start_index, start_tag, end_tag, region)
                )

        if not tag_positions:
            remaining = text[cursor:block.content_end]
            stripped_remaining = remaining.strip()

            if stripped_remaining:
                leading_trim = len(remaining) - len(
                    remaining.lstrip()
                )
                trailing_trim = len(remaining) - len(
                    remaining.rstrip()
                )
                _add_span(
                    spans,
                    cursor + leading_trim,
                    block.content_end - trailing_trim,
                    region="assistant_final_answer",
                    include_in_loss=False,
                )
            break

        start_index, start_tag, end_tag, region = min(
            tag_positions,
            key=lambda item: item[0],
        )

        plain_text = text[cursor:start_index]
        stripped_plain_text = plain_text.strip()

        if stripped_plain_text:
            leading_trim = len(plain_text) - len(
                plain_text.lstrip()
            )
            trailing_trim = len(plain_text) - len(
                plain_text.rstrip()
            )
            _add_span(
                spans,
                cursor + leading_trim,
                start_index - trailing_trim,
                region="assistant_final_answer",
                include_in_loss=False,
            )

        end_index = text.find(
            end_tag,
            start_index + len(start_tag),
            block.content_end,
        )

        if end_index < 0:
            raise ValueError(
                f"Rendered assistant block is missing closing tag for {start_tag}."
            )

        end_index += len(end_tag)

        _add_span(
            spans,
            start_index,
            end_index,
            region=region,
            include_in_loss=(region == "assistant_tool_call"),
        )
        cursor = end_index

    return _merge_adjacent_spans(text, spans)


def _build_loss_mask_spans(
    text: str,
) -> tuple[LossMaskSpan, ...]:
    blocks = _parse_rendered_blocks(text)
    spans: list[LossMaskSpan] = []

    for block in blocks:
        if block.role == "system":
            tools_start = text.find(
                TOOLS_START,
                block.content_start,
                block.content_end,
            )
            tools_end = text.find(
                TOOLS_END,
                block.content_start,
                block.content_end,
            )

            if tools_start >= 0 and tools_end >= 0:
                tools_end += len(TOOLS_END)
                _add_span(
                    spans,
                    block.block_start,
                    tools_start,
                    region="system_prompt",
                    include_in_loss=False,
                )
                _add_span(
                    spans,
                    tools_start,
                    tools_end,
                    region="tool_definitions",
                    include_in_loss=False,
                )
                _add_span(
                    spans,
                    tools_end,
                    block.trailing_end,
                    region="system_prompt",
                    include_in_loss=False,
                )
            else:
                _add_span(
                    spans,
                    block.block_start,
                    block.trailing_end,
                    region="system_prompt",
                    include_in_loss=False,
                )

            continue

        if block.role == "assistant":
            _add_span(
                spans,
                block.block_start,
                block.content_start,
                region="assistant_scaffolding",
                include_in_loss=False,
            )
            spans.extend(_assistant_content_spans(text, block))
            _add_span(
                spans,
                block.content_end,
                block.trailing_end,
                region="assistant_scaffolding",
                include_in_loss=False,
            )
            continue

        if (
            block.role == "user"
            and TOOL_RESPONSE_START
            in text[block.content_start : block.content_end]
        ):
            _add_span(
                spans,
                block.block_start,
                block.trailing_end,
                region="tool_execution_result",
                include_in_loss=False,
            )
            continue

        _add_span(
            spans,
            block.block_start,
            block.trailing_end,
            region="user_request",
            include_in_loss=False,
        )

    return tuple(
        sorted(
            spans,
            key=lambda span: (span.start, span.end),
        )
    )


def _find_region(
    spans: Sequence[LossMaskSpan],
    *,
    start: int,
    end: int,
) -> tuple[str, bool]:
    for span in spans:
        if start >= span.start and end <= span.end:
            return span.region, span.include_in_loss

    return "unclassified", False


def _pad_result(
    *,
    token_ids: list[int],
    labels: list[int],
    tokens: list[LossMaskToken],
    pad_to_length: int,
    pad_token_id: int,
    pad_token_text: str,
) -> None:
    while len(token_ids) < pad_to_length:
        index = len(token_ids)
        token_ids.append(pad_token_id)
        labels.append(IGNORE_INDEX)
        tokens.append(
            LossMaskToken(
                index=index,
                token_id=pad_token_id,
                token_text=pad_token_text,
                label=IGNORE_INDEX,
                region="padding",
                char_start=-1,
                char_end=-1,
            )
        )


def build_expected_loss_mask(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    pad_to_length: int | None = None,
) -> LossMaskResult:
    tokenizer_kwargs = _tokenizer_kwargs(
        tokenizer,
        enable_thinking=enable_thinking,
    )
    rendered_text = tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools) if tools is not None else None,
        tokenize=False,
        add_generation_prompt=False,
        **tokenizer_kwargs,
    )

    if not isinstance(rendered_text, str):
        raise TypeError(
            "Tokenizer returned a non-string chat template when tokenize=False."
        )

    encoding = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(_normalize_input_ids(encoding["input_ids"]))
    offset_mapping = list(encoding["offset_mapping"])
    decoded_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
    )
    spans = _build_loss_mask_spans(rendered_text)
    tokens: list[LossMaskToken] = []
    labels: list[int] = []

    for index, (token_id, (char_start, char_end)) in enumerate(
        zip(token_ids, offset_mapping, strict=True)
    ):
        region, include_in_loss = _find_region(
            spans,
            start=char_start,
            end=char_end,
        )
        token_text = rendered_text[char_start:char_end]
        label = token_id if include_in_loss else IGNORE_INDEX
        labels.append(label)
        tokens.append(
            LossMaskToken(
                index=index,
                token_id=token_id,
                token_text=token_text,
                label=label,
                region=region,
                char_start=char_start,
                char_end=char_end,
            )
        )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_to_length is not None:
        if pad_to_length < len(token_ids):
            raise ValueError(
                f"pad_to_length={pad_to_length} is shorter than the sequence length "
                f"{len(token_ids)}."
            )

        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)

        if pad_token_id is None:
            raise ValueError(
                "Tokenizer does not define pad_token_id or eos_token_id."
            )

        pad_token_text = (
            getattr(tokenizer, "pad_token", None)
            or getattr(tokenizer, "eos_token", None)
            or "<pad>"
        )
        _pad_result(
            token_ids=token_ids,
            labels=labels,
            tokens=tokens,
            pad_to_length=pad_to_length,
            pad_token_id=pad_token_id,
            pad_token_text=pad_token_text,
        )

    return LossMaskResult(
        rendered_text=rendered_text,
        decoded_text=decoded_text,
        input_ids=tuple(token_ids),
        labels=tuple(labels),
        tokens=tuple(tokens),
        spans=spans,
        pad_token_id=pad_token_id,
    )


def build_expected_loss_mask_for_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    enable_thinking: bool = False,
    pad_to_length: int | None = None,
) -> LossMaskResult:
    return build_expected_loss_mask(
        tokenizer,
        record["messages"],
        tools=record["tools"],
        enable_thinking=enable_thinking,
        pad_to_length=pad_to_length,
    )


def format_loss_mask_diagnostic(
    result: LossMaskResult,
    *,
    max_rows: int | None = None,
    focus_on_loss: bool = False,
    context_before_first_labeled: int = 12,
) -> str:
    rows = ["TOKEN                                    LABEL"]
    tokens = result.tokens
    start_index = 0

    if focus_on_loss:
        for index, token in enumerate(tokens):
            if token.label != IGNORE_INDEX:
                start_index = max(
                    0,
                    index - context_before_first_labeled,
                )
                break

    if start_index > 0:
        rows.append("...                                      ...")

    tokens = tokens[start_index:]

    if max_rows is not None:
        tokens = tokens[:max_rows]

    for token in tokens:
        token_text = (
            token.token_text
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
        )

        if token_text == "":
            token_text = "<empty>"

        rows.append(
            f"{token_text:<40} {token.label}"
        )

    return "\n".join(rows)
