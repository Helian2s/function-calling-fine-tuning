from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from function_calling_ft.loss_mask import IGNORE_INDEX, LossMaskResult


@dataclass(frozen=True)
class SelectedAuditRecord:
    record_id: str
    record: dict[str, Any]
    coverage_tags: tuple[str, ...]


def record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("example_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("record must contain a non-empty id or example_id")
    return value


def expected_call_count(record: Mapping[str, Any]) -> int:
    curation = record.get("curation_metadata")
    if isinstance(curation, Mapping) and isinstance(
        curation.get("expected_call_count"),
        int,
    ):
        return int(curation["expected_call_count"])
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(
        metadata.get("expected_call_count"),
        int,
    ):
        return int(metadata["expected_call_count"])
    messages = record.get("messages")
    if isinstance(messages, Sequence):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str):
                return len(tool_calls)
    return 0


def call_category(record: Mapping[str, Any]) -> str:
    curation = record.get("curation_metadata")
    if isinstance(curation, Mapping) and isinstance(curation.get("call_category"), str):
        return str(curation["call_category"])
    count = expected_call_count(record)
    if count <= 0:
        return "no_call"
    if count == 1:
        return "single"
    return "multiple"


def token_count(record: Mapping[str, Any], key: str) -> int:
    split = record.get("split_metadata")
    if not isinstance(split, Mapping):
        return 0
    counts = split.get("token_counts")
    if not isinstance(counts, Mapping):
        return 0
    value = counts.get(key)
    return int(value) if isinstance(value, int | float) else 0


def _has_assistant_tool_call_boundary(record: Mapping[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, Sequence):
        return False
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str):
            return bool(tool_calls)
    return False


def coverage_tags_for_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    category = call_category(record)
    count = expected_call_count(record)
    if count == 1 or category == "single":
        tags.append("single_call")
    if count > 1 or category == "multiple":
        tags.append("multiple_call")
    if category == "parallel":
        tags.append("parallel_call")
    if _has_assistant_tool_call_boundary(record):
        tags.append("boundary_special_tokens")
    if token_count(record, "prompt_schema_tokens") >= 1000:
        tags.append("long_schema")
    if token_count(record, "supervised_target_tokens") >= 100:
        tags.append("long_target")
    return tuple(sorted(set(tags)))


def _pick_first(
    records: Sequence[dict[str, Any]],
    predicate: Any,
    selected_ids: set[str],
) -> dict[str, Any] | None:
    for item in records:
        item_id = record_id(item)
        if item_id in selected_ids:
            continue
        if predicate(item):
            return item
    return None


def _pick_max(
    records: Sequence[dict[str, Any]],
    key: Any,
    selected_ids: set[str],
) -> dict[str, Any] | None:
    candidates = [item for item in records if record_id(item) not in selected_ids]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (key(item), record_id(item)))


def select_loss_mask_audit_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int = 20,
) -> tuple[SelectedAuditRecord, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(records) < count:
        raise ValueError(f"need at least {count} records, got {len(records)}")

    ordered = sorted(records, key=record_id)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    coverage_picks = (
        (
            "single_call",
            lambda record: expected_call_count(record) == 1
            or call_category(record) == "single",
        ),
        (
            "multiple_call",
            lambda record: expected_call_count(record) > 1
            or call_category(record) == "multiple",
        ),
        ("parallel_call", lambda record: call_category(record) == "parallel"),
        ("boundary_special_tokens", _has_assistant_tool_call_boundary),
    )

    for _name, predicate in coverage_picks:
        picked = _pick_first(ordered, predicate, selected_ids)
        if picked is not None:
            selected.append(picked)
            selected_ids.add(record_id(picked))

    for key_name in ("prompt_schema_tokens", "supervised_target_tokens"):
        picked = _pick_max(
            ordered,
            lambda record, key_name=key_name: token_count(record, key_name),
            selected_ids,
        )
        if picked is not None:
            selected.append(picked)
            selected_ids.add(record_id(picked))

    fill_order = sorted(
        ordered,
        key=lambda item: (
            -token_count(item, "full_tokens"),
            call_category(item),
            record_id(item),
        ),
    )
    for item in fill_order:
        if len(selected) >= count:
            break
        item_id = record_id(item)
        if item_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item_id)

    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} records, expected {count}")

    return tuple(
        SelectedAuditRecord(
            record_id=record_id(item),
            record=item,
            coverage_tags=coverage_tags_for_record(item),
        )
        for item in selected
    )


def assert_assistant_only_mask(
    *,
    result: LossMaskResult,
    record: Mapping[str, Any],
    max_sequence_length: int,
) -> list[str]:
    errors: list[str] = []
    supervised = [token for token in result.tokens if token.label != IGNORE_INDEX]
    if not supervised:
        errors.append("no supervised assistant tool-call tokens")
    non_tool_supervised = [
        token.region for token in supervised if token.region != "assistant_tool_call"
    ]
    if non_tool_supervised:
        errors.append(
            "supervised tokens outside assistant_tool_call: "
            + ", ".join(sorted(set(non_tool_supervised))),
        )
    for forbidden_region in (
        "system_prompt",
        "tool_definitions",
        "user_request",
        "tool_execution_result",
        "assistant_final_answer",
        "assistant_scaffolding",
        "padding",
    ):
        if any(
            token.region == forbidden_region and token.label != IGNORE_INDEX
            for token in result.tokens
        ):
            errors.append(f"{forbidden_region} tokens are supervised")

    if len(result.input_ids) > max_sequence_length:
        errors.append(
            f"rendered sequence length {len(result.input_ids)} exceeds {max_sequence_length}",
        )

    split = record.get("split_metadata")
    if isinstance(split, Mapping):
        counts = split.get("token_counts")
        if isinstance(counts, Mapping):
            if bool(counts.get(f"truncation_risk_{max_sequence_length}")):
                errors.append(f"record is marked truncation_risk_{max_sequence_length}")

    if "<|im_start|>assistant" not in result.rendered_text:
        errors.append("rendered text is missing assistant chat boundary")
    if "<tool_call>" not in result.rendered_text:
        errors.append("rendered text is missing tool-call boundary")

    return errors


def mask_statistics(results: Sequence[LossMaskResult]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "full_tokens": {},
            "supervised_tokens": {},
            "ignored_tokens": {},
        }

    def summarize(values: Sequence[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "mean": mean(ordered),
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[int((len(ordered) - 1) * 0.9)],
            "max": ordered[-1],
        }

    full = [len(item.input_ids) for item in results]
    supervised = [item.included_token_count for item in results]
    ignored = [item.ignored_token_count for item in results]
    return {
        "count": len(results),
        "full_tokens": summarize(full),
        "supervised_tokens": summarize(supervised),
        "ignored_tokens": summarize(ignored),
        "supervised_fraction_mean": mean(
            supervised_count / full_count
            for supervised_count, full_count in zip(supervised, full, strict=True)
            if full_count
        ),
    }
