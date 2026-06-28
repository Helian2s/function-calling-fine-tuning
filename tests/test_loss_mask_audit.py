from __future__ import annotations

from function_calling_ft.loss_mask import (
    IGNORE_INDEX,
    LossMaskResult,
    LossMaskSpan,
    LossMaskToken,
)
from function_calling_ft.loss_mask_audit import (
    assert_assistant_only_mask,
    select_loss_mask_audit_records,
)


def _record(
    index: int,
    *,
    call_category: str = "single",
    expected_calls: int = 1,
    prompt_tokens: int = 100,
    target_tokens: int = 20,
) -> dict[str, object]:
    return {
        "id": f"xlam-{index}",
        "messages": [
            {"role": "user", "content": f"request {index}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}_{call_index}",
                        "type": "function",
                        "function": {
                            "name": "tool",
                            "arguments": {"value": call_index},
                        },
                    }
                    for call_index in range(expected_calls)
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "Tool.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "curation_metadata": {
            "call_category": call_category,
            "expected_call_count": expected_calls,
        },
        "split_metadata": {
            "token_counts": {
                "full_tokens": prompt_tokens + target_tokens,
                "prompt_schema_tokens": prompt_tokens,
                "supervised_target_tokens": target_tokens,
                "truncation_risk_2048": False,
            },
        },
    }


def test_select_loss_mask_audit_records_is_deterministic_and_covers_required_tags() -> None:
    records = [
        _record(1, call_category="single", expected_calls=1),
        _record(2, call_category="multiple", expected_calls=2),
        _record(3, call_category="parallel", expected_calls=2),
        _record(4, prompt_tokens=1200, target_tokens=40),
        _record(5, prompt_tokens=200, target_tokens=140),
        *[_record(index, prompt_tokens=300 + index) for index in range(6, 26)],
    ]

    selected = select_loss_mask_audit_records(records, count=20)
    selected_again = select_loss_mask_audit_records(list(reversed(records)), count=20)

    assert [item.record_id for item in selected] == [
        item.record_id for item in selected_again
    ]
    tags = {tag for item in selected for tag in item.coverage_tags}
    assert {
        "single_call",
        "multiple_call",
        "parallel_call",
        "boundary_special_tokens",
        "long_schema",
        "long_target",
    }.issubset(tags)


def _mask_result(*, bad_supervised_region: str | None = None) -> LossMaskResult:
    text = (
        "<|im_start|>user\nrequest<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>{}</tool_call><|im_end|>\n"
    )
    regions = [
        "user_request",
        bad_supervised_region or "assistant_tool_call",
        "assistant_scaffolding",
    ]
    labels = [
        IGNORE_INDEX,
        2,
        IGNORE_INDEX,
    ]
    tokens = tuple(
        LossMaskToken(
            index=index,
            token_id=index + 1,
            token_text=f"tok{index}",
            label=labels[index],
            region=regions[index],
            char_start=index,
            char_end=index + 1,
        )
        for index in range(3)
    )
    return LossMaskResult(
        rendered_text=text,
        decoded_text=text,
        input_ids=(1, 2, 3),
        labels=tuple(labels),
        tokens=tokens,
        spans=(
            LossMaskSpan(0, 1, "user_request", False),
            LossMaskSpan(1, 2, bad_supervised_region or "assistant_tool_call", True),
            LossMaskSpan(2, 3, "assistant_scaffolding", False),
        ),
        pad_token_id=None,
    )


def test_assistant_only_mask_assertion_accepts_tool_call_supervision_only() -> None:
    errors = assert_assistant_only_mask(
        result=_mask_result(),
        record=_record(1),
        max_sequence_length=2048,
    )

    assert errors == []


def test_assistant_only_mask_assertion_rejects_user_supervision() -> None:
    errors = assert_assistant_only_mask(
        result=_mask_result(bad_supervised_region="user_request"),
        record=_record(1),
        max_sequence_length=2048,
    )

    assert any("outside assistant_tool_call" in error for error in errors)
    assert any("user_request tokens are supervised" in error for error in errors)
