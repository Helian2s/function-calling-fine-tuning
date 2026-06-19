from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from function_calling_ft.parser import ParseResult, ToolCall, parse_tool_calls


@dataclass(frozen=True)
class CallScore:
    valid_structure: bool
    correct_function_name: bool
    correct_argument_names: bool
    correct_argument_values: bool
    complete_call_match: bool
    parse_errors: tuple[str, ...] = ()

    @property
    def complete_match(self) -> bool:
        return self.complete_call_match


@dataclass(frozen=True)
class CallSetScore:
    valid_structure: bool
    correct_function_name: bool
    correct_argument_names: bool
    correct_argument_values: bool
    complete_call_match: bool
    predicted_count: int
    expected_count: int
    parse_errors: tuple[str, ...]
    call_scores: tuple[CallScore, ...]
    order_matters: bool

    @property
    def complete_match(self) -> bool:
        return self.complete_call_match


def _as_parse_result(
    value: Any,
) -> ParseResult:
    if isinstance(value, ParseResult):
        return value

    return parse_tool_calls(value)


def _score_normalized_call(
    predicted: ToolCall | None,
    expected: ToolCall | None,
    *,
    predicted_valid: bool,
    expected_valid: bool,
    parse_errors: tuple[str, ...],
) -> CallScore:
    if predicted is None or expected is None:
        return CallScore(
            valid_structure=False,
            correct_function_name=False,
            correct_argument_names=False,
            correct_argument_values=False,
            complete_call_match=False,
            parse_errors=parse_errors,
        )

    valid_structure = (
        predicted_valid
        and expected_valid
        and predicted.name is not None
        and isinstance(predicted.arguments, dict)
        and expected.name is not None
        and isinstance(expected.arguments, dict)
    )

    correct_function_name = (
        predicted.name is not None
        and expected.name is not None
        and predicted.name == expected.name
    )

    if isinstance(predicted.arguments, dict) and isinstance(
        expected.arguments,
        dict,
    ):
        predicted_argument_names = set(predicted.arguments)
        expected_argument_names = set(expected.arguments)
        correct_argument_names = (
            predicted_argument_names == expected_argument_names
        )
        correct_argument_values = (
            predicted.arguments == expected.arguments
        )
    else:
        correct_argument_names = False
        correct_argument_values = False

    complete_call_match = (
        valid_structure
        and correct_function_name
        and correct_argument_names
        and correct_argument_values
    )

    return CallScore(
        valid_structure=valid_structure,
        correct_function_name=correct_function_name,
        correct_argument_names=correct_argument_names,
        correct_argument_values=correct_argument_values,
        complete_call_match=complete_call_match,
        parse_errors=parse_errors,
    )


def score_call(
    predicted: Any,
    expected: Any,
) -> CallScore:
    predicted_result = _as_parse_result(predicted)
    expected_result = _as_parse_result(expected)
    parse_errors = predicted_result.errors + expected_result.errors

    if len(predicted_result.calls) != 1 or len(expected_result.calls) != 1:
        return CallScore(
            valid_structure=False,
            correct_function_name=False,
            correct_argument_names=False,
            correct_argument_values=False,
            complete_call_match=False,
            parse_errors=parse_errors
            + (
                "score_call expects exactly one predicted and one expected call.",
            ),
        )

    return _score_normalized_call(
        predicted_result.calls[0],
        expected_result.calls[0],
        predicted_valid=predicted_result.valid_structure,
        expected_valid=expected_result.valid_structure,
        parse_errors=parse_errors,
    )


def _call_score_weight(score: CallScore) -> int:
    return (
        int(score.complete_call_match) * 1_000
        + int(score.correct_argument_values) * 100
        + int(score.correct_argument_names) * 10
        + int(score.correct_function_name)
    )


def _match_call_scores(
    predicted_calls: tuple[ToolCall, ...],
    expected_calls: tuple[ToolCall, ...],
) -> tuple[CallScore, ...]:
    score_matrix = [
        [
            _score_normalized_call(
                predicted_call,
                expected_call,
                predicted_valid=True,
                expected_valid=True,
                parse_errors=(),
            )
            for predicted_call in predicted_calls
        ]
        for expected_call in expected_calls
    ]

    @lru_cache(maxsize=None)
    def best_assignment(
        expected_index: int,
        used_mask: int,
    ) -> tuple[int, tuple[int | None, ...]]:
        if expected_index == len(expected_calls):
            return 0, ()

        best_weight, best_indices = best_assignment(
            expected_index + 1,
            used_mask,
        )
        best_indices = (None,) + best_indices

        for predicted_index in range(len(predicted_calls)):
            if used_mask & (1 << predicted_index):
                continue

            score = score_matrix[expected_index][predicted_index]
            remaining_weight, remaining_indices = best_assignment(
                expected_index + 1,
                used_mask | (1 << predicted_index),
            )
            total_weight = (
                _call_score_weight(score) + remaining_weight
            )

            if total_weight > best_weight:
                best_weight = total_weight
                best_indices = (
                    predicted_index,
                ) + remaining_indices

        return best_weight, best_indices

    _, assignment = best_assignment(0, 0)
    matched_scores: list[CallScore] = []

    for expected_index, predicted_index in enumerate(assignment):
        if predicted_index is None:
            matched_scores.append(
                CallScore(
                    valid_structure=False,
                    correct_function_name=False,
                    correct_argument_names=False,
                    correct_argument_values=False,
                    complete_call_match=False,
                    parse_errors=(
                        "Expected call has no predicted match.",
                    ),
                )
            )
            continue

        matched_scores.append(
            score_matrix[expected_index][predicted_index]
        )

    return tuple(matched_scores)


def score_calls(
    predicted: Any,
    expected: Any,
    *,
    order_matters: bool = False,
) -> CallSetScore:
    predicted_result = _as_parse_result(predicted)
    expected_result = _as_parse_result(expected)
    parse_errors = predicted_result.errors + expected_result.errors
    predicted_calls = predicted_result.calls
    expected_calls = expected_result.calls

    counts_match = len(predicted_calls) == len(expected_calls)
    valid_structure = (
        predicted_result.valid_structure
        and expected_result.valid_structure
    )

    if (
        not valid_structure
        or not predicted_calls
        or not expected_calls
    ):
        return CallSetScore(
            valid_structure=False,
            correct_function_name=False,
            correct_argument_names=False,
            correct_argument_values=False,
            complete_call_match=False,
            predicted_count=len(predicted_calls),
            expected_count=len(expected_calls),
            parse_errors=parse_errors,
            call_scores=(),
            order_matters=order_matters,
        )

    if order_matters:
        if not counts_match:
            call_scores: tuple[CallScore, ...] = ()
        else:
            call_scores = tuple(
                _score_normalized_call(
                    predicted_call,
                    expected_call,
                    predicted_valid=True,
                    expected_valid=True,
                    parse_errors=(),
                )
                for predicted_call, expected_call in zip(
                    predicted_calls,
                    expected_calls,
                )
            )
    else:
        call_scores = _match_call_scores(
            predicted_calls,
            expected_calls,
        )

    correct_function_name = counts_match and bool(call_scores) and all(
        score.correct_function_name for score in call_scores
    )
    correct_argument_names = counts_match and bool(call_scores) and all(
        score.correct_argument_names for score in call_scores
    )
    correct_argument_values = counts_match and bool(call_scores) and all(
        score.correct_argument_values for score in call_scores
    )
    complete_call_match = (
        valid_structure
        and counts_match
        and bool(call_scores)
        and all(score.complete_call_match for score in call_scores)
    )

    return CallSetScore(
        valid_structure=valid_structure,
        correct_function_name=correct_function_name,
        correct_argument_names=correct_argument_names,
        correct_argument_values=correct_argument_values,
        complete_call_match=complete_call_match,
        predicted_count=len(predicted_calls),
        expected_count=len(expected_calls),
        parse_errors=parse_errors,
        call_scores=call_scores,
        order_matters=order_matters,
    )
