from __future__ import annotations

from function_calling_ft.evaluation_report import build_case_report, requested_metrics


def _record(
    *,
    expected: int,
    predicted: int,
    matched: int,
    executable: bool,
    strict: bool,
    no_tool: bool = False,
    tool_emitted: bool = True,
    valid_structure: bool = True,
    extra_prose: bool = False,
    malformed: bool = False,
) -> dict[str, object]:
    return {
        "missing_prediction": False,
        "generation_error": None,
        "parse": {"valid_structure": valid_structure},
        "emission": {
            "no_tool_call_emitted": no_tool,
            "tool_call_emitted": tool_emitted,
            "extra_prose_with_tool_call": extra_prose,
            "malformed_tool_call": malformed,
            "prose_only_response": no_tool,
        },
        "headline_scores": {
            "executable_complete_match": executable,
            "strict_complete_match": strict,
        },
        "call_metrics": {
            "expected_call_count": expected,
            "predicted_call_count": predicted,
            "matched_call_count": matched,
            "missing_call_count": expected - matched,
            "extra_call_count": predicted - matched,
        },
    }


def test_requested_metrics_use_explicit_denominators() -> None:
    scored_records = [
        _record(expected=1, predicted=1, matched=1, executable=True, strict=True),
        _record(
            expected=2,
            predicted=1,
            matched=1,
            executable=False,
            strict=False,
        ),
        _record(
            expected=1,
            predicted=0,
            matched=0,
            executable=False,
            strict=False,
            no_tool=True,
            tool_emitted=False,
            valid_structure=False,
        ),
        _record(
            expected=0,
            predicted=1,
            matched=0,
            executable=False,
            strict=False,
            no_tool=False,
            tool_emitted=True,
        ),
    ]
    scores = {
        "total_records": 4,
        "executable_complete_match_count": 1,
        "strict_complete_match_count": 1,
        "strict_complete_call_count": 2,
        "expected_call_count": 4,
        "predicted_call_count": 3,
        "matched_call_count": 2,
        "schema_validation_success_count": 2,
        "average_argument_value_accuracy": 0.75,
        "extra_call_count": 1,
        "undeclared_argument_count": 1,
        "malformed_tool_call_count": 0,
    }

    metrics = requested_metrics(
        scored_records=scored_records,
        scores=scores,
    )

    assert (
        metrics["executable_complete_accuracy_record_level"]["value"]
        == 0.25
    )
    assert metrics["strict_complete_accuracy"]["value"] == 0.25
    assert metrics["complete_call_recall"]["value"] == 0.5
    assert metrics["complete_call_precision"]["value"] == 2 / 3
    assert (
        metrics["no_tool_call_rate_on_tool_required_records"]["value"]
        == 1 / 3
    )
    assert metrics["schema_validation_success"]["value"] == 1.0
    assert metrics["argument_value_accuracy"]["value"] == 0.75
    assert metrics["function_name_precision"]["value"] == 2 / 3
    assert metrics["extra_call_rate"]["value"] == 1 / 3
    assert metrics["undeclared_argument_rate"]["value"] == 0.5
    assert metrics["no_tool_false_positive_rate"]["value"] == 1.0
    assert metrics["malformed_call_rate"]["value"] == 0.0


def test_no_tool_false_positive_rate_is_null_without_no_tool_records() -> None:
    metrics = requested_metrics(
        scored_records=[
            _record(
                expected=1,
                predicted=1,
                matched=1,
                executable=True,
                strict=True,
            ),
        ],
        scores={
            "total_records": 1,
            "executable_complete_match_count": 1,
            "strict_complete_match_count": 1,
            "strict_complete_call_count": 1,
            "expected_call_count": 1,
            "predicted_call_count": 1,
            "matched_call_count": 1,
            "schema_validation_success_count": 1,
            "average_argument_value_accuracy": 1.0,
            "extra_call_count": 0,
            "undeclared_argument_count": 0,
            "malformed_tool_call_count": 0,
        },
    )

    assert metrics["no_tool_false_positive_rate"]["value"] is None
    assert metrics["no_tool_false_positive_rate"]["denominator"] == 0


def test_case_report_uses_no_tool_score_for_no_tool_records() -> None:
    scored_records = [
        {
            **_record(
                expected=0,
                predicted=0,
                matched=0,
                executable=False,
                strict=False,
                no_tool=True,
                tool_emitted=False,
                valid_structure=False,
            ),
            "id": "direct",
            "source_id": "source-direct",
            "no_tool_score": {"status": "correct_direct_answer"},
            "expected_calls": [],
        },
        {
            **_record(
                expected=0,
                predicted=1,
                matched=0,
                executable=False,
                strict=False,
                tool_emitted=True,
            ),
            "id": "false-positive",
            "source_id": "source-fp",
            "no_tool_score": {"status": "unnecessary_tool_call"},
            "expected_calls": [],
        },
    ]
    report = build_case_report(
        scored_records=scored_records,
        scores={"total_records": 2},
        metrics={},
    )

    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["failure_reason_counts"] == {"unnecessary_tool_call": 1}
