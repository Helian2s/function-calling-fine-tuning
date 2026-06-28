from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from function_calling_ft.evaluation import (
    SCORED_PREDICTIONS_FILENAME,
    SCORES_FILENAME,
    requested_metrics as canonical_requested_metrics,
)


REQUESTED_METRICS_FILENAME = "requested_metrics.json"
CASE_REPORT_JSON_FILENAME = "case_report.json"
CASE_REPORT_MD_FILENAME = "case_report.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _metric(
    *,
    value: float | None,
    numerator: int | float | None,
    denominator: int | float | None,
    definition: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "definition": definition,
    }


def _call_names(calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for call in calls:
        if isinstance(call, dict):
            name = call.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _failure_reason(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("missing_prediction"):
        return "missing_prediction", "no prediction record was produced"

    generation_error = record.get("generation_error")
    if generation_error:
        return "generation_error", str(generation_error)

    parse = record.get("parse", {})
    emission = record.get("emission", {})
    call_metrics = record.get("call_metrics", {})
    matches = record.get("call_matches", [])

    if not parse.get("valid_structure"):
        errors = parse.get("errors") or ["invalid structure"]
        if emission.get("no_tool_call_emitted"):
            return "no_tool_call", "no parseable tool call emitted"
        if emission.get("malformed_tool_call"):
            return "malformed_tool_call", "; ".join(str(e) for e in errors)
        return "parse_failure", "; ".join(str(e) for e in errors)

    missing_calls = int(call_metrics.get("missing_call_count", 0) or 0)
    extra_calls = int(call_metrics.get("extra_call_count", 0) or 0)
    if missing_calls or extra_calls:
        return (
            "call_count_mismatch",
            f"missing_calls={missing_calls}, extra_calls={extra_calls}",
        )

    if any(not match.get("function_name_match") for match in matches):
        return "function_name_mismatch", "one or more function names differ"

    argument_problems: list[str] = []
    for match in matches:
        expected = match.get("expected_call", {}).get("name", "<unknown>")
        if match.get("missing_required_arguments"):
            argument_problems.append(
                f"{expected}: missing required "
                f"{match['missing_required_arguments']}",
            )
        if match.get("undeclared_arguments"):
            argument_problems.append(
                f"{expected}: undeclared {match['undeclared_arguments']}",
            )
        if not match.get("schema_validation_success", True):
            argument_problems.append(f"{expected}: schema validation failed")
        if not match.get("schema_equivalent_complete_match", False):
            argument_problems.append(
                f"{expected}: argument mismatch "
                f"name_acc={match.get('argument_name_accuracy')} "
                f"type_acc={match.get('argument_type_accuracy')} "
                f"value_acc={match.get('argument_value_accuracy')}",
            )

    if argument_problems:
        return "argument_mismatch", "; ".join(argument_problems)

    if emission.get("extra_prose_with_tool_call"):
        return "extra_prose", "tool call was accompanied by extra prose"

    return (
        "other_failure",
        "failed executable match without a specific diagnostic",
    )


def requested_metrics(
    *,
    scored_records: list[dict[str, Any]],
    scores: dict[str, Any],
) -> dict[str, Any]:
    total_records = int(scores.get("total_records", len(scored_records)) or 0)
    predicted_call_count = int(scores.get("predicted_call_count", 0) or 0)
    expected_call_count = int(scores.get("expected_call_count", 0) or 0)
    matched_call_count = int(scores.get("matched_call_count", 0) or 0)
    strict_complete_call_count = int(
        scores.get("strict_complete_call_count", 0) or 0,
    )
    executable_complete_match_count = int(
        scores.get("executable_complete_match_count", 0) or 0,
    )
    strict_complete_match_count = int(
        scores.get("strict_complete_match_count", 0) or 0,
    )
    schema_validation_success_count = int(
        scores.get("schema_validation_success_count", 0) or 0,
    )
    extra_call_count = int(scores.get("extra_call_count", 0) or 0)
    undeclared_argument_count = int(
        scores.get("undeclared_argument_count", 0) or 0,
    )
    malformed_tool_call_count = int(
        scores.get("malformed_tool_call_count", 0) or 0,
    )

    tool_required_records = [
        record
        for record in scored_records
        if int(
            record.get("call_metrics", {}).get("expected_call_count", 0)
            or 0,
        )
        > 0
    ]
    no_tool_call_on_tool_required = sum(
        int(record.get("emission", {}).get("no_tool_call_emitted", False))
        for record in tool_required_records
    )
    no_tool_records = [
        record
        for record in scored_records
        if int(
            record.get("call_metrics", {}).get("expected_call_count", 0)
            or 0,
        )
        == 0
    ]
    no_tool_false_positive_count = sum(
        int(record.get("emission", {}).get("tool_call_emitted", False))
        for record in no_tool_records
    )
    protocol_clean_count = sum(
        int(
            not record.get("missing_prediction")
            and record.get("generation_error") is None
            and bool(record.get("parse", {}).get("valid_structure"))
            and not bool(
                record.get("emission", {}).get("extra_prose_with_tool_call"),
            )
            and not bool(record.get("emission", {}).get("malformed_tool_call"))
            and not bool(record.get("emission", {}).get("prose_only_response"))
        )
        for record in scored_records
    )

    return {
        "executable_complete_accuracy_record_level": _metric(
            value=_rate(executable_complete_match_count, total_records),
            numerator=executable_complete_match_count,
            denominator=total_records,
            definition=(
                "records where all expected calls are executable complete "
                "matches divided by total records"
            ),
        ),
        "strict_complete_accuracy": _metric(
            value=_rate(strict_complete_match_count, total_records),
            numerator=strict_complete_match_count,
            denominator=total_records,
            definition=(
                "records where all expected calls strictly match divided "
                "by total records"
            ),
        ),
        "complete_call_recall": _metric(
            value=_rate(strict_complete_call_count, expected_call_count),
            numerator=strict_complete_call_count,
            denominator=expected_call_count,
            definition=(
                "strictly complete matched calls divided by expected calls"
            ),
        ),
        "no_tool_call_rate_on_tool_required_records": _metric(
            value=_rate(
                no_tool_call_on_tool_required,
                len(tool_required_records),
            ),
            numerator=no_tool_call_on_tool_required,
            denominator=len(tool_required_records),
            definition=(
                "tool-required records with no emitted tool call divided "
                "by tool-required records"
            ),
        ),
        "complete_call_precision": _metric(
            value=_rate(strict_complete_call_count, predicted_call_count),
            numerator=strict_complete_call_count,
            denominator=predicted_call_count,
            definition=(
                "strictly complete matched calls divided by predicted calls"
            ),
        ),
        "schema_validation_success": _metric(
            value=_rate(schema_validation_success_count, matched_call_count),
            numerator=schema_validation_success_count,
            denominator=matched_call_count,
            definition=(
                "matched calls passing schema validation divided by matched "
                "calls"
            ),
        ),
        "argument_value_accuracy": _metric(
            value=scores.get("average_argument_value_accuracy"),
            numerator=None,
            denominator=matched_call_count,
            definition=(
                "mean argument value accuracy across matched call "
                "comparisons"
            ),
        ),
        "protocol_clean_response_rate": _metric(
            value=_rate(protocol_clean_count, total_records),
            numerator=protocol_clean_count,
            denominator=total_records,
            definition=(
                "records with valid parseable structure, no generation error, "
                "and no extra prose or malformed tool-call emission"
            ),
        ),
        "function_name_precision": _metric(
            value=_rate(matched_call_count, predicted_call_count),
            numerator=matched_call_count,
            denominator=predicted_call_count,
            definition="matched function names divided by predicted calls",
        ),
        "extra_call_rate": _metric(
            value=_rate(extra_call_count, predicted_call_count),
            numerator=extra_call_count,
            denominator=predicted_call_count,
            definition="extra predicted calls divided by predicted calls",
        ),
        "undeclared_argument_rate": _metric(
            value=_rate(undeclared_argument_count, matched_call_count),
            numerator=undeclared_argument_count,
            denominator=matched_call_count,
            definition=(
                "undeclared argument count divided by matched calls; "
                "argument-level denominator is not emitted by the scorer"
            ),
        ),
        "no_tool_false_positive_rate": _metric(
            value=_rate(no_tool_false_positive_count, len(no_tool_records)),
            numerator=no_tool_false_positive_count,
            denominator=len(no_tool_records),
            definition=(
                "records with no expected tool calls but an emitted tool call "
                "divided by records with no expected tool calls"
            ),
        ),
        "malformed_call_rate": _metric(
            value=_rate(malformed_tool_call_count, total_records),
            numerator=malformed_tool_call_count,
            denominator=total_records,
            definition=(
                "records with malformed tool-call emission divided by total "
                "records"
            ),
        ),
    }


def build_case_report(
    *,
    scored_records: list[dict[str, Any]],
    scores: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for record in scored_records:
        no_tool_status = record.get("no_tool_score", {}).get("status")
        tool_call_passed = bool(
            record.get("headline_scores", {}).get("executable_complete_match"),
        )
        no_tool_passed = no_tool_status in {
            "correct_direct_answer",
            "correct_clarification",
        }
        passed = tool_call_passed or no_tool_passed
        if passed:
            reason_category = "passed"
            reason = (
                "correct no-tool response"
                if no_tool_passed
                else "executable complete match"
            )
        elif no_tool_status not in {
            None,
            "not_applicable_tool_required",
        }:
            reason_category = str(no_tool_status)
            reason = f"no_tool_score.status={no_tool_status}"
            reason_counts[reason_category] += 1
        else:
            reason_category, reason = _failure_reason(record)
            reason_counts[reason_category] += 1

        parse_calls = record.get("parse", {}).get("calls", [])
        expected_calls = record.get("expected_calls", [])
        call_metrics = record.get("call_metrics", {})
        cases.append(
            {
                "id": record.get("id"),
                "source_id": record.get("source_id"),
                "passed": passed,
                "reason_category": reason_category,
                "reason": reason,
                "expected_call_count": call_metrics.get(
                    "expected_call_count",
                ),
                "predicted_call_count": call_metrics.get(
                    "predicted_call_count",
                ),
                "matched_call_count": call_metrics.get("matched_call_count"),
                "missing_call_count": call_metrics.get("missing_call_count"),
                "extra_call_count": call_metrics.get("extra_call_count"),
                "expected_functions": _call_names(expected_calls),
                "predicted_functions": _call_names(parse_calls),
            },
        )

    passed_count = sum(1 for case in cases if case["passed"])
    failed_count = len(cases) - passed_count
    return {
        "definition_of_pass": (
            "headline_scores.executable_complete_match"
        ),
        "total_cases": len(cases),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": _rate(passed_count, len(cases)),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "requested_metrics": metrics,
        "scores": scores,
        "cases": cases,
    }


def _write_case_report_markdown(path: Path, report: dict[str, Any]) -> None:
    cases = report["cases"]
    lines = [
        "# Evaluation Case Report",
        "",
        f"Pass definition: `{report['definition_of_pass']}`",
        f"Total: {report['total_cases']}",
        f"Passed: {report['passed']}",
        f"Failed: {report['failed']}",
        f"Pass rate: {report['pass_rate']:.3f}",
        "",
        "## Failure Reasons",
    ]
    for reason, count in report["failure_reason_counts"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "## Cases",
            "| # | id | source_id | result | reason | expected | predicted |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ],
    )
    for index, case in enumerate(cases, start=1):
        result = "PASS" if case["passed"] else "FAIL"
        expected = ", ".join(case["expected_functions"])
        predicted = ", ".join(case["predicted_functions"])
        reason = str(case["reason"]).replace("|", "\\|")
        lines.append(
            "| "
            f"{index} | {case['id']} | {case['source_id']} | {result} | "
            f"{reason} | {expected} | {predicted} |",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path) -> dict[str, Any]:
    scored_records = read_jsonl(output_dir / SCORED_PREDICTIONS_FILENAME)
    scores = json.loads(
        (output_dir / SCORES_FILENAME).read_text(encoding="utf-8"),
    )
    metrics = canonical_requested_metrics(
        scored_records=scored_records,
        scores=scores,
    )
    report = build_case_report(
        scored_records=scored_records,
        scores=scores,
        metrics=metrics,
    )

    (output_dir / REQUESTED_METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CASE_REPORT_JSON_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_case_report_markdown(
        output_dir / CASE_REPORT_MD_FILENAME,
        report,
    )
    return report
