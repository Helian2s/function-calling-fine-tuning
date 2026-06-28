from __future__ import annotations

import pytest

from function_calling_ft.evaluation_compare import compare_scored_records


def _scored(record_id: str, executable: bool) -> dict[str, object]:
    return {
        "id": record_id,
        "headline_scores": {
            "strict_complete_match": executable,
            "schema_equivalent_complete_match": executable,
            "executable_complete_match": executable,
        },
        "call_metrics": {
            "expected_call_count": 1,
            "predicted_call_count": 1,
        },
        "emission": {"tool_call_emitted": True},
    }


def test_paired_comparison_is_deterministic() -> None:
    baseline = [_scored("one", False), _scored("two", True)]
    candidate = [_scored("one", True), _scored("two", True)]

    first, first_deltas = compare_scored_records(
        baseline_records=baseline,
        candidate_records=candidate,
        bootstrap_samples=50,
        seed=123,
    )
    second, second_deltas = compare_scored_records(
        baseline_records=baseline,
        candidate_records=candidate,
        bootstrap_samples=50,
        seed=123,
    )

    assert first == second
    assert first_deltas == second_deltas
    assert first["record_count"] == 2
    metric = first["metrics"]["executable_complete_match"]
    assert metric["baseline_mean"] == 0.5
    assert metric["candidate_mean"] == 1.0
    assert metric["delta_mean"] == 0.5
    assert metric["paired_bootstrap_ci"]["seed"] == 123


def test_paired_comparison_requires_matching_ids() -> None:
    with pytest.raises(ValueError, match="same ids"):
        compare_scored_records(
            baseline_records=[_scored("one", False)],
            candidate_records=[_scored("two", False)],
        )
