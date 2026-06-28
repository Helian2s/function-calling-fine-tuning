from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from function_calling_ft.evaluation import write_jsonl


COMPARISON_SCHEMA_VERSION = "1.0"
DEFAULT_METRICS = (
    "strict_complete_match",
    "schema_equivalent_complete_match",
    "executable_complete_match",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Each scored record must contain a non-empty id")
    return value


def _index_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = _record_id(record)
        if record_id in indexed:
            raise ValueError(f"Duplicate scored record id: {record_id}")
        indexed[record_id] = record
    return indexed


def _metric_value(record: dict[str, Any], metric: str) -> float:
    if metric in DEFAULT_METRICS:
        return float(bool(record.get("headline_scores", {}).get(metric)))
    if metric == "no_tool_false_positive":
        expected = int(
            record.get("call_metrics", {}).get("expected_call_count", 0)
            or 0,
        )
        if expected != 0:
            return 0.0
        return float(bool(record.get("emission", {}).get("tool_call_emitted")))
    if metric == "tool_call_emitted":
        return float(bool(record.get("emission", {}).get("tool_call_emitted")))
    raise ValueError(f"Unsupported comparison metric: {metric}")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _bootstrap_ci(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if not deltas or samples <= 0:
        return {
            "confidence": confidence,
            "lower": None,
            "upper": None,
            "samples": samples,
            "seed": seed,
        }

    rng = random.Random(seed)
    n = len(deltas)
    estimates = sorted(
        _mean([deltas[rng.randrange(n)] for _ in range(n)])
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = min(max(int(alpha * samples), 0), samples - 1)
    upper_index = min(
        max(int((1.0 - alpha) * samples) - 1, 0),
        samples - 1,
    )
    return {
        "confidence": confidence,
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "samples": samples,
        "seed": seed,
    }


def compare_scored_records(
    *,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_by_id = _index_records(baseline_records)
    candidate_by_id = _index_records(candidate_records)
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Scored runs must contain the same ids. "
            f"missing_from_candidate={missing_candidate[:5]}, "
            f"missing_from_baseline={missing_baseline[:5]}",
        )

    ordered_ids = [_record_id(record) for record in baseline_records]
    deltas: list[dict[str, Any]] = []
    metric_deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    metric_baseline: dict[str, list[float]] = {metric: [] for metric in metrics}
    metric_candidate: dict[str, list[float]] = {metric: [] for metric in metrics}

    for record_id in ordered_ids:
        baseline = baseline_by_id[record_id]
        candidate = candidate_by_id[record_id]
        row: dict[str, Any] = {"id": record_id, "metrics": {}}
        for metric in metrics:
            baseline_value = _metric_value(baseline, metric)
            candidate_value = _metric_value(candidate, metric)
            delta = candidate_value - baseline_value
            metric_baseline[metric].append(baseline_value)
            metric_candidate[metric].append(candidate_value)
            metric_deltas[metric].append(delta)
            row["metrics"][metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
            }
        deltas.append(row)

    summary_metrics = {}
    for metric in metrics:
        summary_metrics[metric] = {
            "baseline_mean": _mean(metric_baseline[metric]),
            "candidate_mean": _mean(metric_candidate[metric]),
            "delta_mean": _mean(metric_deltas[metric]),
            "paired_bootstrap_ci": _bootstrap_ci(
                metric_deltas[metric],
                samples=bootstrap_samples,
                seed=seed,
                confidence=confidence,
            ),
        }

    summary = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "record_count": len(ordered_ids),
        "metrics": summary_metrics,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_confidence": confidence,
    }
    return summary, deltas


def write_comparison(
    *,
    baseline_scored_path: Path,
    candidate_scored_path: Path,
    output_dir: Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Path]:
    summary, deltas = compare_scored_records(
        baseline_records=read_jsonl(baseline_scored_path),
        candidate_records=read_jsonl(candidate_scored_path),
        metrics=metrics,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    deltas_path = output_dir / "per_example_deltas.jsonl"
    comparison_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(deltas_path, deltas)
    return {
        "comparison": comparison_path,
        "per_example_deltas": deltas_path,
    }
