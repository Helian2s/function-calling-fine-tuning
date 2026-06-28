from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/experiment_registry.yaml"
ARTIFACT_KEYS = {
    "resolved_config",
    "run_manifest",
    "environment",
    "predictions",
    "per_example_scores",
    "metrics",
    "logs",
    "checksums",
    "report",
}


def _registry() -> dict[str, object]:
    loaded = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_registry_covers_experiments_zero_through_sixteen() -> None:
    registry = _registry()
    experiments = registry["experiments"]
    assert isinstance(experiments, list)

    experiment_ids = {experiment["experiment_id"] for experiment in experiments}

    assert experiment_ids == {f"exp-{index:02d}" for index in range(17)}


def test_registry_entries_have_dependencies_status_and_artifacts() -> None:
    registry = _registry()
    experiments = registry["experiments"]
    assert isinstance(experiments, list)

    for experiment in experiments:
        assert isinstance(experiment.get("dependencies"), list)
        assert isinstance(experiment.get("configs"), list)
        assert isinstance(experiment.get("status"), str)
        assert isinstance(experiment.get("decision_artifacts"), list)
        expected_artifacts = set(experiment.get("expected_artifacts", []))
        assert expected_artifacts
        assert expected_artifacts <= ARTIFACT_KEYS


def test_registry_known_config_paths_exist() -> None:
    registry = _registry()
    experiments = registry["experiments"]
    assert isinstance(experiments, list)

    for experiment in experiments:
        for config_path in experiment.get("configs", []):
            assert (ROOT / config_path).is_file(), config_path
