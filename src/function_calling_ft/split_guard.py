from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "configs/common/split_access.yaml"


class SplitAccessError(ValueError):
    pass


@dataclass(frozen=True)
class SplitAccessDecision:
    allowed: bool
    split_name: str
    split_lock_status: str
    reason: str
    requires_final_evaluation: bool
    requires_final_config: bool = False


def load_split_access_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Split access policy must be a mapping: {path}")
    return loaded


def _normalize_path(path: Path | str) -> str:
    raw = Path(path).as_posix()
    if raw.startswith("./"):
        raw = raw[2:]
    return "/" + raw.lstrip("/")


def _matches_any(path: str, patterns: list[Any]) -> bool:
    lowered = path.lower()
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        normalized_pattern = _normalize_path(pattern).lower()
        if normalized_pattern in lowered or lowered.endswith(
            normalized_pattern,
        ):
            return True
    return False


def classify_split_access(
    dataset_path: Path | str,
    *,
    policy: Mapping[str, Any] | None = None,
    final_evaluation: bool = False,
    final_config: Path | str | None = None,
) -> SplitAccessDecision:
    policy = policy or load_split_access_policy()
    normalized_path = _normalize_path(dataset_path)

    for locked in policy.get("locked_splits", []):
        if not isinstance(locked, Mapping):
            continue
        if not _matches_any(
            normalized_path,
            list(locked.get("path_patterns", [])),
        ):
            continue

        name = str(locked.get("name", "locked-split"))
        status = str(locked.get("split_lock_status", "locked"))
        reason = str(locked.get("reason", "split is locked"))
        requires_final_config = bool(
            locked.get("requires_final_config", False),
        )
        config_ok = True
        if final_evaluation and requires_final_config:
            config_ok = _final_config_references_split(
                final_config,
                dataset_path=dataset_path,
                decision_name=name,
                split_lock_status=status,
            )
        return SplitAccessDecision(
            allowed=final_evaluation and config_ok,
            split_name=name,
            split_lock_status=status,
            reason=(
                reason
                if config_ok
                else (
                    "locked final split requires --final-config that "
                    "references the dataset path, split name, or lock status"
                )
            ),
            requires_final_evaluation=True,
            requires_final_config=requires_final_config,
        )

    for allowed in policy.get("allowed_screening_splits", []):
        if not isinstance(allowed, Mapping):
            continue
        paths = list(allowed.get("paths", []))
        if _matches_any(normalized_path, paths):
            return SplitAccessDecision(
                allowed=True,
                split_name=str(allowed.get("name", "allowed-split")),
                split_lock_status=str(
                    allowed.get("split_lock_status", "screening_allowed"),
                ),
                reason="split is approved for screening",
                requires_final_evaluation=False,
            )

    return SplitAccessDecision(
        allowed=str(policy.get("screening_default", "allow")) == "allow",
        split_name="unregistered",
        split_lock_status="unregistered",
        reason="split path is not registered in split access policy",
        requires_final_evaluation=False,
    )


def assert_split_allowed(
    dataset_path: Path | str,
    *,
    final_evaluation: bool = False,
    final_config: Path | str | None = None,
    command_name: str = "screening command",
) -> SplitAccessDecision:
    decision = classify_split_access(
        dataset_path,
        final_evaluation=final_evaluation,
        final_config=final_config,
    )
    if decision.allowed:
        return decision

    raise SplitAccessError(
        f"{command_name} cannot use locked split {decision.split_name!r} "
        f"({decision.split_lock_status}) without --final-evaluation: "
        f"{decision.reason}",
    )


def _final_config_references_split(
    final_config: Path | str | None,
    *,
    dataset_path: Path | str,
    decision_name: str,
    split_lock_status: str,
) -> bool:
    if final_config is None:
        return False

    path = Path(final_config)
    if not path.is_file():
        return False

    text = path.read_text(encoding="utf-8").lower()
    dataset_name = Path(dataset_path).name.lower()
    dataset_text = str(dataset_path).lower()
    return (
        dataset_name in text
        or dataset_text in text
        or decision_name.lower() in text
        or split_lock_status.lower() in text
    )
