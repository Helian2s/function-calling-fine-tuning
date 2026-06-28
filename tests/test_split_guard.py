from __future__ import annotations

from pathlib import Path

import pytest

from function_calling_ft.split_guard import (
    SplitAccessError,
    assert_split_allowed,
    classify_split_access,
)


def test_smoke_test_split_is_allowed_for_screening() -> None:
    decision = assert_split_allowed(Path("/workspace/data/test.jsonl"))

    assert decision.allowed
    assert decision.split_name == "smoke-v1-test"
    assert decision.split_lock_status == "frozen_smoke"


def test_final_internal_split_requires_final_evaluation_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(SplitAccessError, match="--final-evaluation"):
        assert_split_allowed("/workspace/data/final_internal_test/test.jsonl")

    with pytest.raises(SplitAccessError, match="final-config"):
        assert_split_allowed(
            "/workspace/data/final_internal_test/test.jsonl",
            final_evaluation=True,
        )

    final_config = tmp_path / "final.yaml"
    final_config.write_text(
        "dataset: /workspace/data/final_internal_test/test.jsonl\n",
        encoding="utf-8",
    )
    decision = assert_split_allowed(
        "/workspace/data/final_internal_test/test.jsonl",
        final_evaluation=True,
        final_config=final_config,
    )

    assert decision.allowed
    assert decision.requires_final_evaluation
    assert decision.requires_final_config
    assert decision.split_lock_status == "locked_final_internal_test"


def test_xlam_frozen_screening_splits_are_allowed() -> None:
    decision = assert_split_allowed(
        "data/processed/xlam_splits_v1/dev_eval_1k.jsonl"
    )

    assert decision.allowed
    assert decision.split_name == "xlam-splits-v1-dev-eval-1k"
    assert decision.split_lock_status == "screening_allowed"


def test_xlam_locked_splits_require_final_evaluation_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(SplitAccessError, match="--final-evaluation"):
        assert_split_allowed(
            "data/processed/xlam_splits_v1/internal_test_locked.jsonl"
        )

    final_config = tmp_path / "final.yaml"
    final_config.write_text(
        "dataset: data/processed/xlam_splits_v1/reserved_challenge_locked.jsonl\n",
        encoding="utf-8",
    )
    decision = assert_split_allowed(
        "data/processed/xlam_splits_v1/reserved_challenge_locked.jsonl",
        final_evaluation=True,
        final_config=final_config,
    )

    assert decision.allowed
    assert decision.requires_final_evaluation
    assert decision.split_lock_status == "locked_reserved_challenge"


def test_unregistered_split_uses_policy_default_allow() -> None:
    decision = classify_split_access("/tmp/ad_hoc_eval.jsonl")

    assert decision.allowed
    assert decision.split_name == "unregistered"
    assert decision.split_lock_status == "unregistered"
