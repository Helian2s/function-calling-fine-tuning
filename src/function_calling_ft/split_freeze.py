from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from function_calling_ft.loss_mask import build_expected_loss_mask_for_record


SPLIT_SCHEMA_VERSION = "1.0"
DEFAULT_INPUT_PATH = Path("data/processed/xlam_curated_v1/deduplicated.jsonl")
DEFAULT_CURATION_REPORT_PATH = Path(
    "data/processed/xlam_curated_v1/manifests/curation_report.json",
)
DEFAULT_NORMALIZATION_REPORT_PATH = Path(
    "data/processed/xlam_full_v1/manifests/normalization_report.json",
)
DEFAULT_OUTPUT_DIR = Path("data/processed/xlam_splits_v1")

PRIMARY_SPLITS = (
    "train",
    "validation",
    "internal_test_locked",
    "reserved_challenge_locked",
)
SUBSET_SPLITS = ("train_10k", "train_2k", "dev_eval_1k")
EXCLUDED_SPLIT = "excluded_overlength"
LOCKED_SPLITS = {"internal_test_locked", "reserved_challenge_locked"}
SCREENING_ALLOWED_SPLITS = {
    "train",
    "train_10k",
    "train_2k",
    "validation",
    "dev_eval_1k",
}


@dataclass(frozen=True)
class TokenStats:
    example_id: str
    full_tokens: int
    prompt_schema_tokens: int
    supervised_target_tokens: int
    truncation_risk_2048: bool
    truncation_risk_4096: bool


@dataclass(frozen=True)
class SplitExample:
    record: dict[str, Any]
    example_id: str
    source_id: int
    split_group_id: str
    primary_tool_family: str
    primary_api_category: str
    call_category: str
    tool_count: int
    expected_call_count: int
    schema_complexity_score: int
    schema_complexity_bucket: str
    token_length_bucket: str
    token_stats: TokenStats

    @property
    def balance_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.call_category,
            self.primary_api_category,
            count_bucket(self.tool_count),
            count_bucket(self.expected_call_count),
            self.schema_complexity_bucket,
            self.token_length_bucket,
        )


@dataclass(frozen=True)
class SplitGroup:
    group_id: str
    examples: tuple[SplitExample, ...]
    balance_key: tuple[str, str, str, str, str, str]

    @property
    def size(self) -> int:
        return len(self.examples)

    @property
    def primary_tool_family(self) -> str:
        return majority_value(
            example.primary_tool_family for example in self.examples
        )

    @property
    def max_full_tokens(self) -> int:
        return max(example.token_stats.full_tokens for example in self.examples)

    @property
    def max_supervised_tokens(self) -> int:
        return max(
            example.token_stats.supervised_target_tokens
            for example in self.examples
        )


@dataclass(frozen=True)
class FrozenSplitResult:
    examples: tuple[SplitExample, ...]
    groups: tuple[SplitGroup, ...]
    primary_assignments: dict[str, str]
    subset_memberships: dict[str, tuple[str, ...]]
    selected_max_sequence_length: int
    overlength_group_ids: tuple[str, ...]
    challenge_primary_families: tuple[str, ...]
    config: dict[str, Any]


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_line(value: Any) -> str:
    return canonical_json_dumps(value) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object.",
                )
            yield value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(canonical_json_line(record))


def count_jsonl(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run_git(["rev-parse", "HEAD"])
    diff = run_git(["diff", "--quiet"])
    cached_diff = run_git(["diff", "--cached", "--quiet"])
    untracked = run_git(["ls-files", "--others", "--exclude-standard"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": (
            diff.returncode != 0
            or cached_diff.returncode != 0
            or (untracked.returncode == 0 and bool(untracked.stdout.strip()))
        ),
    }


def stable_hash(*parts: object, seed: int = 0) -> str:
    return sha256_text(
        canonical_json_dumps({"seed": seed, "parts": [str(part) for part in parts]}),
    )


def majority_value(values: Iterable[str]) -> str:
    counter = Counter(values)
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def count_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 5:
        return "4-5"
    return "6+"


def token_length_bucket(tokens: int) -> str:
    if tokens <= 512:
        return "<=512"
    if tokens <= 1024:
        return "513-1024"
    if tokens <= 1536:
        return "1025-1536"
    if tokens <= 2048:
        return "1537-2048"
    if tokens <= 4096:
        return "2049-4096"
    return ">4096"


def schema_complexity_bucket(score: int) -> str:
    if score <= 8:
        return "simple"
    if score <= 24:
        return "moderate"
    if score <= 48:
        return "complex"
    return "very_complex"


def schema_complexity_score(record: Mapping[str, Any]) -> int:
    def walk(value: Any) -> int:
        if isinstance(value, dict):
            score = 1
            properties = value.get("properties")
            if isinstance(properties, dict):
                score += len(properties)
            required = value.get("required")
            if isinstance(required, list):
                score += len(required)
            for child in value.values():
                score += walk(child)
            return score
        if isinstance(value, list):
            return sum(walk(child) for child in value)
        return 0

    total = 0
    tools = record.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            if not isinstance(function, Mapping):
                continue
            total += walk(function.get("parameters", {}))
    return total


def example_id(record: Mapping[str, Any]) -> str:
    value = record.get("example_id") or record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Record does not contain an example id.")
    return value


def source_id(record: Mapping[str, Any]) -> int:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return 2**63 - 1
    value = metadata.get("source_id")
    if value is None:
        return 2**63 - 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2**63 - 1


def curation_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("curation_metadata")
    if not isinstance(value, Mapping):
        raise ValueError(f"Record {example_id(record)} lacks curation_metadata.")
    return value


def split_group_id(record: Mapping[str, Any]) -> str:
    value = curation_metadata(record).get("split_group_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Record {example_id(record)} lacks split_group_id.")
    return value


def measure_token_stats(
    tokenizer: Any,
    record: dict[str, Any],
) -> TokenStats:
    result = build_expected_loss_mask_for_record(
        tokenizer,
        record,
        enable_thinking=False,
    )
    full_tokens = len(result.input_ids)
    supervised = result.included_token_count
    return TokenStats(
        example_id=example_id(record),
        full_tokens=full_tokens,
        prompt_schema_tokens=full_tokens - supervised,
        supervised_target_tokens=supervised,
        truncation_risk_2048=full_tokens > 2048,
        truncation_risk_4096=full_tokens > 4096,
    )


def build_split_examples(
    records: Sequence[dict[str, Any]],
    token_stats_by_id: Mapping[str, TokenStats],
) -> tuple[SplitExample, ...]:
    examples: list[SplitExample] = []
    for record in records:
        metadata = curation_metadata(record)
        record_id = example_id(record)
        stats = token_stats_by_id[record_id]
        complexity_score = schema_complexity_score(record)
        examples.append(
            SplitExample(
                record=record,
                example_id=record_id,
                source_id=source_id(record),
                split_group_id=str(metadata["split_group_id"]),
                primary_tool_family=str(
                    metadata.get("primary_tool_family", "unknown"),
                ),
                primary_api_category=str(
                    metadata.get("primary_api_category", "unknown"),
                ),
                call_category=str(metadata.get("call_category", "unknown")),
                tool_count=int(metadata.get("tool_count", 0)),
                expected_call_count=int(
                    metadata.get("expected_call_count", 0),
                ),
                schema_complexity_score=complexity_score,
                schema_complexity_bucket=schema_complexity_bucket(
                    complexity_score,
                ),
                token_length_bucket=token_length_bucket(stats.full_tokens),
                token_stats=stats,
            )
        )

    return tuple(
        sorted(
            examples,
            key=lambda item: (item.source_id, item.example_id),
        )
    )


def build_groups(examples: Sequence[SplitExample]) -> tuple[SplitGroup, ...]:
    grouped: dict[str, list[SplitExample]] = defaultdict(list)
    for example in examples:
        grouped[example.split_group_id].append(example)

    groups: list[SplitGroup] = []
    for group_id, group_examples in grouped.items():
        balance_key = majority_balance_key(group_examples)
        groups.append(
            SplitGroup(
                group_id=group_id,
                examples=tuple(
                    sorted(
                        group_examples,
                        key=lambda item: (item.source_id, item.example_id),
                    )
                ),
                balance_key=balance_key,
            )
        )

    return tuple(
        sorted(
            groups,
            key=lambda group: (
                min(example.source_id for example in group.examples),
                group.group_id,
            ),
        )
    )


def majority_balance_key(
    examples: Sequence[SplitExample],
) -> tuple[str, str, str, str, str, str]:
    keys = [example.balance_key for example in examples]
    return sorted(Counter(keys).items(), key=lambda item: (-item[1], item[0]))[
        0
    ][0]


def select_max_sequence_length(
    examples: Sequence[SplitExample],
    *,
    preferred: int = 2048,
    fallback: int = 4096,
    coverage_threshold: float = 0.99,
) -> dict[str, Any]:
    lengths = sorted(example.token_stats.full_tokens for example in examples)
    covered = sum(length <= preferred for length in lengths)
    coverage = covered / len(lengths) if lengths else 1.0
    selected = preferred if coverage >= coverage_threshold else fallback
    return {
        "selected": selected,
        "preferred": preferred,
        "fallback": fallback,
        "coverage_threshold": coverage_threshold,
        "preferred_covered_records": covered,
        "preferred_coverage": coverage,
        "fallback_covered_records": sum(length <= fallback for length in lengths),
        "fallback_coverage": (
            sum(length <= fallback for length in lengths) / len(lengths)
            if lengths
            else 1.0
        ),
    }


def group_sort_key(group: SplitGroup, *, seed: int) -> tuple[str, int, str]:
    return (
        stable_hash(group.group_id, seed=seed),
        min(example.source_id for example in group.examples),
        group.group_id,
    )


def challenge_group_score(group: SplitGroup, family_total: int) -> float:
    categories = Counter(
        example.primary_api_category for example in group.examples
    )
    call_categories = Counter(example.call_category for example in group.examples)
    rare_category_bonus = 0.0
    for category in ("health", "search", "media", "travel", "weather", "unknown"):
        rare_category_bonus += categories.get(category, 0) * 1.5
    hard_call_bonus = (
        call_categories.get("multiple_parallel", 0) * 3.0
        + call_categories.get("parallel", 0) * 1.0
        + call_categories.get("multiple", 0) * 1.0
    )
    tool_bonus = sum(max(0, example.tool_count - 4) for example in group.examples)
    call_bonus = sum(
        max(0, example.expected_call_count - 2) for example in group.examples
    )
    length_bonus = sum(
        1.0
        for example in group.examples
        if example.token_stats.full_tokens > 1024
    )
    rarity_bonus = group.size / max(1, family_total)
    return (
        rare_category_bonus
        + hard_call_bonus
        + tool_bonus
        + call_bonus
        + length_bonus
        + rarity_bonus
    )


def select_challenge_groups(
    groups: Sequence[SplitGroup],
    *,
    target_records: int,
    seed: int,
    max_overshoot_records: int,
) -> tuple[set[str], tuple[str, ...]]:
    by_family: dict[str, list[SplitGroup]] = defaultdict(list)
    for group in groups:
        by_family[group.primary_tool_family].append(group)

    family_rows: list[tuple[str, int, float, str]] = []
    for family, family_groups in by_family.items():
        total = sum(group.size for group in family_groups)
        score = sum(
            challenge_group_score(group, total) for group in family_groups
        )
        family_rows.append(
            (family, total, score / max(1, total), stable_hash(family, seed=seed))
        )

    selected_group_ids: set[str] = set()
    selected_families: list[str] = []
    selected_records = 0
    max_records = target_records + max_overshoot_records

    for family, total, score, family_hash in sorted(
        family_rows,
        key=lambda item: (item[1], -item[2], item[3], item[0]),
    ):
        del score, family_hash
        if selected_records >= target_records:
            break
        if selected_records + total > max_records:
            continue
        for group in by_family[family]:
            selected_group_ids.add(group.group_id)
        selected_families.append(family)
        selected_records += total

    if selected_records < target_records:
        selected_group_ids.update(
            allocate_balanced_groups(
                [
                    group
                    for group in groups
                    if group.group_id not in selected_group_ids
                ],
                target_records=target_records - selected_records,
                seed=seed + 17,
                max_overshoot_records=max_overshoot_records,
            )
        )

    return selected_group_ids, tuple(sorted(selected_families))


def target_balance_counts(
    groups: Sequence[SplitGroup],
    target_records: int,
) -> dict[tuple[str, str, str, str, str, str], float]:
    total = sum(group.size for group in groups)
    counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for group in groups:
        counter[group.balance_key] += group.size
    if total == 0:
        return {}
    return {
        key: (count / total) * target_records
        for key, count in counter.items()
    }


def allocate_balanced_groups(
    groups: Sequence[SplitGroup],
    *,
    target_records: int,
    seed: int,
    max_overshoot_records: int,
) -> set[str]:
    if target_records <= 0:
        return set()

    buckets: dict[tuple[str, str, str, str, str, str], list[SplitGroup]] = (
        defaultdict(list)
    )
    for group in groups:
        buckets[group.balance_key].append(group)
    for key in list(buckets):
        buckets[key] = sorted(
            buckets[key],
            key=lambda group: group_sort_key(group, seed=seed),
        )

    targets = target_balance_counts(groups, target_records)
    selected: set[str] = set()
    current: Counter[tuple[str, str, str, str, str, str]] = Counter()
    selected_records = 0
    max_records = target_records + max_overshoot_records

    while buckets and selected_records < target_records:
        available_keys = [key for key, value in buckets.items() if value]
        if not available_keys:
            break

        def key_score(key: tuple[str, str, str, str, str, str]) -> tuple[float, str]:
            target = targets.get(key, 0.0)
            deficit = target - current[key]
            ratio = deficit / max(1.0, target)
            return (ratio, stable_hash(*key, seed=seed))

        best_key = max(available_keys, key=key_score)
        candidates = buckets[best_key]
        chosen_index = 0
        for index, candidate in enumerate(candidates):
            if selected_records + candidate.size <= max_records:
                chosen_index = index
                break
        group = candidates.pop(chosen_index)
        if not candidates:
            buckets.pop(best_key, None)
        selected.add(group.group_id)
        selected_records += group.size
        current[group.balance_key] += group.size

    return selected


def build_frozen_splits(
    records: Sequence[dict[str, Any]],
    token_stats_by_id: Mapping[str, TokenStats],
    config: Mapping[str, Any],
) -> FrozenSplitResult:
    split_config = dict(config.get("splits", {}))
    sequence_config = dict(config.get("sequence_length", {}))
    seed = int(config.get("seed", 20260626))
    target_validation = int(split_config.get("validation_target_records", 5000))
    target_internal = int(split_config.get("internal_test_target_records", 5000))
    target_challenge = int(split_config.get("challenge_target_records", 5000))
    max_overshoot = int(split_config.get("max_overshoot_records", 150))

    examples = build_split_examples(records, token_stats_by_id)
    groups = build_groups(examples)
    sequence_decision = select_max_sequence_length(
        examples,
        preferred=int(sequence_config.get("preferred_max_length", 2048)),
        fallback=int(sequence_config.get("fallback_max_length", 4096)),
        coverage_threshold=float(
            sequence_config.get("preferred_coverage_threshold", 0.99),
        ),
    )
    selected_max_length = int(sequence_decision["selected"])
    overlength_group_ids = {
        group.group_id
        for group in groups
        if group.max_full_tokens > selected_max_length
        or group.max_supervised_tokens > selected_max_length
    }
    eligible_groups = [
        group for group in groups if group.group_id not in overlength_group_ids
    ]
    primary_assignments: dict[str, str] = {}
    for group in groups:
        if group.group_id in overlength_group_ids:
            for example in group.examples:
                primary_assignments[example.example_id] = EXCLUDED_SPLIT

    challenge_group_ids, challenge_families = select_challenge_groups(
        eligible_groups,
        target_records=target_challenge,
        seed=seed + 101,
        max_overshoot_records=max_overshoot,
    )
    remaining_after_challenge = [
        group
        for group in eligible_groups
        if group.group_id not in challenge_group_ids
    ]
    internal_group_ids = allocate_balanced_groups(
        remaining_after_challenge,
        target_records=target_internal,
        seed=seed + 211,
        max_overshoot_records=max_overshoot,
    )
    remaining_after_internal = [
        group
        for group in remaining_after_challenge
        if group.group_id not in internal_group_ids
    ]
    validation_group_ids = allocate_balanced_groups(
        remaining_after_internal,
        target_records=target_validation,
        seed=seed + 307,
        max_overshoot_records=max_overshoot,
    )

    for group in eligible_groups:
        if group.group_id in challenge_group_ids:
            split = "reserved_challenge_locked"
        elif group.group_id in internal_group_ids:
            split = "internal_test_locked"
        elif group.group_id in validation_group_ids:
            split = "validation"
        else:
            split = "train"
        for example in group.examples:
            primary_assignments[example.example_id] = split

    train_groups = [
        group
        for group in eligible_groups
        if all(
            primary_assignments[example.example_id] == "train"
            for example in group.examples
        )
    ]
    validation_groups = [
        group
        for group in eligible_groups
        if all(
            primary_assignments[example.example_id] == "validation"
            for example in group.examples
        )
    ]
    train_10k_group_ids = allocate_balanced_groups(
        train_groups,
        target_records=int(split_config.get("train_10k_target_records", 10000)),
        seed=seed + 401,
        max_overshoot_records=max_overshoot,
    )
    train_2k_group_ids = allocate_balanced_groups(
        [
            group
            for group in train_groups
            if group.group_id in train_10k_group_ids
        ],
        target_records=int(split_config.get("train_2k_target_records", 2000)),
        seed=seed + 503,
        max_overshoot_records=max_overshoot,
    )
    dev_eval_group_ids = allocate_balanced_groups(
        validation_groups,
        target_records=int(split_config.get("dev_eval_target_records", 1000)),
        seed=seed + 607,
        max_overshoot_records=max_overshoot,
    )

    subset_memberships: dict[str, list[str]] = defaultdict(list)
    for group in train_groups:
        for example in group.examples:
            if group.group_id in train_10k_group_ids:
                subset_memberships[example.example_id].append("train_10k")
            if group.group_id in train_2k_group_ids:
                subset_memberships[example.example_id].append("train_2k")
    for group in validation_groups:
        for example in group.examples:
            if group.group_id in dev_eval_group_ids:
                subset_memberships[example.example_id].append("dev_eval_1k")

    return FrozenSplitResult(
        examples=examples,
        groups=groups,
        primary_assignments=primary_assignments,
        subset_memberships={
            key: tuple(sorted(value))
            for key, value in subset_memberships.items()
        },
        selected_max_sequence_length=selected_max_length,
        overlength_group_ids=tuple(sorted(overlength_group_ids)),
        challenge_primary_families=challenge_families,
        config={
            **dict(config),
            "sequence_length_decision": sequence_decision,
        },
    )


def split_lock_status(split_name: str) -> str:
    if split_name == "internal_test_locked":
        return "locked_final_internal_test"
    if split_name == "reserved_challenge_locked":
        return "locked_reserved_challenge"
    if split_name == EXCLUDED_SPLIT:
        return "excluded_overlength"
    return "screening_allowed"


def record_hash(record: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json_dumps(record))


def split_record(
    example: SplitExample,
    *,
    primary_split: str,
    subset_name: str | None,
    result: FrozenSplitResult,
) -> dict[str, Any]:
    record = copy.deepcopy(example.record)
    metadata = record.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["split"] = primary_split
    split_metadata = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "primary_split": primary_split,
        "subset_name": subset_name,
        "subset_memberships": list(
            result.subset_memberships.get(example.example_id, ()),
        ),
        "split_group_id": example.split_group_id,
        "split_lock_status": split_lock_status(primary_split),
        "selected_max_sequence_length": (
            result.selected_max_sequence_length
        ),
        "token_counts": token_stats_record(example.token_stats),
    }
    record["split_metadata"] = split_metadata
    return record


def token_stats_record(stats: TokenStats) -> dict[str, Any]:
    return {
        "full_tokens": stats.full_tokens,
        "prompt_schema_tokens": stats.prompt_schema_tokens,
        "supervised_target_tokens": stats.supervised_target_tokens,
        "truncation_risk_2048": stats.truncation_risk_2048,
        "truncation_risk_4096": stats.truncation_risk_4096,
    }


def manifest_record(
    example: SplitExample,
    *,
    primary_split: str,
    subset_name: str | None,
    result: FrozenSplitResult,
) -> dict[str, Any]:
    metadata = example.record.get("metadata", {})
    source_revision = (
        metadata.get("source_revision") if isinstance(metadata, Mapping) else None
    )
    source_file_sha = (
        metadata.get("source_file_sha256")
        if isinstance(metadata, Mapping)
        else None
    )
    curation = curation_metadata(example.record)
    base_record = {
        "manifest_schema_version": SPLIT_SCHEMA_VERSION,
        "example_id": example.example_id,
        "primary_split": primary_split,
        "subset_name": subset_name,
        "subset_memberships": list(
            result.subset_memberships.get(example.example_id, ()),
        ),
        "split_group_id": example.split_group_id,
        "split_lock_status": split_lock_status(primary_split),
        "source_id": example.source_id,
        "source_revision": source_revision,
        "source_file_sha256": source_file_sha,
        "record_sha256": record_hash(example.record),
        "exact_duplicate_hash": curation.get("exact_duplicate_hash"),
        "call_category": example.call_category,
        "primary_api_category": example.primary_api_category,
        "primary_tool_family": example.primary_tool_family,
        "tool_families": list(curation.get("tool_families", [])),
        "tool_count": example.tool_count,
        "expected_call_count": example.expected_call_count,
        "schema_complexity_score": example.schema_complexity_score,
        "schema_complexity_bucket": example.schema_complexity_bucket,
        "token_length_bucket": example.token_length_bucket,
        "selected_max_sequence_length": result.selected_max_sequence_length,
    }
    base_record.update(token_stats_record(example.token_stats))
    return base_record


def examples_for_split(
    result: FrozenSplitResult,
    split_name: str,
) -> list[SplitExample]:
    if split_name in PRIMARY_SPLITS or split_name == EXCLUDED_SPLIT:
        return [
            example
            for example in result.examples
            if result.primary_assignments[example.example_id] == split_name
        ]

    if split_name in SUBSET_SPLITS:
        return [
            example
            for example in result.examples
            if split_name
            in result.subset_memberships.get(example.example_id, ())
        ]

    raise ValueError(f"Unknown split name: {split_name}")


def primary_split_for_output(split_name: str) -> str:
    if split_name in {"train_10k", "train_2k"}:
        return "train"
    if split_name == "dev_eval_1k":
        return "validation"
    return split_name


def clean_output_dir(output_dir: Path) -> None:
    for child_name in (
        "manifests",
        "DATASET_CARD.md",
        "checksums.sha256",
        "train_full.jsonl",
        "train_10k.jsonl",
        "train_2k.jsonl",
        "validation.jsonl",
        "dev_eval_1k.jsonl",
        "internal_test_locked.jsonl",
        "reserved_challenge_locked.jsonl",
        "excluded_overlength.jsonl",
    ):
        child = output_dir / child_name
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()


def write_split_artifacts(
    *,
    result: FrozenSplitResult,
    output_dir: Path,
    repo_root: Path,
    curation_report_path: Path = DEFAULT_CURATION_REPORT_PATH,
    normalization_report_path: Path = DEFAULT_NORMALIZATION_REPORT_PATH,
    config_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": "train_full.jsonl",
        "train_10k": "train_10k.jsonl",
        "train_2k": "train_2k.jsonl",
        "validation": "validation.jsonl",
        "dev_eval_1k": "dev_eval_1k.jsonl",
        "internal_test_locked": "internal_test_locked.jsonl",
        "reserved_challenge_locked": "reserved_challenge_locked.jsonl",
        EXCLUDED_SPLIT: "excluded_overlength.jsonl",
    }
    all_split_names = tuple(file_map)

    for split_name, file_name in file_map.items():
        examples = examples_for_split(result, split_name)
        primary_split = primary_split_for_output(split_name)
        subset_name = (
            split_name
            if split_name in SUBSET_SPLITS
            else None
        )
        write_jsonl(
            output_dir / file_name,
            (
                split_record(
                    example,
                    primary_split=primary_split,
                    subset_name=subset_name,
                    result=result,
                )
                for example in examples
            ),
        )
        write_jsonl(
            manifests_dir / f"{split_name}_manifest.jsonl",
            (
                manifest_record(
                    example,
                    primary_split=primary_split,
                    subset_name=subset_name,
                    result=result,
                )
                for example in examples
            ),
        )

    write_jsonl(
        manifests_dir / "assignment_manifest.jsonl",
        (
            manifest_record(
                example,
                primary_split=result.primary_assignments[example.example_id],
                subset_name=None,
                result=result,
            )
            for example in result.examples
        ),
    )
    write_jsonl(
        manifests_dir / "token_lengths.jsonl",
        (
            {
                "example_id": example.example_id,
                "split_group_id": example.split_group_id,
                **token_stats_record(example.token_stats),
                "token_length_bucket": example.token_length_bucket,
            }
            for example in result.examples
        ),
    )
    write_jsonl(
        manifests_dir / "overlength_candidates.jsonl",
        (
            manifest_record(
                example,
                primary_split=EXCLUDED_SPLIT,
                subset_name=None,
                result=result,
            )
            for example in examples_for_split(result, EXCLUDED_SPLIT)
        ),
    )

    leakage_report = split_leakage_report(result)
    write_json(manifests_dir / "leakage_audit_report.json", leakage_report)
    report = build_split_report(
        result=result,
        output_dir=output_dir,
        repo_root=repo_root,
        split_names=all_split_names,
        curation_report_path=curation_report_path,
        normalization_report_path=normalization_report_path,
        config_path=config_path,
        leakage_report=leakage_report,
    )
    report["outputs"]["checksums"] = str(output_dir / "checksums.sha256")
    write_json(manifests_dir / "split_report.json", report)
    write_checksums(output_dir)
    write_dataset_card(
        output_dir / "DATASET_CARD.md",
        report=report,
        normalization_report=read_json(normalization_report_path),
        curation_report=read_json(curation_report_path),
    )
    write_checksums(output_dir)
    return report


def split_leakage_report(result: FrozenSplitResult) -> dict[str, Any]:
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for example in result.examples:
        split = result.primary_assignments[example.example_id]
        if split == EXCLUDED_SPLIT:
            continue
        group_to_splits[example.split_group_id].add(split)

    violations = [
        {
            "split_group_id": group_id,
            "splits": sorted(splits),
        }
        for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    ]
    return {
        "audit_schema_version": SPLIT_SCHEMA_VERSION,
        "status": "fail" if violations else "pass",
        "cross_split_group_count": len(violations),
        "violations": sorted(
            violations,
            key=lambda item: str(item["split_group_id"]),
        ),
    }


def distribution_for_examples(
    examples: Sequence[SplitExample],
    field: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for example in examples:
        counter[str(getattr(example, field))] += 1
    return dict(sorted(counter.items()))


def percentile(sorted_values: Sequence[int], pct: float) -> int | None:
    if not sorted_values:
        return None
    index = round((pct / 100) * (len(sorted_values) - 1))
    return sorted_values[int(index)]


def token_distribution(examples: Sequence[SplitExample]) -> dict[str, Any]:
    lengths = sorted(example.token_stats.full_tokens for example in examples)
    supervised = sorted(
        example.token_stats.supervised_target_tokens for example in examples
    )
    return {
        "records": len(examples),
        "full_tokens": {
            "min": lengths[0] if lengths else None,
            "max": lengths[-1] if lengths else None,
            "p50": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
            "p99_5": percentile(lengths, 99.5),
            "p99_9": percentile(lengths, 99.9),
        },
        "supervised_target_tokens": {
            "min": supervised[0] if supervised else None,
            "max": supervised[-1] if supervised else None,
            "p50": percentile(supervised, 50),
            "p90": percentile(supervised, 90),
            "p99": percentile(supervised, 99),
        },
        "length_buckets": distribution_for_examples(
            examples,
            "token_length_bucket",
        ),
    }


def split_counts(result: FrozenSplitResult) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name in (
        "train",
        "train_10k",
        "train_2k",
        "validation",
        "dev_eval_1k",
        "internal_test_locked",
        "reserved_challenge_locked",
        EXCLUDED_SPLIT,
    ):
        examples = examples_for_split(result, split_name)
        counts[split_name] = {
            "records": len(examples),
            "groups": len({example.split_group_id for example in examples}),
        }
    return counts


def check_nesting(result: FrozenSplitResult) -> dict[str, Any]:
    train = {
        example.example_id for example in examples_for_split(result, "train")
    }
    train_10k = {
        example.example_id
        for example in examples_for_split(result, "train_10k")
    }
    train_2k = {
        example.example_id for example in examples_for_split(result, "train_2k")
    }
    validation = {
        example.example_id
        for example in examples_for_split(result, "validation")
    }
    dev_eval = {
        example.example_id
        for example in examples_for_split(result, "dev_eval_1k")
    }
    return {
        "train_2k_subset_of_train_10k": train_2k.issubset(train_10k),
        "train_10k_subset_of_train": train_10k.issubset(train),
        "dev_eval_1k_subset_of_validation": dev_eval.issubset(validation),
    }


def build_split_report(
    *,
    result: FrozenSplitResult,
    output_dir: Path,
    repo_root: Path,
    split_names: Sequence[str],
    curation_report_path: Path,
    normalization_report_path: Path,
    config_path: Path | None,
    leakage_report: Mapping[str, Any],
) -> dict[str, Any]:
    del split_names
    eligible_examples = [
        example
        for example in result.examples
        if result.primary_assignments[example.example_id] != EXCLUDED_SPLIT
    ]
    output_paths = {
        "train_full": str(output_dir / "train_full.jsonl"),
        "train_10k": str(output_dir / "train_10k.jsonl"),
        "train_2k": str(output_dir / "train_2k.jsonl"),
        "validation": str(output_dir / "validation.jsonl"),
        "dev_eval_1k": str(output_dir / "dev_eval_1k.jsonl"),
        "internal_test_locked": str(output_dir / "internal_test_locked.jsonl"),
        "reserved_challenge_locked": str(
            output_dir / "reserved_challenge_locked.jsonl"
        ),
        "excluded_overlength": str(output_dir / "excluded_overlength.jsonl"),
        "assignment_manifest": str(
            output_dir / "manifests" / "assignment_manifest.jsonl"
        ),
        "token_lengths": str(output_dir / "manifests" / "token_lengths.jsonl"),
        "overlength_candidates": str(
            output_dir / "manifests" / "overlength_candidates.jsonl"
        ),
        "leakage_audit_report": str(
            output_dir / "manifests" / "leakage_audit_report.json"
        ),
        "dataset_card": str(output_dir / "DATASET_CARD.md"),
    }
    sequence_decision = dict(result.config["sequence_length_decision"])
    return {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "experiment_id": "exp-01",
        "task_id": "Task04",
        "status": "pass" if leakage_report.get("status") == "pass" else "fail",
        "assignment": {
            "seed": result.config.get("seed"),
            "primary_split_unit": "split_group_id",
            "balancing_objectives": [
                "call_category",
                "primary_api_category",
                "primary_tool_family",
                "tool_count_bucket",
                "expected_call_count_bucket",
                "schema_complexity_bucket",
                "rendered_token_length_bucket",
            ],
            "challenge_strategy": (
                "Hold out complete primary-tool-family bundles, ordered by "
                "family rarity and hard-category score, before balanced "
                "validation/internal-test allocation."
            ),
            "target_policy": (
                "Prioritize validation, locked internal test, and reserved "
                "challenge near 5K each; assign the remaining eligible "
                "records to train."
            ),
        },
        "sequence_length": {
            **sequence_decision,
            "selected_reason": (
                "2,048 covers at least the configured threshold of retained "
                "records under the native Qwen template; overlength groups "
                "are explicitly excluded instead of truncating gold tool calls."
                if int(sequence_decision["selected"]) == 2048
                else (
                    "4,096 selected because 2,048 did not meet the configured "
                    "coverage threshold."
                )
            ),
            "tokenizer": {
                "model_id": "Qwen/Qwen3-1.7B",
                "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "enable_thinking": False,
                "native_chat_template": True,
            },
        },
        "counts": split_counts(result),
        "nesting": check_nesting(result),
        "exclusions": {
            "reason": "full_input_plus_target_tokens_exceed_selected_max_sequence_length_or_target_exceeds_length",
            "records": len(examples_for_split(result, EXCLUDED_SPLIT)),
            "groups": len(result.overlength_group_ids),
            "group_ids": list(result.overlength_group_ids),
        },
        "challenge": {
            "primary_families_held_out": list(
                result.challenge_primary_families,
            ),
            "primary_family_count": len(result.challenge_primary_families),
        },
        "distributions": {
            split_name: {
                "call_category": distribution_for_examples(
                    examples_for_split(result, split_name),
                    "call_category",
                ),
                "primary_api_category": distribution_for_examples(
                    examples_for_split(result, split_name),
                    "primary_api_category",
                ),
                "primary_tool_family_unique": len(
                    {
                        example.primary_tool_family
                        for example in examples_for_split(result, split_name)
                    }
                ),
                "tool_count": distribution_for_examples(
                    examples_for_split(result, split_name),
                    "tool_count",
                ),
                "expected_call_count": distribution_for_examples(
                    examples_for_split(result, split_name),
                    "expected_call_count",
                ),
                "schema_complexity_bucket": distribution_for_examples(
                    examples_for_split(result, split_name),
                    "schema_complexity_bucket",
                ),
                "token_distribution": token_distribution(
                    examples_for_split(result, split_name),
                ),
            }
            for split_name in (
                "train",
                "validation",
                "internal_test_locked",
                "reserved_challenge_locked",
                EXCLUDED_SPLIT,
            )
        },
        "token_distribution_all_retained": token_distribution(
            list(result.examples),
        ),
        "token_distribution_eligible": token_distribution(eligible_examples),
        "leakage_audit": dict(leakage_report),
        "inputs": {
            "deduplicated_dataset": str(DEFAULT_INPUT_PATH),
            "curation_report": str(curation_report_path),
            "curation_report_sha256": (
                sha256_file(curation_report_path)
                if curation_report_path.is_file()
                else None
            ),
            "normalization_report": str(normalization_report_path),
            "normalization_report_sha256": (
                sha256_file(normalization_report_path)
                if normalization_report_path.is_file()
                else None
            ),
            "config": str(config_path) if config_path is not None else None,
            "config_sha256": (
                sha256_file(config_path)
                if config_path is not None and config_path.is_file()
                else None
            ),
        },
        "outputs": output_paths,
        "git": git_metadata(repo_root),
    }


def write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "checksums.sha256"
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != checksum_path.name
    ]
    with checksum_path.open("w", encoding="utf-8") as file:
        for path in files:
            relative = path.relative_to(output_dir)
            file.write(f"{sha256_file(path)}  {relative.as_posix()}\n")
    return checksum_path


def checksum_entries(
    output_dir: Path,
    *,
    excluded_relative_paths: set[str] | None = None,
) -> list[tuple[str, str]]:
    excluded_relative_paths = excluded_relative_paths or set()
    checksum_path = output_dir / "checksums.sha256"
    entries: list[tuple[str, str]] = []
    if not checksum_path.is_file():
        return entries
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        if relative in excluded_relative_paths:
            continue
        entries.append((digest, relative))
    return entries


def write_dataset_card(
    path: Path,
    *,
    report: Mapping[str, Any],
    normalization_report: Mapping[str, Any],
    curation_report: Mapping[str, Any],
) -> None:
    counts = report["counts"]
    sequence = report["sequence_length"]
    leakage = report["leakage_audit"]
    norm_processing = normalization_report.get("processing", {})
    curation_exact = curation_report.get("exact_deduplication", {})
    checksum_table = "\n".join(
        f"| `{relative}` | `{digest}` |"
        for digest, relative in checksum_entries(
            path.parent,
            excluded_relative_paths={"DATASET_CARD.md", "checksums.sha256"},
        )
    )
    if not checksum_table:
        checksum_table = "| pending | pending |"

    content = f"""# xLAM Function-Calling Splits v1

## Source

- Dataset: `Salesforce/xlam-function-calling-60k`
- Source revision: `{normalization_report.get("dataset", {}).get("revision")}`
- Source split/config: `train` / `default`
- License: `{normalization_report.get("dataset", {}).get("license")}`
- Access note: `{normalization_report.get("dataset", {}).get("access")}`
- Model/tokenizer for rendering: `Qwen/Qwen3-1.7B` at revision `{sequence["tokenizer"]["revision"]}`

## Normalization And Curation

- Raw records: `{norm_processing.get("input_records")}`
- Normalized accepted records: `{norm_processing.get("accepted_records")}`
- Quarantined during normalization: `{norm_processing.get("quarantined_records")}`
- Deduplicated retained records: `{curation_exact.get("retained_records")}`
- Exact duplicate groups: `{curation_exact.get("duplicate_groups")}`
- Exact duplicate records removed: `{curation_exact.get("duplicate_records")}`
- Curator comparison status: `{curation_report.get("curator", {}).get("status")}`

## Split Construction

Splits are assigned by whole `split_group_id`, never by individual rows. The
reserved challenge split is selected first by holding out complete
primary-tool-family bundles ordered by family rarity and hard-category score.
Validation and locked internal test are then allocated with deterministic
stratified balancing over call category, API category, tool count, expected
call count, schema complexity, and rendered token length. Train receives the
remaining eligible records. `train_2k` is nested in `train_10k`, which is
nested in `train_full`; `dev_eval_1k` is nested in validation.

| Split | Records | Groups | Lock status |
| --- | ---: | ---: | --- |
| train_full | {counts["train"]["records"]} | {counts["train"]["groups"]} | screening_allowed |
| train_10k | {counts["train_10k"]["records"]} | {counts["train_10k"]["groups"]} | screening_allowed |
| train_2k | {counts["train_2k"]["records"]} | {counts["train_2k"]["groups"]} | screening_allowed |
| validation | {counts["validation"]["records"]} | {counts["validation"]["groups"]} | screening_allowed |
| dev_eval_1k | {counts["dev_eval_1k"]["records"]} | {counts["dev_eval_1k"]["groups"]} | screening_allowed |
| internal_test_locked | {counts["internal_test_locked"]["records"]} | {counts["internal_test_locked"]["groups"]} | locked_final_internal_test |
| reserved_challenge_locked | {counts["reserved_challenge_locked"]["records"]} | {counts["reserved_challenge_locked"]["groups"]} | locked_reserved_challenge |
| excluded_overlength | {counts["excluded_overlength"]["records"]} | {counts["excluded_overlength"]["groups"]} | excluded_overlength |

## Sequence Length Decision

- Native Qwen tool-calling template: enabled
- `enable_thinking`: `False`
- Selected default max sequence length: `{sequence["selected"]}`
- 2,048-token coverage: `{sequence["preferred_coverage"]:.6f}`
- 4,096-token coverage: `{sequence["fallback_coverage"]:.6f}`
- Overlength excluded records: `{report["exclusions"]["records"]}`

No supervised target is silently truncated. Records in overlength groups are
explicitly written to `excluded_overlength.jsonl` and listed in
`manifests/overlength_candidates.jsonl`.

## Leakage Audit

- Status: `{leakage.get("status")}`
- Cross-split group overlaps: `{leakage.get("cross_split_group_count")}`

## Known Limitations

- API/category labels and tool-family names are deterministic derived metadata,
  not human-reviewed taxonomy labels.
- Fuzzy duplicate candidates remain review-only annotations; only exact
  duplicates are removed.
- The reserved challenge emphasizes primary tool-family holdout and hard
  categories, but secondary tool-family overlap can still occur through
  multi-tool examples.
- BFCL is external evaluation data and is not included in these splits.

## Checksums

| Artifact | SHA256 |
| --- | --- |
{checksum_table}
"""
    path.write_text(content, encoding="utf-8")


def validate_frozen_splits(result: FrozenSplitResult) -> dict[str, Any]:
    example_ids = [example.example_id for example in result.examples]
    assigned_ids = set(result.primary_assignments)
    primary_split_by_id = result.primary_assignments
    split_sets = {
        split: {
            example_id
            for example_id, value in primary_split_by_id.items()
            if value == split
        }
        for split in (*PRIMARY_SPLITS, EXCLUDED_SPLIT)
    }
    all_primary_ids = set().union(*split_sets.values()) if split_sets else set()
    duplicate_primary_memberships = len(all_primary_ids) != sum(
        len(value) for value in split_sets.values()
    )
    leakage = split_leakage_report(result)
    nesting = check_nesting(result)
    return {
        "status": (
            "pass"
            if set(example_ids) == assigned_ids
            and not duplicate_primary_memberships
            and leakage["status"] == "pass"
            and all(nesting.values())
            else "fail"
        ),
        "all_records_assigned": set(example_ids) == assigned_ids,
        "duplicate_primary_memberships": duplicate_primary_memberships,
        "leakage": leakage,
        "nesting": nesting,
    }
