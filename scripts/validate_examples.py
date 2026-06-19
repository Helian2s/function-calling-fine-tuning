from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import select_smoke_sample as smoke_selection
from function_calling_ft.validation import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    SPLIT_PRIORITY,
    ExampleValidationResult,
    ValidationIssue,
    validate_raw_example,
)


SPLITS = ("train", "validation", "test")
REPORT_PATH = Path("data/manifests/smoke_v1_validation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the selected 200-example smoke dataset "
            "and deterministically replace invalid examples."
        )
    )
    parser.add_argument(
        "--context-token-limit",
        type=int,
        default=DEFAULT_CONTEXT_TOKEN_LIMIT,
        help=(
            "Reject examples whose estimated rendered sequence "
            "length exceeds this token limit."
        ),
    )
    return parser.parse_args()


def load_selected_from_raw_splits(
    candidate_by_id: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]] | None:
    selected: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }

    for split in SPLITS:
        path = smoke_selection.OUTPUT_DIR / f"{split}.jsonl"
        if not path.is_file():
            return None

        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)
                row_id = int(row["id"])

                if row_id not in candidate_by_id:
                    raise KeyError(
                        f"Selected row id {row_id} from {path} was not found "
                        "in the current candidate pool."
                    )

                selected[split].append(candidate_by_id[row_id])

    return selected


def load_selected_from_manifest(
    candidate_by_id: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]] | None:
    if not smoke_selection.MANIFEST_PATH.is_file():
        return None

    manifest = json.loads(
        smoke_selection.MANIFEST_PATH.read_text(encoding="utf-8")
    )

    selected: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }

    for record in manifest.get("records", []):
        row_id = int(record["id"])
        split = str(record["split"])

        if split not in selected:
            raise ValueError(
                f"Unexpected split {split!r} in selection manifest."
            )

        if row_id not in candidate_by_id:
            raise KeyError(
                f"Selected row id {row_id} from the selection manifest "
                "was not found in the current candidate pool."
            )

        selected[split].append(candidate_by_id[row_id])

    return selected


def load_selected_records(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidate_by_id = {
        int(candidate["id"]): candidate
        for candidate in candidates
    }

    selected = load_selected_from_raw_splits(candidate_by_id)
    if selected is not None:
        return selected

    selected = load_selected_from_manifest(candidate_by_id)
    if selected is not None:
        return selected

    return smoke_selection.select_records(candidates)


def make_duplicate_cross_split_issue(
    candidate: dict[str, Any],
    prior_split: str,
    current_split: str,
) -> ValidationIssue:
    return ValidationIssue(
        category="duplicate_cross_split",
        message=(
            f"Example fingerprint for source id {candidate['id']} appears in "
            f"both {prior_split} and {current_split}."
        ),
    )


def evaluate_selected_records(
    selected: dict[str, list[dict[str, Any]]],
    *,
    context_token_limit: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    fingerprint_owner: dict[str, str] = {}

    for split in SPLIT_PRIORITY:
        for index, candidate in enumerate(selected[split]):
            result = validate_raw_example(
                candidate["raw_row"],
                split=split,
                context_token_limit=context_token_limit,
            )
            issues = list(result.issues)

            fingerprint = str(candidate["fingerprint"])
            prior_split = fingerprint_owner.get(fingerprint)

            if prior_split is None:
                fingerprint_owner[fingerprint] = split
            elif prior_split != split:
                issues.append(
                    make_duplicate_cross_split_issue(
                        candidate,
                        prior_split=prior_split,
                        current_split=split,
                    )
                )

            entries.append(
                {
                    "split": split,
                    "index": index,
                    "candidate": candidate,
                    "estimated_tokens": result.estimated_tokens,
                    "issues": issues,
                }
            )

    return entries


def replacement_report_entry(
    *,
    split: str,
    rejected_entry: dict[str, Any],
    replacement: dict[str, Any],
    result: ExampleValidationResult,
) -> dict[str, Any]:
    return {
        "split": split,
        "generator": replacement["generator"],
        "call_bucket": replacement["call_bucket"],
        "rejected_source_id": rejected_entry["candidate"]["id"],
        "replacement_source_id": replacement["id"],
        "rejected_categories": sorted(
            {
                issue.category
                for issue in rejected_entry["issues"]
            }
        ),
        "rejected_messages": [
            issue.message for issue in rejected_entry["issues"]
        ],
        "replacement_estimated_tokens": result.estimated_tokens,
    }


def find_replacement(
    *,
    split: str,
    rejected_entry: dict[str, Any],
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    used_ids: set[int],
    used_fingerprints: set[str],
    blocked_ids: set[int],
    blocked_fingerprints: set[str],
    replacement_candidate_rejections: Counter[str],
    context_token_limit: int,
) -> tuple[dict[str, Any], ExampleValidationResult]:
    rejected_candidate = rejected_entry["candidate"]
    pool_key = (
        str(rejected_candidate["generator"]),
        str(rejected_candidate["call_bucket"]),
    )
    pool = pools[pool_key]

    for candidate in pool:
        candidate_id = int(candidate["id"])
        fingerprint = str(candidate["fingerprint"])

        if candidate_id in used_ids or candidate_id in blocked_ids:
            continue

        if (
            fingerprint in used_fingerprints
            or fingerprint in blocked_fingerprints
        ):
            continue

        result = validate_raw_example(
            candidate["raw_row"],
            split=split,
            context_token_limit=context_token_limit,
        )

        if not result.is_valid:
            blocked_ids.add(candidate_id)
            blocked_fingerprints.add(fingerprint)
            replacement_candidate_rejections.update(
                {
                    issue.category
                    for issue in result.issues
                }
            )
            continue

        return candidate, result

    raise RuntimeError(
        "Unable to find a deterministic replacement for "
        f"split={split}, generator={pool_key[0]}, "
        f"call_bucket={pool_key[1]}."
    )


def write_selection_artifacts(
    selected: dict[str, list[dict[str, Any]]],
    rejected_candidate_counts: Counter[str],
) -> None:
    smoke_selection.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, records in selected.items():
        smoke_selection.write_jsonl(
            smoke_selection.OUTPUT_DIR / f"{split}.jsonl",
            records,
        )

    summary = smoke_selection.summarize_selection(selected)
    manifest = smoke_selection.create_manifest(
        selected,
        rejected_candidate_counts,
    )

    smoke_selection.MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    smoke_selection.MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    smoke_selection.SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_validation_report(
    *,
    examples_selected: int,
    valid_examples: int,
    rejected_entries: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    final_selected: dict[str, list[dict[str, Any]]],
    selected_rejections: Counter[str],
    replacement_candidate_rejections: Counter[str],
    context_token_limit: int,
) -> dict[str, Any]:
    return {
        "validation_schema_version": "1.0",
        "context_token_limit": context_token_limit,
        "examples_selected": examples_selected,
        "valid_examples": valid_examples,
        "rejected_examples": len(rejected_entries),
        "replacement_examples_added": len(replacements),
        "final_total_examples": sum(
            len(records) for records in final_selected.values()
        ),
        "selected_rejection_category_counts": dict(
            selected_rejections
        ),
        "replacement_candidate_rejection_counts": dict(
            replacement_candidate_rejections
        ),
        "rejected_records": [
            {
                "split": entry["split"],
                "source_id": entry["candidate"]["id"],
                "generator": entry["candidate"]["generator"],
                "call_bucket": entry["candidate"]["call_bucket"],
                "categories": sorted(
                    {issue.category for issue in entry["issues"]}
                ),
                "messages": [
                    issue.message for issue in entry["issues"]
                ],
                "estimated_tokens": entry["estimated_tokens"],
            }
            for entry in rejected_entries
        ],
        "replacements": replacements,
        "final_split_sizes": {
            split: len(records)
            for split, records in final_selected.items()
        },
    }


def main() -> None:
    args = parse_args()
    dataset = smoke_selection.load_raw_dataset()
    candidates, rejected_candidate_counts = (
        smoke_selection.collect_candidates(dataset)
    )
    pools = smoke_selection.build_candidate_pools(candidates)
    selected = load_selected_records(candidates)

    selected_entries = evaluate_selected_records(
        selected,
        context_token_limit=args.context_token_limit,
    )
    rejected_entries = [
        entry for entry in selected_entries if entry["issues"]
    ]
    selected_rejections = Counter(
        category
        for entry in rejected_entries
        for category in {
            issue.category for issue in entry["issues"]
        }
    )

    used_ids = {
        int(entry["candidate"]["id"]) for entry in selected_entries
    }
    used_fingerprints = {
        str(entry["candidate"]["fingerprint"])
        for entry in selected_entries
    }
    blocked_ids: set[int] = set()
    blocked_fingerprints: set[str] = set()
    replacement_candidate_rejections: Counter[str] = Counter()
    replacements: list[dict[str, Any]] = []

    final_selected = {
        split: list(records) for split, records in selected.items()
    }

    for entry in rejected_entries:
        blocked_ids.add(int(entry["candidate"]["id"]))
        blocked_fingerprints.add(
            str(entry["candidate"]["fingerprint"])
        )

    for entry in rejected_entries:
        split = str(entry["split"])
        replacement, result = find_replacement(
            split=split,
            rejected_entry=entry,
            pools=pools,
            used_ids=used_ids,
            used_fingerprints=used_fingerprints,
            blocked_ids=blocked_ids,
            blocked_fingerprints=blocked_fingerprints,
            replacement_candidate_rejections=(
                replacement_candidate_rejections
            ),
            context_token_limit=args.context_token_limit,
        )

        final_selected[split][int(entry["index"])] = replacement
        used_ids.add(int(replacement["id"]))
        used_fingerprints.add(str(replacement["fingerprint"]))
        replacements.append(
            replacement_report_entry(
                split=split,
                rejected_entry=entry,
                replacement=replacement,
                result=result,
            )
        )

    final_entries = evaluate_selected_records(
        final_selected,
        context_token_limit=args.context_token_limit,
    )
    remaining_invalid = [
        entry for entry in final_entries if entry["issues"]
    ]

    if remaining_invalid:
        raise RuntimeError(
            "Validation replacements did not produce a fully valid "
            "200-example smoke dataset."
        )

    write_selection_artifacts(
        final_selected,
        rejected_candidate_counts,
    )

    report = build_validation_report(
        examples_selected=len(selected_entries),
        valid_examples=len(selected_entries) - len(rejected_entries),
        rejected_entries=rejected_entries,
        replacements=replacements,
        final_selected=final_selected,
        selected_rejections=selected_rejections,
        replacement_candidate_rejections=(
            replacement_candidate_rejections
        ),
        context_token_limit=args.context_token_limit,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Examples selected:          {report['examples_selected']}")
    print(f"Valid:                      {report['valid_examples']}")
    print(f"Rejected:                   {report['rejected_examples']}")
    print(
        "Replacement examples added: "
        f"{report['replacement_examples_added']}"
    )
    print(f"Final total:                {report['final_total_examples']}")
    print(f"Validation report:          {REPORT_PATH}")
    print(f"Updated selection manifest: {smoke_selection.MANIFEST_PATH}")
    print(f"Updated selection summary:  {smoke_selection.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
