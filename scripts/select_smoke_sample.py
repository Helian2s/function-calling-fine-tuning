from __future__ import annotations
# ruff: noqa: E402

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.normalization import (
    NormalizationError,
    normalize_xlam_row,
)


RAW_PATH = Path("data/raw/xlam/xlam_function_calling_60k.json")
DOWNLOAD_METADATA_PATH = Path(
    "data/raw/xlam/.cache/huggingface/download/"
    "xlam_function_calling_60k.json.metadata"
)
HF_REPO_CACHE_DIR = (
    Path.home()
    / ".cache/huggingface/hub/"
    / "datasets--Salesforce--xlam-function-calling-60k"
)

OUTPUT_DIR = Path("data/smoke/raw")
MANIFEST_PATH = Path("data/manifests/smoke_v1_selection.json")
SUMMARY_PATH = Path("data/manifests/smoke_v1_summary.json")
SOURCE_MANIFEST_PATH = Path("data/manifests/xlam_source.json")

SEED = 42
GENERATOR_BOUNDARY = 33_659

SPLIT_QUOTAS: dict[str, dict[str, int]] = {
    "train": {
        "single_call": 72,
        "two_calls": 30,
        "three_or_more_calls": 18,
    },
    "validation": {
        "single_call": 24,
        "two_calls": 10,
        "three_or_more_calls": 6,
    },
    "test": {
        "single_call": 24,
        "two_calls": 10,
        "three_or_more_calls": 6,
    },
}

GENERATORS = ("deepseek", "mixtral")
RawDataset = Sequence[dict[str, Any]]
PRIMARY_STRATIFICATION = {
    "field": "len(answers)",
    "bucket_field": "call_bucket",
    "buckets": {
        "single_call": "len(answers) == 1",
        "two_calls": "len(answers) == 2",
        "three_or_more_calls": "len(answers) >= 3",
    },
}


def classify_normalization_rejection(
    error: NormalizationError,
) -> str:
    """Map normalization failures to stable A3 rejection categories."""
    message = str(error).lower()

    if "duplicate tool names" in message:
        return "normalization_duplicate_tool_names"

    if (
        "'callable'" in message
        and "unsupported" in message
        and "parameter type" in message
    ):
        return (
            "normalization_unsupported_callable_parameters"
        )

    if "invalid parameter type expression" in message:
        return (
            "normalization_invalid_parameter_type_expressions"
        )

    if "references unavailable tool" in message:
        return "normalization_unavailable_answer_tools"

    if (
        "unsupported parameter type" in message
        or "unsupported generic parameter type" in message
        or "unsupported type constant" in message
        or "unsupported type expression" in message
    ):
        return "normalization_unsupported_parameter_types"

    return "normalization_other_errors"


def decode_json_field(
    value: Any,
    *,
    field_name: str,
    row_id: Any,
) -> Any:
    """Decode a JSON string while accepting already-decoded values."""
    if isinstance(value, (list, dict)):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"Row {row_id}: {field_name} must be a JSON string, "
            f"list, or dict; received {type(value).__name__}"
        )

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Row {row_id}: {field_name} contains invalid JSON. "
            f"Preview: {value[:200]!r}"
        ) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def infer_dataset_revision() -> str | None:
    if DOWNLOAD_METADATA_PATH.is_file():
        metadata_lines = DOWNLOAD_METADATA_PATH.read_text(
            encoding="utf-8"
        ).splitlines()

        if metadata_lines:
            revision = metadata_lines[0].strip()
            if revision:
                return revision

    refs_main_path = HF_REPO_CACHE_DIR / "refs/main"

    if refs_main_path.is_file():
        revision = refs_main_path.read_text(
            encoding="utf-8"
        ).strip()
        if revision:
            return revision

    snapshots_dir = HF_REPO_CACHE_DIR / "snapshots"

    if snapshots_dir.is_dir():
        snapshot_names = sorted(
            path.name
            for path in snapshots_dir.iterdir()
            if path.is_dir()
        )

        if len(snapshot_names) == 1:
            return snapshot_names[0]

    return None


def build_source_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "repository_id": "Salesforce/xlam-function-calling-60k",
        "repository_type": "dataset",
        "filename": RAW_PATH.name,
        "revision": infer_dataset_revision(),
    }

    if RAW_PATH.is_file():
        stat = RAW_PATH.stat()
        metadata["local_path"] = str(RAW_PATH)
        metadata["size_bytes"] = stat.st_size
        metadata["sha256"] = sha256(RAW_PATH)
        metadata["resolved_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

    return metadata


def load_raw_dataset() -> list[dict[str, Any]]:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}. "
            "Run scripts/download_xlam.py first."
        )

    raw_dataset = json.loads(
        RAW_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(raw_dataset, list):
        raise ValueError(
            f"Expected {RAW_PATH} to contain a JSON array."
        )

    return raw_dataset


def generator_name(row_id: int) -> str:
    return "deepseek" if row_id < GENERATOR_BOUNDARY else "mixtral"


def call_count_bucket(call_count: int) -> str:
    if call_count == 1:
        return "single_call"

    if call_count == 2:
        return "two_calls"

    if call_count >= 3:
        return "three_or_more_calls"

    return "zero_calls"


def extract_parameter_types(tools: list[Any]) -> list[str]:
    parameter_types: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            continue

        for specification in parameters.values():
            if not isinstance(specification, dict):
                continue

            parameter_type = specification.get("type")
            if parameter_type is not None:
                parameter_types.add(str(parameter_type))

    return sorted(parameter_types)


def has_complex_parameters(parameter_types: list[str]) -> bool:
    complex_terms = {
        "array",
        "dict",
        "dictionary",
        "list",
        "object",
    }

    for parameter_type in parameter_types:
        normalized = parameter_type.lower()

        if any(term in normalized for term in complex_terms):
            return True

    return False


def create_fingerprint(
    query: str,
    tools: list[Any],
) -> str:
    """Create a stable fingerprint for exact query-and-tools duplication."""
    normalized = {
        "query": " ".join(query.lower().split()),
        "tools": tools,
    }

    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def parse_candidate(
    row: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    required_fields = {"id", "query", "tools", "answers"}
    missing_fields = required_fields - set(row)

    if missing_fields:
        raise ValueError(
            f"Row index {row_index}: missing fields "
            f"{sorted(missing_fields)}"
        )

    row_id = int(row["id"])
    query = row["query"]

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            f"Row {row_id}: query must be a non-empty string."
        )

    tools = decode_json_field(
        row["tools"],
        field_name="tools",
        row_id=row_id,
    )

    answers = decode_json_field(
        row["answers"],
        field_name="answers",
        row_id=row_id,
    )

    if not isinstance(tools, list):
        raise ValueError(f"Row {row_id}: tools is not a list.")

    if not isinstance(answers, list):
        raise ValueError(f"Row {row_id}: answers is not a list.")

    if not tools:
        raise ValueError(f"Row {row_id}: tools list is empty.")

    if not answers:
        raise ValueError(f"Row {row_id}: answers list is empty.")

    available_tool_names = {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool, dict) and tool.get("name") is not None
    }

    expected_tool_names = [
        str(answer.get("name"))
        for answer in answers
        if isinstance(answer, dict) and answer.get("name") is not None
    ]

    unknown_answer_tools = sorted(
        set(expected_tool_names) - available_tool_names
    )

    parameter_types = extract_parameter_types(tools)
    distinct_expected_tool_count = len(set(expected_tool_names))

    return {
        "row_index": row_index,
        "id": row_id,
        "generator": generator_name(row_id),
        "call_bucket": call_count_bucket(len(answers)),
        "available_tool_count": len(tools),
        "expected_call_count": len(answers),
        "distinct_expected_tool_count": distinct_expected_tool_count,
        "repeated_same_tool": (
            len(answers) > 1
            and distinct_expected_tool_count == 1
        ),
        "multiple_distinct_tools": distinct_expected_tool_count > 1,
        "parameter_types": parameter_types,
        "has_complex_parameters": has_complex_parameters(
            parameter_types
        ),
        "unknown_answer_tools": unknown_answer_tools,
        "fingerprint": create_fingerprint(query, tools),
        "raw_row": {
            "id": row["id"],
            "query": row["query"],
            "tools": row["tools"],
            "answers": row["answers"],
        },
    }


def load_source_metadata() -> dict[str, Any]:
    if SOURCE_MANIFEST_PATH.is_file():
        return json.loads(
            SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
        )

    metadata = build_source_metadata()
    SOURCE_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    SOURCE_MANIFEST_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata


def collect_candidates(
    dataset: RawDataset,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    seen_ids: set[int] = set()

    for row_index in range(len(dataset)):
        row = dataset[row_index]

        try:
            int(row["id"])
        except (KeyError, TypeError, ValueError):
            rejected["invalid_source_id"] += 1
            continue

        try:
            candidate = parse_candidate(row, row_index)
        except (TypeError, ValueError):
            rejected["invalid_record"] += 1
            continue

        if candidate["id"] in seen_ids:
            rejected["duplicate_id"] += 1
            continue

        if candidate["call_bucket"] == "zero_calls":
            rejected["zero_calls"] += 1
            continue

        try:
            normalize_xlam_row(
                candidate["raw_row"],
                split="selection",
            )
        except NormalizationError as error:
            rejected[
                classify_normalization_rejection(error)
            ] += 1
            continue

        seen_ids.add(candidate["id"])
        candidates.append(candidate)

    return candidates, rejected


def build_candidate_pools(
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rng = random.Random(SEED)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for generator in GENERATORS:
        for call_bucket in (
            "single_call",
            "two_calls",
            "three_or_more_calls",
        ):
            pool = [
                candidate
                for candidate in candidates
                if candidate["generator"] == generator
                and candidate["call_bucket"] == call_bucket
            ]

            rng.shuffle(pool)
            pools[(generator, call_bucket)] = pool

    return pools


def select_records(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(SEED)
    pools = build_candidate_pools(candidates)

    selected: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    used_ids: set[int] = set()
    used_fingerprints: set[str] = set()

    for split_name, split_quotas in SPLIT_QUOTAS.items():
        for call_bucket, total_quota in split_quotas.items():
            if total_quota % 2 != 0:
                raise ValueError(
                    f"Quota for {split_name}/{call_bucket} must "
                    "be even to divide equally by generator."
                )

            per_generator_quota = total_quota // 2

            for generator in GENERATORS:
                pool = pools[(generator, call_bucket)]
                chosen: list[dict[str, Any]] = []

                for candidate in pool:
                    if candidate["id"] in used_ids:
                        continue

                    if candidate["fingerprint"] in used_fingerprints:
                        continue

                    chosen.append(candidate)
                    used_ids.add(candidate["id"])
                    used_fingerprints.add(candidate["fingerprint"])

                    if len(chosen) == per_generator_quota:
                        break

                if len(chosen) != per_generator_quota:
                    raise RuntimeError(
                        f"Not enough unique examples for "
                        f"split={split_name}, "
                        f"generator={generator}, "
                        f"bucket={call_bucket}. "
                        f"Needed {per_generator_quota}, "
                        f"found {len(chosen)}."
                    )

                selected[split_name].extend(chosen)

    for split_name in selected:
        rng.shuffle(selected[split_name])

    return selected


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record["raw_row"],
                    ensure_ascii=False,
                    sort_keys=False,
                )
            )
            file.write("\n")


def summarize_selection(
    selected: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    total_records = [
        record
        for records in selected.values()
        for record in records
    ]

    split_sizes = {
        split_name: len(records)
        for split_name, records in selected.items()
    }

    generator_distribution = Counter(
        record["generator"] for record in total_records
    )

    call_bucket_distribution = Counter(
        record["call_bucket"] for record in total_records
    )

    available_tool_distribution = Counter(
        record["available_tool_count"]
        for record in total_records
    )

    expected_call_distribution = Counter(
        record["expected_call_count"]
        for record in total_records
    )

    parameter_type_distribution: Counter[str] = Counter()

    for record in total_records:
        parameter_type_distribution.update(
            record["parameter_types"]
        )

    return {
        "total_selected": len(total_records),
        "recommended_primary_stratification": (
            PRIMARY_STRATIFICATION
        ),
        "split_sizes": split_sizes,
        "generator_distribution": dict(
            sorted(generator_distribution.items())
        ),
        "call_bucket_distribution": dict(
            sorted(call_bucket_distribution.items())
        ),
        "expected_call_count_distribution": dict(
            sorted(expected_call_distribution.items())
        ),
        "available_tool_count_distribution": dict(
            sorted(available_tool_distribution.items())
        ),
        "examples_with_five_or_more_available_tools": sum(
            record["available_tool_count"] >= 5
            for record in total_records
        ),
        "examples_with_complex_parameters": sum(
            record["has_complex_parameters"]
            for record in total_records
        ),
        "multi_call_examples_using_multiple_tools": sum(
            record["multiple_distinct_tools"]
            for record in total_records
        ),
        "multi_call_examples_repeating_same_tool": sum(
            record["repeated_same_tool"]
            for record in total_records
        ),
        "parameter_types": dict(
            parameter_type_distribution.most_common()
        ),
        "unknown_answer_tool_examples": sum(
            bool(record["unknown_answer_tools"])
            for record in total_records
        ),
    }


def create_manifest(
    selected: dict[str, list[dict[str, Any]]],
    rejected: Counter[str],
) -> dict[str, Any]:
    source_metadata = load_source_metadata()

    records: list[dict[str, Any]] = []

    for split_name, split_records in selected.items():
        for record in split_records:
            records.append(
                {
                    "split": split_name,
                    "id": record["id"],
                    "row_index": record["row_index"],
                    "generator": record["generator"],
                    "call_bucket": record["call_bucket"],
                    "available_tool_count": (
                        record["available_tool_count"]
                    ),
                    "expected_call_count": (
                        record["expected_call_count"]
                    ),
                    "distinct_expected_tool_count": (
                        record["distinct_expected_tool_count"]
                    ),
                    "repeated_same_tool": (
                        record["repeated_same_tool"]
                    ),
                    "multiple_distinct_tools": (
                        record["multiple_distinct_tools"]
                    ),
                    "parameter_types": record["parameter_types"],
                    "has_complex_parameters": (
                        record["has_complex_parameters"]
                    ),
                }
            )

    return {
        "manifest_version": "smoke-v1",
        "dataset_repository_id": source_metadata.get(
            "repository_id",
            "Salesforce/xlam-function-calling-60k",
        ),
        "dataset_revision": source_metadata.get("revision"),
        "selection_seed": SEED,
        "generator_boundary": GENERATOR_BOUNDARY,
        "recommended_primary_stratification": (
            PRIMARY_STRATIFICATION
        ),
        "split_quotas": SPLIT_QUOTAS,
        "rejected_candidate_counts": dict(rejected),
        "records": records,
    }


def main() -> None:
    dataset = load_raw_dataset()

    candidates, rejected = collect_candidates(dataset)
    selected = select_records(candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, records in selected.items():
        write_jsonl(
            OUTPUT_DIR / f"{split_name}.jsonl",
            records,
        )

    summary = summarize_selection(selected)
    manifest = create_manifest(selected, rejected)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Dataset rows scanned: {len(dataset)}")
    print(f"Valid candidates: {len(candidates)}")
    print(f"Rejected candidates: {sum(rejected.values())}")
    print()
    print(f"Train examples: {len(selected['train'])}")
    print(f"Validation examples: {len(selected['validation'])}")
    print(f"Test examples: {len(selected['test'])}")
    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Local selected data: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
