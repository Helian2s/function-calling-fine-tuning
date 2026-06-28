from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from unicodedata import normalize as unicode_normalize


CURATION_SCHEMA_VERSION = "1.0"
DEFAULT_INPUT_PATH = Path("data/processed/xlam_full_v1/normalized.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/xlam_curated_v1")

COMMON_ACTION_PREFIXES = {
    "add",
    "assess",
    "build",
    "calculate",
    "check",
    "compute",
    "convert",
    "create",
    "delete",
    "detect",
    "fetch",
    "find",
    "generate",
    "get",
    "has",
    "is",
    "list",
    "make",
    "place",
    "predict",
    "retrieve",
    "search",
    "send",
    "set",
    "update",
    "validate",
}

CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "finance",
        (
            "stock",
            "currency",
            "crypto",
            "exchange",
            "market",
            "price",
            "ticker",
            "forex",
            "bitcoin",
            "ethereum",
            "rate",
        ),
    ),
    (
        "weather",
        ("weather", "forecast", "temperature", "humidity"),
    ),
    (
        "travel",
        (
            "hotel",
            "flight",
            "airport",
            "travel",
            "booking",
            "reservation",
            "restaurant",
        ),
    ),
    (
        "commerce",
        ("order", "cart", "product", "store", "shop", "purchase"),
    ),
    (
        "health",
        ("health", "medical", "diabetes", "calorie", "symptom"),
    ),
    (
        "location",
        ("country", "city", "region", "address", "latitude", "longitude"),
    ),
    (
        "date_time",
        ("date", "time", "timezone", "calendar", "schedule"),
    ),
    (
        "math",
        (
            "calculate",
            "sum",
            "average",
            "distance",
            "equation",
            "number",
            "prime",
        ),
    ),
    (
        "text",
        (
            "text",
            "string",
            "word",
            "sentence",
            "summary",
            "translate",
            "language",
        ),
    ),
    (
        "media",
        ("movie", "music", "video", "image", "pokemon", "game"),
    ),
    (
        "search",
        ("search", "lookup", "find"),
    ),
)


@dataclass(frozen=True)
class CurationOutputs:
    group_metadata_path: Path
    deduplicated_path: Path
    duplicate_map_path: Path
    curator_input_path: Path
    fuzzy_candidates_path: Path
    fuzzy_review_sample_path: Path
    curation_report_path: Path
    leakage_audit_report_path: Path
    checksums_path: Path


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


def short_hash(value: str, *, length: int = 16) -> str:
    return sha256_text(value)[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object."
                )
            yield value


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(canonical_json_line(record))


def file_report(path: Path) -> dict[str, Any]:
    line_count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "line_count": line_count,
        "sha256": sha256_file(path),
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


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit_result = run_git(["rev-parse", "HEAD"])
    diff_result = run_git(["diff", "--quiet"])
    cached_diff_result = run_git(["diff", "--cached", "--quiet"])
    untracked_result = run_git(
        ["ls-files", "--others", "--exclude-standard"]
    )
    return {
        "commit": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else None
        ),
        "dirty": (
            diff_result.returncode != 0
            or cached_diff_result.returncode != 0
            or (
                untracked_result.returncode == 0
                and bool(untracked_result.stdout.strip())
            )
        ),
    }


def source_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except (TypeError, ValueError):
        return 2**63 - 1, str(value)


def example_id(record: dict[str, Any]) -> str:
    value = record.get("example_id") or record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Normalized record does not define an example id.")
    return value


def source_id(record: dict[str, Any]) -> int | None:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    value = metadata.get("source_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalized_user_request(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Normalized record has no messages.")

    first = messages[0]
    if not isinstance(first, dict):
        raise ValueError("First normalized message must be an object.")

    content = first.get("content")
    if not isinstance(content, str):
        raise ValueError("First normalized message must contain text.")

    normalized = unicode_normalize("NFKC", content)
    normalized = normalized.casefold()
    return " ".join(normalized.split())


def request_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def request_shingles(
    text: str,
    *,
    size: int = 5,
) -> frozenset[str]:
    tokens = request_tokens(text)
    if not tokens:
        return frozenset()

    if len(tokens) <= size:
        return frozenset({" ".join(tokens)})

    return frozenset(
        " ".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def tools(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("tools")
    if not isinstance(value, list):
        raise ValueError("Normalized record tools must be a list.")
    return [tool for tool in value if isinstance(tool, dict)]


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Normalized record has no assistant message.")
    assistant = messages[1]
    if not isinstance(assistant, dict):
        raise ValueError("Assistant message must be an object.")
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("Assistant tool_calls must be a list.")
    return [call for call in calls if isinstance(call, dict)]


def function_name_from_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError("Tool has no function object.")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool has no function name.")
    return name.strip()


def function_from_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        raise ValueError("Tool call has no function object.")
    return function


def function_name_from_call(call: dict[str, Any]) -> str:
    name = function_from_call(call).get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool call has no function name.")
    return name.strip()


def normalized_name_tokens(name: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold())
    return tuple(token for token in normalized.split("_") if token)


def tool_family(name: str) -> str:
    if "." in name:
        return normalized_name_tokens(name.split(".", maxsplit=1)[0])[0]

    tokens = normalized_name_tokens(name)
    if not tokens:
        return "unknown"

    if len(tokens) >= 2 and tokens[0] in COMMON_ACTION_PREFIXES:
        return tokens[1]

    return tokens[0]


def category_for_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict):
        return "unknown"

    payload = canonical_json_dumps(
        {
            "name": function.get("name"),
            "description": function.get("description"),
            "parameters": function.get("parameters"),
        }
    ).casefold()

    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in payload for keyword in keywords):
            return category

    return "unknown"


def canonical_tool_for_signature(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "name": function.get("name"),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
    }


def tool_signature_hash(tool: dict[str, Any]) -> str:
    return sha256_text(canonical_json_dumps(canonical_tool_for_signature(tool)))


def schema_shape(value: Any, *, required: bool = False) -> Any:
    if not isinstance(value, dict):
        return {"schema": "unknown", "required": required}

    if "anyOf" in value and isinstance(value["anyOf"], list):
        children = sorted(
            (
                canonical_json_dumps(schema_shape(child))
                for child in value["anyOf"]
            )
        )
        return {
            "schema": "anyOf",
            "required": required,
            "children": children,
        }

    schema_type = value.get("type", "unspecified")
    result: dict[str, Any] = {
        "schema": schema_type,
        "required": required,
    }

    if schema_type == "object":
        properties = value.get("properties", {})
        required_names = set(value.get("required", []))
        if isinstance(properties, dict):
            result["children"] = sorted(
                canonical_json_dumps(
                    schema_shape(
                        child,
                        required=str(name) in required_names,
                    )
                )
                for name, child in properties.items()
            )
        additional = value.get("additionalProperties")
        if isinstance(additional, dict):
            result["additional"] = schema_shape(additional)
        elif isinstance(additional, bool):
            result["additional"] = additional

    if schema_type == "array":
        items = value.get("items")
        prefix_items = value.get("prefixItems")
        if isinstance(items, dict):
            result["items"] = schema_shape(items)
        if isinstance(prefix_items, list):
            result["prefix_items"] = [
                schema_shape(item) for item in prefix_items
            ]
        for key in ("minItems", "maxItems", "uniqueItems"):
            if key in value:
                result[key] = value[key]

    return result


def tool_schema_shape_hash(tool: dict[str, Any]) -> str:
    function = tool["function"]
    return sha256_text(
        canonical_json_dumps(schema_shape(function.get("parameters", {})))
    )


def tool_set_signature_hash(record_tools: list[dict[str, Any]]) -> str:
    signatures = sorted(tool_signature_hash(tool) for tool in record_tools)
    return sha256_text(canonical_json_dumps(signatures))


def schema_shape_fingerprint(record_tools: list[dict[str, Any]]) -> str:
    shapes = sorted(tool_schema_shape_hash(tool) for tool in record_tools)
    return sha256_text(canonical_json_dumps(shapes))


def call_category(calls: list[dict[str, Any]]) -> str:
    if len(calls) == 1:
        return "single"

    counts = Counter(function_name_from_call(call) for call in calls)
    if len(counts) == 1:
        return "multiple"

    if all(count == 1 for count in counts.values()):
        return "parallel"

    return "multiple_parallel"


def exact_dedup_payload(record: dict[str, Any]) -> dict[str, Any]:
    record_tools = tools(record)
    return {
        "user_request": normalized_user_request(record),
        "tools": sorted(
            (
                canonical_tool_for_signature(tool)
                for tool in record_tools
            ),
            key=lambda item: str(item.get("name", "")),
        ),
        "expected_calls": [
            {
                "name": function_name_from_call(call),
                "arguments": function_from_call(call).get(
                    "arguments",
                    {},
                ),
            }
            for call in expected_calls(record)
        ],
    }


def exact_duplicate_hash(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json_dumps(exact_dedup_payload(record)))


def split_group_key(
    *,
    families: list[str],
    categories: list[str],
    tool_set_hash: str,
    schema_shape_hash: str,
) -> dict[str, Any]:
    return {
        "strategy": "tool_set_schema_family_v1",
        "tool_families": sorted(set(families)),
        "api_categories": sorted(set(categories)),
        "tool_set_signature_hash": tool_set_hash,
        "schema_shape_fingerprint": schema_shape_hash,
    }


def compute_curation_metadata(record: dict[str, Any]) -> dict[str, Any]:
    record_tools = tools(record)
    calls = expected_calls(record)
    request = normalized_user_request(record)

    tool_entries: list[dict[str, Any]] = []
    families: list[str] = []
    categories: list[str] = []

    for tool in record_tools:
        name = function_name_from_tool(tool)
        family = tool_family(name)
        category = category_for_tool(tool)
        families.append(family)
        categories.append(category)
        tool_entries.append(
            {
                "name": name,
                "family": family,
                "api_category": category,
                "signature_hash": tool_signature_hash(tool),
                "schema_shape_hash": tool_schema_shape_hash(tool),
            }
        )

    tool_set_hash = tool_set_signature_hash(record_tools)
    schema_hash = schema_shape_fingerprint(record_tools)
    group_key = split_group_key(
        families=families,
        categories=categories,
        tool_set_hash=tool_set_hash,
        schema_shape_hash=schema_hash,
    )
    group_key_json = canonical_json_dumps(group_key)

    metadata = record.get("metadata", {})
    split = metadata.get("split") if isinstance(metadata, dict) else None

    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "example_id": example_id(record),
        "source_id": source_id(record),
        "source_row_index": (
            metadata.get("source_row_index")
            if isinstance(metadata, dict)
            else None
        ),
        "split": split,
        "tool_count": len(record_tools),
        "expected_call_count": len(calls),
        "call_category": call_category(calls),
        "tool_families": sorted(set(families)),
        "primary_tool_family": sorted(set(families))[0]
        if families
        else "unknown",
        "api_categories": sorted(set(categories)),
        "primary_api_category": sorted(set(categories))[0]
        if categories
        else "unknown",
        "tool_signatures": sorted(
            tool_entries,
            key=lambda entry: str(entry["name"]),
        ),
        "tool_set_signature_hash": tool_set_hash,
        "schema_shape_fingerprint": schema_hash,
        "normalized_user_request_fingerprint": sha256_text(request),
        "exact_duplicate_hash": exact_duplicate_hash(record),
        "split_group_key": group_key,
        "split_group_key_hash": sha256_text(group_key_json),
        "split_group_id": f"group-{short_hash(group_key_json)}",
    }


def retained_metadata_for_group(
    metadatas: list[dict[str, Any]],
) -> dict[str, Any]:
    return sorted(
        metadatas,
        key=lambda item: (
            source_sort_key(item.get("source_id")),
            str(item.get("example_id")),
        ),
    )[0]


def exact_duplicate_groups(
    metadatas: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metadata in metadatas:
        groups[str(metadata["exact_duplicate_hash"])].append(metadata)
    return dict(groups)


def duplicate_map_records(
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for duplicate_hash, group in exact_duplicate_groups(metadatas).items():
        retained = retained_metadata_for_group(group)
        sorted_group = sorted(
            group,
            key=lambda item: (
                source_sort_key(item.get("source_id")),
                str(item.get("example_id")),
            ),
        )
        all_source_ids = [item["source_id"] for item in sorted_group]
        duplicate_source_ids = [
            item["source_id"]
            for item in sorted_group
            if item["example_id"] != retained["example_id"]
        ]
        records.append(
            {
                "exact_duplicate_hash": duplicate_hash,
                "retained_example_id": retained["example_id"],
                "retained_source_id": retained["source_id"],
                "all_example_ids": [
                    item["example_id"] for item in sorted_group
                ],
                "all_source_ids": all_source_ids,
                "duplicate_example_ids": [
                    item["example_id"]
                    for item in sorted_group
                    if item["example_id"] != retained["example_id"]
                ],
                "duplicate_source_ids": duplicate_source_ids,
                "group_size": len(sorted_group),
            }
        )

    return sorted(
        records,
        key=lambda item: (
            source_sort_key(item.get("retained_source_id")),
            str(item.get("retained_example_id")),
        ),
    )


def deduplicated_records(
    records: list[dict[str, Any]],
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_id = {
        str(metadata["example_id"]): metadata
        for metadata in metadatas
    }
    record_by_id = {
        example_id(record): record
        for record in records
    }
    retained_ids = {
        str(item["retained_example_id"])
        for item in duplicate_map_records(metadatas)
    }

    output: list[dict[str, Any]] = []
    for retained_id in sorted(
        retained_ids,
        key=lambda value: (
            source_sort_key(metadata_by_id[value].get("source_id")),
            value,
        ),
    ):
        record = copy.deepcopy(record_by_id[retained_id])
        record["curation_metadata"] = metadata_by_id[retained_id]
        output.append(record)

    return output


def curator_input_records(
    records: list[dict[str, Any]],
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_id = {
        str(metadata["example_id"]): metadata
        for metadata in metadatas
    }
    output: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            source_sort_key(source_id(item)),
            example_id(item),
        ),
    ):
        payload = canonical_json_dumps(exact_dedup_payload(record))
        metadata = metadata_by_id[example_id(record)]
        output.append(
            {
                "id": example_id(record),
                "text": payload,
                "exact_duplicate_hash": metadata["exact_duplicate_hash"],
            }
        )
    return output


def minhash_signature(
    shingles: frozenset[str],
    *,
    num_hashes: int,
) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_hashes))

    signature: list[int] = []
    for seed in range(num_hashes):
        minimum = min(
            int(
                hashlib.sha256(
                    f"{seed}:{shingle}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            for shingle in shingles
        )
        signature.append(minimum)
    return tuple(signature)


def fuzzy_candidate_records(
    metadatas: list[dict[str, Any]],
    request_shingles_by_id: dict[str, frozenset[str]],
    *,
    threshold: float = 0.82,
    num_hashes: int = 48,
    bands: int = 12,
    max_bucket_size: int = 50,
    max_candidates: int = 20_000,
) -> list[dict[str, Any]]:
    if num_hashes % bands != 0:
        raise ValueError("num_hashes must be divisible by bands")

    rows = sorted(
        metadatas,
        key=lambda item: (
            source_sort_key(item.get("source_id")),
            str(item.get("example_id")),
        ),
    )
    id_to_metadata = {str(item["example_id"]): item for item in rows}
    band_size = num_hashes // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for metadata in rows:
        current_id = str(metadata["example_id"])
        signature = minhash_signature(
            request_shingles_by_id[current_id],
            num_hashes=num_hashes,
        )
        for band in range(bands):
            start = band * band_size
            key = (band, signature[start : start + band_size])
            buckets[key].append(current_id)

    pairs: set[tuple[str, str]] = set()
    for bucket_ids in buckets.values():
        if len(bucket_ids) < 2 or len(bucket_ids) > max_bucket_size:
            continue
        for left, right in itertools.combinations(sorted(bucket_ids), 2):
            pairs.add((left, right))
            if len(pairs) >= max_candidates:
                break
        if len(pairs) >= max_candidates:
            break

    candidates: list[dict[str, Any]] = []
    for left, right in sorted(pairs):
        score = jaccard(
            request_shingles_by_id[left],
            request_shingles_by_id[right],
        )
        if score < threshold:
            continue
        left_metadata = id_to_metadata[left]
        right_metadata = id_to_metadata[right]
        candidates.append(
            {
                "left_example_id": left,
                "right_example_id": right,
                "left_source_id": left_metadata["source_id"],
                "right_source_id": right_metadata["source_id"],
                "jaccard": round(score, 6),
                "same_split_group": (
                    left_metadata["split_group_id"]
                    == right_metadata["split_group_id"]
                ),
                "left_split_group_id": left_metadata["split_group_id"],
                "right_split_group_id": right_metadata["split_group_id"],
                "disposition": "review_only",
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            -float(item["jaccard"]),
            source_sort_key(item["left_source_id"]),
            source_sort_key(item["right_source_id"]),
        ),
    )


def audit_group_leakage(
    metadatas: Iterable[dict[str, Any]],
    *,
    ignored_splits: set[str] | None = None,
) -> dict[str, Any]:
    ignored = ignored_splits or {"full", "unknown", ""}
    groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"splits": set(), "examples": set()}
    )

    for metadata in metadatas:
        split = str(metadata.get("split") or "unknown")
        if split in ignored:
            continue
        group_id = str(metadata["split_group_id"])
        groups[group_id]["splits"].add(split)
        groups[group_id]["examples"].add(str(metadata["example_id"]))

    violations: list[dict[str, Any]] = []
    for group_id, data in groups.items():
        if len(data["splits"]) <= 1:
            continue
        violations.append(
            {
                "split_group_id": group_id,
                "splits": sorted(data["splits"]),
                "example_ids": sorted(data["examples"]),
            }
        )

    return {
        "audit_schema_version": CURATION_SCHEMA_VERSION,
        "status": "fail" if violations else "pass",
        "cross_split_group_count": len(violations),
        "violations": sorted(
            violations,
            key=lambda item: str(item["split_group_id"]),
        ),
    }


def curator_report(output_dir: Path, curator_image: str) -> dict[str, Any]:
    comparison_path = (
        output_dir / "manifests" / "curator_comparison_report.json"
    )
    ec2_report_path = output_dir / "manifests" / "curator_ec2_report.json"
    s3_uri_path = output_dir / "manifests" / "curator_ec2_s3_uri.txt"
    comparison = read_optional_json(comparison_path)
    ec2_report = read_optional_json(ec2_report_path)

    report: dict[str, Any] = {
        "status": "pending_ec2_gpu_execution",
        "docker_image": curator_image,
        "comparison_required": True,
    }
    if comparison is not None:
        report["comparison_report"] = {
            "path": str(comparison_path),
            "status": comparison.get("status"),
            "sha256": sha256_file(comparison_path),
        }
    if ec2_report is not None:
        report["ec2_report"] = {
            "path": str(ec2_report_path),
            "status": ec2_report.get("status"),
            "run_id": ec2_report.get("run_id"),
            "curator_version": ec2_report.get("curator_version"),
            "curator_image_digest": ec2_report.get(
                "curator_image_digest"
            ),
            "sha256": sha256_file(ec2_report_path),
        }
    if s3_uri_path.is_file():
        report["s3_uri"] = s3_uri_path.read_text(
            encoding="utf-8",
        ).strip()

    if (
        comparison is not None
        and comparison.get("status") == "pass"
        and ec2_report is not None
        and ec2_report.get("status") == "pass"
    ):
        report["status"] = "pass"
        report["comparison_required"] = False

    return report


def build_curation_report(
    *,
    input_path: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    metadatas: list[dict[str, Any]],
    duplicate_maps: list[dict[str, Any]],
    fuzzy_candidates: list[dict[str, Any]],
    leakage_report: dict[str, Any],
    repo_root: Path,
    curator_image: str,
) -> dict[str, Any]:
    duplicate_group_count = sum(
        1 for item in duplicate_maps if int(item["group_size"]) > 1
    )
    duplicate_record_count = sum(
        max(0, int(item["group_size"]) - 1) for item in duplicate_maps
    )
    split_group_distribution = Counter(
        str(metadata["split_group_id"]) for metadata in metadatas
    )
    call_category_distribution = Counter(
        str(metadata["call_category"]) for metadata in metadatas
    )
    category_distribution = Counter(
        str(metadata["primary_api_category"]) for metadata in metadatas
    )

    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "input": {
            "normalized_path": str(input_path),
            "records": len(records),
            "sha256": sha256_file(input_path),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "deduplicated": str(output_dir / "deduplicated.jsonl"),
            "duplicate_map": str(output_dir / "duplicate_map.jsonl"),
            "group_metadata": str(output_dir / "group_metadata.jsonl"),
            "fuzzy_candidates": str(output_dir / "fuzzy_candidates.jsonl"),
            "fuzzy_review_sample": str(
                output_dir / "fuzzy_review_sample.jsonl"
            ),
            "leakage_audit_report": str(
                output_dir / "manifests" / "leakage_audit_report.json"
            ),
            "curator_input": str(
                output_dir / "curator_input" / "exact_dedup_input.jsonl"
            ),
        },
        "exact_deduplication": {
            "method": "independent_sha256_canonical_payload",
            "input_records": len(records),
            "retained_records": len(duplicate_maps),
            "duplicate_groups": duplicate_group_count,
            "duplicate_records": duplicate_record_count,
            "no_silent_removal": (
                len(records)
                == len(duplicate_maps) + duplicate_record_count
            ),
        },
        "grouping": {
            "strategy": "tool_set_schema_family_v1",
            "split_group_count": len(split_group_distribution),
            "largest_split_group_size": max(
                split_group_distribution.values(),
                default=0,
            ),
            "tradeoff": (
                "The split group key includes exact presented tool-set "
                "signatures and schema-shape fingerprints. This is "
                "conservative for leakage prevention but may reduce later "
                "split balance when common tool sets dominate."
            ),
        },
        "fuzzy_candidates": {
            "method": "request_text_minhash_lsh_jaccard",
            "candidate_pairs": len(fuzzy_candidates),
            "disposition": "review_only",
        },
        "leakage_audit": leakage_report,
        "distributions": {
            "call_category": dict(sorted(call_category_distribution.items())),
            "primary_api_category": dict(
                sorted(category_distribution.items())
            ),
        },
        "curator": curator_report(output_dir, curator_image),
        "git": git_metadata(repo_root),
    }


def build_curation_outputs(output_dir: Path) -> CurationOutputs:
    return CurationOutputs(
        group_metadata_path=output_dir / "group_metadata.jsonl",
        deduplicated_path=output_dir / "deduplicated.jsonl",
        duplicate_map_path=output_dir / "duplicate_map.jsonl",
        curator_input_path=(
            output_dir / "curator_input" / "exact_dedup_input.jsonl"
        ),
        fuzzy_candidates_path=output_dir / "fuzzy_candidates.jsonl",
        fuzzy_review_sample_path=output_dir / "fuzzy_review_sample.jsonl",
        curation_report_path=(
            output_dir / "manifests" / "curation_report.json"
        ),
        leakage_audit_report_path=(
            output_dir / "manifests" / "leakage_audit_report.json"
        ),
        checksums_path=output_dir / "checksums.sha256",
    )


def curate_normalized_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    repo_root: Path,
    fuzzy_threshold: float = 0.82,
    fuzzy_review_sample_size: int = 100,
    curator_image: str = "nvcr.io/nvidia/nemo-curator:25.09",
) -> dict[str, Any]:
    records = list(read_jsonl(input_path))
    metadatas = [
        compute_curation_metadata(record)
        for record in records
    ]
    stable_metadatas = sorted(
        metadatas,
        key=lambda item: (
            source_sort_key(item.get("source_id")),
            str(item.get("example_id")),
        ),
    )
    duplicate_maps = duplicate_map_records(metadatas)
    deduped_records = deduplicated_records(records, metadatas)
    request_shingles_by_id = {
        example_id(record): request_shingles(normalized_user_request(record))
        for record in records
    }
    fuzzy_candidates = fuzzy_candidate_records(
        stable_metadatas,
        request_shingles_by_id,
        threshold=fuzzy_threshold,
    )
    leakage_report = audit_group_leakage(stable_metadatas)

    outputs = build_curation_outputs(output_dir)
    write_jsonl(outputs.group_metadata_path, stable_metadatas)
    write_jsonl(outputs.deduplicated_path, deduped_records)
    write_jsonl(outputs.duplicate_map_path, duplicate_maps)
    write_jsonl(
        outputs.curator_input_path,
        curator_input_records(records, metadatas),
    )
    write_jsonl(outputs.fuzzy_candidates_path, fuzzy_candidates)
    write_jsonl(
        outputs.fuzzy_review_sample_path,
        fuzzy_candidates[:fuzzy_review_sample_size],
    )
    write_json(outputs.leakage_audit_report_path, leakage_report)

    report = build_curation_report(
        input_path=input_path,
        output_dir=output_dir,
        records=records,
        metadatas=stable_metadatas,
        duplicate_maps=duplicate_maps,
        fuzzy_candidates=fuzzy_candidates,
        leakage_report=leakage_report,
        repo_root=repo_root,
        curator_image=curator_image,
    )
    write_json(outputs.curation_report_path, report)
    write_checksums(output_dir)
    return report


def verify_stable_under_shuffle(
    *,
    input_path: Path,
    output_dir: Path,
    repo_root: Path,
    curator_image: str = "nvcr.io/nvidia/nemo-curator:25.09",
) -> dict[str, Any]:
    records = list(read_jsonl(input_path))
    shuffled = sorted(
        records,
        key=lambda record: sha256_text(example_id(record)),
        reverse=True,
    )
    shuffled_path = output_dir / "shuffled_input.jsonl"
    write_jsonl(shuffled_path, shuffled)
    return curate_normalized_dataset(
        input_path=shuffled_path,
        output_dir=output_dir / "shuffled_output",
        repo_root=repo_root,
        curator_image=curator_image,
    )


def compare_curation_hashes(
    left_output_dir: Path,
    right_output_dir: Path,
) -> dict[str, Any]:
    relative_paths = [
        "deduplicated.jsonl",
        "duplicate_map.jsonl",
        "group_metadata.jsonl",
        "curator_input/exact_dedup_input.jsonl",
        "fuzzy_candidates.jsonl",
    ]
    comparisons = []
    for relative_path in relative_paths:
        left = left_output_dir / relative_path
        right = right_output_dir / relative_path
        comparisons.append(
            {
                "path": relative_path,
                "left_sha256": sha256_file(left),
                "right_sha256": sha256_file(right),
                "match": sha256_file(left) == sha256_file(right),
            }
        )

    return {
        "stable": all(item["match"] for item in comparisons),
        "comparisons": comparisons,
    }
