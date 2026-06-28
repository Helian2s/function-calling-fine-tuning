from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

from function_calling_ft.normalization import (
    SCHEMA_VERSION as NORMALIZED_RECORD_SCHEMA_VERSION,
)
from function_calling_ft.validation import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    ValidationIssue,
    validate_raw_example,
)


FULL_NORMALIZATION_SCHEMA_VERSION = "1.0"
QUARANTINE_SCHEMA_VERSION = "1.0"
SOURCE_DATASET_REPOSITORY_ID = "Salesforce/xlam-function-calling-60k"
SOURCE_DATASET_CONFIG = "default"
SOURCE_DATASET_SPLIT = "train"
DEFAULT_OUTPUT_SPLIT = "full"


@dataclass(frozen=True)
class FullNormalizationResult:
    accepted: dict[str, Any] | None
    quarantine: dict[str, Any] | None
    estimated_tokens: int | None


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_line(value: Any) -> str:
    return canonical_json_dumps(value) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


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


def stable_example_id(source_id: int | str) -> str:
    return f"xlam-{int(source_id)}"


def load_source_manifest(path: Path, raw_path: Path) -> dict[str, Any]:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    if not isinstance(manifest, dict):
        raise ValueError(f"Source manifest must be an object: {path}")

    source: dict[str, Any] = {
        "repository_id": manifest.get(
            "repository_id",
            SOURCE_DATASET_REPOSITORY_ID,
        ),
        "repository_type": manifest.get("repository_type", "dataset"),
        "dataset_config": manifest.get(
            "dataset_config",
            SOURCE_DATASET_CONFIG,
        ),
        "dataset_split": manifest.get(
            "dataset_split",
            SOURCE_DATASET_SPLIT,
        ),
        "filename": manifest.get("filename", raw_path.name),
        "revision": manifest.get("revision"),
        "license": manifest.get("license"),
        "access": manifest.get("access"),
        "local_path": manifest.get("local_path", str(raw_path)),
        "size_bytes": manifest.get("size_bytes", raw_path.stat().st_size),
        "sha256": manifest.get("sha256", sha256_file(raw_path)),
        "raw_data_committed_to_git": manifest.get(
            "raw_data_committed_to_git",
            False,
        ),
    }

    return source


def _fill_buffer(
    file: TextIO,
    *,
    buffer: str,
    chunk_size: int,
) -> tuple[str, bool]:
    chunk = file.read(chunk_size)
    if not chunk:
        return buffer, False

    return buffer + chunk, True


def iter_json_array(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading it fully."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def fill(file: TextIO) -> bool:
        nonlocal buffer, eof
        if eof:
            return False

        buffer, added = _fill_buffer(
            file,
            buffer=buffer,
            chunk_size=chunk_size,
        )
        eof = not added
        return added

    def skip_whitespace(file: TextIO) -> bool:
        nonlocal position

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if position < len(buffer):
                return True

            if not fill(file):
                return False

    with path.open(encoding="utf-8") as file:
        if not skip_whitespace(file) or buffer[position] != "[":
            raise ValueError(f"Expected top-level JSON array in {path}")

        position += 1
        first = True

        while True:
            if not skip_whitespace(file):
                raise ValueError(f"Unterminated JSON array in {path}")

            if buffer[position] == "]":
                return

            if first:
                first = False
            else:
                if buffer[position] != ",":
                    raise ValueError(
                        f"Expected ',' before next JSON object in {path}."
                    )
                position += 1

                if not skip_whitespace(file):
                    raise ValueError(f"Unterminated JSON array in {path}")

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if fill(file):
                        continue
                    raise ValueError(
                        f"Invalid JSON array item in {path}: {exc}"
                    ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected each source array item to be an object; "
                    f"received {type(value).__name__}."
                )

            yield value

            buffer = buffer[end:]
            position = 0


def issue_reason_code(issue: ValidationIssue) -> str:
    message = issue.message.lower()

    if "duplicate tool name" in message or "duplicate tool names" in message:
        return "duplicate_tool_names"

    if "missing required fields" in message:
        return "missing_required_fields"

    if "query must be a non-empty string" in message:
        return "invalid_query"

    if "unsupported parameter type" in message:
        return "unsupported_parameter_type"

    if "invalid parameter type expression" in message:
        return "invalid_parameter_type_expression"

    if issue.category == "invalid_tool_schema":
        return "invalid_tool_schema"

    return issue.category


def source_id_from_row(row: dict[str, Any]) -> int | None:
    try:
        return int(row["id"])
    except (KeyError, TypeError, ValueError):
        return None


def make_quarantine_record(
    *,
    row: dict[str, Any],
    source_row_index: int,
    source: dict[str, Any],
    issues: tuple[ValidationIssue, ...],
) -> dict[str, Any]:
    source_id = source_id_from_row(row)
    reason_codes = sorted(
        {
            issue_reason_code(issue)
            for issue in issues
        }
    )
    raw_canonical = canonical_json_dumps(row)

    return {
        "quarantine_schema_version": QUARANTINE_SCHEMA_VERSION,
        "source": {
            "repository_id": source.get("repository_id"),
            "dataset_config": source.get("dataset_config"),
            "dataset_split": source.get("dataset_split"),
            "revision": source.get("revision"),
            "filename": source.get("filename"),
            "source_row_index": source_row_index,
            "source_id": source_id,
        },
        "example_id": (
            stable_example_id(source_id)
            if source_id is not None
            else None
        ),
        "reason_codes": reason_codes,
        "issues": [
            {
                "category": issue.category,
                "reason_code": issue_reason_code(issue),
                "message": issue.message,
            }
            for issue in issues
        ],
        "raw_record_sha256": sha256_text(raw_canonical),
        "raw_record": row,
    }


def augment_full_record(
    normalized: dict[str, Any],
    *,
    source_row_index: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    record = copy.deepcopy(normalized)
    metadata = record.setdefault("metadata", {})

    source_id = int(metadata["source_id"])
    example_id = stable_example_id(source_id)

    record["example_id"] = example_id
    metadata.update(
        {
            "source_repository_id": source.get("repository_id"),
            "source_revision": source.get("revision"),
            "source_file": source.get("filename"),
            "source_file_sha256": source.get("sha256"),
            "source_row_index": source_row_index,
            "source_dataset_config": source.get("dataset_config"),
            "source_split": source.get("dataset_split"),
            "normalization_warnings": [],
        }
    )

    return record


def normalize_full_xlam_row(
    row: dict[str, Any],
    *,
    source_row_index: int,
    source: dict[str, Any],
    output_split: str = DEFAULT_OUTPUT_SPLIT,
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> FullNormalizationResult:
    result = validate_raw_example(
        row,
        split=output_split,
        context_token_limit=context_token_limit,
    )

    if not result.is_valid:
        return FullNormalizationResult(
            accepted=None,
            quarantine=make_quarantine_record(
                row=row,
                source_row_index=source_row_index,
                source=source,
                issues=result.issues,
            ),
            estimated_tokens=result.estimated_tokens,
        )

    if result.normalized is None:
        issue = ValidationIssue(
            category="normalization_error",
            message="Validation succeeded without a normalized record.",
        )
        return FullNormalizationResult(
            accepted=None,
            quarantine=make_quarantine_record(
                row=row,
                source_row_index=source_row_index,
                source=source,
                issues=(issue,),
            ),
            estimated_tokens=result.estimated_tokens,
        )

    return FullNormalizationResult(
        accepted=augment_full_record(
            result.normalized,
            source_row_index=source_row_index,
            source=source,
        ),
        quarantine=None,
        estimated_tokens=result.estimated_tokens,
    )


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


def write_checksum_file(output_dir: Path) -> Path:
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
    commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0
        else None
    )

    diff_result = run_git(["diff", "--quiet"])
    cached_diff_result = run_git(["diff", "--cached", "--quiet"])
    untracked_result = run_git(
        ["ls-files", "--others", "--exclude-standard"]
    )

    dirty = (
        diff_result.returncode != 0
        or cached_diff_result.returncode != 0
        or (
            untracked_result.returncode == 0
            and bool(untracked_result.stdout.strip())
        )
    )

    return {
        "commit": commit,
        "dirty": dirty,
    }


def update_distribution(
    counter: Counter[str],
    key: Any,
) -> None:
    counter[str(key)] += 1


def build_normalization_report(
    *,
    raw_path: Path,
    output_dir: Path,
    normalized_path: Path,
    quarantine_path: Path,
    source_manifest_path: Path,
    source: dict[str, Any],
    input_records: int,
    accepted_records: int,
    quarantined_records: int,
    reason_counts: Counter[str],
    available_tool_counts: Counter[str],
    expected_call_counts: Counter[str],
    parameter_type_counts: Counter[str],
    estimated_token_counts: list[int],
    limit: int | None,
    context_token_limit: int,
    repo_root: Path,
) -> dict[str, Any]:
    token_summary: dict[str, int | None]
    if estimated_token_counts:
        token_summary = {
            "min": min(estimated_token_counts),
            "max": max(estimated_token_counts),
            "mean_floor": sum(estimated_token_counts)
            // len(estimated_token_counts),
        }
    else:
        token_summary = {
            "min": None,
            "max": None,
            "mean_floor": None,
        }

    return {
        "normalization_schema_version": (
            FULL_NORMALIZATION_SCHEMA_VERSION
        ),
        "normalized_record_schema_version": (
            NORMALIZED_RECORD_SCHEMA_VERSION
        ),
        "dataset": {
            "repository_id": source.get("repository_id"),
            "repository_type": source.get("repository_type"),
            "config": source.get("dataset_config"),
            "split": source.get("dataset_split"),
            "revision": source.get("revision"),
            "license": source.get("license"),
            "access": source.get("access"),
            "raw_path": str(raw_path),
            "raw_size_bytes": source.get("size_bytes"),
            "raw_sha256": source.get("sha256"),
            "source_manifest_path": str(source_manifest_path),
        },
        "normalizer": {
            "module": "function_calling_ft.full_normalization",
            "entry_point": "scripts/normalize_xlam_full.py",
            "git": git_metadata(repo_root),
        },
        "processing": {
            "streaming": True,
            "limit": limit,
            "input_records": input_records,
            "accepted_records": accepted_records,
            "quarantined_records": quarantined_records,
            "reconciled": (
                input_records == accepted_records + quarantined_records
            ),
            "output_split": DEFAULT_OUTPUT_SPLIT,
            "context_token_limit": context_token_limit,
            "canonical_serialization": {
                "format": "jsonl",
                "ensure_ascii": False,
                "sort_keys": True,
                "separators": [",", ":"],
            },
        },
        "distributions": {
            "quarantine_reasons": dict(sorted(reason_counts.items())),
            "available_tool_counts": dict(
                sorted(available_tool_counts.items())
            ),
            "expected_call_counts": dict(
                sorted(expected_call_counts.items())
            ),
            "parameter_types": dict(
                parameter_type_counts.most_common()
            ),
            "estimated_tokens": token_summary,
        },
        "outputs": {
            "output_dir": str(output_dir),
            "normalized": file_report(normalized_path),
            "quarantine": file_report(quarantine_path),
            "report_path": str(
                output_dir / "manifests" / "normalization_report.json"
            ),
            "checksums_path": str(output_dir / "checksums.sha256"),
        },
    }


def record_distributions(
    record: dict[str, Any],
    *,
    available_tool_counts: Counter[str],
    expected_call_counts: Counter[str],
    parameter_type_counts: Counter[str],
) -> None:
    metadata = record["metadata"]
    update_distribution(
        available_tool_counts,
        metadata["available_tool_count"],
    )
    update_distribution(
        expected_call_counts,
        metadata["expected_call_count"],
    )

    for tool in record["tools"]:
        function = tool["function"]
        parameters = function["parameters"]
        properties = parameters.get("properties", {})

        if not isinstance(properties, dict):
            continue

        for schema in properties.values():
            if not isinstance(schema, dict):
                continue

            schema_type = schema.get("type")
            if isinstance(schema_type, str):
                parameter_type_counts[schema_type] += 1
            elif "anyOf" in schema:
                parameter_type_counts["anyOf"] += 1
            else:
                parameter_type_counts["unspecified"] += 1


def normalize_full_dataset(
    *,
    raw_path: Path,
    output_dir: Path,
    source_manifest_path: Path,
    repo_root: Path,
    limit: int | None = None,
    chunk_size: int = 1024 * 1024,
    context_token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> dict[str, Any]:
    source = load_source_manifest(source_manifest_path, raw_path)
    normalized_path = output_dir / "normalized.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    report_path = output_dir / "manifests" / "normalization_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    input_records = 0
    accepted_records = 0
    quarantined_records = 0
    reason_counts: Counter[str] = Counter()
    available_tool_counts: Counter[str] = Counter()
    expected_call_counts: Counter[str] = Counter()
    parameter_type_counts: Counter[str] = Counter()
    estimated_token_counts: list[int] = []

    with normalized_path.open("w", encoding="utf-8") as normalized_file:
        with quarantine_path.open("w", encoding="utf-8") as quarantine_file:
            for source_row_index, row in enumerate(
                iter_json_array(raw_path, chunk_size=chunk_size),
            ):
                if limit is not None and input_records >= limit:
                    break

                input_records += 1
                result = normalize_full_xlam_row(
                    row,
                    source_row_index=source_row_index,
                    source=source,
                    context_token_limit=context_token_limit,
                )

                if result.estimated_tokens is not None:
                    estimated_token_counts.append(
                        result.estimated_tokens
                    )

                if result.accepted is not None:
                    accepted_records += 1
                    record_distributions(
                        result.accepted,
                        available_tool_counts=available_tool_counts,
                        expected_call_counts=expected_call_counts,
                        parameter_type_counts=parameter_type_counts,
                    )
                    normalized_file.write(
                        canonical_json_line(result.accepted)
                    )
                    continue

                if result.quarantine is None:
                    raise RuntimeError(
                        "Full normalization produced neither accepted "
                        "nor quarantine output."
                    )

                quarantined_records += 1
                for reason_code in result.quarantine["reason_codes"]:
                    reason_counts[str(reason_code)] += 1
                quarantine_file.write(
                    canonical_json_line(result.quarantine)
                )

    report = build_normalization_report(
        raw_path=raw_path,
        output_dir=output_dir,
        normalized_path=normalized_path,
        quarantine_path=quarantine_path,
        source_manifest_path=source_manifest_path,
        source=source,
        input_records=input_records,
        accepted_records=accepted_records,
        quarantined_records=quarantined_records,
        reason_counts=reason_counts,
        available_tool_counts=available_tool_counts,
        expected_call_counts=expected_call_counts,
        parameter_type_counts=parameter_type_counts,
        estimated_token_counts=estimated_token_counts,
        limit=limit,
        context_token_limit=context_token_limit,
        repo_root=repo_root,
    )
    write_json(report_path, report)
    write_checksum_file(output_dir)

    return report
