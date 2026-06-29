#!/usr/bin/env python3
"""Delete approved fine-tuning S3 artifacts, including noncurrent versions.

This script is intentionally manifest-driven and defaults to dry-run mode. It
does not infer cleanup targets from bucket contents.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("results/fine_tuning_closure/cleanup_candidates_s3.json")
DEFAULT_OUTPUT = Path("results/fine_tuning_closure/s3_cleanup_execution.json")


def run_json(command: list[str]) -> Any:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def list_versions(bucket: str, profile: str) -> list[dict[str, Any]]:
    versions: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    while True:
        command = [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--profile",
            profile,
            "--output",
            "json",
        ]
        if key_marker:
            command.extend(["--key-marker", key_marker])
        if version_marker:
            command.extend(["--version-id-marker", version_marker])
        page = run_json(command)
        for item in page.get("Versions", []):
            versions.append(
                {
                    "key": item["Key"],
                    "version_id": item["VersionId"],
                    "size": int(item.get("Size") or 0),
                    "kind": "version",
                }
            )
        for item in page.get("DeleteMarkers", []):
            versions.append(
                {
                    "key": item["Key"],
                    "version_id": item["VersionId"],
                    "size": 0,
                    "kind": "delete_marker",
                }
            )
        if not page.get("IsTruncated"):
            break
        key_marker = page.get("NextKeyMarker")
        version_marker = page.get("NextVersionIdMarker")
    return versions


def selected_for_delete(
    versions: list[dict[str, Any]],
    *,
    delete_prefixes: list[str],
    keep_prefixes: list[str],
) -> list[dict[str, Any]]:
    selected = []
    for item in versions:
        key = str(item["key"])
        if any(key.startswith(prefix) for prefix in keep_prefixes):
            continue
        if any(key.startswith(prefix) for prefix in delete_prefixes):
            selected.append(item)
    return selected


def delete_batch(bucket: str, profile: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "Objects": [{"Key": item["key"], "VersionId": item["version_id"]} for item in batch],
        "Quiet": True,
    }
    command = [
        "aws",
        "s3api",
        "delete-objects",
        "--bucket",
        bucket,
        "--profile",
        profile,
        "--delete",
        json.dumps(payload),
        "--output",
        "json",
    ]
    return run_json(command)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", default="finetuning-local")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bucket = manifest["bucket"]
    keep_prefixes = list(manifest["keep_prefixes"])
    delete_prefixes = list(manifest["delete_prefixes"])
    all_versions = list_versions(bucket, args.profile)
    selected = selected_for_delete(
        all_versions,
        delete_prefixes=delete_prefixes,
        keep_prefixes=keep_prefixes,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "bucket": bucket,
        "execute": args.execute,
        "delete_prefixes": delete_prefixes,
        "keep_prefixes": keep_prefixes,
        "objects_considered": len(all_versions),
        "objects_selected": len(selected),
        "bytes_selected": sum(int(item["size"]) for item in selected),
        "deleted_batches": [],
        "errors": [],
    }
    if args.execute:
        for start in range(0, len(selected), 1000):
            batch = selected[start : start + 1000]
            response = delete_batch(bucket, args.profile, batch)
            report["deleted_batches"].append(
                {
                    "start": start,
                    "count": len(batch),
                    "response": response,
                }
            )
            if response and response.get("Errors"):
                report["errors"].extend(response["Errors"])
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
