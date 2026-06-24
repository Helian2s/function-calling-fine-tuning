#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import posixpath
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_members: list[str] | None = None,
    forbidden_members: list[str] | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"Archive is empty: {path}")

    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "Archive checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}",
        )

    member_names: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            normalized = posixpath.normpath(name)

            if not name or name.startswith("/") or normalized.startswith("../"):
                raise ValueError(
                    f"Unsafe archive member path rejected: {name!r}",
                )

            if normalized == ".." or "/../" in f"/{normalized}/":
                raise ValueError(
                    f"Unsafe archive member path rejected: {name!r}",
                )

            if member.issym() or member.islnk():
                raise ValueError(
                    f"Archive links are not allowed: {name!r}",
                )

            member_names.append(name)

    members = set(member_names)
    missing = sorted(set(expected_members or []) - members)
    forbidden_present = sorted(set(forbidden_members or []) & members)

    if missing:
        raise ValueError(f"Archive is missing expected members: {missing}")

    if forbidden_present:
        raise ValueError(
            f"Archive contains forbidden members: {forbidden_present}",
        )

    return {
        "path": str(path),
        "bytes": size,
        "sha256": actual_sha256,
        "member_count": len(member_names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Experiment 0 source archive safety.",
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--forbid", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_archive(
        args.archive,
        expected_sha256=args.sha256,
        expected_members=args.expect,
        forbidden_members=args.forbid,
    )
    print(f"archive={report['path']}")
    print(f"bytes={report['bytes']}")
    print(f"sha256={report['sha256']}")
    print(f"member_count={report['member_count']}")


if __name__ == "__main__":
    main()
