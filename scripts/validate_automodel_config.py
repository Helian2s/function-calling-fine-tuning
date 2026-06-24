#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.smoke_config import validate_smoke_config


def _walk_targets(value: Any) -> list[str]:
    targets: list[str] = []

    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str):
            targets.append(target)
        for child in value.values():
            targets.extend(_walk_targets(child))
    elif isinstance(value, list):
        for child in value:
            targets.extend(_walk_targets(child))

    return targets


def _resolve_dotted_path(path: str) -> object:
    parts = path.split(".")
    last_error: Exception | None = None

    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            resolved: object = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue

        for attribute in parts[index:]:
            resolved = getattr(resolved, attribute)
        return resolved

    if last_error is not None:
        raise last_error

    raise ModuleNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate smoke config invariants and resolve AutoModel _target_ "
            "imports inside the pinned container."
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--allow-missing-automodel",
        action="store_true",
        help="Return success when nemo_automodel is not installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invariant_result = validate_smoke_config(args.config)
    if not invariant_result.ok:
        for error in invariant_result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    try:
        importlib.import_module("nemo_automodel")
    except ModuleNotFoundError:
        if args.allow_missing_automodel:
            print("nemo_automodel is not installed; import validation skipped.")
            return
        raise

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    targets = sorted(set(_walk_targets(config)))
    for target in targets:
        _resolve_dotted_path(target)
        print(f"resolved={target}")

    print(f"automodel_targets_resolved={len(targets)}")


if __name__ == "__main__":
    main()
