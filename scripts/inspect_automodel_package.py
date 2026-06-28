#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pinned NeMo AutoModel package files for config keys.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-files", type=int, default=80)
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_interesting_files(package_root: Path, *, max_files: int) -> list[dict[str, Any]]:
    patterns = ("*.yaml", "*.yml", "*.py")
    keywords = (
        "activation",
        "checkpoint",
        "clip",
        "grad_clip",
        "gradient_clip",
        "max_grad_norm",
        "peft",
        "lora",
        "finetune",
    )
    results: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(package_root.rglob(pattern)):
            if len(results) >= max_files:
                return results
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches = [
                {
                    "line": index,
                    "text": line.strip()[:240],
                }
                for index, line in enumerate(text.splitlines(), start=1)
                if any(keyword in line.lower() for keyword in keywords)
            ][:40]
            if matches:
                results.append(
                    {
                        "path": str(path),
                        "relative_path": str(path.relative_to(package_root)),
                        "matches": matches,
                    },
                )
    return results


def _training_semantics(package_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        from nemo_automodel.components.training.step_scheduler import StepScheduler
    except Exception as exc:
        payload["step_scheduler_signature_error"] = repr(exc)
    else:
        signature = inspect.signature(StepScheduler)
        payload["step_scheduler_parameters"] = list(signature.parameters)
        payload["step_scheduler_accepts_gradient_clip"] = any(
            "clip" in parameter.lower() or "grad_norm" in parameter.lower()
            for parameter in signature.parameters
        )

    train_ft = package_root / "recipes/llm/train_ft.py"
    try:
        lines = train_ft.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        payload["train_ft_read_error"] = repr(exc)
    else:
        gradient_clip_hits = [
            {
                "line": line_number,
                "text": text.strip(),
            }
            for line_number, text in enumerate(lines, start=1)
            if "max_grad_norm" in text or "_run_train_optim_step(batches" in text
        ]
        payload["train_ft_gradient_clip_hits"] = gradient_clip_hits
        payload["train_ft_hardcoded_gradient_clip_norm"] = any(
            "_run_train_optim_step(batches, 1.0)" in str(hit.get("text", ""))
            for hit in gradient_clip_hits
        )
    return payload


def main() -> None:
    args = parse_args()
    try:
        import nemo_automodel
    except Exception as exc:
        payload = {
            "schema_version": "1.0",
            "import_ok": False,
            "error": repr(exc),
        }
    else:
        package_file = Path(nemo_automodel.__file__).resolve()
        package_root = package_file.parent
        payload = {
            "schema_version": "1.0",
            "import_ok": True,
            "python": sys.version,
            "nemo_automodel_version": _package_version("nemo_automodel"),
            "package_file": str(package_file),
            "package_root": str(package_root),
            "training_semantics": _training_semantics(package_root),
            "interesting_files": _find_interesting_files(
                package_root,
                max_files=args.max_files,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"automodel_package_report={args.output}")
    if not payload.get("import_ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
