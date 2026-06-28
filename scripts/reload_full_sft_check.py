#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.evaluation import evaluate_predictions  # noqa: E402
from function_calling_ft.generation import (  # noqa: E402
    DecodingConfig,
    generate_prediction_records,
    read_jsonl,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a full-SFT checkpoint in a clean process and score small slices.",
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tool-dataset", required=True, type=Path)
    parser.add_argument("--no-tool-dataset", required=True, type=Path)
    parser.add_argument("--tool-limit", type=int, default=100)
    parser.add_argument("--no-tool-limit", type=int, default=100)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _candidate_model_dirs(checkpoint_path: Path) -> list[Path]:
    preferred = [
        checkpoint_path,
        checkpoint_path / "model",
        checkpoint_path / "LATEST" / "model",
        checkpoint_path / "LOWEST_VAL" / "model",
    ]
    candidates: list[Path] = []
    for path in preferred:
        if (path / "config.json").is_file():
            candidates.append(path)
    candidates.extend(
        sorted(
            {
                path.parent
                for path in checkpoint_path.rglob("config.json")
                if path.is_file()
            },
        ),
    )
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def locate_model_dir(checkpoint_path: Path) -> Path:
    candidates = _candidate_model_dirs(checkpoint_path)
    if not candidates:
        raise FileNotFoundError(
            "No config.json found under full-SFT checkpoint path: "
            f"{checkpoint_path}",
        )
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate == checkpoint_path / "LATEST" / "model":
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates[:10])
    raise ValueError(f"Multiple checkpoint model dirs found: {joined}")


def inspect_loaded_model(model: Any) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    adapter_modules: list[str] = []
    total_parameter_count = 0
    trainable_parameter_count = 0
    dtype_counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        class_name = f"{type(module).__module__}.{type(module).__name__}"
        class_counts[class_name] += 1
        lowered = class_name.lower()
        if "peft" in lowered or "lora" in lowered:
            adapter_modules.append(name)
    for _, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameter_count += count
        dtype_counts[str(parameter.dtype)] += count
        if parameter.requires_grad:
            trainable_parameter_count += count
    return {
        "class_counts": dict(sorted(class_counts.items())),
        "adapter_module_count": len(adapter_modules),
        "adapter_module_name_sample": adapter_modules[:50],
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_dtype_counts": dict(sorted(dtype_counts.items())),
    }


def _generate_and_score(
    *,
    name: str,
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    output_dir: Path,
    model_path: Path,
    seed: int,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    first = generate_prediction_records(
        records=records,
        tokenizer=tokenizer,
        model=model,
        model_name=str(model_path),
        model_revision="local-full-sft-checkpoint",
        adapter_path=None,
        seed=seed,
        max_new_tokens=max_new_tokens,
        decoding=DecodingConfig(),
        batch_size=batch_size,
    )
    second = generate_prediction_records(
        records=records[: min(5, len(records))],
        tokenizer=tokenizer,
        model=model,
        model_name=str(model_path),
        model_revision="local-full-sft-checkpoint",
        adapter_path=None,
        seed=seed,
        max_new_tokens=max_new_tokens,
        decoding=DecodingConfig(),
        batch_size=batch_size,
    )
    deterministic = [
        left["raw_generation"] == right["raw_generation"]
        and left["generation_error"] == right["generation_error"]
        for left, right in zip(first[: len(second)], second, strict=True)
    ]
    split_dir = output_dir / name
    predictions_path = split_dir / "predictions.jsonl"
    write_jsonl(predictions_path, first)
    score_outputs = evaluate_predictions(
        dataset_path=split_dir / "dataset.jsonl",
        predictions_path=predictions_path,
        output_dir=split_dir,
    )
    return {
        "name": name,
        "records": len(records),
        "predictions": str(predictions_path),
        "scores": str(score_outputs.scores_path),
        "deterministic": all(deterministic),
        "per_record_deterministic": deterministic,
    }


def _write_dataset_slice(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    model_dir = locate_model_dir(args.checkpoint_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=args.tokenizer_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True,
    )
    model.eval()
    model_inspection = inspect_loaded_model(model)
    if model_inspection["adapter_module_count"]:
        raise RuntimeError("Full-SFT reload unexpectedly found adapter modules")

    tool_records = read_jsonl(args.tool_dataset)[: args.tool_limit]
    no_tool_records = read_jsonl(args.no_tool_dataset)[: args.no_tool_limit]
    tool_dir = args.output_dir / "tool_reload_eval"
    no_tool_dir = args.output_dir / "no_tool_reload_eval"
    _write_dataset_slice(tool_dir / "dataset.jsonl", tool_records)
    _write_dataset_slice(no_tool_dir / "dataset.jsonl", no_tool_records)

    tool_report = _generate_and_score(
        name="tool_reload_eval",
        records=tool_records,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        model_path=model_dir,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    no_tool_report = _generate_and_score(
        name="no_tool_reload_eval",
        records=no_tool_records,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        model_path=model_dir,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    report = {
        "schema_version": "1.0",
        "started_at": started_at,
        "ended_at": utc_now(),
        "process_id": os.getpid(),
        "checkpoint_path": str(args.checkpoint_path),
        "resolved_model_dir": str(model_dir),
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "torch_dtype": args.torch_dtype,
        "model": model_inspection,
        "tool_reload_eval": tool_report,
        "no_tool_reload_eval": no_tool_report,
        "deterministic": (
            tool_report["deterministic"] and no_tool_report["deterministic"]
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "reload_check.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"full_sft_reload_check={report_path}")
    if not report["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
