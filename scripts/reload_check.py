#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import (
    generate_prediction_records,
    load_transformers_model,
    read_jsonl,
    validate_adapter_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a saved adapter and verify deterministic output.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_adapter = validate_adapter_path(args.adapter_path)
    records = read_jsonl(args.dataset)[: args.limit]
    tokenizer, model, loaded_adapter_path = load_transformers_model(
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=resolved_adapter,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )
    first = generate_prediction_records(
        records=records,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=loaded_adapter_path,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    second = generate_prediction_records(
        records=records,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=loaded_adapter_path,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    deterministic = [
        left["raw_generation"] == right["raw_generation"]
        and left["generation_error"] == right["generation_error"]
        for left, right in zip(first, second, strict=True)
    ]
    report = {
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_path": loaded_adapter_path,
        "records_checked": len(records),
        "deterministic": all(deterministic),
        "per_record_deterministic": deterministic,
        "first": first,
        "second": second,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"reload_check={args.output}")

    if not report["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
