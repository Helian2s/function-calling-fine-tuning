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
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic function-call predictions.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional JSON path for generation metadata.",
    )
    parser.add_argument(
        "--validate-adapter-only",
        action="store_true",
        help="Validate adapter layout and exit without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_adapter_path: str | None = None

    if args.adapter_path is not None:
        resolved_adapter_path = str(validate_adapter_path(args.adapter_path))

    if args.validate_adapter_only:
        print(f"adapter_path={resolved_adapter_path}")
        return

    records = read_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    tokenizer, model, loaded_adapter_path = load_transformers_model(
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )
    predictions = generate_prediction_records(
        records=records,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=loaded_adapter_path,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    write_jsonl(args.output, predictions)

    metadata = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "records_requested": len(records),
        "records_written": len(predictions),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_path": loaded_adapter_path,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
    }

    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"predictions={args.output}")
    print(f"records={len(predictions)}")


if __name__ == "__main__":
    main()
