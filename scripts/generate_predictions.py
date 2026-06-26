#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import (
    generate_prediction_records,
    iter_prediction_records,
    load_transformers_model,
    read_jsonl,
    validate_adapter_base_model,
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
        "--stream-output",
        action="store_true",
        help="Write and flush each prediction as soon as it is generated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --stream-output, skip IDs already present in output.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print streaming progress every N processed records.",
    )
    parser.add_argument(
        "--validate-adapter-only",
        action="store_true",
        help="Validate adapter layout and exit without loading the model.",
    )
    return parser.parse_args()


def _prediction_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()

    return {str(record.get("id", "")) for record in read_jsonl(path)}


def _write_streaming_predictions(
    *,
    output: Path,
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    model_name: str,
    model_revision: str,
    adapter_path: str | None,
    seed: int,
    max_new_tokens: int,
    device: str | None,
    resume: bool,
    progress_interval: int,
) -> tuple[int, int]:
    existing_ids = _prediction_ids(output) if resume else set()
    mode = "a" if resume and output.is_file() else "w"
    written = 0
    skipped = 0
    processed = 0

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(mode, encoding="utf-8") as file:
        for prediction in iter_prediction_records(
            records=records,
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_revision=model_revision,
            adapter_path=adapter_path,
            seed=seed,
            max_new_tokens=max_new_tokens,
            device=device,
        ):
            processed += 1
            prediction_id = str(prediction.get("id", ""))

            if prediction_id in existing_ids:
                skipped += 1
            else:
                file.write(
                    json.dumps(
                        prediction,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                file.flush()
                written += 1

            if progress_interval > 0 and (
                processed % progress_interval == 0
                or processed == len(records)
            ):
                print(
                    "progress="
                    f"{processed}/{len(records)} "
                    f"written={written} skipped={skipped}",
                    flush=True,
                )

    return written, skipped


def main() -> None:
    args = parse_args()
    resolved_adapter_path: str | None = None

    if args.adapter_path is not None:
        resolved_adapter_path = str(
            validate_adapter_base_model(args.adapter_path, args.model_name),
        )

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
    if args.stream_output:
        records_written, records_skipped = _write_streaming_predictions(
            output=args.output,
            records=records,
            tokenizer=tokenizer,
            model=model,
            model_name=args.model_name,
            model_revision=args.model_revision,
            adapter_path=loaded_adapter_path,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            resume=args.resume,
            progress_interval=args.progress_interval,
        )
    else:
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
        records_written = len(predictions)
        records_skipped = 0

    metadata = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "records_requested": len(records),
        "records_written": records_written,
        "records_skipped_existing": records_skipped,
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
    print(f"records={records_written}")
    if records_skipped:
        print(f"records_skipped_existing={records_skipped}")


if __name__ == "__main__":
    main()
