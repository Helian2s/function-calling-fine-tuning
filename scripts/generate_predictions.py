#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import (
    DecodingConfig,
    generate_prediction_records,
    iter_prediction_record_batches,
    load_transformers_model,
    read_jsonl,
    validate_adapter_base_model,
    write_jsonl,
)
from function_calling_ft.split_guard import assert_split_allowed


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
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of deterministic greedy decoding.",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate together.",
    )
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
        "--progress-file",
        type=Path,
        help="Optional JSON path overwritten with the latest progress state.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help=(
            "Optional path checked before each batch; if it exists, "
            "generation stops cleanly."
        ),
    )
    parser.add_argument(
        "--validate-adapter-only",
        action="store_true",
        help="Validate adapter layout and exit without loading the model.",
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Permit explicitly locked final-evaluation splits.",
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        help=(
            "Frozen final-evaluation config required with "
            "--final-evaluation on locked final splits."
        ),
    )
    return parser.parse_args()


def _prediction_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()

    return {str(record.get("id", "")) for record in read_jsonl(path)}


def _write_progress_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cuda_memory_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "peak_allocated_vram_gb": None,
            "peak_reserved_vram_gb": None,
            "cuda_memory_error": "torch-not-installed",
        }

    try:
        if not torch.cuda.is_available():
            return {
                "peak_allocated_vram_gb": None,
                "peak_reserved_vram_gb": None,
                "cuda_memory_error": "cuda-not-available",
            }
        return {
            "peak_allocated_vram_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "peak_reserved_vram_gb": (
                torch.cuda.max_memory_reserved() / 1024**3
            ),
            "cuda_memory_error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive GPU metadata path
        return {
            "peak_allocated_vram_gb": None,
            "peak_reserved_vram_gb": None,
            "cuda_memory_error": str(exc),
        }


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
    decoding: DecodingConfig,
    batch_size: int,
    device: str | None,
    resume: bool,
    progress_interval: int,
    progress_file: Path | None,
    stop_file: Path | None,
) -> tuple[int, int, bool, dict[str, Any]]:
    existing_ids = _prediction_ids(output) if resume else set()
    records_to_generate = [
        record
        for record in records
        if str(record.get("id", "")) not in existing_ids
    ]
    mode = "a" if resume and output.is_file() else "w"
    written = 0
    skipped = len(records) - len(records_to_generate)
    processed = skipped
    generation_errors = 0
    generated_tokens = 0
    start_time = time.monotonic()
    last_report_time = start_time
    last_report_processed = processed
    stopped_early = False
    last_reported_processed: int | None = None

    def report_progress(
        *,
        force: bool = False,
        batch_records: int = 0,
        stop_requested: bool = False,
    ) -> None:
        nonlocal last_report_time
        nonlocal last_report_processed
        nonlocal last_reported_processed

        if progress_interval <= 0 and progress_file is None and not force:
            return

        elapsed = max(time.monotonic() - start_time, 0.0)
        recent_elapsed = max(time.monotonic() - last_report_time, 0.0)
        recent_processed = processed - last_report_processed
        records_remaining = max(len(records) - processed, 0)
        records_per_second = processed / elapsed if elapsed else None
        recent_records_per_second = (
            recent_processed / recent_elapsed
            if recent_elapsed and recent_processed
            else None
        )
        eta_seconds = (
            records_remaining / records_per_second
            if records_per_second
            else None
        )
        payload = {
            "batch_records": batch_records,
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": (
                generated_tokens / elapsed if elapsed else None
            ),
            "generation_errors": generation_errors,
            "output": str(output),
            "processed": processed,
            "recent_records_per_second": recent_records_per_second,
            "records_per_second": records_per_second,
            "remaining": records_remaining,
            "skipped_existing": skipped,
            "stop_file": str(stop_file) if stop_file is not None else None,
            "stop_requested": stop_requested,
            "total_records": len(records),
            "written": written,
        }

        should_print = force or (
            progress_interval > 0
            and (
                last_reported_processed is None
                or processed - last_reported_processed >= progress_interval
                or processed == len(records)
                or stop_requested
            )
        )
        if should_print:
            print(
                "progress_json="
                + json.dumps(payload, sort_keys=True),
                flush=True,
            )
            last_reported_processed = processed
            last_report_time = time.monotonic()
            last_report_processed = processed

        if progress_file is not None:
            _write_progress_file(progress_file, payload)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(mode, encoding="utf-8") as file:
        prediction_batches = iter_prediction_record_batches(
            records=records_to_generate,
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            model_revision=model_revision,
            adapter_path=adapter_path,
            seed=seed,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            batch_size=batch_size,
            device=device,
        )

        while True:
            if stop_file is not None and stop_file.exists():
                stopped_early = True
                report_progress(force=True, stop_requested=True)
                break

            try:
                prediction_batch = next(prediction_batches)
            except StopIteration:
                break

            for prediction in prediction_batch:
                processed += 1
                generated_count = int(
                    prediction.get("generated_token_count", 0) or 0,
                )
                generated_tokens += generated_count
                if prediction.get("generation_error") is not None:
                    generation_errors += 1
                file.write(
                    json.dumps(
                        prediction,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                written += 1

            file.flush()
            report_progress(batch_records=len(prediction_batch))

        if not records_to_generate:
            report_progress(force=True)

    elapsed_seconds = max(time.monotonic() - start_time, 0.0)
    return (
        written,
        skipped,
        stopped_early,
        {
            "generation_wall_time_seconds": elapsed_seconds,
            "generated_tokens": generated_tokens,
            "generation_errors": generation_errors,
            "records_per_second": (
                processed / elapsed_seconds if elapsed_seconds else None
            ),
            "generated_tokens_per_second": (
                generated_tokens / elapsed_seconds if elapsed_seconds else None
            ),
        },
    )


def main() -> None:
    args = parse_args()
    resolved_adapter_path: str | None = None

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.do_sample and any(
        value is not None for value in (args.temperature, args.top_p, args.top_k)
    ):
        raise SystemExit(
            "--temperature, --top-p, and --top-k require --do-sample",
        )
    decoding = DecodingConfig(
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if args.adapter_path is not None:
        resolved_adapter_path = str(
            validate_adapter_base_model(args.adapter_path, args.model_name),
        )

    if args.validate_adapter_only:
        print(f"adapter_path={resolved_adapter_path}")
        return

    split_decision = assert_split_allowed(
        args.dataset,
        final_evaluation=args.final_evaluation,
        final_config=args.final_config,
        command_name="generation",
    )
    records = read_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    load_start = time.monotonic()
    tokenizer, model, loaded_adapter_path = load_transformers_model(
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )
    load_time_seconds = time.monotonic() - load_start
    generation_start = time.monotonic()
    if args.stream_output:
        (
            records_written,
            records_skipped,
            stopped_early,
            generation_stats,
        ) = _write_streaming_predictions(
            output=args.output,
            records=records,
            tokenizer=tokenizer,
            model=model,
            model_name=args.model_name,
            model_revision=args.model_revision,
            adapter_path=loaded_adapter_path,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            decoding=decoding,
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
            progress_interval=args.progress_interval,
            progress_file=args.progress_file,
            stop_file=args.stop_file,
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
            decoding=decoding,
            batch_size=args.batch_size,
            device=args.device,
        )
        write_jsonl(args.output, predictions)
        records_written = len(predictions)
        records_skipped = 0
        stopped_early = False
        generated_tokens = sum(
            int(prediction.get("generated_token_count", 0) or 0)
            for prediction in predictions
        )
        generation_wall = max(time.monotonic() - generation_start, 0.0)
        generation_stats = {
            "generation_wall_time_seconds": generation_wall,
            "generated_tokens": generated_tokens,
            "generation_errors": sum(
                int(prediction.get("generation_error") is not None)
                for prediction in predictions
            ),
            "records_per_second": (
                len(predictions) / generation_wall if generation_wall else None
            ),
            "generated_tokens_per_second": (
                generated_tokens / generation_wall if generation_wall else None
            ),
        }

    metadata = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "records_requested": len(records),
        "records_written": records_written,
        "records_skipped_existing": records_skipped,
        "stopped_early": stopped_early,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_path": loaded_adapter_path,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "load_time_seconds": load_time_seconds,
        "batch_size": args.batch_size,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "split_lock_status": split_decision.split_lock_status,
        "split_name": split_decision.split_name,
        "enable_thinking": False,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        **generation_stats,
        **_cuda_memory_metadata(),
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
