from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.dataset import (
    DEFAULT_MODEL_NAME,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TEMPLATE_CACHE_DIR,
    load_smoke_records,
    select_representative_examples,
)
from function_calling_ft.loss_mask import (
    build_expected_loss_mask,
    build_expected_loss_mask_for_record,
    format_loss_mask_diagnostic,
)


REPORT_PATH = Path(
    "data/manifests/smoke_v1_loss_mask_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local predicted/reference loss mask for Qwen "
            "tool-calling examples and print a token-label diagnostic."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local tokenizer path.",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=DEFAULT_NORMALIZED_DIR,
        help="Directory containing normalized smoke JSONL files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_CACHE_DIR,
        help="Local Hugging Face cache directory.",
    )
    parser.add_argument(
        "--smoke-count",
        type=int,
        default=1,
        help="Number of representative smoke examples to inspect.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=120,
        help="Maximum token rows to print per example.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Where to write the loss-mask diagnostic report.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking mode if the tokenizer template supports it. "
            "Defaults to disabled."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    return parser.parse_args()


def _load_tokenizer(
    *,
    model_name: str,
    cache_dir: Path,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is not installed. Install development "
            "dependencies first, then rerun scripts/inspect_loss_mask.py."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )


def _synthetic_tool_result_case() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What is the weather in Denver?",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Denver"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": '{"temperature":72,"unit":"F"}',
        },
        {
            "role": "assistant",
            "content": "The weather in Denver is 72 F.",
        },
    ]
    return messages, tools


def _result_to_report(
    *,
    name: str,
    source: str,
    result: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "rendered_text": result.rendered_text,
        "decoded_text": result.decoded_text,
        "input_ids": list(result.input_ids),
        "labels": list(result.labels),
        "included_token_count": result.included_token_count,
        "ignored_token_count": result.ignored_token_count,
        "spans": [
            {
                "start": span.start,
                "end": span.end,
                "region": span.region,
                "include_in_loss": span.include_in_loss,
            }
            for span in result.spans
        ],
        "tokens": [
            {
                "index": token.index,
                "token_id": token.token_id,
                "token_text": token.token_text,
                "label": token.label,
                "region": token.region,
                "char_start": token.char_start,
                "char_end": token.char_end,
            }
            for token in result.tokens
        ],
    }


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(
        model_name=args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    dataset_source, records = load_smoke_records(
        normalized_dir=args.normalized_dir,
    )
    smoke_examples = select_representative_examples(
        records,
        count=args.smoke_count,
    )
    smoke_results = [
        (
            example.record_id,
            build_expected_loss_mask_for_record(
                tokenizer,
                example.record,
                enable_thinking=args.enable_thinking,
            ),
        )
        for example in smoke_examples
    ]

    synthetic_messages, synthetic_tools = _synthetic_tool_result_case()
    synthetic_result = build_expected_loss_mask(
        tokenizer,
        synthetic_messages,
        tools=synthetic_tools,
        enable_thinking=args.enable_thinking,
    )

    report = {
        "model_name": args.model,
        "dataset_source": dataset_source,
        "normalized_dir": str(args.normalized_dir),
        "cache_dir": str(args.cache_dir),
        "thinking_mode_enabled": args.enable_thinking,
        "smoke_examples": [
            _result_to_report(
                name=record_id,
                source="smoke",
                result=result,
            )
            for record_id, result in smoke_results
        ],
        "synthetic_example": _result_to_report(
            name="synthetic_tool_result_and_final_answer",
            source="synthetic",
            result=synthetic_result,
        ),
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Model: {args.model}")
    print(f"Dataset source: {dataset_source}")
    print(f"Smoke examples inspected: {len(smoke_results)}")
    print(f"Report: {args.report_path}")

    for record_id, result in smoke_results:
        print()
        print(f"Smoke example: {record_id}")
        print(
            f"Included tokens: {result.included_token_count} | "
            f"Ignored tokens: {result.ignored_token_count}"
        )
        print(
            format_loss_mask_diagnostic(
                result,
                max_rows=args.max_rows,
                focus_on_loss=True,
            )
        )

    print()
    print("Synthetic example: synthetic_tool_result_and_final_answer")
    print(
        f"Included tokens: {synthetic_result.included_token_count} | "
        f"Ignored tokens: {synthetic_result.ignored_token_count}"
    )
    print(
        format_loss_mask_diagnostic(
            synthetic_result,
            max_rows=args.max_rows,
            focus_on_loss=True,
        )
    )


if __name__ == "__main__":
    main()
