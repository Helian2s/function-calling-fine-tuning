#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import read_jsonl, write_jsonl
from function_calling_ft.prompt_audit import (
    PromptAuditTokenizer,
    prompt_audit_record,
    summarize_prompt_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Qwen native prompts and write stable prompt hashes.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Render with thinking enabled. Primary runs must leave this false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    audit_tokenizer = cast(PromptAuditTokenizer, tokenizer)
    records = read_jsonl(args.dataset)
    audit_records = [
        prompt_audit_record(
            tokenizer=audit_tokenizer,
            record=record,
            enable_thinking=args.enable_thinking,
        )
        for record in records
    ]
    summary = summarize_prompt_audit(audit_records)
    if summary["hidden_expected_response_count"]:
        raise SystemExit("expected response text was rendered into prompts")
    if summary["target_tool_call_leak_count"]:
        raise SystemExit("target tool calls were rendered into prompts")
    write_jsonl(args.output, audit_records)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prompt_hashes={args.output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
