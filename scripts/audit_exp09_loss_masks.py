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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import read_jsonl
from function_calling_ft.loss_mask import (
    build_expected_loss_mask_for_record,
    format_loss_mask_diagnostic,
)
from function_calling_ft.loss_mask_audit import (
    assert_assistant_only_mask,
    mask_statistics,
    select_loss_mask_audit_records,
)
from function_calling_ft.reference_lora import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SEQUENCE_LENGTH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task 14 assistant-only loss-mask audit artifacts.",
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("/workspace/data/processed/xlam_splits_v1/train_10k.jsonl"),
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path("/workspace/data/processed/xlam_splits_v1/validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=EXPECTED_MODEL_NAME)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("/root/.cache/huggingface"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=180)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment gate
        raise SystemExit("transformers is required for loss-mask audit") from exc

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    return AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir),
        trust_remote_code=args.trust_remote_code,
    )


def _source_records(train: Path, validation: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_name, path in (("train_10k", train), ("validation", validation)):
        for record in read_jsonl(path):
            copied = dict(record)
            copied["_audit_source_split"] = source_name
            records.append(copied)
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Experiment 9A Loss-Mask Audit",
        "",
        f"Model: `{report['model_name']}`",
        f"Revision: `{report['model_revision']}`",
        f"Records audited: `{report['record_count']}`",
        f"Status: `{report['status']}`",
        "",
        "## Coverage",
        "",
    ]
    for tag, count in sorted(report["coverage_counts"].items()):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(
        [
            "",
            "## Aggregate Tokens",
            "",
            "```json",
            json.dumps(report["aggregate_statistics"], indent=2, sort_keys=True),
            "```",
            "",
            "## Records",
            "",
            "| ID | Source | Tags | Full | Supervised | Ignored | Assertions |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ],
    )
    for item in report["records"]:
        lines.append(
            "| {record_id} | {source} | {tags} | {full} | {supervised} | {ignored} | {status} |".format(
                record_id=item["record_id"],
                source=item["source_split"],
                tags=", ".join(item["coverage_tags"]),
                full=item["full_tokens"],
                supervised=item["supervised_tokens"],
                ignored=item["ignored_tokens"],
                status="pass" if not item["assertion_errors"] else "fail",
            ),
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args)
    records = _source_records(args.train_dataset, args.validation_dataset)
    selected = select_loss_mask_audit_records(records, count=args.count)

    diagnostics_dir = args.output_dir / "visualizations"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    result_objects = []
    record_reports = []
    coverage_counts: dict[str, int] = {}

    for selected_record in selected:
        result = build_expected_loss_mask_for_record(
            tokenizer,
            selected_record.record,
            enable_thinking=args.enable_thinking,
        )
        result_objects.append(result)
        assertion_errors = assert_assistant_only_mask(
            result=result,
            record=selected_record.record,
            max_sequence_length=EXPECTED_SEQUENCE_LENGTH,
        )
        for tag in selected_record.coverage_tags:
            coverage_counts[tag] = coverage_counts.get(tag, 0) + 1
        diagnostic_path = diagnostics_dir / f"{selected_record.record_id}.txt"
        diagnostic_path.write_text(
            format_loss_mask_diagnostic(
                result,
                max_rows=args.max_rows,
                focus_on_loss=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_reports.append(
            {
                "record_id": selected_record.record_id,
                "source_split": selected_record.record.get("_audit_source_split"),
                "coverage_tags": list(selected_record.coverage_tags),
                "full_tokens": len(result.input_ids),
                "supervised_tokens": result.included_token_count,
                "ignored_tokens": result.ignored_token_count,
                "assertion_errors": assertion_errors,
                "diagnostic_path": str(diagnostic_path),
                "regions": sorted({token.region for token in result.tokens}),
                "span_summary": [
                    {
                        "start": span.start,
                        "end": span.end,
                        "region": span.region,
                        "include_in_loss": span.include_in_loss,
                    }
                    for span in result.spans
                ],
            },
        )

    required_tags = {
        "single_call",
        "multiple_call",
        "parallel_call",
        "boundary_special_tokens",
        "long_schema",
        "long_target",
    }
    missing_tags = sorted(tag for tag in required_tags if coverage_counts.get(tag, 0) == 0)
    assertion_failures = [
        item
        for item in record_reports
        if item["assertion_errors"]
    ]
    status = (
        "pass"
        if not missing_tags and not assertion_failures
        else "fail"
    )
    report = {
        "schema_version": "1.0",
        "experiment_id": "exp-09a",
        "task_id": "task-14",
        "status": status,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "thinking_mode_enabled": args.enable_thinking,
        "train_dataset": str(args.train_dataset),
        "validation_dataset": str(args.validation_dataset),
        "record_count": len(record_reports),
        "coverage_counts": coverage_counts,
        "missing_required_coverage_tags": missing_tags,
        "aggregate_statistics": mask_statistics(result_objects),
        "records": record_reports,
    }

    _write_json(args.output_dir / "loss_mask_audit_report.json", report)
    _write_markdown(args.output_dir / "loss_mask_audit_report.md", report)
    print("loss_mask_audit_report=" + str(args.output_dir / "loss_mask_audit_report.json"))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
