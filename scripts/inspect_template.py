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

from function_calling_ft.dataset import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TEMPLATE_CACHE_DIR,
    load_smoke_records,
    render_template_example,
    rendered_example_to_report,
    select_representative_examples,
)


REPORT_PATH = Path(
    "data/manifests/smoke_v1_template_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render representative smoke examples through the "
            "target tokenizer chat template and verify the output."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local tokenizer path.",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Exact Hugging Face model revision to load.",
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
        "--count",
        type=int,
        default=5,
        help="Number of representative examples to inspect.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Where to write the template inspection report.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking mode if the tokenizer template "
            "supports it. Defaults to disabled."
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
    model_revision: str,
    cache_dir: Path,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is not installed. Install development "
            "dependencies first, then rerun scripts/inspect_template.py."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    return AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )


def main() -> None:
    args = parse_args()
    dataset_source, records = load_smoke_records(
        normalized_dir=args.normalized_dir,
    )
    examples = select_representative_examples(
        records,
        count=args.count,
    )
    tokenizer = _load_tokenizer(
        model_name=args.model,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    rendered_examples = [
        render_template_example(
            tokenizer,
            example,
            enable_thinking=args.enable_thinking,
        )
        for example in examples
    ]
    report_examples = [
        rendered_example_to_report(rendered_example)
        for rendered_example in rendered_examples
    ]
    failures = [
        {
            "id": report["id"],
            "split": report["split"],
            "failures": report["checks"]["failures"],
        }
        for report in report_examples
        if report["checks"]["failures"]
    ]
    report = {
        "model_name": args.model,
        "model_revision": args.model_revision,
        "dataset_source": dataset_source,
        "normalized_dir": str(args.normalized_dir),
        "cache_dir": str(args.cache_dir),
        "examples_requested": args.count,
        "examples_rendered": len(report_examples),
        "thinking_mode_enabled": args.enable_thinking,
        "total_failures": len(failures),
        "failures": failures,
        "examples": report_examples,
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
    print(f"Model revision: {args.model_revision}")
    print(f"Dataset source: {dataset_source}")
    print(
        f"Rendered {len(rendered_examples)} representative example(s)."
    )
    print(f"Thinking mode enabled: {args.enable_thinking}")
    print(f"Report: {args.report_path}")

    for index, report_example in enumerate(
        report_examples,
        start=1,
    ):
        print()
        print(
            "=" * 20
            + f" Example {index}: {report_example['id']} "
            + "=" * 20
        )
        print(
            f"Split: {report_example['split']} | "
            f"Source ID: {report_example['source_id']} | "
            f"Features: {', '.join(report_example['feature_tags'])}"
        )
        print(
            f"Token count: {report_example['token_count']} | "
            f"Failures: {len(report_example['checks']['failures'])}"
        )
        print("-- Rendered text --")
        print(report_example["rendered_text"])
        print("-- Decoded text --")
        print(report_example["decoded_text"])

    if failures:
        print()
        print(
            f"Template inspection failed for {len(failures)} "
            "example(s)."
        )
        raise SystemExit(1)

    print()
    print("Template inspection completed successfully.")


if __name__ == "__main__":
    main()
