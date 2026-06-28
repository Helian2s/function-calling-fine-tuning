#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.split_freeze import (
    DEFAULT_CURATION_REPORT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_NORMALIZATION_REPORT_PATH,
    DEFAULT_OUTPUT_DIR,
    TokenStats,
    build_frozen_splits,
    measure_token_stats,
    read_jsonl,
    validate_frozen_splits,
    write_split_artifacts,
)


DEFAULT_CONFIG_PATH = Path("configs/exp01_dataset/split_freeze.yaml")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_CACHE_DIR = Path(".cache/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze group-aware xLAM train/validation/test/challenge splits "
            "with Qwen native-template token accounting."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--curation-report",
        type=Path,
        default=DEFAULT_CURATION_REPORT_PATH,
    )
    parser.add_argument(
        "--normalization-report",
        type=Path,
        default=DEFAULT_NORMALIZATION_REPORT_PATH,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading tokenizer files missing from cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10000,
        help="Print tokenization progress every N records.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Split freeze config must be a mapping: {path}")
    return loaded


def load_tokenizer(
    *,
    model_name: str,
    model_revision: str,
    cache_dir: Path,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is not installed. Install development dependencies "
            "before running split freeze."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except Exception:
        if local_files_only:
            raise
        return AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
        )


def measure_all_token_stats(
    *,
    tokenizer: Any,
    records: list[dict[str, Any]],
    progress_interval: int,
) -> dict[str, TokenStats]:
    stats: dict[str, TokenStats] = {}
    for index, record in enumerate(records, start=1):
        token_stats = measure_token_stats(tokenizer, record)
        stats[token_stats.example_id] = token_stats
        if progress_interval > 0 and index % progress_interval == 0:
            print(f"Measured token lengths: {index}", flush=True)
    return stats


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Deduplicated curated dataset not found: {args.input}. "
            "Run make xlam-curate first."
        )
    config = load_config(args.config)
    records = list(read_jsonl(args.input))
    tokenizer = load_tokenizer(
        model_name=args.model,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    token_stats_by_id = measure_all_token_stats(
        tokenizer=tokenizer,
        records=records,
        progress_interval=args.progress_interval,
    )
    result = build_frozen_splits(
        records,
        token_stats_by_id,
        config,
    )
    validation = validate_frozen_splits(result)
    if validation["status"] != "pass":
        raise SystemExit(
            "Frozen split validation failed: "
            + json.dumps(validation, sort_keys=True)
        )
    report = write_split_artifacts(
        result=result,
        output_dir=args.output_dir,
        repo_root=ROOT,
        curation_report_path=args.curation_report,
        normalization_report_path=args.normalization_report,
        config_path=args.config,
    )

    counts = report["counts"]
    print("xLAM split freeze completed.")
    print(f"Selected max sequence length: {report['sequence_length']['selected']}")
    print(f"Train full records:          {counts['train']['records']}")
    print(f"Train 10K records:           {counts['train_10k']['records']}")
    print(f"Train 2K records:            {counts['train_2k']['records']}")
    print(f"Validation records:          {counts['validation']['records']}")
    print(f"Dev eval records:            {counts['dev_eval_1k']['records']}")
    print(
        "Internal test records:      "
        f"{counts['internal_test_locked']['records']}"
    )
    print(
        "Challenge records:          "
        f"{counts['reserved_challenge_locked']['records']}"
    )
    print(
        "Excluded overlength records:"
        f" {counts['excluded_overlength']['records']}"
    )
    print(f"Report:                      {args.output_dir / 'manifests' / 'split_report.json'}")
    print(f"Dataset card:                {args.output_dir / 'DATASET_CARD.md'}")
    print(f"Checksums:                   {args.output_dir / 'checksums.sha256'}")
    print(
        json.dumps(
            {
                "selected_max_sequence_length": report["sequence_length"][
                    "selected"
                ],
                "train_records": counts["train"]["records"],
                "validation_records": counts["validation"]["records"],
                "internal_test_records": counts["internal_test_locked"][
                    "records"
                ],
                "challenge_records": counts["reserved_challenge_locked"][
                    "records"
                ],
                "excluded_overlength_records": counts[
                    "excluded_overlength"
                ]["records"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
