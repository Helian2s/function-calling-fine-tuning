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

from function_calling_ft.generation import (
    generate_prediction_records,
    load_transformers_model,
    read_jsonl,
    validate_adapter_base_model,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def inspect_adapter(adapter_path: Path) -> dict[str, Any]:
    adapter_config_path = adapter_path / "adapter_config.json"
    automodel_config_path = adapter_path / "automodel_peft_config.json"
    weight_files = sorted(
        [
            *adapter_path.glob("adapter_model*.safetensors"),
            *adapter_path.glob("adapter_model*.bin"),
        ],
    )
    adapter_config = read_json_file(adapter_config_path)
    automodel_config = read_json_file(automodel_config_path)

    return {
        "resolved_adapter_path": str(adapter_path),
        "adapter_config_path": str(adapter_config_path),
        "automodel_peft_config_path": (
            str(automodel_config_path) if automodel_config is not None else None
        ),
        "weight_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for path in weight_files
        ],
        "adapter_config": adapter_config,
        "automodel_peft_config": automodel_config,
        "base_model_name_or_path": (
            adapter_config.get("base_model_name_or_path")
            if adapter_config is not None
            else None
        ),
        "peft_type": (
            adapter_config.get("peft_type") if adapter_config is not None else None
        ),
        "target_modules": (
            adapter_config.get("target_modules")
            if adapter_config is not None
            else None
        ),
    }


def inspect_loaded_model(model: Any) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    four_bit_module_names: list[str] = []
    trainable_parameter_count = 0
    total_parameter_count = 0
    trainable_dtype_counts: Counter[str] = Counter()
    parameter_dtype_counts: Counter[str] = Counter()

    for name, module in model.named_modules():
        class_name = f"{type(module).__module__}.{type(module).__name__}"
        class_counts[class_name] += 1
        lowered = class_name.lower()
        if "4bit" in lowered or "4-bit" in lowered:
            four_bit_module_names.append(name)

    for _, parameter in model.named_parameters():
        count = int(parameter.numel())
        dtype = str(parameter.dtype)
        total_parameter_count += count
        parameter_dtype_counts[dtype] += count
        if parameter.requires_grad:
            trainable_parameter_count += count
            trainable_dtype_counts[dtype] += count

    quantization_attrs = {
        name: getattr(model, name, None)
        for name in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized")
        if hasattr(model, name)
    }
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        for name in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized"):
            if hasattr(base_model, name):
                quantization_attrs[f"base_model.{name}"] = getattr(base_model, name)

    four_bit_observed = bool(four_bit_module_names) or any(
        bool(value)
        for key, value in quantization_attrs.items()
        if key.endswith("is_loaded_in_4bit")
    )

    return {
        "class_counts": dict(sorted(class_counts.items())),
        "four_bit_module_count": len(four_bit_module_names),
        "four_bit_module_name_sample": four_bit_module_names[:50],
        "four_bit_quantization_observed": four_bit_observed,
        "quantization_attrs": {
            key: bool(value) if isinstance(value, bool) else str(value)
            for key, value in quantization_attrs.items()
        },
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_ratio": (
            trainable_parameter_count / total_parameter_count
            if total_parameter_count
            else None
        ),
        "parameter_dtype_counts": dict(sorted(parameter_dtype_counts.items())),
        "trainable_dtype_counts": dict(sorted(trainable_dtype_counts.items())),
    }


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
        default=False,
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    resolved_adapter = validate_adapter_base_model(
        args.adapter_path,
        args.model_name,
    )
    adapter_inspection = inspect_adapter(resolved_adapter)
    records = read_jsonl(args.dataset)[: args.limit]
    tokenizer, model, loaded_adapter_path = load_transformers_model(
        model_name=args.model_name,
        model_revision=args.model_revision,
        adapter_path=resolved_adapter,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )
    model_inspection = inspect_loaded_model(model)
    if args.load_in_4bit and not model_inspection["four_bit_quantization_observed"]:
        raise RuntimeError("Expected 4-bit quantized modules were not observed after reload")
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
        "started_at": started_at,
        "ended_at": utc_now(),
        "process_id": os.getpid(),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_path": loaded_adapter_path,
        "input_adapter_path": str(args.adapter_path),
        "adapter": adapter_inspection,
        "records_checked": len(records),
        "record_ids": [record.get("id") for record in records],
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "loaded_model": model_inspection,
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
