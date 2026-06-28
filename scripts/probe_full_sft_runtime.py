#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.generation import read_jsonl  # noqa: E402
from function_calling_ft.loss_mask import build_expected_loss_mask_for_record  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe full-SFT load/forward/optimizer memory with assistant-only labels.",
    )
    parser.add_argument("--stage", choices=("load", "forward", "step"), required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-records", type=int, default=1)
    return parser.parse_args()


def _gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024 / 1024 / 1024, 6)


def _cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"cuda_available": False, "error": repr(exc)}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    return {
        "cuda_available": True,
        "allocated_vram_gb": _gb(torch.cuda.memory_allocated()),
        "reserved_vram_gb": _gb(torch.cuda.memory_reserved()),
        "peak_allocated_vram_gb": _gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_gb": _gb(torch.cuda.max_memory_reserved()),
    }


def _inspect_model(model: Any) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    total_parameter_count = 0
    trainable_parameter_count = 0
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    dtype_counts: dict[str, int] = {}
    adapter_module_names: list[str] = []
    for name, module in model.named_modules():
        class_name = f"{type(module).__module__}.{type(module).__name__}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        lowered = class_name.lower()
        if "peft" in lowered or "lora" in lowered:
            adapter_module_names.append(name)
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        dtype = str(parameter.dtype)
        total_parameter_count += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        if parameter.requires_grad:
            trainable_parameter_count += count
            if len(trainable_names) < 50:
                trainable_names.append(name)
        elif len(frozen_names) < 50:
            frozen_names.append(name)
    return {
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": total_parameter_count - trainable_parameter_count,
        "trainable_parameter_ratio": (
            trainable_parameter_count / total_parameter_count
            if total_parameter_count
            else None
        ),
        "parameter_dtype_counts": dict(sorted(dtype_counts.items())),
        "trainable_parameter_name_sample": trainable_names,
        "frozen_parameter_name_sample": frozen_names,
        "adapter_module_count": len(adapter_module_names),
        "adapter_module_name_sample": adapter_module_names[:50],
        "class_counts": dict(sorted(class_counts.items())),
    }


def _optimizer_state_summary(optimizer: Any) -> dict[str, Any]:
    state_tensor_count = 0
    state_numel = 0
    for state in optimizer.state.values():
        for value in state.values():
            if hasattr(value, "numel"):
                state_tensor_count += 1
                state_numel += int(value.numel())
    optimizer_parameter_numel = sum(
        int(parameter.numel())
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    return {
        "optimizer_parameter_numel": optimizer_parameter_numel,
        "optimizer_state_tensor_count": state_tensor_count,
        "optimizer_state_numel": state_numel,
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    started_at = utc_now()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.torch_dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    load_memory = _cuda_memory()
    model_inspection = _inspect_model(model)

    records = read_jsonl(args.dataset)[: args.max_records]
    if not records:
        raise ValueError(f"No records available in {args.dataset}")

    losses: list[float] = []
    optimizer_summary: dict[str, Any] | None = None
    if args.stage in {"forward", "step"}:
        record = records[0]
        mask = build_expected_loss_mask_for_record(
            tokenizer,
            record,
            enable_thinking=False,
        )
        input_ids = torch.tensor([list(mask.input_ids)], dtype=torch.long, device=device)
        labels = torch.tensor([list(mask.labels)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        losses.append(float(loss.detach().cpu()))
        if args.stage == "step":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_summary = _optimizer_state_summary(optimizer)

    payload = {
        "schema_version": "1.0",
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        "stage": args.stage,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "dataset": str(args.dataset),
        "record_ids": [record.get("id") for record in records],
        "torch_dtype": args.torch_dtype,
        "device": str(device),
        "loss_history": losses,
        "losses_are_finite": all(torch.isfinite(torch.tensor(losses)).tolist()) if losses else None,
        "model": model_inspection,
        "optimizer": optimizer_summary,
        "memory_after_load": load_memory,
        "memory_final": _cuda_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"full_sft_runtime_probe={args.output}")


if __name__ == "__main__":
    main()
