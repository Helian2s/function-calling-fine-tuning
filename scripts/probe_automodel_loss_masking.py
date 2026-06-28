#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe pinned AutoModel loss-mask controls before Exp 09A training.",
    )
    parser.add_argument("--assistant-config", type=Path, required=True)
    parser.add_argument("--full-sequence-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-version", default="0.2.0rc0")
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_hits(source: str, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        lowered = line.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            hits.append({"line": line_number, "text": line.strip()[:240]})
    return hits


def _config_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _mapping(config.get("dataset"))
    validation = _mapping(config.get("validation_dataset"))
    return {
        "dataset_target": dataset.get("_target_"),
        "validation_target": validation.get("_target_"),
        "dataset_loss_mask_policy": dataset.get("loss_mask_policy"),
        "validation_loss_mask_policy": validation.get("loss_mask_policy"),
        "dataset_path": dataset.get("path_or_dataset_id"),
        "validation_path": validation.get("path_or_dataset_id"),
    }


def _load_target(path: str) -> Any:
    module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Invalid _target_: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(_mapping(config.get("dataset")))
    dataset.pop("_target_", None)
    dataset["path_or_dataset_id"] = str(dataset["path_or_dataset_id"])
    return dataset


def _probe_configured_datasets(
    assistant_config: Mapping[str, Any],
    full_config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - container gate
        return {"status": "fail", "error": repr(exc)}

    assistant_dataset_cfg = _mapping(assistant_config.get("dataset"))
    full_dataset_cfg = _mapping(full_config.get("dataset"))
    target = str(assistant_dataset_cfg.get("_target_", ""))
    if target != str(full_dataset_cfg.get("_target_", "")):
        return {"status": "fail", "error": "dataset targets differ"}
    dataset_cls = _load_target(target)
    model = _mapping(assistant_config.get("model"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model.get("pretrained_model_name_or_path")),
        revision=str(model.get("revision")),
        cache_dir="/root/.cache/huggingface",
    )
    assistant_dataset = dataset_cls(
        tokenizer=tokenizer,
        **_dataset_kwargs(assistant_config),
    )
    full_dataset = dataset_cls(
        tokenizer=tokenizer,
        **_dataset_kwargs(full_config),
    )
    assistant_item = assistant_dataset[0]
    full_item = full_dataset[0]
    assistant_labels = list(assistant_item["labels"])
    full_labels = list(full_item["labels"])
    same_inputs = list(assistant_item["input_ids"]) == list(full_item["input_ids"])
    assistant_ignored = sum(label == -100 for label in assistant_labels)
    assistant_supervised = sum(label != -100 for label in assistant_labels)
    full_ignored = sum(label == -100 for label in full_labels)
    full_supervised = sum(label != -100 for label in full_labels)
    errors: list[str] = []
    if not same_inputs:
        errors.append("assistant/full configured datasets produced different input_ids")
    if assistant_ignored <= 0 or assistant_supervised <= 0:
        errors.append("assistant-only labels must contain ignored and supervised tokens")
    if full_ignored != 0:
        errors.append("full-sequence labels unexpectedly contain ignored tokens")
    if full_supervised <= assistant_supervised:
        errors.append("full-sequence labels should supervise more tokens")
    return {
        "status": "pass" if not errors else "fail",
        "dataset_class": f"{dataset_cls.__module__}.{dataset_cls.__name__}",
        "dataset_signature": str(inspect.signature(dataset_cls)),
        "record_count": len(assistant_dataset),
        "same_inputs": same_inputs,
        "assistant_ignored_labels": assistant_ignored,
        "assistant_supervised_labels": assistant_supervised,
        "full_ignored_labels": full_ignored,
        "full_supervised_labels": full_supervised,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    assistant_config = _read_yaml(args.assistant_config)
    full_config = _read_yaml(args.full_sequence_config)

    payload: dict[str, Any]
    try:
        import nemo_automodel
    except Exception as exc:  # pragma: no cover - container gate
        payload = {
            "schema_version": "1.0",
            "status": "fail",
            "import_ok": False,
            "error": repr(exc),
        }
    else:
        version = _package_version("nemo_automodel")
        package_file = Path(nemo_automodel.__file__).resolve()
        assistant_policy = _config_policy(assistant_config)
        full_policy = _config_policy(full_config)
        dataset_probe = _probe_configured_datasets(assistant_config, full_config)
        expected_target = "function_calling_ft.automodel_datasets.ToolCallChatDataset"
        errors: list[str] = []
        if version != args.require_version:
            errors.append(
                f"nemo_automodel version is {version!r}, expected {args.require_version!r}",
            )
        if assistant_policy["dataset_target"] != expected_target:
            errors.append(f"assistant config dataset._target_ must be {expected_target}")
        if full_policy["dataset_target"] != expected_target:
            errors.append(f"full-sequence config dataset._target_ must be {expected_target}")
        if assistant_policy["dataset_loss_mask_policy"] != "assistant_only":
            errors.append("assistant config dataset.loss_mask_policy must be assistant_only")
        if assistant_policy["validation_loss_mask_policy"] != "assistant_only":
            errors.append(
                "assistant config validation_dataset.loss_mask_policy must be assistant_only",
            )
        if full_policy["dataset_loss_mask_policy"] != "full_sequence":
            errors.append("full-sequence config dataset.loss_mask_policy must be full_sequence")
        if full_policy["validation_loss_mask_policy"] != "full_sequence":
            errors.append(
                "full-sequence config validation_dataset.loss_mask_policy must be full_sequence",
            )
        if dataset_probe.get("status") != "pass":
            errors.extend(str(error) for error in dataset_probe.get("errors", []))
        payload = {
            "schema_version": "1.0",
            "status": "pass" if not errors else "fail",
            "import_ok": True,
            "python": sys.version,
            "nemo_automodel_version": version,
            "package_file": str(package_file),
            "assistant_config_policy": assistant_policy,
            "full_sequence_config_policy": full_policy,
            "configured_dataset_probe": dataset_probe,
            "errors": errors,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("automodel_loss_mask_probe=" + str(args.output))
    if payload.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
