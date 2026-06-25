#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.smoke_config import (  # noqa: E402
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    validate_smoke_config,
)


DEFAULT_MODEL_CONFIG = ROOT / "configs/common/model_qwen3_1_7b.yaml"
DEFAULT_SMOKE_CONFIG = ROOT / "configs/exp00_smoke/smoke_qlora.yaml"
DEFAULT_ENV_FILE = ROOT / "configs/common/exp00.env"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return loaded


def _read_env_default(path: Path, name: str) -> str:
    pattern = re.compile(
        rf'^{re.escape(name)}="\$\{{{re.escape(name)}:-(?P<value>.*)\}}"$',
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group("value")
    raise ValueError(f"Missing default for {name} in {path}")


def resolve_config(
    *,
    model_config_path: Path,
    smoke_config_path: Path,
    env_file_path: Path,
) -> dict[str, Any]:
    model_config = _load_yaml(model_config_path)
    smoke_config = _load_yaml(smoke_config_path)
    model = model_config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Missing model mapping in {model_config_path}")

    smoke_model = smoke_config.get("model")
    dataset = smoke_config.get("dataset")
    validation_dataset = smoke_config.get("validation_dataset")
    checkpoint = smoke_config.get("checkpoint")
    if not isinstance(smoke_model, dict):
        raise ValueError(f"Missing model mapping in {smoke_config_path}")
    if not isinstance(dataset, dict):
        raise ValueError(f"Missing dataset mapping in {smoke_config_path}")
    if not isinstance(validation_dataset, dict):
        raise ValueError(
            f"Missing validation_dataset mapping in {smoke_config_path}",
        )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Missing checkpoint mapping in {smoke_config_path}")

    validation = validate_smoke_config(smoke_config_path)
    if not validation.ok:
        joined = "; ".join(validation.errors)
        raise ValueError(f"Smoke config validation failed: {joined}")

    env_model_name = _read_env_default(env_file_path, "SMOKE_MODEL_NAME")
    env_model_revision = _read_env_default(
        env_file_path,
        "SMOKE_MODEL_REVISION",
    )
    env_adapter_path = _read_env_default(env_file_path, "SMOKE_ADAPTER_PATH")

    resolved = {
        "experiment": "exp00_smoke",
        "base_model": model.get("name"),
        "tokenizer": model.get("name"),
        "model_revision": model.get("revision"),
        "training_mode": (
            "qlora" if "qlora" in smoke_config_path.name else "lora"
        ),
        "train_path": dataset.get("path_or_dataset_id"),
        "validation_path": validation_dataset.get("path_or_dataset_id"),
        "test_path": "/workspace/data/test.jsonl",
        "checkpoint_path": checkpoint.get("checkpoint_dir"),
        "adapter_path": env_adapter_path,
        "log_path": "/workspace/logs/exp-00/training.log",
        "result_path": "/workspace/results/exp-00",
        "env_model": env_model_name,
        "env_model_revision": env_model_revision,
        "smoke_model": smoke_model.get("pretrained_model_name_or_path"),
        "smoke_model_revision": smoke_model.get("revision"),
    }

    expected_pairs = {
        "common model": resolved["base_model"],
        "env model": resolved["env_model"],
        "smoke model": resolved["smoke_model"],
        "validator model": EXPECTED_MODEL_NAME,
    }
    revision_pairs = {
        "common revision": resolved["model_revision"],
        "env revision": resolved["env_model_revision"],
        "smoke revision": resolved["smoke_model_revision"],
        "validator revision": EXPECTED_MODEL_REVISION,
    }
    if set(expected_pairs.values()) != {EXPECTED_MODEL_NAME}:
        raise ValueError(f"Model mismatch: {expected_pairs}")
    if set(revision_pairs.values()) != {EXPECTED_MODEL_REVISION}:
        raise ValueError(f"Revision mismatch: {revision_pairs}")

    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve Experiment 0 model and path configuration.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
    )
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=DEFAULT_SMOKE_CONFIG,
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = resolve_config(
        model_config_path=args.model_config,
        smoke_config_path=args.smoke_config,
        env_file_path=args.env_file,
    )

    if args.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    labels = [
        ("Experiment", "experiment"),
        ("Base model", "base_model"),
        ("Tokenizer", "tokenizer"),
        ("Model revision", "model_revision"),
        ("Training mode", "training_mode"),
        ("Train split", "train_path"),
        ("Validation split", "validation_path"),
        ("Test split", "test_path"),
        ("Checkpoint directory", "checkpoint_path"),
        ("Log path", "log_path"),
        ("Result path", "result_path"),
    ]
    for label, key in labels:
        print(f"{label}: {resolved[key]}")


if __name__ == "__main__":
    main()
