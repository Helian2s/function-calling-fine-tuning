from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml

from function_calling_ft.smoke_config import validate_smoke_config


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "Qwen/Qwen3-1.7B"
EXPECTED_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OLD_MODEL = "Qwen/Qwen3-8B"


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_common_model_config_resolves_to_qwen3_1_7b() -> None:
    config = _load_yaml(ROOT / "configs/common/model_qwen3_1_7b.yaml")
    model = config["model"]
    assert isinstance(model, dict)

    assert model["name"] == EXPECTED_MODEL
    assert model["revision"] == EXPECTED_REVISION


def test_exp00_lora_and_qlora_configs_resolve_to_qwen3_1_7b() -> None:
    for relative_path in (
        "configs/exp00_smoke/smoke_qlora.yaml",
        "configs/exp00_smoke/smoke_lora.yaml",
    ):
        validation = validate_smoke_config(ROOT / relative_path)

        assert validation.ok
        assert validation.model_name == EXPECTED_MODEL
        assert validation.model_revision == EXPECTED_REVISION


def test_baseline_training_and_evaluation_resolve_same_model() -> None:
    resolver = _load_script(
        ROOT / "scripts/resolve_exp00_config.py",
        "resolve_exp00_config_for_tests",
    )

    resolved = resolver.resolve_config(
        model_config_path=ROOT / "configs/common/model_qwen3_1_7b.yaml",
        smoke_config_path=ROOT / "configs/exp00_smoke/smoke_qlora.yaml",
        env_file_path=ROOT / "configs/common/exp00.env",
    )

    assert resolved["base_model"] == EXPECTED_MODEL
    assert resolved["tokenizer"] == EXPECTED_MODEL
    assert resolved["env_model"] == EXPECTED_MODEL
    assert resolved["smoke_model"] == EXPECTED_MODEL
    assert resolved["model_revision"] == EXPECTED_REVISION
    assert resolved["env_model_revision"] == EXPECTED_REVISION
    assert resolved["smoke_model_revision"] == EXPECTED_REVISION
    assert resolved["training_mode"] == "qlora"
    assert resolved["train_path"] == "/workspace/data/train.jsonl"
    assert resolved["validation_path"] == "/workspace/data/validation.jsonl"
    assert resolved["test_path"] == "/workspace/data/test.jsonl"


def test_active_configs_do_not_reference_qwen3_8b() -> None:
    active_paths = [
        ROOT / "configs/common/exp00.env",
        ROOT / "configs/common/model_qwen3_1_7b.yaml",
        ROOT / "configs/exp00_smoke/smoke_qlora.yaml",
        ROOT / "configs/exp00_smoke/smoke_lora.yaml",
        ROOT / "scripts/collect_run_metadata.py",
        ROOT / "src/function_calling_ft/dataset.py",
        ROOT / "src/function_calling_ft/smoke_config.py",
    ]

    for path in active_paths:
        assert OLD_MODEL not in path.read_text(encoding="utf-8")


def test_common_model_config_filename_reflects_active_model() -> None:
    assert (ROOT / "configs/common/model_qwen3_1_7b.yaml").is_file()
    assert not (ROOT / "configs/common/model_qwen3_8b.yaml").exists()
