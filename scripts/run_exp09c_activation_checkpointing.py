#!/usr/bin/env python3
from __future__ import annotations
# ruff: noqa: E402

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from function_calling_ft.activation_checkpointing import (
    CANONICAL_EXP09C_CHECKPOINT_ROOT,
    EXP09C_PROFILE_BY_NAME,
    build_tradeoff_summary,
    compare_primary_configs,
    compact_training_metrics,
    policy_from_tradeoff,
    validate_activation_checkpointing_config,
    validation_to_dict,
    write_json,
)
from function_calling_ft.reference_lora import load_yaml_config, write_yaml_config
from function_calling_ft.split_guard import assert_split_allowed
from scripts.run_exp03_reference_lora import _run_training_stage


DEFAULT_CONFIG_ROOT = Path("configs/exp09c_activation_checkpointing")
DEFAULT_RESULTS_ROOT = Path("/workspace/results/exp-09c")
DEFAULT_LOGS_ROOT = Path("/workspace/logs/exp-09c")
DEFAULT_CHECKPOINT_ROOT = Path("/workspace/checkpoints/exp-09c")
DEFAULT_CACHE_DIR = Path("/root/.cache/huggingface")
TRAIN_PATH = "/workspace/data/processed/xlam_splits_v1/train_10k.jsonl"
VALIDATION_PATH = "/workspace/data/processed/xlam_splits_v1/validation.jsonl"
PRIMARY_PROFILES = ("lora_off", "lora_on")
SECONDARY_PROFILE = "lora_on_microbatch8"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp09C activation-checkpointing benchmark.",
    )
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--automodel-bin", default="automodel")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--run-secondary-microbatch", action="store_true")
    parser.add_argument("--disable-memory-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _config_path(config_root: Path, profile_name: str) -> Path:
    return config_root / f"{profile_name}.yaml"


def _validate_configs(
    *,
    config_root: Path,
    results_root: Path,
    include_secondary: bool,
) -> dict[str, Any]:
    profile_names = [*PRIMARY_PROFILES]
    if include_secondary:
        profile_names.append(SECONDARY_PROFILE)
    validations: dict[str, Any] = {}
    for profile_name in profile_names:
        validation = validate_activation_checkpointing_config(
            _config_path(config_root, profile_name),
            profile_name=profile_name,
            checkpoint_root=CANONICAL_EXP09C_CHECKPOINT_ROOT,
        )
        validations[profile_name] = validation_to_dict(validation)
    write_json(results_root / "config_validation.json", validations)
    errors = [
        f"{name}: {error}"
        for name, payload in validations.items()
        for error in payload["errors"]
    ]
    if errors:
        raise ValueError("; ".join(errors))
    return validations


def _write_staged_config(
    *,
    base_config: Mapping[str, Any],
    profile_name: str,
    output_path: Path,
    checkpoint_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    config = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in dict(base_config).items()
    }
    profile = EXP09C_PROFILE_BY_NAME[profile_name]
    scheduler = dict(config["step_scheduler"])
    scheduler["local_batch_size"] = profile.local_batch_size
    scheduler["global_batch_size"] = profile.global_batch_size
    scheduler["max_steps"] = max_steps
    scheduler["ckpt_every_steps"] = max_steps
    scheduler["val_every_steps"] = max_steps
    config["step_scheduler"] = scheduler
    lr_scheduler = dict(config["lr_scheduler"])
    lr_scheduler["lr_warmup_steps"] = max(1, round(max_steps * 0.03))
    config["lr_scheduler"] = lr_scheduler
    checkpoint = dict(config["checkpoint"])
    checkpoint["checkpoint_dir"] = str(checkpoint_dir)
    checkpoint["enabled"] = True
    config["checkpoint"] = checkpoint
    policy = dict(config.get("task16_policy", {}))
    policy["max_steps"] = max_steps
    policy["staged_at_utc"] = utc_now()
    config["task16_policy"] = policy
    write_yaml_config(output_path, config)
    return config


def _profile_validator(profile_name: str, checkpoint_root: Path) -> Any:
    def validator(path: Path, *, allow_alternate_validation_path: bool = False) -> Any:
        _ = allow_alternate_validation_path
        return validate_activation_checkpointing_config(
            path,
            profile_name=profile_name,
            checkpoint_root=checkpoint_root,
        )

    return validator


def _run_command(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        log_path.write_text(
            json.dumps({"dry_run": True, "command": command}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}: {command}")


def _inspect_automodel_package(
    *,
    results_root: Path,
    logs_root: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/inspect_automodel_package.py",
        "--output",
        str(results_root / "automodel_checkpointing_inspection.json"),
        "--max-files",
        "120",
    ]
    _run_command(command, log_path=logs_root / "automodel-package-inspection.log", dry_run=dry_run)


def _inspect_targets(
    *,
    config: Path,
    results_root: Path,
    logs_root: Path,
    cache_dir: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "scripts/inspect_lora_targets.py",
        "--config",
        str(config),
        "--output",
        str(results_root / "lora_target_inspection.json"),
        "--cache-dir",
        str(cache_dir),
        "--activation-checkpointing-profile",
        "lora_off",
    ]
    _run_command(command, log_path=logs_root / "lora-target-inspection.log", dry_run=dry_run)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_profile(
    *,
    profile_name: str,
    config_root: Path,
    results_root: Path,
    logs_root: Path,
    checkpoint_root: Path,
    automodel_bin: str,
    max_steps: int,
    memory_trace: bool,
    dry_run: bool,
) -> dict[str, Any]:
    stage_root = results_root / profile_name
    checkpoint_dir = checkpoint_root / profile_name
    base_config = load_yaml_config(_config_path(config_root, profile_name))
    config_path = stage_root / "resolved_config.yaml"
    staged = _write_staged_config(
        base_config=base_config,
        profile_name=profile_name,
        output_path=config_path,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
    )
    validation = validate_activation_checkpointing_config(
        config_path,
        profile_name=profile_name,
        checkpoint_root=checkpoint_root,
    )
    write_json(stage_root / "staged_config_validation.json", validation_to_dict(validation))
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    profile = EXP09C_PROFILE_BY_NAME[profile_name]
    metrics = _run_training_stage(
        stage_name=profile_name,
        base_config=staged,
        results_root=results_root,
        logs_root=logs_root,
        checkpoint_root=checkpoint_root,
        automodel_bin=automodel_bin,
        local_batch_size=profile.local_batch_size,
        global_batch_size=profile.global_batch_size,
        max_steps=max_steps,
        ckpt_every_steps=max_steps,
        val_every_steps=max_steps + 1,
        validation_path=None,
        checkpoint_enabled=True,
        memory_trace=memory_trace,
        dry_run=dry_run,
        patch_qlora_peft_state_dict=False,
        validator=_profile_validator(profile_name, checkpoint_root),
        stop_after_step=max_steps - 1,
    )
    if dry_run:
        metrics = {
            "dry_run": True,
            "return_code": 0,
            "losses_are_finite": True,
            "global_step": max_steps - 1,
            "stage_name": profile_name,
            "duration_seconds": 1.0,
            "checkpoint_exists_after": True,
        }
        write_json(stage_root / "training_metrics.json", metrics)
    compact = compact_training_metrics(metrics)
    write_json(stage_root / "compact_training_metrics.json", compact)
    return metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        lines.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_prior_full_sft_context() -> dict[str, Any]:
    roots = [Path("/workspace/results/exp-05a"), Path("/workspace/results/exp-05b")]
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("**/training_metrics.json", "**/training_torch_memory.json", "**/completion_summary.json"):
            for path in sorted(root.glob(pattern)):
                try:
                    payload = _read_json(path)
                except (OSError, json.JSONDecodeError):
                    continue
                files.append(
                    {
                        "path": str(path),
                        "keys": sorted(payload)[:60],
                        "compact_metrics": compact_training_metrics(payload),
                    },
                )
    return {
        "schema_version": "1.0",
        "source": "retained_workspace_prior_exp05a_exp05b",
        "available": bool(files),
        "files": files,
        "note": (
            "No new full-SFT checkpointing probe was run in Exp09C; this section "
            "incorporates retained full-SFT context when present."
        ),
    }


def _write_markdown_report(
    *,
    path: Path,
    summary: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    tradeoff = summary.get("primary_tradeoff", {})
    lines = [
        "# Exp09C Activation Checkpointing Benchmark",
        "",
        "This benchmark measures activation checkpointing as a runtime/memory trade-off.",
        "It does not tune model quality and it does not include packing interaction.",
        "",
        "## Primary Trade-Off",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "allocated_vram_saved_gb",
        "reserved_vram_saved_gb",
        "allocated_vram_saved_pct",
        "reserved_vram_saved_pct",
        "step_time_slowdown_pct",
        "tokens_per_second_delta_pct",
    ):
        value = tradeoff.get(key)
        lines.append(f"| `{key}` | {value if value is not None else 'n/a'} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            f"- PEFT on L40S: `{policy.get('peft_l40s_default')}`",
            f"- PEFT on L4: `{policy.get('peft_l4_default')}`",
            f"- Full SFT on L40S: `{policy.get('full_sft_l40s_default')}`",
            f"- Full SFT on L4: `{policy.get('full_sft_l4_default')}`",
            f"- Reason: {policy.get('primary_reason')}",
            "- Packing interaction measured: `false`",
            "",
            "## Stage Metrics",
            "",
        ],
    )
    for name, metrics in dict(summary.get("stages", {})).items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Duration seconds: `{metrics.get('duration_seconds')}`",
                f"- Mean step time seconds: `{metrics.get('step_time_mean_seconds')}`",
                f"- Peak allocated VRAM GB: `{metrics.get('peak_allocated_vram_gb')}`",
                f"- Peak reserved VRAM GB: `{metrics.get('peak_reserved_vram_gb')}`",
                f"- Loss finite: `{metrics.get('losses_are_finite')}`",
                "",
            ],
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    if not args.validate_only:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    train_decision = assert_split_allowed(
        TRAIN_PATH,
        command_name="exp09c-activation-checkpointing",
    )
    validation_decision = assert_split_allowed(
        VALIDATION_PATH,
        command_name="exp09c-activation-checkpointing",
    )
    write_json(
        args.results_root / "run_plan.json",
        {
            "schema_version": "1.0",
            "task_id": "task-16",
            "experiment_id": "exp-09c",
            "created_at_utc": utc_now(),
            "config_root": str(args.config_root),
            "results_root": str(args.results_root),
            "logs_root": str(args.logs_root),
            "checkpoint_root": str(args.checkpoint_root),
            "max_steps": args.max_steps,
            "memory_trace": not args.disable_memory_trace,
            "run_secondary_microbatch": args.run_secondary_microbatch,
            "train_split": train_decision.__dict__,
            "validation_split": validation_decision.__dict__,
            "packing_interaction_measured": False,
        },
    )

    validations = _validate_configs(
        config_root=args.config_root,
        results_root=args.results_root,
        include_secondary=args.run_secondary_microbatch,
    )
    off_config = load_yaml_config(_config_path(args.config_root, "lora_off"))
    on_config = load_yaml_config(_config_path(args.config_root, "lora_on"))
    config_diff = compare_primary_configs(off_config, on_config)
    write_json(args.results_root / "config_diff.json", config_diff)
    if not config_diff["only_activation_checkpointing_and_output_paths_differ"]:
        raise ValueError("primary configs differ beyond activation checkpointing/output paths")

    _inspect_automodel_package(
        results_root=args.results_root,
        logs_root=args.logs_root,
        dry_run=args.dry_run or args.validate_only,
    )
    _inspect_targets(
        config=_config_path(args.config_root, "lora_off"),
        results_root=args.results_root,
        logs_root=args.logs_root,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run or args.validate_only,
    )

    if args.validate_only:
        write_json(
            args.results_root / "activation_checkpointing_summary.json",
            {
                "schema_version": "1.0",
                "status": "validate_only_complete",
                "config_validation": validations,
                "config_diff": config_diff,
            },
        )
        return

    stages: dict[str, Any] = {}
    metrics_by_profile: dict[str, dict[str, Any]] = {}
    for profile_name in PRIMARY_PROFILES:
        metrics = _run_profile(
            profile_name=profile_name,
            config_root=args.config_root,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            max_steps=args.max_steps,
            memory_trace=not args.disable_memory_trace,
            dry_run=args.dry_run,
        )
        metrics_by_profile[profile_name] = metrics
        stages[profile_name] = compact_training_metrics(metrics)

    secondary_metrics = None
    if args.run_secondary_microbatch:
        secondary_metrics = _run_profile(
            profile_name=SECONDARY_PROFILE,
            config_root=args.config_root,
            results_root=args.results_root,
            logs_root=args.logs_root,
            checkpoint_root=args.checkpoint_root,
            automodel_bin=args.automodel_bin,
            max_steps=args.max_steps,
            memory_trace=not args.disable_memory_trace,
            dry_run=args.dry_run,
        )
        stages[SECONDARY_PROFILE] = compact_training_metrics(secondary_metrics)

    tradeoff = build_tradeoff_summary(
        off_metrics=metrics_by_profile["lora_off"],
        on_metrics=metrics_by_profile["lora_on"],
    )
    policy = policy_from_tradeoff(tradeoff)
    full_sft_context = _collect_prior_full_sft_context()
    summary = {
        "schema_version": "1.0",
        "task_id": "task-16",
        "experiment_id": "exp-09c",
        "status": "complete" if not args.dry_run else "dry_run_complete",
        "created_at_utc": utc_now(),
        "primary_profiles": list(PRIMARY_PROFILES),
        "secondary_profile": SECONDARY_PROFILE if args.run_secondary_microbatch else None,
        "config_validation": validations,
        "config_diff": config_diff,
        "stages": stages,
        "primary_tradeoff": tradeoff,
        "policy": policy,
        "full_sft_context": full_sft_context,
        "packing_interaction_measured": False,
        "artifact_roots": {
            "results": str(args.results_root),
            "logs": str(args.logs_root),
            "checkpoints": str(args.checkpoint_root),
        },
    }
    write_json(args.results_root / "activation_checkpointing_summary.json", summary)
    write_json(args.results_root / "activation_checkpointing_policy.json", policy)
    _write_markdown_report(
        path=args.results_root / "activation_checkpointing_policy.md",
        summary=summary,
        policy=policy,
    )
    write_json(
        args.results_root / "run_manifest.json",
        {
            "schema_version": "1.0",
            "task_id": "task-16",
            "experiment_id": "exp-09c",
            "run_id": "exp09c-activation-checkpointing-benchmark",
            "status": summary["status"],
            "created_at_utc": utc_now(),
            "model_id": "Qwen/Qwen3-1.7B",
            "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "container": {
                "tag": "nvcr.io/nvidia/nemo-automodel:25.11.00",
                "digest": "sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2",
                "observed_nemo_automodel": "0.2.0rc0",
            },
            "artifact_paths": {
                "summary": str(args.results_root / "activation_checkpointing_summary.json"),
                "policy": str(args.results_root / "activation_checkpointing_policy.md"),
            },
        },
    )
    _write_checksums(args.results_root)
    print("exp09c_activation_checkpointing_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
