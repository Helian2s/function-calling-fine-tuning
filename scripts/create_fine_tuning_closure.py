#!/usr/bin/env python3
"""Create the durable fine-tuning closure package.

The closure package intentionally stores conclusions, metrics, checksums, and
cleanup manifests rather than bulky checkpoints or raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BUCKET = "finetuning-lab-1-037678282394-us-west-2-an"
S3_RESULTS_PREFIX = f"s3://{BUCKET}/finetuning/results"
S3_CHECKPOINT_PREFIX = f"s3://{BUCKET}/finetuning/checkpoints"
S3_LOG_PREFIX = f"s3://{BUCKET}/finetuning/logs"


DECISION_ARTIFACTS: dict[str, str] = {
    "exp02_decision.json": "exp-02/decision.json",
    "exp03_reference_lora_decision.json": "exp-03/reference_lora_decision.json",
    "exp04_lora_vs_qlora_decision.json": "exp-04/reference_lora_vs_qlora_decision.json",
    "exp05a_gate_decision.json": "exp-05a/gate_decision.json",
    "exp05b_completion_summary.json": "exp-05b/completion_summary.json",
    "exp05b_method_selection.json": "exp-05b/method_selection.json",
    "exp06_rank_selection.json": "exp-06/rank_selection.json",
    "exp07_target_module_selection.json": "exp-07/target_module_selection.json",
    "exp08_sample_efficiency_selection.json": "exp-08/sample_efficiency_selection.json",
    "exp09a_loss_mask_ablation_summary.json": "exp-09a/loss_mask_ablation_summary.json",
    "exp09c_activation_checkpointing_summary.json": "exp-09c/activation_checkpointing_summary.json",
    "exp09c_activation_checkpointing_policy.json": "exp-09c/activation_checkpointing_policy.json",
}


METRIC_FIELDS = [
    "experiment",
    "variant",
    "evaluation_set",
    "records",
    "strict_complete_match_rate",
    "schema_equivalent_complete_match_rate",
    "executable_complete_match_rate",
    "complete_call_f1",
    "complete_call_precision",
    "complete_call_recall",
    "function_name_f1",
    "function_name_precision",
    "function_name_recall",
    "average_argument_value_accuracy",
    "schema_validation_success_rate",
    "protocol_clean_response_rate",
    "no_tool_false_positive_rate",
    "missing_call_count",
    "extra_call_count",
    "malformed_tool_call_count",
    "peak_reserved_vram_gb",
    "peak_allocated_vram_gb",
    "training_duration_seconds",
    "training_gpu_hours",
    "notes",
]


@dataclass(frozen=True)
class AwsContext:
    profile: str
    bucket: str = BUCKET


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def run_json(command: list[str]) -> Any:
    output = run_text(command)
    return json.loads(output) if output.strip() else None


def aws_s3_cp_to_file(ctx: AwsContext, s3_key: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = f"s3://{ctx.bucket}/finetuning/results/{s3_key}"
    result = subprocess.run(
        ["aws", "s3", "cp", source, str(dest), "--profile", ctx.profile],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    *,
    experiment: str,
    variant: str,
    evaluation_set: str,
    metrics: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    record_count = metrics.get("total_records")
    if record_count is None:
        if evaluation_set == "validation":
            record_count = 5001
        elif evaluation_set == "validation_slice":
            record_count = 1000
        elif evaluation_set == "no_tool_dev":
            record_count = 100
        elif evaluation_set == "tool_dev":
            record_count = 1000
    return {
        "experiment": experiment,
        "variant": variant,
        "evaluation_set": evaluation_set,
        "records": record_count,
        "strict_complete_match_rate": metrics.get("strict_complete_match_rate"),
        "schema_equivalent_complete_match_rate": metrics.get(
            "schema_equivalent_complete_match_rate"
        ),
        "executable_complete_match_rate": metrics.get("executable_complete_match_rate"),
        "complete_call_f1": metrics.get("complete_call_f1"),
        "complete_call_precision": metrics.get("complete_call_precision"),
        "complete_call_recall": metrics.get("complete_call_recall"),
        "function_name_f1": metrics.get("function_name_f1"),
        "function_name_precision": metrics.get("function_name_precision"),
        "function_name_recall": metrics.get("function_name_recall"),
        "average_argument_value_accuracy": metrics.get("average_argument_value_accuracy"),
        "schema_validation_success_rate": metrics.get("schema_validation_success_rate"),
        "protocol_clean_response_rate": metrics.get("protocol_clean_response_rate"),
        "no_tool_false_positive_rate": metrics.get("no_tool_false_positive_rate"),
        "missing_call_count": metrics.get("missing_call_count"),
        "extra_call_count": metrics.get("extra_call_count"),
        "malformed_tool_call_count": metrics.get("malformed_tool_call_count"),
        "peak_reserved_vram_gb": metrics.get("peak_reserved_vram_gb"),
        "peak_allocated_vram_gb": metrics.get("peak_allocated_vram_gb"),
        "training_duration_seconds": metrics.get("training_duration_seconds")
        or metrics.get("duration_seconds"),
        "training_gpu_hours": metrics.get("training_gpu_hours"),
        "notes": notes,
    }


def build_metric_table(decision_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    exp02 = load_json(decision_dir / "exp02_decision.json")
    for variant, run in exp02.get("runs", {}).items():
        for evaluation_set, metrics in run.items():
            rows.append(
                metric_row(
                    experiment="exp-02",
                    variant=variant,
                    evaluation_set=evaluation_set,
                    metrics=metrics,
                    notes="1K development evaluation for feasibility matrix.",
                )
            )

    exp03 = load_json(decision_dir / "exp03_reference_lora_decision.json")
    rows.extend(
        [
            metric_row(
                experiment="exp-03",
                variant="bf16_lora_r8_alpha16_attention",
                evaluation_set="tool_dev",
                metrics=exp03.get("tool_dev_scores", {}),
                notes="Original reference LoRA on 1K dev set.",
            ),
            metric_row(
                experiment="exp-03",
                variant="bf16_lora_r8_alpha16_attention",
                evaluation_set="no_tool_dev",
                metrics=exp03.get("no_tool_dev_scores", {}),
                notes="Shows severe no-tool regression.",
            ),
        ]
    )

    exp04 = load_json(decision_dir / "exp04_lora_vs_qlora_decision.json")
    for evaluation_set, metrics in exp04.get("evaluation", {}).items():
        rows.append(
            metric_row(
                experiment="exp-04",
                variant="nf4_qlora_r8_alpha16_attention",
                evaluation_set=evaluation_set,
                metrics=metrics,
                notes="QLoRA matched-token-budget run.",
            )
        )

    exp05b = load_json(decision_dir / "exp05b_method_selection.json")
    for variant, metrics in exp05b.get("advanced_score_table", {}).items():
        evaluation_set = "no_tool_dev" if variant.endswith("_no_tool_dev") else "tool_dev"
        rows.append(
            metric_row(
                experiment="exp-05b",
                variant=variant,
                evaluation_set=evaluation_set,
                metrics=metrics,
                notes="Method comparison table: LoRA, QLoRA, full SFT.",
            )
        )

    exp06 = load_json(decision_dir / "exp06_rank_selection.json")
    for metrics in exp06.get("aggregate_metrics", []):
        rank = metrics.get("rank")
        alpha = metrics.get("alpha")
        rows.append(
            metric_row(
                experiment="exp-06",
                variant=f"lora_rank{rank}_alpha{alpha}_attention",
                evaluation_set=str(metrics.get("dataset")),
                metrics=metrics,
                notes="Full validation rank sweep.",
            )
        )

    exp07 = load_json(decision_dir / "exp07_target_module_selection.json")
    for metrics in exp07.get("aggregate_metrics", []):
        rows.append(
            metric_row(
                experiment="exp-07",
                variant=f"lora_{metrics.get('target_profile')}",
                evaluation_set=str(metrics.get("dataset")),
                metrics=metrics,
                notes="Attention-only versus attention+MLP target placement.",
            )
        )

    exp08 = load_json(decision_dir / "exp08_sample_efficiency_selection.json")
    for metrics in exp08.get("aggregate_metrics", []):
        rows.append(
            metric_row(
                experiment="exp-08",
                variant=str(metrics.get("sample_profile")),
                evaluation_set=str(metrics.get("dataset")),
                metrics=metrics,
                notes="One epoch per dataset size.",
            )
        )

    exp09a = load_json(decision_dir / "exp09a_loss_mask_ablation_summary.json")
    for metrics in exp09a.get("aggregate_metrics", []):
        rows.append(
            metric_row(
                experiment="exp-09a",
                variant=str(metrics.get("loss_mask_profile")),
                evaluation_set=str(metrics.get("dataset")),
                metrics=metrics,
                notes="300-step masking ablation; diagnostic only.",
            )
        )

    return rows


def s3_version_inventory(ctx: AwsContext) -> dict[str, Any]:
    payload = run_json(
        [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            ctx.bucket,
            "--profile",
            ctx.profile,
            "--output",
            "json",
        ]
    )
    versions = payload.get("Versions", []) if isinstance(payload, dict) else []
    current_prefixes: dict[str, dict[str, Any]] = {}
    for version in versions:
        key = str(version.get("Key", ""))
        parts = key.split("/")
        prefix = "/".join(parts[:3]) if len(parts) >= 3 else key
        item = current_prefixes.setdefault(prefix, {"prefix": prefix, "count": 0, "bytes": 0})
        item["count"] += 1
        item["bytes"] += int(version.get("Size") or 0)
    prefixes = sorted(current_prefixes.values(), key=lambda item: item["bytes"], reverse=True)
    return {
        "bucket": ctx.bucket,
        "created_at": datetime.now(UTC).isoformat(),
        "version_count": len(versions),
        "delete_marker_count": len(payload.get("DeleteMarkers", []) if isinstance(payload, dict) else []),
        "total_versioned_bytes": sum(int(version.get("Size") or 0) for version in versions),
        "prefixes": prefixes,
    }


def build_experiment_decisions(decision_dir: Path) -> dict[str, Any]:
    exp02 = load_json(decision_dir / "exp02_decision.json")
    exp04 = load_json(decision_dir / "exp04_lora_vs_qlora_decision.json")
    exp05a = load_json(decision_dir / "exp05a_gate_decision.json")
    exp05b = load_json(decision_dir / "exp05b_completion_summary.json")
    exp06 = load_json(decision_dir / "exp06_rank_selection.json")
    exp07 = load_json(decision_dir / "exp07_target_module_selection.json")
    exp08 = load_json(decision_dir / "exp08_sample_efficiency_selection.json")
    exp09a = load_json(decision_dir / "exp09a_loss_mask_ablation_summary.json")
    exp09c = load_json(decision_dir / "exp09c_activation_checkpointing_summary.json")
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "project_decision": "Close fine-tuning experiments and move to inference optimization / GPU acceleration projects.",
        "final_practical_recommendation": {
            "primary_next_model": "Qwen/Qwen3-1.7B base for TensorRT-LLM optimization benchmark",
            "optional_retained_adapter": "Best LoRA adapter from exp-08 train-full for optional deployment comparison",
            "do_not_retain_by_default": "Full-SFT checkpoints; metrics are retained, checkpoint is not needed for next tasks",
        },
        "decisions": [
            {
                "experiment": "exp-02",
                "title": "Base, decoding, and quantization feasibility",
                "decision": exp02.get("decision"),
            },
            {
                "experiment": "exp-04",
                "title": "BF16 LoRA versus NF4 QLoRA",
                "decision": exp04.get("decision"),
                "deviations": exp04.get("deviations"),
            },
            {
                "experiment": "exp-05a",
                "title": "Full-SFT feasibility pilot",
                "decision": exp05a.get("gate_decision") or exp05a.get("decision") or exp05a,
            },
            {
                "experiment": "exp-05b",
                "title": "Full-SFT 10K comparison",
                "decision": {
                    "status": exp05b.get("status"),
                    "reload_succeeded": exp05b.get("reload_succeeded"),
                    "selected_checkpoint_policy": exp05b.get("selected_checkpoint_policy"),
                },
            },
            {
                "experiment": "exp-06",
                "title": "LoRA rank sweep",
                "decision": exp06.get("decision"),
            },
            {
                "experiment": "exp-07",
                "title": "Target-module selection",
                "decision": exp07.get("decision"),
            },
            {
                "experiment": "exp-08",
                "title": "Dataset size and saturation",
                "decision": exp08.get("decision"),
            },
            {
                "experiment": "exp-09a",
                "title": "Assistant-only masking and full-sequence ablation",
                "decision": {
                    "production_policy": exp09a.get("production_policy"),
                    "ablation_policy": exp09a.get("ablation_policy"),
                    "interpretation": exp09a.get("interpretation"),
                },
            },
            {
                "experiment": "exp-09c",
                "title": "Activation checkpointing benchmark",
                "decision": exp09c.get("decision"),
                "policy": exp09c.get("policy"),
            },
        ],
    }


def build_retention_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "principle": "Git stores knowledge and reproducibility metadata; S3/EBS store only artifacts needed for immediate next tasks.",
        "retain_in_git": [
            "docs/fine_tuning_final_report.md",
            "docs/fine_tuning_artifact_retention.md",
            "results/fine_tuning_closure/final_metric_table.csv",
            "results/fine_tuning_closure/final_metric_table.json",
            "results/fine_tuning_closure/experiment_decisions.json",
            "results/fine_tuning_closure/decision_artifacts/*.json",
            "results/fine_tuning_closure/artifact_inventory.json",
            "results/fine_tuning_closure/checksums.sha256",
        ],
        "retain_on_s3": [
            {
                "prefix": "finetuning/checkpoints/exp-08/train-full/",
                "reason": "Small best LoRA adapter from the full-data run; optional TensorRT-LLM deployment comparison.",
            }
        ],
        "retain_on_ebs": [
            {
                "path": "Hugging Face cache for Qwen/Qwen3-1.7B if present",
                "reason": "Speeds up TensorRT-LLM benchmark setup; reproducible from Hugging Face if absent.",
            },
            {
                "path": "best LoRA adapter if present",
                "reason": "Optional future fine-tuned deployment benchmark.",
            },
        ],
        "do_not_retain": [
            "Full-SFT checkpoints and optimizer state",
            "Pilot checkpoints",
            "Intermediate LoRA/QLoRA/rank/target checkpoints",
            "Repeated source/workspace bundles",
            "Temporary Curator payloads",
            "S3 noncurrent versions for deleted artifacts",
            "EBS result/log/checkpoint duplicates after git closure is committed",
        ],
    }


def build_cleanup_candidates() -> dict[str, Any]:
    keep_prefixes = ["finetuning/checkpoints/exp-08/train-full/"]
    delete_prefixes = [
        "finetuning/checkpoints/exp-00/",
        "finetuning/checkpoints/exp-03/",
        "finetuning/checkpoints/exp-04/",
        "finetuning/checkpoints/exp-05a/",
        "finetuning/checkpoints/exp-05b/",
        "finetuning/checkpoints/exp-06/",
        "finetuning/checkpoints/exp-07/",
        "finetuning/checkpoints/exp-08/train-2k/",
        "finetuning/checkpoints/exp-08/train-10k/",
        "finetuning/checkpoints/exp-09a/",
        "finetuning/checkpoints/exp-09c/",
        "finetuning/results/",
        "finetuning/logs/",
        "finetuning/source-bundles/",
        "finetuning/source-updates/",
        "finetuning/tmp/",
        "finetuning/data/",
        "finetuning/configs/",
        "finetuning/checksums-exp00.sha256",
    ]
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "bucket": BUCKET,
        "mode": "delete all versions under delete_prefixes; preserve keep_prefixes",
        "keep_prefixes": keep_prefixes,
        "delete_prefixes": delete_prefixes,
        "notes": [
            "S3 bucket versioning is enabled; current-object deletion alone is insufficient.",
            "Deletion should remove explicit VersionId entries under each cleanup prefix.",
            "The repository closure package must be committed before cleanup.",
        ],
    }


def build_ebs_cleanup_plan() -> str:
    return """# EBS Cleanup Plan

The existing stopped instance keeps two attached gp3 volumes:

- root volume: 100 GiB, delete-on-termination true
- retained workspace volume: 250 GiB, delete-on-termination false

Cleaning files from EBS frees filesystem space but does not reduce the EBS bill
unless the volume is shrunk, replaced, or deleted. Because the next TensorRT-LLM
task can reuse the same 250 GiB workspace, the recommended action is filesystem
cleanup only, followed by a later storage resize decision if needed.

## Keep

- Hugging Face cache for `Qwen/Qwen3-1.7B` if present.
- Best LoRA adapter if present and small.
- Shell history is not collected or preserved.

## Remove

- `/workspace/checkpoints` except the best full-data LoRA adapter if present.
- `/workspace/results`
- `/workspace/logs`
- `/workspace/source-bundles`
- `/workspace/source-updates`
- `/workspace/tmp`
- duplicated fine-tuning datasets under `/workspace/data`
- Docker build cache and stopped containers if present.

## Required Postconditions

- EC2 instance is stopped.
- Workspace free-space report is captured.
- No secrets are printed or persisted.
"""


def write_docs(output_dir: Path, metric_rows: list[dict[str, Any]], decisions: dict[str, Any]) -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    best_validation = [
        row
        for row in metric_rows
        if row["evaluation_set"] == "validation"
        and row["executable_complete_match_rate"] is not None
    ]
    best_validation.sort(key=lambda row: float(row["executable_complete_match_rate"]), reverse=True)
    top_rows = best_validation[:8]
    report_lines = [
        "# Fine-Tuning Final Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Scope",
        "",
        "This report closes the Qwen3-1.7B fine-tuning workstream. The next workstream is model optimization and GPU acceleration, so this repository now preserves conclusions and reproducibility metadata rather than bulky training artifacts.",
        "",
        "## Main Decisions",
        "",
        "- Use the base `Qwen/Qwen3-1.7B` model first for the TensorRT-LLM optimization benchmark.",
        "- Preserve only the small best LoRA adapter as an optional future deployment comparison.",
        "- Do not preserve full-SFT checkpoints by default; the checkpoint is expensive and not needed for the next two projects.",
        "- Keep BF16 deterministic generation as the primary comparable evaluation policy.",
        "- Treat no-tool behavior as a known regression area for tool-call fine-tuning.",
        "- Do not use activation checkpointing with the pinned BF16 LoRA path; it failed before the first step.",
        "",
        "## Best Validation Rows",
        "",
        "| experiment | variant | exec complete | complete-call F1 | arg value accuracy | no-tool FP | notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in top_rows:
        report_lines.append(
            "| {experiment} | {variant} | {exec:.4f} | {f1:.4f} | {arg:.4f} | {fp} | {notes} |".format(
                experiment=row["experiment"],
                variant=row["variant"],
                exec=float(row["executable_complete_match_rate"] or 0.0),
                f1=float(row["complete_call_f1"] or 0.0),
                arg=float(row["average_argument_value_accuracy"] or 0.0),
                fp="" if row["no_tool_false_positive_rate"] is None else f"{float(row['no_tool_false_positive_rate']):.4f}",
                notes=row["notes"],
            )
        )
    report_lines.extend(
        [
            "",
            "## Experiment Conclusions",
            "",
            "- Base BF16 deterministic inference is the clean baseline for future optimization.",
            "- NF4 inference reduced memory but hurt quality enough that it was not accepted as the primary comparison mode.",
            "- BF16 LoRA improved tool-call accuracy over base, but caused severe no-tool false positives.",
            "- QLoRA produced similar tool-call quality to reference LoRA under controlled settings, with stack-specific deviations documented.",
            "- Full-parameter SFT improved tool-call metrics over PEFT on the 1K development comparison but was operationally much more expensive.",
            "- Rank 4 / alpha 8 attention-only was selected as the smallest adequate LoRA configuration on full validation.",
            "- Adding MLP adapters did not improve validation quality enough to justify the extra cost and worsened no-tool behavior.",
            "- The full training pool produced the best tool-call validation scores, but no candidate satisfied all no-tool guardrails.",
            "- Assistant-only masking remains the production policy; full-sequence loss was diagnostic only.",
            "- Activation checkpointing failed in the pinned BF16 LoRA path because checkpointed inputs had no `requires_grad=True`.",
            "",
            "## Durable Artifacts",
            "",
            "- `results/fine_tuning_closure/final_metric_table.csv`",
            "- `results/fine_tuning_closure/final_metric_table.json`",
            "- `results/fine_tuning_closure/experiment_decisions.json`",
            "- `results/fine_tuning_closure/decision_artifacts/`",
            "- `results/fine_tuning_closure/artifact_inventory.json`",
            "- `results/fine_tuning_closure/retained_artifacts_manifest.json`",
            "",
            "## Cleanup Decision",
            "",
            "S3 and EBS cleanup should remove checkpoints, optimizer states, repeated source bundles, temporary payloads, and raw run duplicates after this closure package is committed. The only optional model artifact retained outside git is the full-data LoRA adapter.",
            "",
            "## Next Workstreams",
            "",
            "1. TensorRT-LLM inference optimization benchmark on base Qwen3-1.7B.",
            "2. Megatron/Megatron Core multi-GPU training benchmark using a synthetic small GPT-style setup.",
        ]
    )
    (docs_dir / "fine_tuning_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    retention = load_json(output_dir / "retained_artifacts_manifest.json")
    retention_lines = [
        "# Fine-Tuning Artifact Retention",
        "",
        "## Policy",
        "",
        retention["principle"],
        "",
        "## Keep In Git",
        "",
    ]
    retention_lines.extend(f"- `{item}`" for item in retention["retain_in_git"])
    retention_lines.extend(["", "## Keep Outside Git Temporarily", ""])
    for item in retention["retain_on_s3"]:
        retention_lines.append(f"- S3 `{item['prefix']}`: {item['reason']}")
    for item in retention["retain_on_ebs"]:
        retention_lines.append(f"- EBS `{item['path']}`: {item['reason']}")
    retention_lines.extend(["", "## Cleanup Candidates", ""])
    retention_lines.extend(f"- {item}" for item in retention["do_not_retain"])
    retention_lines.extend(
        [
            "",
            "## Cost Note",
            "",
            "S3 cleanup reduces object storage cost directly. EBS filesystem cleanup frees space but does not reduce gp3 volume charges unless the volume is resized, replaced, or deleted.",
        ]
    )
    (docs_dir / "fine_tuning_artifact_retention.md").write_text(
        "\n".join(retention_lines) + "\n", encoding="utf-8"
    )


def write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "checksums.sha256"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="finetuning-local")
    parser.add_argument("--output-dir", default="results/fine_tuning_closure")
    args = parser.parse_args()

    ctx = AwsContext(profile=args.profile)
    output_dir = Path(args.output_dir)
    decision_dir = output_dir / "decision_artifacts"
    decision_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for filename, s3_key in DECISION_ARTIFACTS.items():
        if not aws_s3_cp_to_file(ctx, s3_key, decision_dir / filename):
            missing.append(s3_key)
    if missing:
        raise RuntimeError(f"Missing required decision artifacts: {missing}")

    artifact_inventory = s3_version_inventory(ctx)
    write_json(output_dir / "artifact_inventory.json", artifact_inventory)

    metric_rows = build_metric_table(decision_dir)
    write_json(output_dir / "final_metric_table.json", metric_rows)
    write_csv(output_dir / "final_metric_table.csv", metric_rows, METRIC_FIELDS)

    decisions = build_experiment_decisions(decision_dir)
    write_json(output_dir / "experiment_decisions.json", decisions)
    write_json(output_dir / "retained_artifacts_manifest.json", build_retention_manifest())
    write_json(output_dir / "cleanup_candidates_s3.json", build_cleanup_candidates())
    (output_dir / "cleanup_candidates_ebs.md").write_text(build_ebs_cleanup_plan(), encoding="utf-8")

    readme = """# Fine-Tuning Closure Package

This directory is the durable repository record for the Qwen3-1.7B fine-tuning
workstream. It stores conclusions, metric tables, decision artifacts, inventory,
and cleanup manifests. It deliberately excludes bulky model checkpoints, raw
predictions, optimizer state, and duplicated cloud artifacts.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_docs(output_dir, metric_rows, decisions)
    write_checksums(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
