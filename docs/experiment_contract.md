# Experiment Contract

This repository contract freezes active Experiment 0 behavior for
`Qwen/Qwen3-1.7B` and defines the metadata surfaces required before additional
experiments are implemented.

## Active Inventory

Active configuration files:

- `configs/common/model_qwen3_1_7b.yaml`
- `configs/common/dataset_xlam.yaml`
- `configs/common/evaluation.yaml`
- `configs/common/exp00.env`
- `configs/common/split_access.yaml`
- `configs/exp01_dataset/full_normalization.yaml`
- `configs/exp01_dataset/curation_leakage.yaml`
- `configs/exp00_smoke/smoke_lora.yaml`
- `configs/exp00_smoke/smoke_qlora.yaml`
- `configs/experiment_registry.yaml`

Compatibility placeholders:

- `configs/exp00_smoke/lora.yaml`
- `configs/exp00_smoke/qlora.yaml`

Active entry points:

- Local validation: `make preflight`
- Config resolution dry-run: `scripts/resolve_exp00_config.py --json`
- Baseline generation and scoring: `make smoke-baseline`
- Stratified 1,000-case baseline: `make smoke-baseline-1000`
- Smoke training: `make smoke-train`
- Adapter reload check: `make smoke-reload-check`
- Full 40-record evaluation: `make smoke-evaluate`
- End-to-end smoke pipeline wrapper: `make smoke-run`
- Legacy metadata migration: `scripts/migrate_smoke_run_metadata.py`
- Split access check: `scripts/check_split_access.py --dataset PATH`
- General prediction scoring: `scripts/evaluate.py --dataset DATASET --predictions PREDICTIONS --output-dir OUTPUT`
- Paired evaluation comparison:
  `scripts/compare_evaluations.py --baseline-scored BASE --candidate-scored CANDIDATE --output-dir OUTPUT`
- Full xLAM normalization: `scripts/normalize_xlam_full.py`
- Full xLAM curation metadata: `scripts/curate_xlam_groups.py`
- Leakage audit: `scripts/audit_xlam_leakage.py`
- Scoped Curator Docker exact-dedup: `scripts/run_curator_exact_dedup_docker.sh`

## Model And Runtime

The only active base model is `Qwen/Qwen3-1.7B`.

- Model revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Tokenizer revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Primary prompt path: Qwen native chat template with tools passed through
  `apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)`
- Primary decoding: deterministic, `do_sample=False`, seed `42`,
  `max_new_tokens=256`

The comparable runtime is fixed:

- Container: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- Digest:
  `sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2`
- Runtime-observed package: `nemo_automodel==0.2.0rc0`

Do not change the container to match a previously expected package version.

## Dataset And Split Locking

The frozen smoke split is `smoke-v1`:

- Train: `data/smoke/normalized/train.jsonl`, 120 records
- Validation: `data/smoke/normalized/validation.jsonl`, 40 records
- Test: `data/smoke/normalized/test.jsonl`, 40 records

Manifests and hashes:

- `data/manifests/smoke_v1_selection.json`
- `data/manifests/smoke_v1_summary.json`
- `data/manifests/smoke_v1_normalization_report.json`
- `data/manifests/smoke_v1_validation_report.json`
- `data/manifests/smoke_v1_split_verification.json`
- `data/manifests/smoke_v1_template_report.json`
- `data/manifests/smoke_v1_loss_mask_report.json`
- `data/manifests/smoke_v1_checksums.json`
- `data/manifests/xlam_source.json`

`configs/common/split_access.yaml` is the machine-readable split policy.
Generation, scoring, and explicit split checks block final internal,
reserved challenge, and final no-tool split paths unless the caller passes
`--final-evaluation` and a frozen final-evaluation config reference through
`--final-config`.

The full source-normalized xLAM dataset for Experiment 1A is generated locally
under `data/processed/xlam_full_v1/`. This generated directory is ignored by
Git. The canonical outputs are `normalized.jsonl`, `quarantine.jsonl`,
`manifests/normalization_report.json`, and `checksums.sha256`.

Experiment 1B curation metadata is generated locally under
`data/processed/xlam_curated_v1/`. The mandatory local outputs are
`deduplicated.jsonl`, `duplicate_map.jsonl`, `group_metadata.jsonl`,
`fuzzy_candidates.jsonl`, `fuzzy_review_sample.jsonl`,
`manifests/curation_report.json`, `manifests/leakage_audit_report.json`, and
`checksums.sha256`. Fuzzy candidates are review-only annotations and must not be
used for automatic removal without an approved reviewed-removal manifest.

The final split group key is deliberately conservative:
`tool_set_schema_family_v1` combines exact presented tool-set signatures,
schema-shape fingerprints independent of function names, derived tool-family
labels, and derived API/category labels. This reduces leakage risk by keeping
closely related tool/schema groups together, but it can make later split
balancing harder.

## Stale Reference Classification

| Pattern | Active status |
| --- | --- |
| `Qwen/Qwen3-8B` | Historical README note or test fixture only. Active configs use `Qwen/Qwen3-1.7B`. |
| `qwen3_8b` | No active execution config should use this name. |
| `nemo_automodel==0.3.0` | Historical expectation only; corrected to observed `0.2.0rc0`. |
| `nemo-automodel:26.02.00` | Not active. |
| `4096` / `4,096` | Active smoke configs use `seq_length: 4096` as an explicit smoke training setting, not a project-wide mandatory assumption. |
| rank 32 | Present only as a planned rank-sweep config, not as the active default. |

## Run Manifest

Every future GPU run must write `run_manifest.json` using schema version `1.0`.
The manifest is validated by `function_calling_ft.run_manifest` and must record:

- Experiment ID, task ID, run ID, status, parent/comparison run IDs
- Git commit and dirty-tree state
- Container tag, digest, and relevant package versions
- Model ID, model revision, and tokenizer revision
- Dataset manifest paths and hashes, split name, and split-lock status
- Method: `base`, `bf16_lora`, `nf4_qlora`, or `full_parameter_sft`
- Precision, quantization, sequence length, packing, checkpointing
- Microbatch, gradient accumulation, supervised-token budget, optimizer, LR,
  warm-up, seed
- Decoding configuration and thinking mode
- Instance type, GPU, host memory, peak VRAM, wall time, throughput, optional
  cost
- Canonical artifact paths and checksums

`scripts/migrate_smoke_run_metadata.py` converts existing
`run_metadata.json`-style smoke metadata into the new manifest shape without
regenerating model outputs.

## Canonical Artifact Bundle

Each comparable run must provide these artifact groups:

- `resolved_config`
- `run_manifest`
- `environment`
- `predictions`
- `per_example_scores`
- `metrics`
- `logs`
- `checksums`
- `report`

The generalized evaluator writes versioned local artifacts:

- `scored_predictions.jsonl`
- `parse_failures.jsonl`
- `scores.json`
- `requested_metrics.json`
- `failure_sample.jsonl`
- `summary.md`
- `checksums.sha256`

`scores.json` includes `metric_schema_version: "1.0"` and metrics by call
category, primary tool family, primary API category, seen/unseen status when
available, rendered length bucket, expected-call bucket, tool-count bucket,
split lock status, and primary split. Paired comparisons write
`comparison.json` and `per_example_deltas.jsonl` with deterministic paired
bootstrap confidence intervals.

Experiment 0 uses the established workspace and S3 layout:

- Workspace results: `/workspace/results/exp-00`
- Workspace logs: `/workspace/logs/exp-00`
- Workspace run-info: `/workspace/run-info`
- Workspace adapter: `/workspace/checkpoints/exp-00/smoke-lora`
- S3 results prefix:
  `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/exp-00/`
- S3 logs prefix:
  `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/logs/exp-00/`
- S3 adapter prefix:
  `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/checkpoints/exp-00/smoke-lora/`

GPU host selection and cost-control rules are documented in
`docs/ec2_instance_policy.md`. Normal GPU work reuses one stopped EC2 instance
and changes its instance type before launch instead of creating parallel
task-specific instances.

## Experiment Registry

`configs/experiment_registry.yaml` is the machine-readable experiment index.
It maps `exp-00` through `exp-16` to dependencies, known configs, status, and
expected artifacts. Entries with no concrete repo definition are marked
`pending_definition` instead of guessing future experiment semantics.
