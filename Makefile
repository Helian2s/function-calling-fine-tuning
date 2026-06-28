SHELL := /bin/bash

ifndef PYTHON
ifneq ($(wildcard .venv/bin/python),)
PYTHON := .venv/bin/python
else
PYTHON := python3
endif
endif

RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest
SHELLCHECK_BIN := $(shell if command -v shellcheck >/dev/null 2>&1; then command -v shellcheck; elif [ -x .venv/bin/shellcheck ]; then printf '%s' '.venv/bin/shellcheck'; fi)

.PHONY: help lint typecheck unit-tests validate-smoke verify-splits render-templates \
	inspect-loss-masks test-parser test-scorer test-evaluation test-generation \
	test-infrastructure calculate-checksums preflight smoke-preflight \
	smoke-baseline smoke-baseline-1000 smoke-train smoke-reload-check \
	smoke-evaluate smoke-run shellcheck validate-smoke-configs \
	select-eval-1000 validate-eval-1000 publish-eval-1000-dry-run \
	sync-source-dry-run normalize-xlam-full normalize-xlam-full-check \
	xlam-curate xlam-curate-check xlam-leakage-audit \
	xlam-curator-docker xlam-curator-compare xlam-freeze-splits \
	xlam-freeze-splits-check exp02-build-no-tool exp02-preflight \
	exp02-matrix exp03-validate exp03-dry-run exp03-pilot exp03-full \
	exp04-validate exp04-dry-run exp04-pilot exp04-full \
	exp05a-validate exp05a-dry-run exp05a-pilot \
	exp05b-validate exp05b-dry-run exp05b-full \
	exp06-validate exp06-dry-run exp06-full \
	exp07-validate exp07-dry-run exp07-full \
	exp08-validate exp08-dry-run exp08-full \
	exp09-validate exp09-dry-run exp09-full

help:
	@printf '%s\n' \
		'make preflight   Run local validation gates for the smoke dataset pipeline' \
		'make smoke-preflight  Run EC2/container preflight checks before GPU smoke' \
		'make smoke-baseline   Generate and score fixed baseline predictions' \
		'make smoke-baseline-1000  Generate and score the 1000-case baseline' \
		'make smoke-train      Run 30-step AutoModel smoke training' \
		'make smoke-evaluate   Generate and score the full 40-record test split' \
		'make select-eval-1000  Build the stratified 1000-case eval dataset' \
		'make normalize-xlam-full  Normalize the full local xLAM source dataset' \
		'make normalize-xlam-full-check  Run a small full-normalizer local check' \
		'make xlam-curate  Build grouping, dedup, and leakage metadata' \
		'make xlam-curate-check  Run a small curation metadata check' \
		'make xlam-freeze-splits  Freeze group-aware dataset splits' \
		'make exp02-preflight  Build/check Exp 02 configs without GPU generation' \
		'make exp03-validate  Validate the reference LoRA config' \
		'make exp03-pilot  Run Exp 03 batch probes and 100-step pilot' \
		'make exp03-full   Run Exp 03 pilot plus approved full epoch' \
		'make exp04-validate  Validate the reference QLoRA config' \
		'make exp04-pilot  Run Exp 04 QLoRA batch probes and pilot' \
		'make exp04-full   Run Exp 04 pilot plus approved full epoch' \
		'make exp05a-validate  Validate the full-parameter SFT pilot config' \
		'make exp05a-pilot  Run Exp 05A full-parameter SFT feasibility pilot' \
		'make exp05b-validate  Validate the full-parameter SFT 10K config' \
		'make exp05b-full  Run Exp 05B controlled full-parameter SFT' \
		'make exp06-validate  Validate Exp 06 LoRA rank sweep configs' \
		'make exp06-full  Run Exp 06 LoRA rank sweep' \
		'make exp07-validate  Validate Exp 07 LoRA target-module configs' \
		'make exp07-full  Run Exp 07 LoRA target-module comparison' \
		'make exp08-validate  Validate Exp 08 sample-efficiency configs' \
		'make exp08-full  Run Exp 08 dataset-size sample-efficiency comparison' \
		'make exp09-validate  Validate Exp 09A loss-mask ablation configs' \
		'make exp09-full  Run Exp 09A loss-mask proof and short ablation' \
		'make smoke-run   Run the container/EC2 smoke pipeline scaffold' \
		'make sync-source-dry-run   Preview source publication to S3' \
		'make lint        Run Ruff' \
		'make typecheck   Run mypy'

lint:
	$(RUFF) check src scripts tests

typecheck:
	$(MYPY) src scripts

unit-tests:
	$(PYTEST) \
		tests/test_normalization.py \
		tests/test_splits.py \
		tests/test_validation.py \
		tests/test_template.py \
		tests/test_loss_mask.py \
		tests/test_loss_mask_audit.py \
		tests/test_model_config.py \
		tests/test_run_manifest.py \
		tests/test_split_guard.py \
		tests/test_reference_lora.py \
		tests/test_exp00_completion.py \
		tests/test_preflight_scripts.py \
		-v

validate-smoke:
	$(PYTHON) scripts/validate_examples.py

verify-splits:
	$(PYTHON) scripts/verify_smoke_splits.py

render-templates:
	$(PYTHON) scripts/inspect_template.py --count 5

inspect-loss-masks:
	$(PYTHON) scripts/inspect_loss_mask.py --smoke-count 1

test-parser:
	$(PYTEST) tests/test_parser.py -v

test-scorer:
	$(PYTEST) tests/test_scorer.py -v

test-evaluation:
	$(PYTEST) tests/test_evaluation.py tests/test_evaluation_report.py tests/test_evaluation_compare.py -v

test-generation:
	$(PYTEST) tests/test_generation.py -v

test-infrastructure:
	$(PYTEST) tests/test_infrastructure_scripts.py -v

calculate-checksums:
	$(PYTHON) scripts/calculate_checksums.py

shellcheck:
	bash -n scripts/bootstrap_instance.sh scripts/smoke_run.sh scripts/train_smoke.sh scripts/train_reference_lora.sh scripts/train_exp04_qlora.sh scripts/train_exp05a_full_sft.sh scripts/train_exp05b_full_sft.sh scripts/train_exp06_lora_rank.sh scripts/train_exp07_target_modules.sh scripts/train_exp08_sample_efficiency.sh scripts/train_exp09_loss_masking.sh scripts/sync_results.sh scripts/run_automodel_container.sh infrastructure/aws/bootstrap/bootstrap_instance.sh infrastructure/aws/bootstrap/shutdown_and_sync.sh scripts/publish_exp00_source_bundle.sh scripts/build_exp00_source_bundle.sh scripts/audit_launch_template.sh scripts/sync_source_to_s3.sh scripts/publish_eval_dataset.sh scripts/run_curator_exact_dedup_docker.sh
	$(PYTHON) -m py_compile scripts/resolve_exp00_config.py scripts/select_stratified_eval_sample.py scripts/migrate_smoke_run_metadata.py scripts/check_split_access.py scripts/audit_exp00_completion.py scripts/summarize_evaluation_report.py scripts/compare_evaluations.py scripts/run_training_with_monitor.py scripts/python_sitecustomize/sitecustomize.py scripts/normalize_xlam_full.py scripts/curate_xlam_groups.py scripts/audit_xlam_leakage.py scripts/compare_curator_exact_dedup.py scripts/freeze_xlam_splits.py scripts/build_no_tool_relevance_set.py scripts/render_prompt_hashes.py scripts/run_exp02_matrix.py scripts/validate_reference_lora_config.py scripts/validate_full_sft_config.py scripts/inspect_lora_targets.py scripts/inspect_automodel_package.py scripts/probe_full_sft_runtime.py scripts/probe_automodel_loss_masking.py scripts/audit_exp09_loss_masks.py scripts/reload_full_sft_check.py scripts/run_exp03_reference_lora.py scripts/run_exp05a_full_sft.py scripts/run_exp05b_full_sft.py scripts/run_exp06_lora_rank.py scripts/run_exp07_target_modules.py scripts/run_exp08_sample_efficiency.py scripts/run_exp09_loss_masking.py
	@if [ -n "$(SHELLCHECK_BIN)" ]; then \
		"$(SHELLCHECK_BIN)" -x scripts/bootstrap_instance.sh scripts/smoke_run.sh scripts/train_smoke.sh scripts/train_reference_lora.sh scripts/train_exp04_qlora.sh scripts/train_exp05a_full_sft.sh scripts/train_exp05b_full_sft.sh scripts/train_exp06_lora_rank.sh scripts/train_exp07_target_modules.sh scripts/train_exp08_sample_efficiency.sh scripts/train_exp09_loss_masking.sh scripts/sync_results.sh scripts/run_automodel_container.sh infrastructure/aws/bootstrap/bootstrap_instance.sh infrastructure/aws/bootstrap/shutdown_and_sync.sh scripts/publish_exp00_source_bundle.sh scripts/build_exp00_source_bundle.sh scripts/audit_launch_template.sh scripts/sync_source_to_s3.sh scripts/publish_eval_dataset.sh scripts/run_curator_exact_dedup_docker.sh; \
	else \
		printf '%s\n' 'shellcheck not installed; bash -n completed and shellcheck was skipped.'; \
	fi

normalize-xlam-full:
	$(PYTHON) scripts/normalize_xlam_full.py

normalize-xlam-full-check:
	$(PYTHON) scripts/normalize_xlam_full.py --limit 100 --output-dir data/processed/xlam_full_v1_check

xlam-curate:
	$(PYTHON) scripts/curate_xlam_groups.py --verify-shuffle-stability

xlam-curate-check:
	$(PYTHON) scripts/curate_xlam_groups.py --input data/processed/xlam_full_v1_check/normalized.jsonl --output-dir data/processed/xlam_curated_v1_check --verify-shuffle-stability

xlam-leakage-audit:
	$(PYTHON) scripts/audit_xlam_leakage.py

xlam-curator-docker:
	bash scripts/run_curator_exact_dedup_docker.sh

xlam-curator-compare:
	$(PYTHON) scripts/compare_curator_exact_dedup.py

xlam-freeze-splits:
	$(PYTHON) scripts/freeze_xlam_splits.py

xlam-freeze-splits-check:
	$(PYTHON) scripts/freeze_xlam_splits.py --input data/processed/xlam_curated_v1_check/deduplicated.jsonl --output-dir data/processed/xlam_splits_v1_check --config configs/exp01_dataset/split_freeze_check.yaml --local-files-only --progress-interval 0

exp02-build-no-tool:
	$(PYTHON) scripts/build_no_tool_relevance_set.py

exp02-preflight: exp02-build-no-tool
	$(PYTHON) scripts/run_exp02_matrix.py --dry-run

exp02-matrix: exp02-build-no-tool
	$(PYTHON) scripts/run_exp02_matrix.py

exp03-validate:
	$(PYTHON) scripts/validate_reference_lora_config.py configs/exp03_reference_lora/lora_r8_attention.yaml

exp03-dry-run:
	EXP03_DRY_RUN=1 EXP03_RESULTS_ROOT=/tmp/exp03-reference-lora-dry-run/results EXP03_LOGS_ROOT=/tmp/exp03-reference-lora-dry-run/logs bash scripts/train_reference_lora.sh

exp03-pilot:
	bash scripts/train_reference_lora.sh

exp03-full:
	EXP03_FULL=1 bash scripts/train_reference_lora.sh

exp04-validate:
	$(PYTHON) scripts/validate_reference_lora_config.py --method qlora configs/exp04_lora_vs_qlora/qlora.yaml

exp04-dry-run:
	EXP04_DRY_RUN=1 EXP04_RESULTS_ROOT=/tmp/exp04-qlora-dry-run/results EXP04_LOGS_ROOT=/tmp/exp04-qlora-dry-run/logs bash scripts/train_exp04_qlora.sh

exp04-pilot:
	bash scripts/train_exp04_qlora.sh

exp04-full:
	EXP04_FULL=1 bash scripts/train_exp04_qlora.sh

exp05a-validate:
	$(PYTHON) scripts/validate_full_sft_config.py configs/exp05a_full_sft/full_sft_pilot.yaml

exp05a-dry-run:
	EXP05A_DRY_RUN=1 EXP05A_RESULTS_ROOT=/tmp/exp05a-full-sft-dry-run/results EXP05A_LOGS_ROOT=/tmp/exp05a-full-sft-dry-run/logs bash scripts/train_exp05a_full_sft.sh

exp05a-pilot:
	bash scripts/train_exp05a_full_sft.sh

exp05b-validate:
	$(PYTHON) scripts/validate_full_sft_config.py --profile exp05b configs/exp05b_full_sft/full_sft_10k.yaml

exp05b-dry-run:
	EXP05B_DRY_RUN=1 EXP05B_RESULTS_ROOT=/tmp/exp05b-full-sft-dry-run/results EXP05B_LOGS_ROOT=/tmp/exp05b-full-sft-dry-run/logs EXP05B_CHECKPOINT_ROOT=/tmp/exp05b-full-sft-dry-run/checkpoints bash scripts/train_exp05b_full_sft.sh

exp05b-full:
	bash scripts/train_exp05b_full_sft.sh

exp06-validate:
	$(PYTHON) scripts/run_exp06_lora_rank.py --validate-only --results-root /tmp/exp06-lora-rank-validate/results --logs-root /tmp/exp06-lora-rank-validate/logs

exp06-dry-run:
	EXP06_DRY_RUN=1 EXP06_RESULTS_ROOT=/tmp/exp06-lora-rank-dry-run/results EXP06_LOGS_ROOT=/tmp/exp06-lora-rank-dry-run/logs EXP06_CHECKPOINT_ROOT=/tmp/exp06-lora-rank-dry-run/checkpoints bash scripts/train_exp06_lora_rank.sh

exp06-full:
	bash scripts/train_exp06_lora_rank.sh

exp07-validate:
	$(PYTHON) scripts/run_exp07_target_modules.py --validate-only --results-root /tmp/exp07-target-modules-validate/results --logs-root /tmp/exp07-target-modules-validate/logs

exp07-dry-run:
	EXP07_DRY_RUN=1 EXP07_RESULTS_ROOT=/tmp/exp07-target-modules-dry-run/results EXP07_LOGS_ROOT=/tmp/exp07-target-modules-dry-run/logs EXP07_CHECKPOINT_ROOT=/tmp/exp07-target-modules-dry-run/checkpoints bash scripts/train_exp07_target_modules.sh

exp07-full:
	bash scripts/train_exp07_target_modules.sh

exp08-validate:
	$(PYTHON) scripts/run_exp08_sample_efficiency.py --validate-only --results-root /tmp/exp08-sample-efficiency-validate/results --logs-root /tmp/exp08-sample-efficiency-validate/logs

exp08-dry-run:
	EXP08_DRY_RUN=1 EXP08_RESULTS_ROOT=/tmp/exp08-sample-efficiency-dry-run/results EXP08_LOGS_ROOT=/tmp/exp08-sample-efficiency-dry-run/logs EXP08_CHECKPOINT_ROOT=/tmp/exp08-sample-efficiency-dry-run/checkpoints bash scripts/train_exp08_sample_efficiency.sh

exp08-full:
	bash scripts/train_exp08_sample_efficiency.sh

exp09-validate:
	$(PYTHON) scripts/run_exp09_loss_masking.py --validate-only --results-root /tmp/exp09-loss-masking-validate/results --logs-root /tmp/exp09-loss-masking-validate/logs

exp09-dry-run:
	EXP09_DRY_RUN=1 EXP09_RESULTS_ROOT=/tmp/exp09-loss-masking-dry-run/results EXP09_LOGS_ROOT=/tmp/exp09-loss-masking-dry-run/logs bash scripts/train_exp09_loss_masking.sh

exp09-full:
	bash scripts/train_exp09_loss_masking.sh

sync-source-dry-run:
	./scripts/sync_source_to_s3.sh --dry-run

select-eval-1000:
	$(PYTHON) scripts/select_stratified_eval_sample.py

validate-eval-1000:
	cd data/eval/stratified_1000 && sha256sum -c checksums.sha256

publish-eval-1000-dry-run:
	./scripts/publish_eval_dataset.sh --dry-run

validate-smoke-configs:
	$(PYTHON) scripts/validate_smoke_config.py configs/exp00_smoke/smoke_qlora.yaml
	$(PYTHON) scripts/validate_smoke_config.py configs/exp00_smoke/smoke_lora.yaml
	$(PYTHON) scripts/resolve_exp00_config.py --json

preflight:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) unit-tests
	$(MAKE) test-evaluation
	$(MAKE) test-generation
	$(MAKE) test-infrastructure
	$(MAKE) validate-smoke-configs
	$(MAKE) validate-smoke
	$(MAKE) verify-splits
	$(MAKE) render-templates
	$(MAKE) inspect-loss-masks
	$(MAKE) test-parser
	$(MAKE) test-scorer
	$(MAKE) calculate-checksums
	$(MAKE) shellcheck

smoke-preflight:
	$(MAKE) unit-tests
	$(PYTEST) tests/test_model_config.py -v
	$(MAKE) test-parser
	$(MAKE) test-scorer
	$(MAKE) test-evaluation
	$(MAKE) test-generation
	$(MAKE) test-infrastructure
	$(MAKE) validate-smoke-configs
	$(MAKE) shellcheck
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/validate_automodel_config.py "$${SMOKE_CONFIG_PATH}"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/inspect_template.py --normalized-dir "$${CONTAINER_DATA_ROOT}" --cache-dir /root/.cache/huggingface --count 5'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/inspect_loss_mask.py --normalized-dir "$${CONTAINER_DATA_ROOT}" --cache-dir /root/.cache/huggingface --smoke-count 1'

smoke-baseline:
	bash -c 'source configs/common/exp00.env; mkdir -p "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline" "$${CONTAINER_LOGS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/generate_predictions.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --output "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline/predictions.jsonl" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --limit "$${SMOKE_BASELINE_LIMIT}" --seed "$${SMOKE_SEED}" --max-new-tokens "$${SMOKE_MAX_NEW_TOKENS}" --cache-dir /root/.cache/huggingface --metadata-output "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline/generation_metadata.json"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/evaluate.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --predictions "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline/predictions.jsonl" --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline"'

smoke-baseline-1000:
	bash -o pipefail -c 'source configs/common/exp00.env; dataset_path="$${EVAL1000_DATASET_PATH:-$${CONTAINER_DATA_ROOT}/eval/stratified_1000/normalized/test.jsonl}"; output_dir="$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000"; log_path="$${CONTAINER_LOGS_ROOT}/exp-00/baseline-1000.log"; stop_path="$${output_dir}/STOP"; mkdir -p "$${output_dir}" "$${CONTAINER_LOGS_ROOT}/exp-00"; rm -f "$${stop_path}"; "$${PYTHON_BIN:-python3}" scripts/generate_predictions.py --dataset "$${dataset_path}" --output "$${output_dir}/predictions.jsonl" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --seed "$${SMOKE_SEED}" --max-new-tokens "$${SMOKE_MAX_NEW_TOKENS}" --batch-size "$${GENERATION_BATCH_SIZE:-16}" --cache-dir /root/.cache/huggingface --metadata-output "$${output_dir}/generation_metadata.json" --stream-output --resume --progress-interval "$${EVAL1000_PROGRESS_INTERVAL:-16}" --progress-file "$${output_dir}/progress.json" --stop-file "$${stop_path}" 2>&1 | tee "$${log_path}"'
	bash -c 'source configs/common/exp00.env; dataset_path="$${EVAL1000_DATASET_PATH:-$${CONTAINER_DATA_ROOT}/eval/stratified_1000/normalized/test.jsonl}"; "$${PYTHON_BIN:-python3}" scripts/evaluate.py --dataset "$${dataset_path}" --predictions "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000/predictions.jsonl" --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/summarize_evaluation_report.py --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000"'

smoke-train:
	./scripts/train_smoke.sh

smoke-reload-check:
	bash -c 'source configs/common/exp00.env; mkdir -p "$${CONTAINER_RESULTS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/reload_check.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --output "$${CONTAINER_RESULTS_ROOT}/exp-00/reload-check.json" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --seed "$${SMOKE_SEED}" --cache-dir /root/.cache/huggingface'

smoke-evaluate:
	bash -c 'source configs/common/exp00.env; mkdir -p "$${CONTAINER_RESULTS_ROOT}/exp-00" "$${CONTAINER_LOGS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/generate_predictions.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --output "$${CONTAINER_RESULTS_ROOT}/exp-00/predictions.jsonl" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --seed "$${SMOKE_SEED}" --max-new-tokens "$${SMOKE_MAX_NEW_TOKENS}" --cache-dir /root/.cache/huggingface --no-load-in-4bit --metadata-output "$${CONTAINER_RESULTS_ROOT}/exp-00/generation_metadata.json" 2>&1 | tee "$${CONTAINER_LOGS_ROOT}/exp-00/evaluation.log"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/evaluate.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --predictions "$${CONTAINER_RESULTS_ROOT}/exp-00/predictions.jsonl" --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/summarize_evaluation_report.py --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/collect_run_metadata.py --run-info-dir "$${CONTAINER_RUN_INFO_ROOT}" --results-dir "$${CONTAINER_RESULTS_ROOT}/exp-00" --training-log "$${CONTAINER_LOGS_ROOT}/exp-00/training.log" --config "$${SMOKE_CONFIG_PATH}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}"'

smoke-run:
	./scripts/smoke_run.sh
