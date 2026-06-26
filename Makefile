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
	sync-source-dry-run

help:
	@printf '%s\n' \
		'make preflight   Run local validation gates for the smoke dataset pipeline' \
		'make smoke-preflight  Run EC2/container preflight checks before GPU smoke' \
		'make smoke-baseline   Generate and score fixed baseline predictions' \
		'make smoke-baseline-1000  Generate and score the 1000-case baseline' \
		'make smoke-train      Run 30-step AutoModel smoke training' \
		'make smoke-evaluate   Generate and score the full 40-record test split' \
		'make select-eval-1000  Build the stratified 1000-case eval dataset' \
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
		tests/test_model_config.py \
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
	$(PYTEST) tests/test_evaluation.py -v

test-generation:
	$(PYTEST) tests/test_generation.py -v

test-infrastructure:
	$(PYTEST) tests/test_infrastructure_scripts.py -v

calculate-checksums:
	$(PYTHON) scripts/calculate_checksums.py

shellcheck:
	bash -n scripts/bootstrap_instance.sh scripts/smoke_run.sh scripts/train_smoke.sh scripts/sync_results.sh scripts/run_automodel_container.sh infrastructure/aws/bootstrap/bootstrap_instance.sh infrastructure/aws/bootstrap/shutdown_and_sync.sh scripts/publish_exp00_source_bundle.sh scripts/build_exp00_source_bundle.sh scripts/audit_launch_template.sh scripts/sync_source_to_s3.sh scripts/publish_eval_dataset.sh
	$(PYTHON) -m py_compile scripts/resolve_exp00_config.py scripts/select_stratified_eval_sample.py
	@if [ -n "$(SHELLCHECK_BIN)" ]; then \
		"$(SHELLCHECK_BIN)" -x scripts/bootstrap_instance.sh scripts/smoke_run.sh scripts/train_smoke.sh scripts/sync_results.sh scripts/run_automodel_container.sh infrastructure/aws/bootstrap/bootstrap_instance.sh infrastructure/aws/bootstrap/shutdown_and_sync.sh scripts/publish_exp00_source_bundle.sh scripts/build_exp00_source_bundle.sh scripts/audit_launch_template.sh scripts/sync_source_to_s3.sh scripts/publish_eval_dataset.sh; \
	else \
		printf '%s\n' 'shellcheck not installed; bash -n completed and shellcheck was skipped.'; \
	fi

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
	bash -o pipefail -c 'source configs/common/exp00.env; dataset_path="$${EVAL1000_DATASET_PATH:-$${CONTAINER_DATA_ROOT}/eval/stratified_1000/normalized/test.jsonl}"; output_dir="$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000"; log_path="$${CONTAINER_LOGS_ROOT}/exp-00/baseline-1000.log"; mkdir -p "$${output_dir}" "$${CONTAINER_LOGS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/generate_predictions.py --dataset "$${dataset_path}" --output "$${output_dir}/predictions.jsonl" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --seed "$${SMOKE_SEED}" --max-new-tokens "$${SMOKE_MAX_NEW_TOKENS}" --cache-dir /root/.cache/huggingface --metadata-output "$${output_dir}/generation_metadata.json" --stream-output --resume --progress-interval "$${EVAL1000_PROGRESS_INTERVAL:-25}" 2>&1 | tee "$${log_path}"'
	bash -c 'source configs/common/exp00.env; dataset_path="$${EVAL1000_DATASET_PATH:-$${CONTAINER_DATA_ROOT}/eval/stratified_1000/normalized/test.jsonl}"; "$${PYTHON_BIN:-python3}" scripts/evaluate.py --dataset "$${dataset_path}" --predictions "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000/predictions.jsonl" --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00/baseline-1000"'

smoke-train:
	./scripts/train_smoke.sh

smoke-reload-check:
	bash -c 'source configs/common/exp00.env; mkdir -p "$${CONTAINER_RESULTS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/reload_check.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --output "$${CONTAINER_RESULTS_ROOT}/exp-00/reload-check.json" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --seed "$${SMOKE_SEED}" --cache-dir /root/.cache/huggingface'

smoke-evaluate:
	bash -c 'source configs/common/exp00.env; mkdir -p "$${CONTAINER_RESULTS_ROOT}/exp-00" "$${CONTAINER_LOGS_ROOT}/exp-00"; "$${PYTHON_BIN:-python3}" scripts/generate_predictions.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --output "$${CONTAINER_RESULTS_ROOT}/exp-00/predictions.jsonl" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --seed "$${SMOKE_SEED}" --max-new-tokens "$${SMOKE_MAX_NEW_TOKENS}" --cache-dir /root/.cache/huggingface --metadata-output "$${CONTAINER_RESULTS_ROOT}/exp-00/generation_metadata.json" 2>&1 | tee "$${CONTAINER_LOGS_ROOT}/exp-00/evaluation.log"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/evaluate.py --dataset "$${CONTAINER_DATA_ROOT}/test.jsonl" --predictions "$${CONTAINER_RESULTS_ROOT}/exp-00/predictions.jsonl" --output-dir "$${CONTAINER_RESULTS_ROOT}/exp-00"'
	bash -c 'source configs/common/exp00.env; "$${PYTHON_BIN:-python3}" scripts/collect_run_metadata.py --run-info-dir "$${CONTAINER_RUN_INFO_ROOT}" --results-dir "$${CONTAINER_RESULTS_ROOT}/exp-00" --training-log "$${CONTAINER_LOGS_ROOT}/exp-00/training.log" --config "$${SMOKE_CONFIG_PATH}" --adapter-path "$${SMOKE_ADAPTER_PATH}" --model-name "$${SMOKE_MODEL_NAME}" --model-revision "$${SMOKE_MODEL_REVISION}"'

smoke-run:
	./scripts/smoke_run.sh
