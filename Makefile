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

.PHONY: help lint typecheck unit-tests validate-smoke verify-splits render-templates \
	inspect-loss-masks test-parser test-scorer calculate-checksums preflight smoke-run \
	shellcheck

help:
	@printf '%s\n' \
		'make preflight   Run local validation gates for the smoke dataset pipeline' \
		'make smoke-run   Run the container/EC2 smoke pipeline scaffold' \
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

calculate-checksums:
	$(PYTHON) scripts/calculate_checksums.py

shellcheck:
	bash -n scripts/bootstrap_instance.sh scripts/smoke_run.sh scripts/train_smoke.sh scripts/sync_results.sh

preflight:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) unit-tests
	$(MAKE) validate-smoke
	$(MAKE) verify-splits
	$(MAKE) render-templates
	$(MAKE) inspect-loss-masks
	$(MAKE) test-parser
	$(MAKE) test-scorer
	$(MAKE) calculate-checksums
	$(MAKE) shellcheck

smoke-run:
	./scripts/smoke_run.sh
