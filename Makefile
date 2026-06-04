ROOT_DIR := $(shell pwd)
VENV := $(ROOT_DIR)/.venv

PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install
install:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: clean
clean:
	rm -rf .pytest_cache .ruff_cache build dist runs
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: force-clean
force-clean: clean
	rm -rf .venv

.PHONY: lint
lint:
	$(PYTHON) -m ruff check src

.PHONY: format
format:
	$(PYTHON) -m ruff format src

.PHONY: check
check:
	$(PYTHON) -m ruff check --fix src
	$(PYTHON) -m ruff format src

.PHONY: test
test:
	$(PYTHON) -m pytest

.PHONY: train
train:
	$(PYTHON) -m show_ball.training.train_model

.PHONY: evaluate
evaluate:
	$(PYTHON) -m show_ball.training.evaluate_model

.PHONY: export_to_tensorrt
export_to_tensorrt:
	$(PYTHON) -m tools.export_weight_to_tensorrt

.DEFAULT_GOAL := install