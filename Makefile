# PHCA — developer entry points
#
#   make setup            create .venv and install the dev extra
#   make test-all         run the full suite (no MuJoCo)
#   make ci-local         lint + tests + causal smoke (what CI runs)
#   make demo             the demonstration from docs/DEMO.md
#   make bench            geometry vs learned selector, 5 and 10, 3 seeds

SHELL := /bin/bash
PYTHON ?= python3
PYTEST := MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH $(PYTHON) -m pytest
FAST_IGNORES := --ignore=python/tests/test_mujoco_env.py \
                --ignore=python/tests/test_cycle_with_mujoco.py \
                --ignore=python/tests/test_continuous_actions.py
# pytest-timeout is in the dev extra; omit the flag when it is not installed.
_PYTEST_TIMEOUT := $(shell $(PYTHON) -c "import pytest_timeout" 2>/dev/null && echo "--timeout=60")

.PHONY: all setup install-deps install-mujoco-deps test-all test-python test-mujoco \
        lint ci-local causal-smoke demo bench clean \
        docker-build docker-build-core docker-build-mujoco docker-run-test \
        docker-run-ci-local docker-bash

all: lint test-all

# ── Setup ────────────────────────────────────────────────────

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev,stats]"
	@echo "Done. Run 'make test-all' to verify."

install-deps:
	$(PYTHON) -m pip install -e ".[dev,stats]"

install-mujoco-deps:
	$(PYTHON) -m pip install -e ".[mujoco]"

# ── Tests ────────────────────────────────────────────────────

test-all: test-python

test-python:
	@echo "Running Python tests (no MuJoCo — fast path)..."
	$(PYTEST) python/tests/ python/phca/ $(FAST_IGNORES) \
	    -v --tb=short --maxfail=5 $(_PYTEST_TIMEOUT)

test-mujoco:
	@echo "Running MuJoCo integration tests (needs gymnasium[mujoco])..."
	$(PYTEST) python/tests/test_mujoco_env.py \
	    python/tests/test_cycle_with_mujoco.py \
	    python/tests/test_continuous_actions.py \
	    -v --tb=short --maxfail=5

lint:
	@command -v ruff >/dev/null 2>&1 || (echo "ruff not installed — run: make install-deps" && exit 1)
	ruff check python/ --no-cache

ci-local: lint
	$(PYTEST) python/tests/ python/phca/ $(FAST_IGNORES) \
	    -v --tb=short --maxfail=5 $(_PYTEST_TIMEOUT)
	$(MAKE) causal-smoke

causal-smoke:
	@mkdir -p .tmp
	@echo "Causal smoke (level2, 10 cycles × 1 seed)..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH $(PYTHON) scripts/phca_causal_eval.py \
	    --levels level2 --cycles 10 --seeds 1 --output .tmp/causal_smoke.json
	@echo "See .tmp/causal_smoke.json"

# ── Demo and benchmark ───────────────────────────────────────

demo:
	PYTHONPATH=python $(PYTHON) -m phca run --cycles 100 --seed 0 --relocate-goal-every 25

bench:
	PYTHONPATH=python $(PYTHON) -m phca bench --sizes 5,10 --seeds 0,1,2 \
	    --cycles 200 --relocate-goal-every 25 --out .tmp/bench.json

# ── Housekeeping ─────────────────────────────────────────────

clean:
	@echo "Cleaning build and cache artifacts..."
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['.gate-dist', 'build', 'python/phca.egg-info'])]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Done."

# ── Docker ───────────────────────────────────────────────────

DOCKER_TAG ?= phca:latest

docker-build:
	docker build -t $(DOCKER_TAG) .

docker-build-core:
	docker build --target phca-core -t phca:core .

docker-build-mujoco:
	docker build --target phca-mujoco -t phca:mujoco .

docker-run-test: docker-build-core
	docker compose run --rm test

docker-run-ci-local: docker-build-core
	docker compose run --rm ci-local

docker-bash: docker-build-mujoco
	docker compose run --rm bash

help:
	@grep -E '^[a-zA-Z0-9_-]+:' Makefile | sed 's/:.*//' | sort | head -40
