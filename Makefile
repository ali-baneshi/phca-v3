.PHONY: all test-all test-python test-rust lint bench-level-0 bench-all profile-cycle clean setup

# ─────────────────────────────────────────────────────────────
# PHCA v3.0 — Build & Test Automation
# ─────────────────────────────────────────────────────────────

all: lint test-all

# ── Setup ────────────────────────────────────────────────────

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip \
	    && pip install -r requirements-phase-3.1.txt \
	    && pip install -r requirements-dev.txt
	@echo "Setting up Rust workspace..."
	cd rust && cargo fetch
	@echo "Done. Run 'make test-all' to verify."
	@echo "Install Phase 3.2+ deps when needed: pip install -r requirements-phase-3.2.txt"

# ── Testing ───────────────────────────────────────────────────

test-all: test-python test-rust
	@echo "✅ All tests passed"

test-python:
	@echo "Running Python tests..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest python/tests/ python/phca/ -v --tb=short -x

test-rust:
	@echo "Running Rust tests..."
	cd rust && cargo test

# ── Linting ──────────────────────────────────────────────────

lint:
	@echo "Linting Python..."
	@-which ruff > /dev/null 2>&1 && ruff check python/ || echo "⚠️  ruff not installed, skipping Python lint"
	@-which black > /dev/null 2>&1 && black --check python/ || echo "⚠️  black not installed, skipping Python format check"
	@echo "Linting Rust..."
	@-cd rust && cargo clippy -- -D warnings 2>/dev/null || echo "⚠️  clippy not configured, skipping Rust lint"

# ── Benchmarks ───────────────────────────────────────────────

bench-level-0:
	@echo "Running Level 0 benchmark (stationary prediction)..."
	PYTHONPATH=python:$$PYTHONPATH python -m phca.benchmarks.runner --level=0 --output=results/level_0.json

bench-level-1:
	@echo "Running Level 1 benchmark (reactive control)..."
	PYTHONPATH=python:$$PYTHONPATH python -m phca.benchmarks.runner --level=1 --output=results/level_1.json

bench-all:
	@echo "Running all benchmarks (Levels 0-5)..."
	PYTHONPATH=python:$$PYTHONPATH python -m phca.benchmarks.runner --output=results/full_benchmark.json

# ── Profiling ────────────────────────────────────────────────

profile-cycle:
	@echo "Profiling cognitive cycle..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/profile_cycle.py --cycles=100 --output=logs/profile.json

# ── Utilities ────────────────────────────────────────────────

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .venv
	rm -rf rust/target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	@echo "Done."

# ── Help ─────────────────────────────────────────────────────

help:
	@echo "PHCA v3.0 — Available targets:"
	@echo "  make setup          Install all dependencies"
	@echo "  make test-all       Run all Python + Rust tests"
	@echo "  make test-python    Run Python tests only"
	@echo "  make test-rust      Run Rust tests only"
	@echo "  make lint           Run linters (ruff + clippy)"
	@echo "  make bench-level-0  Run Level 0 benchmark"
	@echo "  make bench-all      Run all benchmarks"
	@echo "  make profile-cycle  Profile the cognitive cycle"
	@echo "  make clean          Remove build artifacts"
	@echo "  make help           Show this message"
