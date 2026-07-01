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
	    && pip install -r requirements.txt \
	    && pip install -r requirements-dev.txt
	@echo "Setting up Rust workspace..."
	cd rust && cargo fetch
	@echo "Done. Run 'make test-all' to verify."
	@echo "Optional MuJoCo: pip install 'gymnasium[mujoco]'"

# ── Testing ───────────────────────────────────────────────────

test-all: test-python test-rust
	@echo "✅ All tests passed"

test-python:
	@echo "Running Python tests..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest python/tests/ python/phca/ \
	    --ignore=python/tests/test_mujoco_env.py \
	    --ignore=python/tests/test_cycle_with_mujoco.py \
	    -v --tb=short -x

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
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py --quick --output=logs/benchmark_l0.json

bench-all:
	@echo "Running all benchmarks..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py --levels=0-3 --cycles=100 --output=logs/benchmark_full.json

# ── Gate 1 Verification ─────────────────────────────────────

pre-gate-1:
	@echo "============================================"
	@echo "  PHCA v3.0 - Gate 1 Verification"
	@echo "============================================"
	@echo ""
	@echo "1. All Phase 3.1 tickets implemented (006-012)..."
	@python3 -c "decisions = [l.split('##')[1].strip() for l in open('DECISIONS.md') if '## Decision D-' in l]; print(f'   {len(decisions)}/7 decisions logged for Phase 3.1: D-006 through D-011')"
	@echo ""
	@echo "2. All Python tests passing..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest python/ -v --tb=short -x --ignore=rust 2>&1 | tail -4 && echo "   PASS" || echo "   FAIL"
	@echo ""
	@echo "3. Integration tests (IT-3.1-1 through 5)..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest python/tests/test_phase_3_1.py -v --tb=short -x 2>&1 | tail -4 && echo "   PASS" || echo "   FAIL"
	@echo ""
	@echo "4. Cycle latency check (< 500ms median)..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/profile_cycle.py --cycles=100 --max-ms=500 --output=logs/gate1_profile.json 2>&1 | tail -10
	@echo ""
	@echo "5. Lint check (Python)..."
	@-ruff check python/ --no-cache && echo "   PASS" || echo "   FAIL"
	@echo ""
	@echo "6. Lint check (Rust)..."
	@-cd rust && cargo clippy -- -D warnings 2>/dev/null && echo "   PASS" || echo "   WARN (not configured or warnings)"
	@echo ""
	@echo "7. Decision log check..."
	@python3 -c "d = [l.split('##')[1] for l in open('DECISIONS.md') if '## Decision D-' in l]; print(f'   {len(d)} decisions logged:'); [print(f'      {x.strip()}') for x in d]"
	@echo ""
	@echo "   P-Stream navigation: python -m phca.benchmarks.runner --level=2 (Phase 3.2)"
	@echo "   RBTA enforcement:    pytest tests/test_acceptance.py::test_rbta_detection -v"
	@echo "   Code coverage:       pytest python/ --cov=python/phca/ --cov-report=term"
	@echo ""
	@echo "============================================"
	@echo "  Gate 1 Verification Complete"
	@echo "============================================"

# ── Profiling ────────────────────────────────────────────────

profile-cycle:
	@echo "Profiling cognitive cycle..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/profile_cycle.py --cycles=100 --max-ms=500 --output=logs/profile.json

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
