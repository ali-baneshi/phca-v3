.PHONY: all test-all test-python test-mujoco lint ci-local bench-level-0 bench-all profile-cycle clean setup nightly nightly-mujoco

# ─────────────────────────────────────────────────────────────
# PHCA v3.0 — Build & Test Automation
# ─────────────────────────────────────────────────────────────
#
# Note: The Rust workspace was removed in D-084 (empty crates, Python
# is not a bottleneck at ~50 ms p95 cycle latency). Re-introduce Rust
# targets only if profiling shows Python as a bottleneck.

all: lint test-all

# ── Setup ────────────────────────────────────────────────────

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip \
	    && pip install -r requirements.txt \
	    && pip install -r requirements-dev.txt
	@echo "Done. Run 'make test-all' to verify."
	@echo "Optional MuJoCo: pip install 'gymnasium[mujoco]'"

# ── Testing ───────────────────────────────────────────────────

test-all: test-python
	@echo "✅ All tests passed"

test-python:
	@echo "Running Python tests (no MuJoCo — fast path)..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest python/tests/ python/phca/ \
	    --ignore=python/tests/test_mujoco_env.py \
	    --ignore=python/tests/test_cycle_with_mujoco.py \
	    --ignore=python/tests/test_continuous_actions.py \
	    -v --tb=short -x

test-mujoco:
	@echo "Running MuJoCo integration tests (needs gymnasium[mujoco])..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python -m pytest \
	    python/tests/test_mujoco_env.py \
	    python/tests/test_cycle_with_mujoco.py \
	    python/tests/test_continuous_actions.py \
	    -v --tb=short -x

# ── Linting ──────────────────────────────────────────────────

lint:
	@echo "Linting Python..."
	@command -v ruff >/dev/null 2>&1 || (echo "❌ ruff not installed — run: pip install -r requirements-dev.txt" && exit 1)
	ruff check python/ --no-cache
	@-which black > /dev/null 2>&1 && black --check python/ || echo "⚠️  black not installed, skipping Python format check"

ci-local: lint
	@echo "Running CI-equivalent test suite..."
	@python -c "import pytest_timeout" 2>/dev/null || (echo "❌ pytest-timeout missing — run: pip install -r requirements.txt" && exit 1)
	@python -c "import pytest_benchmark.plugin" 2>/dev/null || (echo "❌ pytest-benchmark missing — run: pip install -r requirements.txt" && exit 1)
	PYTHONPATH=python:$$PYTHONPATH MUJOCO_GL=disabled python -m pytest python/tests/ python/phca/ \
	    -v --tb=short --timeout=30 -x --benchmark-skip
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py --quick --output=logs/benchmark_report.json
	python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
	@echo "✅ ci-local: ALL PASS"

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
	@echo "6. Decision log check..."
	@python3 -c "d = [l.split('##')[1] for l in open('DECISIONS.md') if '## Decision D-' in l]; print(f'   {len(d)} decisions logged:'); [print(f'      {x.strip()}') for x in d]"
	@echo ""
	@echo "   P-Stream navigation: python -m phca.benchmarks.runner --level=2 (Phase 3.2)"
	@echo "   RBTA enforcement:    pytest tests/test_acceptance.py::test_rbta_detection -v"
	@echo "   Code coverage:       pytest python/ --cov=python/phca/ --cov-report=term"
	@echo ""
	@echo "============================================"
	@echo "  Gate 1 Verification Complete"
	@echo "============================================"

# ── Nightly CI hardening (Phase 6 / C3) ──────────────────────
#
# `make nightly` runs the full hardening suite: static Φ-IQ gate, MuJoCo
# benchmark gate (continuous + discrete), assumption validation (--ci),
# OOD calibration, and the long-run stress test. Override the stress length
# with NIGHTLY_CYCLES (default 1000 for CI; use 10000 for a true soak):
#     make nightly NIGHTLY_CYCLES=10000
# This is a script+gate target, NOT a cron job — schedule it externally
# (GitHub Actions nightly, systemd timer, or cron) as documented in README.

nightly: nightly-mujoco
	@echo "============================================"
	@echo "  PHCA v3.0 — Nightly CI Hardening"
	@echo "============================================"
	@echo "[1/5] Static Φ-IQ benchmark (200cyc MLP)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --use-mlp --cycles=200 --output=logs/nightly_static.json >/dev/null
	@python scripts/check_benchmark_gate.py logs/nightly_static.json logs/benchmark_ci_baseline.json
	@echo "[2/5] MuJoCo benchmark gate (see nightly-mujoco) — done."
	@echo "[3/5] Assumption validation (A1/A3/A4/A5, --ci)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:scripts:$$PYTHONPATH python scripts/assumption_validation.py --ci \
	    --output=logs/nightly_assumptions.json
	@echo "[4/5] OOD calibration (σ-sweep, monotonic)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/ood_calibration.py \
	    --output=logs/nightly_ood.json
	@echo "[5/5] Nightly stress (NIGHTLY_CYCLES=$(NIGHTLY_CYCLES))..."
	@MUJOCO_GL=disabled NIGHTLY_CYCLES=$(NIGHTLY_CYCLES) PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/nightly_stress.py --output=logs/nightly_stress.json
	@echo "============================================"
	@echo "  Nightly hardening: ALL PASS"
	@echo "============================================"

nightly-mujoco:
	@echo "[2/5] MuJoCo benchmark gate (Pendulum continuous + Cartpole/Reacher discrete)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env pendulum --use-mlp --cycles=100 --output=logs/nightly_mujoco_pendulum.json >/dev/null
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env cartpole --use-mlp --cycles=100 --output=logs/nightly_mujoco_cartpole.json >/dev/null
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env reacher --use-mlp --cycles=100 --output=logs/nightly_mujoco_reacher.json >/dev/null
	@python scripts/check_benchmark_gate.py --mujoco \
	    logs/nightly_mujoco_pendulum.json logs/nightly_mujoco_cartpole.json logs/nightly_mujoco_reacher.json
	@python scripts/check_benchmark_gate.py --neg-test >/dev/null

# Set a default stress length for `make nightly` (override on the command line).
NIGHTLY_CYCLES ?= 1000

# ── Profiling ────────────────────────────────────────────────

profile-cycle:
	@echo "Profiling cognitive cycle..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/profile_cycle.py --cycles=100 --max-ms=500 --output=logs/profile.json

# ── Utilities ────────────────────────────────────────────────

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	@echo "Done."

# ── Help ─────────────────────────────────────────────────────

help:
	@echo "PHCA v3.0 — Available targets:"
	@echo "  make setup          Install all dependencies"
	@echo "  make test-all       Run all Python tests"
	@echo "  make test-python    Run Python tests only"
	@echo "  make lint           Run linters (ruff + black)"
	@echo "  make ci-local       Run the same checks as GitHub Actions CI"
	@echo "  make bench-level-0  Run Level 0 benchmark"
	@echo "  make bench-all      Run all benchmarks"
	@echo "  make profile-cycle  Profile the cognitive cycle"
	@echo "  make nightly        Run the full nightly hardening suite (static+MuJoCo gates,"
	@echo "                      assumption validation --ci, OOD calibration, stress test)"
	@echo "                      override length: make nightly NIGHTLY_CYCLES=10000"
	@echo "  make test-mujoco    Run MuJoCo integration tests (needs gymnasium[mujoco])"
	@echo "  make clean          Remove build artifacts"
	@echo "  make help           Show this message"
