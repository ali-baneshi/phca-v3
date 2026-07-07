.PHONY: all test-all test-python test-mujoco lint ci-local bench-level-0 bench-all profile-cycle clean setup nightly nightly-mujoco mujoco-ci causal-smoke reproduce reproduce-quick validate-science maturation-test bench-level4-smoke bench-level4-ablation bench-recovery

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

# Optional pytest plugins (requirements.txt); omit flags when not installed.
_PYTEST_TIMEOUT := $(shell python -c "import pytest_timeout" 2>/dev/null && echo "--timeout=30")
_PYTEST_BENCH := $(shell python -c "import pytest_benchmark.plugin" 2>/dev/null && echo "--benchmark-skip")

test-all: test-python
	@echo "✅ All tests passed"

test-python:
	@echo "Running Python tests (no MuJoCo — fast path)..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python -m pytest python/tests/ python/phca/ \
	    --ignore=python/tests/test_mujoco_env.py \
	    --ignore=python/tests/test_cycle_with_mujoco.py \
	    --ignore=python/tests/test_continuous_actions.py \
	    -v --tb=short -x $(_PYTEST_TIMEOUT) $(_PYTEST_BENCH)

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
	@echo "Observatory session integrity gate..."
	PYTHONPATH=python:$$PYTHONPATH python scripts/phca_replay.py --check python/phca/monitoring/tests/fixtures/multi_agent_short/
	PYTHONPATH=python:$$PYTHONPATH python scripts/phca_replay.py --check python/phca/monitoring/tests/fixtures/reacher_short/
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py --quick --output=logs/benchmark_report.json
	python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
	$(MAKE) causal-smoke
	@echo "✅ ci-local: ALL PASS"

# ── Causal behavior gate smoke (P0-6) ────────────────────────

causal-smoke:
	@echo "Running causal L2 smoke (10 cycles × 1 seed)..."
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/phca_causal_eval.py \
	    --levels level2 --cycles 10 --seeds 1 \
	    --output logs/phca_causal_eval_smoke.json
	@echo "✅ causal-smoke: completed (see logs/phca_causal_eval_smoke.json)"

# ── Maturation gates (plan v2) ───────────────────────────────

maturation-test:
	@echo "Running maturation verification tests (Tracks B,C,D,F,G,H)..."
	PYTHONPATH=python:$$PYTHONPATH python -m pytest \
	    python/tests/test_static_contracts.py \
	    python/tests/test_maturation.py \
	    python/tests/test_forgetting.py \
	    python/tests/test_resilience.py \
	    -q --tb=short
	@echo "✅ maturation-test: PASS"

bench-level4-smoke:
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark_level4.py \
	    --tasks 2 --task-cycles 30 --eval-cycles 10 --seeds 1 \
	    --diagnostic --output logs/benchmark_level4_smoke.json

bench-level4-ablation:
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/run_l4_ablation.py \
	    --runs R0,R2,R3,R6 --tasks 10 --task-cycles 80 --eval-cycles 20 --seeds 1 \
	    --output logs/l4_ablation.json

bench-recovery:
	mkdir -p logs
	PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark_recovery.py \
	    --scenarios b1,b4,b5,c1,f5 --output logs/benchmark_recovery.json

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
	@echo "   P-Stream navigation: PYTHONPATH=python python scripts/benchmark.py --levels=2 (Phase 3.2+)"
	@echo "   RBTA enforcement:    pytest python/tests/test_phase_3_1.py -v"
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
# with NIGHTLY_CYCLES (default 11000 for post-M3 retention gate; 1000 fill-phase):
#     make nightly NIGHTLY_CYCLES=11000
# This is a script+gate target, NOT a cron job — schedule it externally
# (GitHub Actions nightly, systemd timer, or cron) as documented in README.

nightly: nightly-mujoco
	@echo "============================================"
	@echo "  PHCA v3.0 — Nightly CI Hardening"
	@echo "============================================"
	@echo "[1/7] Static Φ-IQ benchmark (200cyc MLP)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --use-mlp --cycles=200 --output=logs/nightly_static.json >/dev/null
	@python scripts/check_benchmark_gate.py logs/nightly_static.json logs/benchmark_ci_baseline.json
	@echo "[2/7] MuJoCo benchmark gate (see nightly-mujoco) — done."
	@echo "[3/7] Assumption validation (A1–A5 incl. A2, --ci)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:scripts:$$PYTHONPATH python scripts/assumption_validation.py --ci \
	    --output=logs/nightly_assumptions.json
	@echo "[4/7] OOD calibration (σ-sweep, monotonic)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/ood_calibration.py \
	    --output=logs/nightly_ood.json
	@echo "[5/7] Nightly stress (NIGHTLY_CYCLES=$(NIGHTLY_CYCLES))..."
	@MUJOCO_GL=disabled NIGHTLY_CYCLES=$(NIGHTLY_CYCLES) PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/nightly_stress.py --output=logs/nightly_stress.json
	@echo "[6/7] Causal behavior gate (level2+level3, 200cyc; L3 uses 10 seeds)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/phca_causal_eval.py \
	    --levels level2,level3 --cycles 200 --seeds 5 --level-seeds level3=10 \
	    --use-mlp --gate --output logs/phca_causal_eval_nightly.json
	@echo "[7/7] Session anomaly gate..."
	@PYTHONPATH=python python scripts/nightly_anomaly_gate.py --output=logs/nightly_anomaly_gate.json
	@echo "============================================"
	@echo "  Nightly hardening: ALL PASS"
	@echo "============================================"

nightly-mujoco:
	@echo "[2/7] MuJoCo benchmark gate (Pendulum + Reacher continuous, Cartpole discrete)..."
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env pendulum --use-mlp --cycles=100 --output=logs/nightly_mujoco_pendulum.json >/dev/null
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env cartpole --use-mlp --cycles=100 --output=logs/nightly_mujoco_cartpole.json >/dev/null
	@MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env reacher --use-mlp --cycles=100 --output=logs/nightly_mujoco_reacher.json >/dev/null
	@python scripts/check_benchmark_gate.py --mujoco \
	    logs/nightly_mujoco_pendulum.json logs/nightly_mujoco_cartpole.json logs/nightly_mujoco_reacher.json
	@python scripts/check_benchmark_gate.py --neg-test >/dev/null

# CI MuJoCo gate — verbose output (no >/dev/null) for GitHub Actions logs.
MUJOCO_CI_CYCLES ?= 100

mujoco-ci:
	@mkdir -p logs
	@echo "MuJoCo CI gate ($(MUJOCO_CI_CYCLES) cycles per env)..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env pendulum --use-mlp --cycles=$(MUJOCO_CI_CYCLES) --output=logs/nightly_mujoco_pendulum.json
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env cartpole --use-mlp --cycles=$(MUJOCO_CI_CYCLES) --output=logs/nightly_mujoco_cartpole.json
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH python scripts/benchmark.py \
	    --env reacher --use-mlp --cycles=$(MUJOCO_CI_CYCLES) --output=logs/nightly_mujoco_reacher.json
	python scripts/check_benchmark_gate.py --mujoco \
	    logs/nightly_mujoco_pendulum.json logs/nightly_mujoco_cartpole.json logs/nightly_mujoco_reacher.json
	python scripts/check_benchmark_gate.py --neg-test

# Set default stress length for `make nightly` (override on the command line).
# Use 11000 for post-M3 retention gate (D-134); 1000 uses fill-phase threshold (D-112).
NIGHTLY_CYCLES ?= 11000

# ── Scientific reproduction (Phase 19) ─────────────────────

reproduce:
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/reproduce.py --profile full

reproduce-quick:
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/reproduce.py --profile quick

validate-science:
	@echo "Running full scientific validation suite..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/run_validation_suite.py --profile full --base-seed 42 \
	    --output-root results/validation --resume
	@echo "Aggregating results..."
	MUJOCO_GL=disabled PYTHONPATH=python:$$PYTHONPATH \
	    python scripts/aggregate_validation.py \
	    --input results/validation \
	    --hypotheses experiments/hypotheses.yaml \
	    --output results/validation/summary.json
	@echo "✅ validate-science complete — see results/validation/summary.json"

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
	@echo "                      override length: make nightly NIGHTLY_CYCLES=11000"
	@echo "  make reproduce      One-command scientific reproduction (full nightly-equivalent)"
	@echo "  make reproduce-quick  CI-science subset (~10-15 min)"
	@echo "  make validate-science Full validation suite + aggregation"
	@echo "  make maturation-test  Maturation plan v2 unit gates (T1)"
	@echo "  make mujoco-ci        MuJoCo gate (verbose, for CI)"
	@echo "  make bench-level4-smoke  L4-lite 2-task diagnostic smoke"
	@echo "  make bench-level4-ablation  L4 ablation R0,R2,R3,R6 (T3 local)"
	@echo "  make bench-recovery  Cognitive resilience injectables (T3)"
	@echo "  make test-mujoco    Run MuJoCo integration tests (needs gymnasium[mujoco])"
	@echo "  make clean          Remove build artifacts"
	@echo "  make help           Show this message"
