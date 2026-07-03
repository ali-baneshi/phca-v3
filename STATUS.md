# PHCA v3.0 — Project Status

**Last updated:** 2026-07-03  
**Phase:** 7 IN PROGRESS — observability hardening + retention soak (Phase 6 complete; see docs/phase6_completion_report.md)  
**Decision log:** [DECISIONS.md](DECISIONS.md) (D-001 through D-107+)

---

## Audit Passes

| Pass | Focus | Status |
|------|-------|--------|
| 1 | Specification compliance | Complete |
| 2 | Architectural debt | Complete |
| 3 | Performance & scalability | Complete |
| 4 | Test coverage & resilience | Complete |
| 5 | Documentation & onboarding | Complete |
| 6 | Strategic v2.0 zero-trust re-audit | Complete (P0/P1/P2 closed; L2 bottleneck closed) |
| 7 | Phase 7 observability + retention | In progress |

---

## Issue Registry

### P0 — Resolved

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| TC-1 | CI missing requirements file | ✅ Fixed | D-074 — `requirements.txt` in CI/Makefile |
| DO-1 | No STATUS.md | ✅ Fixed | This file |
| P0-1 | MLP empowerment was a constant stub; closure report overclaimed "MC Dropout MI" | ✅ Fixed | D-077 — real MC-Dropout MI in `mlp.py`; closure report corrected |
| P0-2 | GAP-013 empowerment-blend inverted (D6 rose with prediction error) | ✅ Fixed | D-078 — `mdim.py:237` blend uses `1.0 - min(error,1.0)` |
| P0-3 | D-072/74/75/76 uncommitted; no .gitignore; .pyc tracked | ✅ Staged | D-079 — `.gitignore` added; pyc untracked; staged for maintainer commit |
| D-085 | structlog `filter_by_level` + `PrintLoggerFactory` crashed benchmark | ✅ Fixed | D-085 — removed incompatible processor; benchmark runs |

### P1 — Resolved / Documented

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| PS-1 | MLP hidden_dim 64 vs 128 | ✅ Fixed | D-072 — default restored to 128 |
| PS-2 | L2 Φ-IQ < 0.5 | ✅ Fixed | D-086 + D-087 — L2 0.477 → 0.778 (static 200-cyc) |
| TC-2 | MuJoCo breaks test-all | ✅ Fixed | D-074 — importorskip + ignore in CI |
| TC-3 | No CI Φ-IQ gate | ✅ Fixed | D-075 — `check_benchmark_gate.py` |
| AD-2 | README path drift | ✅ Fixed | D-076 |
| SC-1 | Grounding adapter missing | Deferred | limitations.md |

### P1 — Phase 7 (in progress)

| ID | Issue | Status | Target |
|----|-------|--------|--------|
| P7-1 | Reacher-v5 continuous control | ✅ Done | D-107 — 2D `ContinuousSpace`, Gate A PASS |
| P7-2 | M3/M4 retention soak (late RSS ≤ 500 B/cyc) | 🔴 Open | `make nightly` fails retention gate on this machine (~4650 B/cyc late slope) |
| P7-3 | Operational docs sync | ✅ Done | 2026-07-03 — README, STATUS, architecture, limitations, observability, SETUP, CONTRIBUTING |
| P7-4 | Observatory JSONL/replay/report parity | In progress | `cognitive_panels.py` shared near-bound helpers; integrity tests |

### P2 — Backlog (scheduled)

| ID | Issue | Target phase | Owner |
|----|-------|--------------|-------|
| AD-3 | Unify benchmark entry points | 4.1 | — |
| AD-4 | M5 procedural memory | 4.3 | — |
| AD-5 | EnvironmentProtocol goal distance hook | 4.1 | — |
| SC-2 | Pareto over config vectors | 4.1 | — |
| SC-3 | PID freeze by gain timescale | 4.1 | — |
| SC-4 | True H(WM) disruption detection | 4.2 | — |
| PS-3 | MC Dropout + empowerment FLOP cap | 4.1 | (partially addressed by D-077 cap) |
| PS-4 | Mid-cycle RBTA preemption | 4.2 | — |
| TC-4 | 10K-cycle nightly stress job | ✅ Done (Phase 6) | D-102 — `scripts/nightly_stress.py` + `make nightly` |
| TC-5 | Assumption validation experiments | ✅ Done (Phase 6) | D-101 — `scripts/assumption_validation.py --ci` (A1/A3/A4/A5) |
| TC-6 | `pytest-mock` undeclared | ✅ Fixed | D-088 — `pytest-mock>=3.12` in `requirements-dev.txt` |

---

## Test Status

| Suite | Last run | Result |
|-------|----------|--------|
| Python (`make test-python`) | 2026-07-03 | **524 passed**, 0 errors (core + monitoring; MuJoCo files ignored) |
| MuJoCo (`make test-mujoco`, `MUJOCO_GL=disabled`) | 2026-07-03 | **36 passed**, 0 errors |
| Monitoring only | 2026-07-03 | **225 passed** (`pytest python/phca/monitoring/tests/`) |
| **Total (both suites)** | 2026-07-03 | **560 passed**, 0 errors |
| Rust | N/A (workspace removed D-084) | — |
| CI Φ-IQ gate | 2026-07-03 | PASS (Overall 0.7403 ≥ floor 0.5486) |
| Assumption validation `--ci` | 2026-07-03 | 4/4 PASS |
| `make nightly NIGHTLY_CYCLES=1000` | 2026-07-03 | **FAIL** — retention gate (late RSS slope ~4650 B/cyc > 500 B/cyc); other stages pass |

**Notes:**
- Phase 7 / D-107: Reacher-v5 continuous (+3 net MuJoCo tests vs Phase 6 baseline).
- Monitoring suite (225 tests) is included in `make test-python` but reported separately for clarity.
- Nightly stress now gates on **late-half** RSS slope (`LEAK_SLOPE_LATE = 500 B/cyc` in `scripts/nightly_stress.py`).

**Failing tests:** None (unit/integration). Nightly orchestration fails on retention soak only.

---

## Benchmark Status

| Metric | Value | Baseline / target | Source |
|--------|-------|-------------------|--------|
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.7403** | ≥ 0.5486 floor — PASS | `logs/benchmark_report.json` (2026-07-03) |
| L0 / L1 / L2 / L3 Φ-IQ | 0.7032 / 0.7125 / 0.7773 / 0.7683 | L2 ≥ 0.5 — PASS | same |
| Pendulum continuous 100-cyc | PASS | D-106: 7.1 ms, 0 violations | `logs/phase7_a0_pendulum.json` |
| Reacher continuous 100-cyc | PASS | D-107: 4.4 ms, 0 violations, error 105.7→8.4 | `logs/phase7_a1_reacher.json` |
| Cartpole 100-cyc (discrete) | PASS | 5.9 ms, 0 violations | `logs/phase7_a1_cartpole.json` |
| Dynamic every-75 L2 | 0.6444 | ≥ 0.50 — PASS (validated cadence) | `logs/phase6_baseline_dyn75.json` |
| OOD calibration | monotonic, drop 0.71 | drop > 0 — PASS | `logs/nightly_ood.json` |
| Assumption validation | A1/A3/A4/A5 PASS | `--ci` exit 0 | `logs/assumption_validation.json` |
| Nightly stress 1000-cyc | **FAIL** retention | late slope ~4650 B/cyc (threshold 500) | `logs/nightly_stress.json` |

---

## Execution Log (recent)

| Date | Fix | Outcome |
|------|-----|---------|
| 2026-07-03 | Zero-trust re-measurement + operational doc sync | 560 tests; Φ-IQ 0.7403; limitations/README/architecture updated |
| 2026-07-01 | Phase 7 / D-106: baseline re-measurement | 332 tests; Φ-IQ 0.7416; `make nightly` exit 0 (pre-tightened retention gate) |
| 2026-07-01 | Phase 7 / D-107: Reacher continuous + Gate A | 335 tests reported; Reacher 2D continuous PASS |
| 2026-07-01 | Phase 6 complete (D-095–D-105) | Continuous Pendulum; OOD/assumption/nightly hardening |
