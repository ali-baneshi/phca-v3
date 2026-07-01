# PHCA v3.0 — Project Status

**Last updated:** 2026-07-01  
**Phase:** 4 Weeks 1–4 hardening (W1–W2 complete — long-run + TC-6 + MuJoCo CI; W3 in progress)  
**Decision log:** [DECISIONS.md](DECISIONS.md) (D-001 through D-089)

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
| PS-2 | L2 Φ-IQ < 0.5 | ✅ Fixed | D-086 + D-087 — metric ceiling aligned with L0/L1; PGA ramp onset lowered; L2 0.477 → 0.764 |
| TC-2 | MuJoCo breaks test-all | ✅ Fixed | D-074 — importorskip + ignore in CI |
| TC-3 | No CI Φ-IQ gate | ✅ Fixed | D-075 — `check_benchmark_gate.py` |
| AD-2 | README path drift | ✅ Fixed | D-076 |
| SC-1 | Grounding adapter missing | Deferred | limitations.md — Phase 4.2 |

### P1 — In Progress (Strategic v2.0)

| ID | Issue | Status | Target |
|----|-------|--------|--------|
| P1-4 / DO-5 | F1-F19 dead-code sweep (grounding_level, schema_version, TSPL-E/S) | ✅ Done | D-083 — engine grounding_level=2 dead path removed; schema_version/TSPL-E/S already cleaned in Phase 3.3; StateVector.grounding_level field kept (harmless, documented) |
| P1-2 / G-017 | MLP online vs replay double-learning | ✅ Done | D-081 — hybrid schedule documented; LR unified to lr*0.5; early-return prevents same-cycle double-learning |
| P1-1 / G-002 | MLP confidence = exp(-MSE), not probabilistic | ✅ Done | D-080 — epistemic MC-Dropout variance already in predict(); documented aleatoric/epistemic split; OOD confidence-drop test added |
| P1-3 / G-010 | SQLite M3 hardening (VACUUM, checkpoint, integrity) | ✅ Done | D-082 — VACUUM + integrity_check already present; added wal_checkpoint(TRUNCATE) on close + in-memory fallback on corrupt/init failure |
| P1-5 / AD-1 | Empty Rust workspace decision | ✅ Done | D-084 — deleted empty rust/ + Cargo + 221MB target/; updated SETUP/Makefile/CI |

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
| TC-4 | 10K-cycle nightly stress job | 4.1 | — |
| TC-5 | Assumption validation experiments | 4.1 | — |
| TC-6 | `pytest-mock` undeclared — `test_engine.py` errors on missing `mocker` fixture | ✅ Fixed | D-088 — `pytest-mock>=3.12` added to `requirements-dev.txt`; 6 errors cleared |

---

## Test Status

| Suite | Last run | Result |
|-------|----------|--------|
| Python core | 2026-07-01 | **299 passed**, 0 errors (TC-6 closed via D-088) |
| MuJoCo (`MUJOCO_GL=disabled`) | 2026-07-01 | **23 passed**, 0 errors |
| Total | 2026-07-01 | **322 passed**, 0 errors |
| Rust | N/A (workspace removed D-084) | — |
| CI gate script | 2026-07-01 | PASS (1000-cyc Φ-IQ 0.7919 ≥ floor 0.5486) |

**Notes:**
- TC-6 closed: `pytest-mock` installed + declared; the 6 `mocker` fixture errors in `test_engine.py` are gone.
- 1000-cycle long-run probe (`logs/longrun_probe.json`): RSS +3.23% (bounded, no leak); latency 16.6ms→57ms plateau (D-081 warm-up→replay transition, not creep); p95 max 66ms << 500ms A1 bound.
- W4 optimisation target identified: steady-state replay path is 3.4× slower than warm-up.

**Failing tests:** None.

---

## Benchmark Status

| Metric | Value | Baseline / target | Source |
|--------|-------|-------------------|--------|
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.740** | ≥ 0.5 (and ≥ 0.635 = 95% of 0.668) | `logs/benchmark_final.json` |
| L2 Φ-IQ (MLP, 200 cyc) | **0.764** | ≥ 0.5 — PASS (was 0.477) | `logs/benchmark_final.json` |
| L2 goal_rate | 0.95 | — | same |
| CI quick Φ-IQ | 0.577 | ≥ 0.548 (95% floor) | `logs/benchmark_ci_baseline.json` |
| Gate result (post-P0) | PASS | report 0.6861 ≥ floor 0.5486 | `check_benchmark_gate.py` |

---

## Execution Log

| Date | Fix | Outcome |
|------|-----|---------|
| 2026-07-01 | STATUS.md created | DO-1 complete |
| 2026-07-01 | CI/Makefile + MuJoCo isolation | D-074; 284 tests pass |
| 2026-07-01 | MLP hidden_dim=128 | D-072 |
| 2026-07-01 | L2 benchmark 500 cycles | D-073; Φ-IQ 0.477 |
| 2026-07-01 | CI Φ-IQ gate + baseline | D-075 |
| 2026-07-01 | README + architecture sync | D-076 |
| 2026-07-01 | GAP-013 empowerment-blend inversion fixed | D-078; 30 MDIM tests pass |
| 2026-07-01 | MLP empowerment MC-Dropout MI (replaces constant stub) | D-077; 22 MLP tests pass; Φ-IQ 0.668→0.686 |
| 2026-07-01 | structlog filter_by_level removed (unblocks benchmark) | D-085; benchmark + gate PASS |
| 2026-07-01 | .gitignore + pyc untrack + staging | D-079; staged for maintainer commit |
| 2026-07-01 | L2 adaptation_speed metric aligned with L0/L1 (ceiling fix) | D-086; L2 Φ-IQ 0.477→0.764; overall 0.669→0.740; gate PASS |
| 2026-07-01 | PGA ramp onset lowered (50→150 cycles); Iteration B (goal randomization) tested + rejected | D-087; L0/L1/L3 unchanged; 293 tests pass |
| 2026-07-01 | W1: TC-6 closed (pytest-mock) + 1000-cycle stability probe | D-088; 322 passed 0 errors; Φ-IQ 0.7919 @ 1000cyc; RSS +3.23% no leak |
| 2026-07-01 | W2: MuJoCo into CI + requirements-mujoco.txt + `--env` flag | D-089; Cartpole+Pendulum PASS C1/C3/C4/C6; 16ms latency, error 8.4→0.3 |

