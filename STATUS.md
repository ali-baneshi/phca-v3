# PHCA v3.0 — Project Status

**Last updated:** 2026-07-01  
**Phase:** 4 (Principal Architect audit — execution complete)  
**Decision log:** [DECISIONS.md](DECISIONS.md) (D-001 through D-076)

---

## Audit Passes

| Pass | Focus | Status |
|------|-------|--------|
| 1 | Specification compliance | Complete |
| 2 | Architectural debt | Complete |
| 3 | Performance & scalability | Complete |
| 4 | Test coverage & resilience | Complete |
| 5 | Documentation & onboarding | Complete |

---

## Issue Registry

### P0 — Resolved

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| TC-1 | CI missing requirements file | ✅ Fixed | D-074 — `requirements.txt` in CI/Makefile |
| DO-1 | No STATUS.md | ✅ Fixed | This file |

### P1 — Resolved / Documented

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| PS-1 | MLP hidden_dim 64 vs 128 | ✅ Fixed | D-072 — default restored to 128 |
| PS-2 | L2 Φ-IQ < 0.5 | ⚠️ Documented | D-073 — 0.477 @ 500 cycles; Phase 4.1 target |
| TC-2 | MuJoCo breaks test-all | ✅ Fixed | D-074 — importorskip + ignore in CI |
| TC-3 | No CI Φ-IQ gate | ✅ Fixed | D-075 — `check_benchmark_gate.py` |
| AD-2 | README path drift | ✅ Fixed | D-076 |
| SC-1 | Grounding adapter missing | Deferred | limitations.md — Phase 4.2 |

### P2 — Backlog (scheduled)

| ID | Issue | Target phase | Owner |
|----|-------|--------------|-------|
| AD-1 | Empty Rust workspace | 4.1 | — |
| AD-3 | Unify benchmark entry points | 4.1 | — |
| AD-4 | M5 procedural memory | 4.3 | — |
| AD-5 | EnvironmentProtocol goal distance hook | 4.1 | — |
| SC-2 | Pareto over config vectors | 4.1 | — |
| SC-3 | PID freeze by gain timescale | 4.1 | — |
| SC-4 | True H(WM) disruption detection | 4.2 | — |
| PS-3 | MC Dropout + empowerment FLOP cap | 4.1 | — |
| PS-4 | Mid-cycle RBTA preemption | 4.2 | — |
| TC-4 | 10K-cycle nightly stress job | 4.1 | — |
| TC-5 | Assumption validation experiments | 4.1 | — |
| DO-5 | F1–F19 dead code sweep | 4.1 | — |

---

## Test Status

| Suite | Last run | Result |
|-------|----------|--------|
| Python core | 2026-07-01 | **284 passed**, 2 skipped (MuJoCo) |
| Rust | Not re-run (empty `.rs` sources) | N/A |
| CI gate script | 2026-07-01 | PASS |

**Failing tests:** None

---

## Benchmark Status

| Metric | Value | Baseline / target | Source |
|--------|-------|-------------------|--------|
| Overall Φ-IQ (4-level MLP, 200 cyc) | 0.668 | ≥ 0.5 | `logs/benchmark_phase4_fix3.json` |
| L2 Φ-IQ (MLP, 500 cyc) | **0.477** | ≥ 0.5 (exception) | `logs/benchmark_l2_ps1.json` |
| L2 goal_rate | 0.936 | — | same |
| CI quick Φ-IQ | 0.577 | ≥ 0.548 (95% floor) | `logs/benchmark_ci_baseline.json` |

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
