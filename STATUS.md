# PHCA v3.0 — Project Status

**Last updated:** 2026-07-03  
**Phase:** 8 LARGELY COMPLETE — Observatory replay/scrub hardening (Phase 7 observability stabilization complete; see docs/observability.md)  
**Decision log:** [DECISIONS.md](DECISIONS.md) (D-001 through D-107+; D-108+ referenced in code but not yet logged)

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
| 7 | Phase 7 observability stabilization | Complete |
| 8 | Phase 8 replay/scrub hardening | Largely complete |
| 9 | Phase 9 observability schema governance | Complete |

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

### Phase 7 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P7-1 | Reacher-v5 continuous control | ✅ Done | D-107 — 2D `ContinuousSpace`, Gate A PASS |
| P7-3 | Operational docs sync (round 1) | ✅ Done | 2026-07-03 |
| P7-4 | Observatory JSONL/replay/report parity | ✅ Done | `cognitive_panels.py`, `session_report.py`, integrity tests |

### Phase 7 — Open

| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| P7-2 | M3/M4 retention soak (late RSS ≤ 500 B/cyc) | 🔴 Open | D-108/D-109 in code; nightly late slope ~4817 B/cyc (2026-07-03) |

### Phase 8 — Largely Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P8-1 | Seek/scrub without panel desync | ✅ Done | `PlaybackClock` + `rebuild_histories()` on 11 views |
| P8-2 | Accurate replay banners | ✅ Done | Action, Flow, Phase Space, Memory panels |
| P8-3 | Transport + keyboard controls | ✅ Done | `_TransportBar`, Space/Arrows/Home/End/Esc |
| P8-4 | Frame/JSON immutability on scrub | ✅ Done | `test_dashboard_controller_scrub_500_jsonl_frames_no_mutation` |
| P8-5 | Phase 8 operational docs sync | ✅ Done | 2026-07-03 |

### Phase 9 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P9-1 | `schema_version` + migration policy | ✅ Done | JSONL `schema_version=1`, legacy v0 normalization, unknown/mixed schema checks |

### Phase 10+ — Backlog

| ID | Issue | Target |
|----|-------|--------|
| P10-1 | Full offline report parity with dashboard | Phase 10 |
| P11-1 | 3000+ cycle scrub without severe lag | Phase 11 |
| DO-6 | Log D-108+ in DECISIONS.md | Process debt |

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
| TC-4 | 10K-cycle nightly stress job | ✅ Done (Phase 6) | D-102 |
| TC-5 | Assumption validation experiments | ✅ Done (Phase 6) | D-101 |
| TC-6 | `pytest-mock` undeclared | ✅ Fixed | D-088 |

---

## Test Status

| Suite | Last run | Result |
|-------|----------|--------|
| Python (`make test-python`) | 2026-07-03 | **524 passed**, 0 errors |
| MuJoCo (`make test-mujoco`, `MUJOCO_GL=disabled`) | 2026-07-03 | **36 passed**, 0 errors |
| Monitoring only | 2026-07-03 | **232 passed** (`pytest python/phca/monitoring/tests/`) |
| **Total (both suites)** | 2026-07-03 | **560 passed**, 0 errors |
| CI Φ-IQ gate | 2026-07-03 | PASS (Overall 0.7403 ≥ floor 0.5486) |
| Causal behavior gate | 2026-07-04 | Mixed — L1 PASS; L2/L3 FAIL versus `greedy_observed` |
| Assumption validation `--ci` | 2026-07-03 | 4/4 PASS |
| `make nightly NIGHTLY_CYCLES=1000` | 2026-07-03 | **FAIL** — retention gate only (late RSS ~4817 B/cyc > 500); latency/violations/Φ-IQ pass |

**Failing tests:** None (unit/integration). Nightly orchestration fails on retention soak only.

---

## Benchmark Status

| Metric | Value | Baseline / target | Source |
|--------|-------|-------------------|--------|
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.7403** | ≥ 0.5486 floor — PASS | `logs/benchmark_report.json` |
| L0 / L1 / L2 / L3 Φ-IQ | 0.7032 / 0.7125 / 0.7773 / 0.7683 | L2 ≥ 0.5 — PASS | same |
| Causal behavior gate (GridWorld levels 1–3, 200 cyc × 5 seeds) | **Mixed** | L1 PASS; L2 FAIL (`0/4` vs observed greedy); L3 FAIL (`1/6` vs observed greedy) | `scripts/phca_causal_eval.py` |
| Nightly stress 1000-cyc | **FAIL** retention | late slope ~4817 B/cyc (threshold 500) | `logs/nightly_stress.json` |
| OOD calibration | monotonic, drop 0.71 | PASS | `logs/nightly_ood.json` |
| Assumption validation | A1/A3/A4/A5 PASS | `--ci` exit 0 | `logs/assumption_validation.json` |

---

## Execution Log (recent)

| Date | Fix | Outcome |
|------|-----|---------|
| 2026-07-04 | Three-level causal behavior gate | L1 passes; L2/L3 expose gap vs observed greedy |
| 2026-07-03 | Phase 8 operational doc sync | Observatory replay/scrub documented; 560 tests green |
| 2026-07-03 | phase7-audit-11 … phase-7-15 | Observatory hardening: scrub, banners, session_report, panel tests |
| 2026-07-03 | Zero-trust re-measurement (round 1) | Φ-IQ 0.7403; limitations/README/architecture updated |
| 2026-07-01 | Phase 7 / D-107: Reacher continuous | Gate A PASS |
| 2026-07-01 | Phase 6 complete (D-095–D-105) | Continuous Pendulum; OOD/assumption/nightly hardening |
