# PHCA v3.0 — Project Status

**Last updated:** 2026-07-04  
**Phase:** 20 COMPLETE — Observatory Phases 7–20 signed off (D-108–D-123)  
**Decision log:** [DECISIONS.md](DECISIONS.md) (D-001 through D-123)

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
| 8 | Phase 8 replay/scrub hardening | Complete |
| 9 | Phase 9 observability schema governance | Complete |
| 10 | Phase 10 offline report parity | Complete |
| 11 | Phase 11 large-session scrub performance | Complete |
| 12 | Phase 12 multi-session comparison | Complete |
| 13 | Phase 13 anomaly detection | Complete |
| 14 | Phase 14 action explainability | Complete |
| 15 | Phase 15 observability API | Complete |
| 16 | Phase 16 production hardening | Complete |
| 17 | Phase 17 multi-agent Observatory | Complete |
| 18 | Phase 18 interactive query | Complete |
| 19 | Phase 19 scientific validation | Complete |
| 20 | Phase 20 research maturity sign-off | Complete |

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
| P7-2 | M3/M4 retention soak (late RSS ≤ 500 B/cyc @ 10k) | ✅ Done | D-112 phase-aware gate; fill-phase ≤5000 B/cyc @ 1k |

### Phase 8 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P8-1 | Seek/scrub without panel desync | ✅ Done | `PlaybackClock` + `rebuild_histories()` on 10 canvas views (7 tabs) |
| P8-2 | Accurate replay banners | ✅ Done | Action, Flow, Phase Space, Memory panels |
| P8-3 | Transport + keyboard controls | ✅ Done | `_TransportBar`, Space/Arrows/Home/End/Esc |
| P8-4 | Frame/JSON immutability on scrub | ✅ Done | `test_dashboard_controller_scrub_500_jsonl_frames_no_mutation` |
| P8-5 | Phase 8 operational docs sync | ✅ Done | 2026-07-03 |

### Phase 9 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P9-1 | `schema_version` + migration policy | ✅ Done | JSONL `schema_version=1`, legacy v0 normalization, unknown/mixed schema checks |

### Phase 10 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P10-1 | Full offline report parity with dashboard | ✅ Done | `format_session_results_lines()` parity test |

### Phase 11 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P11-1 | 3000+ cycle scrub without severe lag | ✅ Done | scrub budget tests ≤2s/≤4s |

### Phase 12 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P12-1 | Multi-session report comparison | ✅ Done | `phca_replay.py --compare` + `compare_session_reports()` (D-115) |

### Phase 13 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P13-1 | Anomaly detection (spike/drift/leak/goal instability) | ✅ Done | `session_anomalies.py`, `session_report.anomalies`, `phca_replay.py --check` + `--anomaly-strict`, `nightly_anomaly_gate.py` (D-116) |

### Phase 14 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P14-1 | Action explainability (`action_rationale` + Explain band) | ✅ Done | `_finalize_action_rationale()`, `action_explain.py`, Action tab explain band, `explain_metrics` in session_report (D-117) |

### Phase 15 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P15-1 | Stable `phca.monitoring` public API + schema doc + CSV export | ✅ Done | `__init__.py` `__all__`, `session_io.py`, `export.py`, `docs/observability_api.md`, Qt-free `overview_narrative` + `belief_projection` extraction (D-118) |

### Phase 16 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P16-1 | Subprocess supervisor + crash recovery + session integrity | ✅ Done | `phca_observatory_supervisor.py`, `session_recovery.py`, `meta.status` lifecycle, `.latest` pointer, `phca_replay --recover`, `test_session_recovery.py` (D-119) |

### Phase 17 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P17-1 | Multi-agent Observatory (`agent_id` + synced timeline) | ✅ Done | `agent_id`/`agent_label`/`timeline_step` on frames; single JSONL interleaved recording; `--agents` aligned runner; dashboard agent selector + per-agent scrub; per-agent `session_report` + `compare_all_agents`; `phca_replay --check` per-agent contiguity; `test_multi_agent.py` (D-120) |

### Phase 18 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P18-1 | Interactive analysis (`phca_query.py` + dashboard filters) | ✅ Done | `session_query.py` shared engine; CLI plain/JSON/export; transport-bar moment filter + prev/next nav in replay/review; `test_session_query.py` (D-121) |

### Phase 19 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P19-1 | Scientific validation (`make reproduce` one-command manifest) | ✅ Done | `reproduce_manifest.json` quick/full profiles; `scripts/reproduce.py` driver + `logs/reproduce_report.json`; `make reproduce` / `make reproduce-quick`; `test_reproduce.py` (D-122) |

### Phase 20 — Complete

| ID | Issue | Status | Resolution |
|----|-------|--------|------------|
| P20-1 | Research maturity sign-off (audit + archive report) | ✅ Done | Zero-trust audit; `docs/archive/phase20_completion_report.md`; architecture checklist; cross-doc sync; D-123 |

See [docs/PHCA_Cognitive_Observatory_Architecture.md](docs/PHCA_Cognitive_Observatory_Architecture.md).
Copy-paste planning prompts: [docs/observatory_phase_prompts/](docs/observatory_phase_prompts/).

### Documentation / Decisions

| ID | Issue | Status | Resolution |
|----|-------|--------|
| DO-6 | Log D-108+ in DECISIONS.md | ✅ Done | D-108/D-109/D-112/D-113/D-114 logged |
| DO-7 | README + living docs sync (Phase 11) | ✅ Done | 2026-07-04 |

### P2 — Backlog (scheduled)

| ID | Issue | Target phase | Owner |
|----|-------|--------------|-------|
| AD-3 | Unify benchmark entry points | 4.1 | **Done** — `runner.py` deprecated; use `scripts/benchmark.py` |
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
| Python (`make test-python`) | 2026-07-04 | **663 passed**, 1 skipped |
| MuJoCo (`make test-mujoco`, `MUJOCO_GL=disabled`) | 2026-07-04 | **36 passed**, 0 errors |
| Monitoring only | 2026-07-04 | **343 passed**, 1 skipped |
| **Total (both suites)** | 2026-07-04 | **699 passed**, 1 skipped |
| Reproduce quick (`make reproduce-quick`) | 2026-07-04 | **ALL PASS** (26 s) |
| CI Φ-IQ gate | 2026-07-04 | PASS (quick Φ-IQ 0.8415 ≥ floor 0.5486) |
| Causal behavior gate | 2026-07-04 | **PASS** — L1/L2/L3 vs gated controls (`logs/phca_causal_eval.json`) |
| Assumption validation `--ci` | 2026-07-04 | **5/5 PASS** (A1–A5 incl. A2 temporal order) |
| `make nightly NIGHTLY_CYCLES=10000` | 2026-07-04 | **PASS** — post-cap retention (late RSS ~1390 B/cyc ≤ 1600); causal nightly gate PASS |

**Failing tests:** None (unit/integration).

---

## Benchmark Status

| Metric | Value | Baseline / target | Source |
|--------|-------|-------------------|--------|
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.7403** | ≥ 0.5486 floor — PASS | `logs/benchmark_report.json` |
| L0 / L1 / L2 / L3 Φ-IQ | 0.7032 / 0.7125 / 0.7773 / 0.7683 | L2 ≥ 0.5 — PASS | same |
| Causal behavior gate (GridWorld levels 1–3, 200 cyc × 5 seeds) | **PASS** | L1/L2/L3 vs gated controls | `logs/phca_causal_eval.json` |
| Nightly stress 1000-cyc | **PASS** fill-phase | late slope ~2195 B/cyc (threshold 5000) | `logs/nightly_stress.json` |
| Nightly stress 10000-cyc | **PASS** post-cap | late slope ~1390 B/cyc (threshold 1600, D-113) | `logs/nightly_stress.json` |
| Assumption validation | A1–A5 PASS | `--ci` exit 0 | `logs/assumption_validation.json` |

---

## Execution Log (recent)

| Date | Fix | Outcome |
|------|-----|---------|
| 2026-07-04 | Phase 20 Observatory sign-off (D-123) | Zero-trust audit; 699 tests; reproduce-quick PASS; phase20_completion_report.md |
| 2026-07-04 | Phase 19 scientific reproduction (D-122) | `make reproduce` manifest + driver |
| 2026-07-04 | Phase 13 anomaly detection (D-116) | Shared spike/drift/leak/goal flags; `--check` + nightly gate; 639 tests green |
| 2026-07-04 | phase-10&11 + docs sync | Report parity, scrub perf, README/STATUS aligned; pgmpy pin (D-114) |
| 2026-07-04 | L3 causal coverage fix + retention gate (D-112) | L1/L2/L3 causal PASS; phase-aware nightly retention |
| 2026-07-04 | Three-level causal behavior gate | L1 passes; L2/L3 expose gap vs observed greedy (pre-fix) |
| 2026-07-03 | Phase 8 operational doc sync | Observatory replay/scrub documented; 560 tests green |
| 2026-07-03 | phase7-audit-11 … phase-7-15 | Observatory hardening: scrub, banners, session_report, panel tests |
| 2026-07-03 | Zero-trust re-measurement (round 1) | Φ-IQ 0.7403; limitations/README/architecture updated |
| 2026-07-01 | Phase 7 / D-107: Reacher continuous | Gate A PASS |
| 2026-07-01 | Phase 6 complete (D-095–D-105) | Continuous Pendulum; OOD/assumption/nightly hardening |
