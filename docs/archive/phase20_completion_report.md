# PHCA v3.0 — Observatory Phase 20 Research Maturity Sign-off

> **Historical snapshot (2026-07-04).** Post–Phase 20 remediation added D-124–D-127 and +6 monitoring tests (360 monitoring / 715 total as of 2026-07-05). See [STATUS.md](../../STATUS.md) for current counts.

**Date:** 2026-07-04  
**Phase:** 20 — Product/Research Maturity (Observatory capstone)  
**Status:** **COMPLETE — Observatory Phases 7–20 signed off**  
**Decision log:** [DECISIONS.md](../../DECISIONS.md) D-108 through D-123

---

## 1. Mission recap

Phase 20 is the **Observatory capstone**, not a claim that the entire PHCA v3.0
blueprint is finished. Phases 7–19 delivered the cognitive observability stack:
JSONL recording, PyQt dashboard, replay/scrub, schema governance, report parity,
scrub performance, multi-session compare, anomaly detection, action explainability,
stable `phca.monitoring` API, production hardening, multi-agent timelines,
interactive query, and one-command scientific reproduction.

Phase 20 delivers:

| Workstream | Goal | Outcome |
| :--- | :--- | :--- |
| **A** Zero-trust audit | Re-run tests, gates, integrity checks; record actual numbers | **PASS** — see §2 |
| **B** Cross-doc consistency | Fix drift in living docs only | **DONE** — STATUS, README, architecture, satellites |
| **C** Architecture checklist | Verify Observatory design rules item-by-item | **PASS** — see §5 |
| **D** Sign-off artifact | Citable completion report + D-123 | **DONE** — this report |

**Scope boundary:** Observatory Phases 7–20 are **complete**. Whole-PHCA backlog
(M5 procedural memory, Φ-IQ L4–L5, grounding adapter, V/S modules, failure
recovery matrix, etc.) remains open per [IMPLEMENTATION_STATUS.md](../../IMPLEMENTATION_STATUS.md).

---

## 2. Final measured state (zero-trust audit, 2026-07-04)

Machine: Linux x86_64, Python 3.14.5 (venv), `MUJOCO_GL=disabled`.

| Check | Command | Result |
| :--- | :--- | :--- |
| Core tests | `make test-python` | **663 passed**, 1 skipped |
| MuJoCo tests | `make test-mujoco` | **36 passed** |
| Combined unique | both suites | **699 passed**, 1 skipped |
| Monitoring only | `pytest python/phca/monitoring/tests/` | **343 passed**, 1 skipped |
| Reproduce quick | `make reproduce-quick` | **ALL PASS** (26 s) |
| Replay reacher | `phca_replay --check` on `reacher_short` fixture | exit **0**, anomaly PASS |
| Replay grid | `phca_replay --check` on `grid_short` fixture | exit **0**, anomaly PASS |

### Reproduce-quick step summary (`logs/reproduce_report.json`)

| Step | Status | Notes |
| :--- | :--- | :--- |
| lint (ruff) | PASS | |
| test_python | PASS | 699 passed, 1 skipped |
| benchmark_quick_gate | PASS | Φ-IQ **0.8415** ≥ floor 0.5486 |
| causal_smoke | PASS | 10 cyc × 1 seed (smoke only; no `--gate`) |
| assumption_ci | PASS | A1–A5 |
| ood_calibration | PASS | monotonic, drop 0.7234 |
| anomaly_gate | PASS | 5/5 synthetic checks |

### Full nightly soak (not re-run this session)

Last verified PASS on 2026-07-04 per [STATUS.md](../../STATUS.md): `make nightly
NIGHTLY_CYCLES=10000` — post-cap retention late RSS ~1390 B/cyc ≤ 1600; causal
nightly L2+L3 gate PASS. Use `make reproduce` for full re-validation.

---

## 3. Observatory Phases 13–20 deliverables

| Phase | Gate | Decision |
| :--- | :--- | :--- |
| 13 Anomaly detection | `detect_session_anomalies()`, `--check`, nightly gate | D-116 |
| 14 Action explainability | `action_rationale`, Explain band, report metrics | D-117 |
| 15 Observability API | `phca.monitoring` `__all__`, CSV export, schema doc | D-118 |
| 16 Production hardening | Supervisor, crash recovery, session integrity | D-119 |
| 17 Multi-agent | `agent_id`, interleaved JSONL, aligned runner | D-120 |
| 18 Interactive query | `session_query.py`, `phca_query.py`, moment nav | D-121 |
| 19 Scientific validation | `make reproduce` / `reproduce_manifest.json` | D-122 |
| 20 Research maturity | This report + audit PASS | D-123 |

---

## 4. Gates table (Observatory + scientific)

| Gate | Script / command | Audit status |
| :--- | :--- | :--- |
| Φ-IQ regression (L0 quick) | `check_benchmark_gate.py` | **PASS** (0.8415) |
| Assumption validation A1–A5 | `assumption_validation.py --ci` | **PASS** |
| OOD calibration | `ood_calibration.py` | **PASS** |
| Session anomaly | `nightly_anomaly_gate.py` | **PASS** |
| Session integrity | `phca_replay.py --check` | **PASS** (fixtures) |
| Reproduce quick | `make reproduce-quick` | **PASS** |
| Causal L2+L3 (200×5, `--gate`) | `phca_causal_eval.py --gate` | PASS (2026-07-04 nightly) |
| MuJoCo benchmark | `check_benchmark_gate.py --mujoco` | PASS (2026-07-04 nightly) |
| Nightly stress 10k | `nightly_stress.py` | PASS (2026-07-04 nightly) |

---

## 5. Architecture checklist verification

From [PHCA_Cognitive_Observatory_Architecture.md](../PHCA_Cognitive_Observatory_Architecture.md):

| Question | Status | Evidence |
| :--- | :--- | :--- |
| Live/replay data same? | PASS | `PlaybackClock`, `rebuild_histories()`, `test_playback_store.py` |
| Live-only has banner? | PASS | Rollouts, explain band in `qt_dashboard.py` |
| Logic also in report? | PASS | `session_report.py`, `format_session_results_lines()` parity |
| Shared helper? | PASS | `cognitive_panels.py`, `session_anomalies.py`, `session_query.py` |
| JSONL not bloated? | PASS | `OBSERVABILITY_SCHEMA_VERSION=1`; query/export read-only |
| Frame not mutated? | PASS | Scrub immutability + `test_session_query.py` |
| Seek/scrub rebuild correct? | PASS | Phase 8; ≤2s/≤4s @ 3000 cycles |
| Incomplete session fails? | PASS | `phca_replay --check`; recovery uses `--allow-incomplete` |
| Behavioral test? | PASS | Per-phase monitoring test modules |
| Documentation updated? | PASS | Phase 20 doc sync |

---

## 6. Honest limitations

### Observatory scope

- PyQt5 **local desktop** UI only; no distributed multi-node Observatory cluster.
- Causal smoke (10 cyc) does not substitute for full `--gate` nightly eval.
- Legacy fixture sessions may show `explain: WARN` (pre-Phase-14 rationale).

### Whole PHCA blueprint (not signed off here)

From [IMPLEMENTATION_STATUS.md](../../IMPLEMENTATION_STATUS.md):

| Item | Status |
| :--- | :--- |
| M5 procedural memory | Not implemented |
| Φ-IQ L4–L5 | Not implemented |
| Grounding adapter (L0/L2) | Not implemented |
| V (VSA), S (script library) | Not implemented |
| Failure matrix A–F | Stub only |
| Forgetting rate (100 tasks) | Not measured |
| Criticality maintenance (Φ band) | Not implemented |
| Failure recovery ≥80% / 10 cyc | Not implemented |

### Audit fixes applied during Phase 20

- Ruff unused-import cleanup (Phases 17–18 drift).
- `frame_from_json` re-export in `render.py` (Phase 15 API move).
- `_flow_bottleneck_key`, `_overview_spike` re-exports in `qt_dashboard.py`.

---

## 7. Sign-off

**Observatory Phases 7–20 are COMPLETE.**

All zero-trust audit checks passed on 2026-07-04. The Cognitive Observatory is a
shipped, test-backed observability layer: live dashboard, JSONL replay, offline
reports, anomaly detection, explainability, multi-agent recording, interactive
query, and manifest-driven reproduction.

The **PHCA v3.0 core blueprint** remains partially implemented (M5, L4–L5,
grounding, resilience, etc.). Future work should not conflate Observatory maturity
with whole-architecture completion.

Optional release tag (not created in this phase): `v3.0-observatory`.
