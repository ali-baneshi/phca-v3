# PHCA v3.0 — Documentation Update Plan

**Author:** Chief Architect
**Date:** 2026-06-30
**Status:** PLAN MODE — no files modified
**Audit scope:** All 12 files in `docs/` + `README.md` + `DECISIONS.md` + source code cross-reference

---

## 0. Executive Summary

**Total documents audited:** 12 in `docs/`, plus `README.md` and `DECISIONS.md`

| Status | Count | Documents |
|--------|-------|-----------|
| ✅ **Accurate (historical)** | 4 | `docs/top5_fixes_plan.md`, `docs/monitoring_plan.md`, `docs/architectural_audit_report.md`, `docs/gap_closure_plan.md` |
| ⚠️ **Needs minor updates** | 5 | `docs/architecture.md`, `docs/phase4_gap_report.md`, `docs/phase3.3_release_notes.md`, `docs/phase3.3_completion_report.md`, `docs/phase3.3_full_completion_report.md` |
| 🔴 **Needs major updates** | 2 | `docs/mujoco_integration_plan.md`, `docs/monitoring_completion_report.md` |
| 📦 **Should archive** | 1 | `docs/phase3.2_post_validation_report.md` |
| 📝 **Missing new document** | 1 | `docs/phase4_gap_closure_report.md` (needs to be created after fixes applied) |

---

## 1. Documents Requiring Updates

### 1.1 `docs/architecture.md` — Module Map & Names Stale

**Problem:** The module map references old file paths and class names that were changed in Phase 3.3 gap closure and Phase 4 gap audit. Critical renames (G-001, G-004) are not reflected.

| Location | Current (Wrong) | Should Be |
|----------|----------------|-----------|
| `Module Map` table — ASI path | `phca/perception/asi.py` | `phca/asi/sanitizer.py` |
| `Module Map` — M1 path | `phca/working_memory/m1_sensory.py` | `phca/memory/m1_sensory.py` |
| `Module Map` — M2 path | `phca/working_memory/m2_working.py` | `phca/memory/m2_working.py` |
| `Module Map` — G' Engine path | `phca/world_model/engine.py` | `phca/prediction/engine.py` |
| `Module Map` — PEU path | `phca/learning/peu.py` | `phca/prediction/error_unit.py` |
| `Module Map` — MDIM path | Correct ✅ | — |
| `Module Map` — CR path | `phca/motivation/criticality.py` | `phca/regulation/pid_controller.py` (class renamed to `AdaptiveParameterController`) |
| `Module Map` — ATTN path | `phca/learning/attention.py` | `phca/attention/attention.py` |
| `Module Map` — HPM path | Correct ✅ | — |
| `Module Map` — RBTA path | `phca/governance/rbta_enforcer.py` | `phca/regulation/rbta_enforcer.py` |
| `Module Map` — PID entry | `phca/governance/pid_controller.py` | Remove — PID was merged into `AdaptiveParameterController` in regulation/ |
| `Module Map` — M3 path | `phca/episodic_memory/m3_episodic.py` | `phca/memory/m3_episodic.py` |
| `Module Map` — Consolidation path | `phca/episodic_memory/consolidation.py` | `phca/consolidation/scheduler.py` |
| `Module Map` — Config | Correct ✅ | — |
| System Diagram | Lists 12-step cycle, missing Steps 8 and 20 status | Update to reflect 12 active steps; note Step 8 (reserved), Step 20 (removed) |
| Resource Bounds table | Time=500ms bound | Update if bounds changed (check config.py DEFAULT_MODULE_BOUNDS) |
| `Key Design Decisions` | Missing references to unified `build(env)` | Add note about G-008 refactor |
| `Key Design Decisions` | Missing references to `error_volatility` rename | Add note about G-001 rename |
| `Key Design Decisions` | Missing references to `AdaptiveParameterController` | Add note about G-004 rename |
| `Key Design Decisions` | Missing Pareto front wiring (G-006/G-012) | Add note |
| `Up-to-Date Reference` | Points to D-044 | Update to D-049 (add D-045 through D-049) |
| `Up-to-Date Reference` | Points to `phase3.3_full_completion_report.md` | Also reference `phase4_gap_report.md` and `phase4_gap_closure_report.md` |

**Changes required:** ~30 edits across the document. Effort: 1 hour.

**Justification:** The module map is the primary navigation tool for developers. Stale paths and names cause confusion and wasted time.

---

### 1.2 `docs/phase4_gap_report.md` — Mark Resolved Findings

**Problem:** This report lists 26 findings (4 critical, 9 major, 8 minor, 5 tech debt). Several have been resolved in the Phase 4 gap audit execution (G-001, G-004, G-005, G-006/G-012, G-007, G-008, G-020). The report should be updated to mark resolved items and provide an accurate remaining-findings snapshot.

| Finding | Status | Resolution |
|---------|--------|-----------|
| G-001 | ✅ RESOLVED | Renamed `phi` → `error_volatility` (D-045) |
| G-002 | 🔴 DEFERRED to Phase 4.2 | MC dropout |
| G-003 | ❌ PENDING | Proper empowerment |
| G-004 | ✅ RESOLVED | Renamed to `AdaptiveParameterController` (D-046) |
| G-005 | ✅ RESOLVED | Goal-driven attention biasing (D-049) |
| G-006 | ✅ RESOLVED | Pareto front wired into meta-stable suppression (D-048) |
| G-007 | ✅ RESOLVED | Docstrings updated "statistical" (D patch) |
| G-008 | ✅ RESOLVED | Unified `build(env)` (D-047) |
| G-009 | ❌ PENDING | Deep-copy metrics |
| G-010 | ❌ PENDING | SQLite VACUUM |
| G-011 | ❌ PENDING | FLOP-based energy |
| G-012 | ✅ RESOLVED | Pareto front activated (D-048) |
| G-013 | ❌ PENDING | Dead branches in `_result_to_dict` |
| G-014 | ❌ PENDING | `schema_version` table |
| G-015 | ❌ PENDING | E/S-Stream docstring remnants |
| G-016 | ❌ PENDING | Magic numbers |
| G-017 | ❌ PENDING | Online vs replay learning conflict |
| G-018 | ❌ PENDING | Grounding levels |
| G-019 | ❌ PENDING | TSPL theta drift |
| G-020 | ✅ RESOLVED | `__import__("time")` fix (D patch) |
| TD-001–012 | ❌ PENDING | Tech debt items |

**Changes required:**
- Add a "Status" column to the Surgical Repair Plan tables (Section 4) showing which items are resolved
- Move resolved items to a "Completed" subsection
- Update the "Acceptance after week 2" checklist to mark completed items
- Update the open-sourcing recommendations (items 3, 5 are now done)

**Justification:** Without status tracking, readers cannot distinguish between "known issues to fix" and "already fixed" items.

---

### 1.3 `docs/phase3.3_release_notes.md` — Stale Test Counts & Names

**Problem:** Multiple stale references:

| Location | Current | Should Be |
|----------|---------|-----------|
| Summary table | "Tests: 398 → 289 (−27.4%)" | "Tests: 398 → 284" (G-005 added 1 test, SkillLibrary removed 6 tests) |
| Note block | "Tests further reduced to **283**" | "Tests currently at **284**" |
| Benchmark Results table | References `benchmark_mlp_final.json` and `benchmark_gaussian_final.json` | These files exist but benchmarks may need re-running after gap audit fixes |
| Known Limitations #1 | "benchmark runner fixed" ✅ | OK — already marked fixed |
| Known Limitations #2 | "L2 Φ-IQ ~0.32" | Update if benchmarks re-run |
| Known Limitations #4 | "MLP mode needs ≥200 cycles" | OK — still accurate |

**Changes required:** Update test counts. Effort: 10 minutes.

**Justification:** New readers should see the current test count, not a historical one.

---

### 1.4 `docs/phase3.3_completion_report.md` — Stale Benchmark & Test References

**Problem:**

| Location | Current | Should Be |
|----------|---------|-----------|
| Issue Resolution table | No mention of Phase 4 gap audit fixes (G-001 through G-020) | Add a note that the Phase 4 gap audit identified 26 additional findings, some of which are resolved |
| Benchmark Runner Validation table | Φ-IQ scores from quick 5-cycle test | These were pre-gap-closure. May not represent current state |
| Test Results | "289 pre-gap-closure" | Add note about current 284 tests |
| Postscript | References `phase3.3_full_completion_report.md` | OK — still accurate |

**Changes required:** Minor — add a note referencing the Phase 4 gap audit. Effort: 15 minutes.

**Justification:** Maintains the document's value as a historical record while not misleading about the current state.

---

### 1.5 `docs/phase3.3_full_completion_report.md` — Test Count Stale

**Problem:**

| Location | Current | Should Be |
|----------|---------|-----------|
| Summary | "283 tests" | "284 tests" (G-005 added 1) |
| Test Results | `283 passed` | `284 passed` |
| Invariant Verification | No mention of Phase 4 gap audit changes | Add note about D-045 through D-049 |
| Key Metrics | "All tests pass — 283" | Update to 284 |
| Go/No-Go | References 283 tests | Update |

**Changes required:** Update test count. Add a postscript about Phase 4 gap audit. Effort: 15 minutes.

**Justification:** Consistency with actual test suite.

---

### 1.6 `docs/mujoco_integration_plan.md` — Stale Class Names & Architecture

**Problem:** This document was written before the Phase 3.3 gap closure and Phase 4 gap audit. Multiple class names and code paths are stale:

| Location | Current (Wrong) | Should Be |
|----------|----------------|-----------|
| §4.1 `build_for_mujoco()` method | Uses `criticality_regulator = CriticalityRegulator()` | `adaptive_controller = AdaptiveParameterController()` |
| §4.1 `build_for_mujoco()` method | `from environments.mujoco_env` | `from phca.environments.mujoco_env` (package move B-004) |
| §4.1 `build_for_mujoco()` method | Manually wires all 13 modules | Should delegate to `cls.build(env=env, ...)` (G-008 refactor) |
| §4.2 `_compute_distance_gain()` description | Says "no code changes required" | Still accurate — fallback returns 0.5 ✅ |
| §4.5 RBTA Bounds table | Lists standalone bounds | These are now set in `build()` via parameters, not hardcoded |
| §7 Risk 1 | References `build_for_env()` vs `build_for_mujoco()` duplication | These now delegate to `build()`, so risk is lower |
| §8 Checklist — Phase 2 items | "Add CognitiveCycle.build_for_mujoco()" | Done ✅ |
| §8 Checklist — Phase 2 items | "Update RBTA bounds" | Done ✅ |
| §8 Checklist — Phase 2 items | "Test fallback" | Done ✅ |

**Changes required:** Either:
- **Option A (Recommended):** Mark the document as **HISTORICAL** with a prominent note that the implementation was completed via `CognitiveCycle.build(env)` (G-008) and the specific class names/architecture details are superseded. Add a pointer to `docs/architecture.md` for current implementation.
- **Option B:** Rewrite §4 with the current `build()` architecture.

**Effort:** Option A: 15 minutes. Option B: 2 hours.

**Justification:** The plan was executed, but the document as-is describes a different implementation than what exists.

---

### 1.7 `docs/monitoring_completion_report.md` — Test Count Stale

**Problem:**

| Location | Current | Should Be |
|---------|---------|-----------|
| Header Status | "283 tests passing" | "284 tests passing" |
| M3 criterion | "289 tests pass" | "284 tests pass" |
| Performance Impact section | All costs still accurate ✅ | — |

**Changes required:** Update test count. Effort: 5 minutes.

**Justification:** Consistency.

---

### 1.8 `docs/phase3.2_post_validation_report.md` — Should Be Archived

**Problem:** This document describes the system at Phase 3.2 completion. All its findings (C1–C3, M1–M4, m1–m12) have been addressed in Phase 3.3, gap closure, or Phase 4 gap audit. It is a historical record.

**Recommended action:** Add a prominent header: **⚠️ HISTORICAL — Phase 3.2 State. All findings addressed.** Move to `docs/archive/` if an archive directory exists, or add archival note.

**Effort:** 5 minutes.

**Justification:** New readers should not think these are current issues.

---

## 2. Non-`docs/` Documents Requiring Updates

### 2.1 `README.md` — Test Count, Cycle Steps, Benchmark Scores

**Currently accurate:** ✅
- Architecture description
- Quick start guide
- Test run commands
- Build method references

**Check:** Verify test count (should be 284), verify cycle steps count (12 active, removed Step 20), verify benchmark score references if any.

**Effort:** 10 minutes to verify and update.

### 2.2 `DECISIONS.md` — D-048/D-049 Already Added

**Currently:** D-001 through D-049 are recorded. D-044 has a formatting issue (decision body split from header by the "End of Decision Log" marker).

**Fix needed:**
- D-044 body is orphaned after the "End of Decision Log" marker from D-044's original position. The body text is:
  ```
  - **Date:** 2026-06-30
  - **Author:** Chief Architect
  - **Category:** Tier 2
  - **Option chosen:** Reverted `empowerment_blend = 0.7 * empowerment + 0.3 * (1.0 - min(error, 1.0))` back to the original `0.7 * empowerment + 0.3 * min(error * 0.5, 1.0)`.
  - **Alternatives:** Keep inverted logic; use different formula
  - **Rationale:** The original formula causes D6 deficit to be small when error is high...
  - **v3.0 trace:** §3.3 Def 3.5
  ```
  This should be moved back under the D-044 header.

**Effort:** 5 minutes.

---

## 3. Missing Documents to Create

### 3.1 `docs/phase4_gap_closure_report.md` — Post-Fix Validation

**Need:** After all Phase 4 gap audit fixes are applied (or the selected subset), a closure report certifying Phase 4 readiness should be created. This would:
- List which G-001 through G-020 findings were resolved
- Show benchmark results after fixes
- State Go/No-Go for Phase 4

**Content template:**
```markdown
# PHCA v3.0 — Phase 4 Gap Closure Report

**Date:** 2026-06-30
**Status:** ✅ CONDITIONAL GO (list which items remain)

## Resolved Findings
- G-001, G-004, G-005, G-006/G-012, G-007, G-008, G-020 ✅

## Remaining for Phase 4
- G-002, G-003, G-009, G-010, G-011, G-013, G-014, G-015, G-016, G-017, G-018, G-019

## Benchmark Results (Post-Fix)
| Level | Φ-IQ | Pass? |
...

## Go / No-Go Recommendation
...
```

**Effort:** 30 minutes to create.

---

## 4. Suggested Execution Order

```
Phase 1 — Quick fixes (numbers only, ~30 min total):
├── docs/phase3.3_release_notes.md      [10 min] — test counts
├── docs/phase3.3_completion_report.md   [15 min] — test counts + Phase 4 gap audit note
├── docs/phase3.3_full_completion_report.md [15 min] — test counts + postscript
├── docs/monitoring_completion_report.md [5 min]  — test count
├── README.md                            [10 min] — verify test count
├── DECISIONS.md                         [5 min]  — fix D-044 formatting
└── docs/phase3.2_post_validation_report.md [5 min] — add archival header

Phase 2 — Structural updates (~2-3 hours total):
├── docs/architecture.md                [1 hour]  — module map paths, names, new decisions
├── docs/mujoco_integration_plan.md      [15 min]  — mark as historical / architecturally superseded
└── docs/phase4_gap_report.md            [30 min]  — mark resolved findings

Phase 3 — New document:
└── docs/phase4_gap_closure_report.md    [30 min]  — certification of Phase 4 readiness
```

**Total estimated effort:** ~4 hours for a single engineer.

---

## 5. Phase 4 Gap Closure Additions (S-006/S-007)

After execution of the above plan, two surgical fixes were applied to resolve L2 goal pursuit performance:

| ID | Description | File | Change |
|----|-------------|------|--------|
| S-006 | Remove terminal-on-goal | `grid_world.py:147` | `terminal = at_goal or ...` → `terminal = ...` |
| S-007 | Fix goal-state distance_gain | `cycle.py:734-735` | STAY preferred at goal via signed gain |

These are reflected in:
- `DECISIONS.md` → D-070, D-071 added
- `docs/phase4_gap_closure_report.md` → Round 2 section added
- `docs/phase4_gap_report.md` → Section 4 updated with new findings table
- `docs/architecture.md` → Key Design Decisions updated, system diagram note added

Benchmark improvement: L2 Φ-IQ 0.331 → **0.476** (+44%).

## 6. Dependency Map

```
Quick fixes (Phase 1)
    │
    ▼
docs/architecture.md  ← depends on phase4_gap_report.md (to reference resolved findings)
    │
    ▼
docs/mujoco_integration_plan.md  ← depends on architecture.md (for current class names)
    │
    ▼
docs/phase4_gap_report.md  ← depends on all above (to have accurate resolved-finding list)
    │
    ▼
docs/phase4_gap_closure_report.md  ← depends on phase4_gap_report.md (uses it as source data)
```

**Parallel execution possible:** Phase 1 items have no dependencies on each other. All can be done simultaneously.

---

*End of Documentation Update Plan — 14 files audited, 8 action items identified, ~4 hours estimated.*
