# PHCA v3.0 — Gap Closure Execution Plan

**Date:** 2026-06-30  
**Author:** Chief Architect  
**Status:** ✅ **COMPLETED** — all items resolved  
**Based on:** `architectural_audit_report.md` (28 issues found)  
**Execution:** See `docs/phase3.3_full_completion_report.md` for results

> **This plan has been fully executed.** All Phase 3.3a (9 items), Phase 3.3b (6 items),
> and selected Phase 3.3c items are resolved. 284 tests pass. See the completion
> report for details, and the `DECISIONS.md` entries D-036 through D-044 for
> the specific decisions made during gap closure.

---

## Execution Rules

1. **Fix in order** within each phase section. Dependencies are noted.
2. **Run `make test-all`** after each fix. All 289 tests must pass.
3. **Record each decision** in `DECISIONS.md` with the date, what was done, why, and result.
4. **Do not add new features.** Only the planned fixes.
5. **Do not change files unrelated to the current issue.**
6. **Prefer surgical fixes** (2-10 lines) over refactors.
7. **Run benchmark Level 0** after Phase 3.3a completion to verify no regression.

---

## Phase 3.3a — Critical & Major Fixes (9 issues)

These issues must be fixed before Phase 3.3 can be considered complete. They represent P0/P1 severity items from the audit.

---

### Fix A-001: RBTA Composition Tree Energy Key Mismatch (GAP-001, P0)

**Files:**
- `python/phca/regulation/rbta_enforcer.py` (~line 256, ~line 303)

**What:**  
`_check_composition_tree()` and `_compute_subtree_energy()` read `runtime_log.get(child + "_energy", 1.0)`, but `_collect_runtime_log()` writes to `self.energy_log[mod]` (bare module ID), not `runtime_log[mod + "_energy"]`.

**Fix (Option A — Recommended):**  
Change both `_check_composition_tree` and `_compute_subtree_energy` to read energy from runtime_log using the bare module ID, matching the time-reading pattern:

In `_check_composition_tree()` (~line 256):
```python
# BEFORE:
child_energies.append(runtime_log.get(child + "_energy", 1.0))
# AFTER:
child_energies.append(runtime_log.get(child, 0.0))
```

In `_compute_subtree_energy()` (~line 303):
```python
# BEFORE:
child_energies.append(runtime_log.get(child + "_energy", 1.0))
# AFTER:
child_energies.append(runtime_log.get(child, 0.0))
```

**Why:** This aligns energy reading with time reading — both use bare module IDs from `_collect_runtime_log()`. The fallback `1.0` → `0.0` ensures that missing modules don't contribute phantom energy.

**Acceptance:**  
1. Unit test: create a sample runtime_log with known module energies, call `_check_composition_tree`, verify energy violations are correctly detected.
2. Integration test: set a tight B_energy bound, run one cycle, verify RBTA flags ENERGY violation.
3. All 289 tests pass.

**Effort:** 30 minutes.

---

### Fix A-002: Wire Monitoring Into Default Usage Path (GAP-002, P0)

**Files:**
- `scripts/benchmark.py`
- `scripts/phca-monitor.py`

**What:**  
The monitoring system (`MetricsStore`, file logging) is not wired into the default benchmark and dashboard scripts. Users following the README flow get no logs and no dashboard visibility.

**Fix:**  
1. In `scripts/benchmark.py`, add `from phca.logging import ensure_logging` at top. Call `ensure_logging()` at the start of `main()`. Pass a `MetricsStore` to the cycle builder.

2. In `scripts/phca-monitor.py`, ensure `ensure_logging()` is called before building the cycle.

**Detailed steps:**

For `scripts/benchmark.py`:
```python
# At top:
from phca.logging import ensure_logging
from phca.monitoring.metrics_store import MetricsStore

# In main(), before building cycle:
ensure_logging()

# When building cycle:
store = MetricsStore(maxlen=2000)
cycle = CognitiveCycle.build_for_env(..., metrics_store=store)
```

For `scripts/phca-monitor.py`:
```python
# At top, ensure ensure_logging is imported and called
from phca.logging import ensure_logging

# In run_monitor(), before building cycle:
ensure_logging()
```

**Acceptance:**  
1. Run `python scripts/benchmark.py --quick` and verify `logs/phca.log` exists with entries.
2. Run `python scripts/phca-monitor.py --cycles=10 --text` and verify it works without errors.
3. All 289 tests pass.

**Effort:** 45 minutes.

---

### Fix A-003: Orphaned Audit Report — Review and Integrate (GAP-003, P0)

**Files:**
- `.opencode/plans/silent_failure_and_fallacy_report.md`
- `.gitignore`

**What:**  
An audit report exists at `.opencode/plans/silent_failure_and_fallacy_report.md` outside the project's managed documentation. Its findings may contain critical issues not addressed in this audit.

**Fix:**  
1. Read the orphaned report.
2. Cross-reference its findings with this audit.
3. If critical issues are found, add them to this plan.
4. Either:
   - Archive the report by moving to `docs/archive/` or
   - Add `.opencode/` to `.gitignore` and add a note in README

**Detailed steps:**

```bash
cat .opencode/plans/silent_failure_and_fallacy_report.md
```

Read its findings. If any issues are not already captured in this audit:
- Add them as new entries to `architectural_audit_report.md`
- Add corresponding fix steps to this plan

Then:
```bash
mkdir -p docs/archive
cp .opencode/plans/silent_failure_and_fallacy_report.md docs/archive/
```
And update `.gitignore` to exclude `.opencode/`.

**Effort:** 1 hour.

---

### Fix A-004: `_compute_subtree_energy` Pass `energy_log` Instead of `runtime_log` (GAP-004, P1)

**Files:**
- `python/phca/regulation/rbta_enforcer.py` (lines 280-316)

**What:**  
Same key mismatch as A-001, but in the `_compute_subtree_energy()` function which is called recursively from `_check_composition_tree()`. It receives `runtime_log` and reads `child + "_energy"` keys that don't exist.

**Fix:**  
Modify `_check_composition_tree()` to read from `energy_log` (the cycle's actual energy measurement) for energy checks, and pass it down to `_compute_subtree_energy()`.

The simplest approach: In `_check_composition_tree()`, when computing leaf energy for a child string module, read from `runtime_log` using the bare child name (same as time), not `child + "_energy"`.

```python
# In _check_composition_tree(), around line 256:
# BEFORE:
child_energies.append(runtime_log.get(child + "_energy", 1.0))
# AFTER:
child_energies.append(runtime_log.get(child, 0.0))  # same key pattern as time
```

This mirrors the time reading and matches the `_collect_runtime_log()` key convention.

**Acceptance:** Same as A-001. Combined test with A-001.

**Effort:** 15 minutes (combined with A-001).

---

### Fix A-005: Remove Dead `compute_pareto_front()` Call (GAP-005, P1)

**Files:**
- `python/phca/motivation/mdim.py` (line 537)

**What:**  
`pareto = self.compute_pareto_front()` is called but the result is never used. The Pareto front output should inform `_update_meta_stable()`.

**Fix (Option B — Proper):**  
Wire the Pareto front output into meta-stability logic. In `_update_meta_stable()`, instead of suppressing ALL of D1/D3/D5, only suppress drives that are **NOT on the Pareto front**.

In `_update_meta_stable()` (mdim.py):
```python
def _update_meta_stable(self) -> None:
    deficits = [self.drives[d].deficit for d in range(1, 7)]
    all_satisfied = all(d < self._meta_stable_threshold for d in deficits)
    # ...
```

And in `_is_deeply_meta_stable()`, only suppress non-Pareto drives. Wait — `_is_deeply_meta_stable()` is called in `generate_goal()`, before `_update_meta_stable()`. So compute the Pareto front in `_update_meta_stable()` and store which drives are on it, then read that in `_is_deeply_meta_stable()`.

**Alternative (Option A — Minimal):** Comment out the `pareto = self.compute_pareto_front()` call and add a TODO:
```python
# TODO: Wire Pareto front into meta-stability (deferred from Phase 3.3)
# pareto = self.compute_pareto_front()
```

**Recommended:** Option A for now. Phase 4 can complete the Pareto→meta-stability wiring. The Pareto front computation itself is not harmful — it's just wasted cycles — but fixing the wiring would require changing the meta-stability contract, which is Phase 3.3 already closed.

**Effort:** 5 minutes.

---

### Fix A-006: Remove Dead `SkillLibrary` (GAP-006, P1)

**Files:**
- `python/phca/learning/skill_compilation.py` (entire file)
- Test file for skill_compilation (if exists)
- Any imports of `SkillLibrary` in other files

**What:**  
`SkillLibrary` class (50 lines + tests) is never instantiated in production code. TSPL uses its own inlined compilation logic.

**Fix:**  
1. Delete `python/phca/learning/skill_compilation.py`
2. Search for `import` statements referencing `SkillLibrary` or `skill_compilation` and remove them
3. Delete the corresponding test file (if it exists as a separate file)

**Search:**
```bash
grep -r "skill_compilation\|SkillLibrary" python/ --include="*.py"
```

**Effort:** 30 minutes.

**Acceptance:**  
1. `grep "SkillLibrary" python/ -r` returns no results
2. All 289 tests pass
3. The TSPL still compiles skills correctly

---

### Fix A-007: Add Energy Estimation TODO — Replace `runtime_s * 50.0` (GAP-007, P1)

**Files:**
- `python/phca/core/cycle.py` (~line 530)

**What:**  
Energy estimation uses hardcoded `max(0.1, min(10.0, runtime_s * 50.0))` with no physical basis.

**Fix:**  
Add a detailed TODO documenting the limitation and suggesting a FLOP-based estimate:

```python
# Energy estimates from actual module runtimes (scaled to match original magnitude)
# TODO (Phase 4): Replace runtime_s * 50.0 with actual energy measurement:
#   - For MLP: FLOPs/cycle = 3 * hidden_dim^2 + 2 * hidden_dim * state_dim
#   - For Gaussian G': FLOPs/cycle = O(n^3) for n = state_dim
#   - For SQLite M3: FLOPs = ~1000 * rows_written
#   For now, scale factor 50.0 keeps values in 0.1-10.0 range to match bounds.
self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
```

**Effort:** 5 minutes.

---

### Fix A-008: Step 20 Redundancy — Remove Sleep Cycle (GAP-009, P2)

**Files:**
- `python/phca/core/cycle.py` (lines ~490-500)

**What:**  
Consolidation runs every 10 cycles (Steps 16-18). Step 20 triggers another consolidation every 50 cycles (same function, force=True). Every 50th cycle, consolidation runs twice.

**Fix (Option C — Recommended):**  
Remove the Step 20 sleep-cycle code entirely. Consolidation is already handled by Steps 16-18 with `consolidation_interval=10`. The sleep cycle was meant for a different design where consolidation ran less frequently.

```python
# Remove these lines from step():
# Step 20: Sleep-cycle check (full consolidation every N cycles)
if self.cycle_count % self._get_sleep_interval() == 0:
    t_sleep = time.perf_counter()
    _log(logger, "debug", "cycle.sleep_cycle", cycle=self.cycle_count)
    self.consolidation.step(self.cycle_count, force=True)
    ...
```

Also remove `_get_sleep_interval()` static method if it becomes unused.

**Effort:** 10 minutes.

**Acceptance:** All 289 tests pass. Consolidation still fires every 10 cycles.

---

### Fix A-009: Fix D5 Energy Drive — Replace Hardcoded `energy_cost` (GAP-012, P2)

**Files:**
- `python/phca/core/cycle.py` (line ~351)
- `python/phca/motivation/mdim.py` (line ~170)

**What:**  
MDIM context always passes `"energy_cost": 0.1`. D5 (Energy Efficiency) drive deficit is always `max(0.0, 0.1 - 0.2) = 0.0`, making D5 never selectable.

**Fix:**  
Compute actual energy cost from the cycle's energy_log:

In `cycle.py`:
```python
# BEFORE:
"energy_cost": 0.1,
# AFTER:
"energy_cost": max(0.0, self.energy_log.get("CYCLE", 0.0) / 10.0),
```

This gives a dynamic energy cost between 0.0 and ~1.0 (since energy_log max is clamped to 10.0).

**Effort:** 5 minutes.

---

## Phase 3.3b — Structural Improvements (6 issues)

These issues should be fixed before the final Phase 3.3 release but do not block Phase 4 start.

---

### Fix B-001: Remove TSPL-E and TSPL-S from Memory/Energy Logs (GAP-010, P2)

**Files:**
- `python/phca/core/cycle.py` (lines ~510-530)

**What:**  
`_collect_runtime_log()` populates `memory_log` with `TSPL-E` and `TSPL-S` entries, but these streams were removed in Phase 3.3 (D-020). RBTA checks their bounds every cycle unnecessarily.

**Fix:**  
Remove TSPL-E and TSPL-S from the `memory_log` dict and the `baseline` dict in `energy_log`.

**Effort:** 5 minutes.

---

### Fix B-002: Wire `attention_focus` Into D2 or Remove (GAP-011, P2)

**Files:**
- `python/phca/core/cycle.py` (line ~343)
- `python/phca/motivation/mdim.py` (line ~300)

**What:**  
`attention_focus` is computed from M2 salience CV and passed in MDIM context, but no drive reads it.

**Fix:**  
Either:
1. Wire `attention_focus` into D2 (criticality seeking) as a secondary signal, or
2. Remove `attention_focus` from the context dict

**Recommended:** Option 2 — minimal. If D2 needs attention data in Phase 4, it can be added then.

**Effort:** 5 minutes.

---

### Fix B-003: Fix D6 Empowerment Blend Logic (GAP-013, P2)

**Files:**
- `python/phca/motivation/mdim.py` (line ~230)

**What:**  
D6 blend uses `prediction_error * 0.5` which increases empowerment when error is HIGH — opposite of expected.

**Fix:**  
```python
# BEFORE:
empowerment_blend = 0.7 * empowerment + 0.3 * min(prediction_error * 0.5, 1.0)
# AFTER:
empowerment_blend = 0.7 * empowerment + 0.3 * (1.0 - min(prediction_error, 1.0))
```

**Effort:** 2 minutes.

**Acceptance:** D6 deficit decreases when prediction error is high (the agent should seek empowerment when it understands action-outcome relationships).

---

### Fix B-004: Move `environments/` Into `python/phca/environments/` (GAP-016, P2)

**Files:**
- `python/environments/` directory
- `python/phca/core/cycle.py`
- `scripts/benchmark.py`
- `python/tests/test_*` files that import from environments

**What:**  
The GridWorld and protocol modules live outside the `python/phca/` package, breaking the package boundary.

**Fix:**  
1. `mv python/environments/ python/phca/environments/`
2. Update all `from environments.xxx import ...` to `from phca.environments.xxx import ...`
3. Update PYTHONPATH usage (may no longer need separate `python/` in PYTHONPATH for environments)

**Files to update imports in:**
- `python/phca/core/cycle.py`: `from environments.grid_world` → `from phca.environments.grid_world`
- `python/phca/core/cycle.py`: `from environments.protocol` → `from phca.environments.protocol`
- `python/phca/core/cycle.py`: `from environments.mujoco_env` → `from phca.environments.mujoco_env`
- Any test files importing from `environments.`

**Effort:** 30 minutes.

**Acceptance:** All 289 tests pass after the move.

---

### Fix B-005: Clean Up Makefile Benchmark Targets (GAP-027, P3)

**Files:**
- `Makefile` (lines ~55-65)

**What:**  
`bench-level-0` target calls `python -m phca.benchmarks.runner` which is not the recommended benchmark path.

**Fix:**  
Update to use `scripts/benchmark.py`:
```makefile
bench-level-0:
	@echo "Running Level 0 benchmark (stationary prediction)..."
	PYTHONPATH=python $$PYTHONPATH python scripts/benchmark.py --quick
```

Or remove the bench-* targets entirely and direct users to the README.

**Effort:** 5 minutes.

---

### Fix B-006: Remove Stale Requirement Files (GAP-024, P3)

**Files:**
- `requirements-phase-3.1.txt`
- `requirements-phase-3.2.txt`

**What:**  
Two requirement files named after completed phases. No `requirements-phase-3.3.txt` exists.

**Fix:**  
Either:
1. Consolidate into `requirements.txt` with a note: "All PHCA phases — use this single file."
2. Or keep separate files but add `requirements-phase-3.3.txt` and a deprecation notice on the old ones.

**Recommended:** Consolidate into `requirements.txt` and delete the phase-specific files.

**Effort:** 10 minutes.

---

## Phase 3.3c — Cleanup & Documentation (8 issues)

These are cosmetic or documentation-only issues. Fix as time permits.

---

### Fix C-001: Update cycle.py Docstring — 12 Active Steps (GAP-008, P3)

**Files:**
- `python/phca/core/cycle.py` (docstring)

**What:**  
Header documents 21-step cycle but only 12 active.

**Fix:**  
Rewrite the docstring to list the 12 actual steps with accurate Phase mapping.

---

### Fix C-002: Clean Up `_result_to_dict()` Legacy Branches (GAP-019, P3)

**Files:**
- `python/phca/world_model/graph.py` (~line 415)

**What:**  
Two legacy code branches for old pgmpy versions that are never hit.

**Fix:**  
Remove `isinstance(result, dict)` and `isinstance(result, DiscreteFactor)` checks.

---

### Fix C-003: Remove Schema Version Table or Implement Migration (GAP-020, P3)

**Files:**
- `python/phca/memory/m3_episodic.py`

**What:**  
`schema_version` table created but never written to.

**Fix:**  
Remove the `schema_version` table from `M3_SCHEMA_SQL`.

---

### Fix C-004: Add `_attention_weights` Clear in Gaussian.learn() (GAP-021, P3)

**Files:**
- `python/phca/world_model/graph.py` (~line 447)

**What:**  
Inconsistent `_attention_weights` cleanup between MLP and Gaussian.

**Fix:**  
Add `if hasattr(self, '_attention_weights'): self._attention_weights = None` at end of Gaussian.learn().

---

### Fix C-005: Replace Hardcoded `STAY` Index 4 With `self.env.stay_action` (GAP-022, P3)

**Files:**
- `python/phca/core/cycle.py` (line ~605)

**What:**  
`action_idx == 4` hardcodes STAY as GridWorld-specific index 4.

**Fix:**  
Replace with `action_idx == self.env.stay_action`.

---

### Fix C-006: Add `research/` README.md (GAP-026, P3)

**Files:**
- `research/outputs/README.md` (new file)

**What:**  
No guide for which documents describe current architecture.

**Fix:**  
Create `research/outputs/README.md` stating: "Historical documents. See `docs/` for current architecture."

---

### Fix C-007: Add Log Cleanup Script (GAP-025, P3)

**Files:**
- `scripts/clean_logs.py` (new file)

**What:**  
`logs/` has 25+ benchmark files with no rotation.

**Fix:**  
Create a cleanup script or add a `make clean-logs` target.

---

### Fix C-008: Update DECISIONS.md Phase Labels (GAP-023, P3)

**Files:**
- `DECISIONS.md`

**What:**  
Phase references in D-001 through D-035 are now historical.

**Fix:**  
Add a note at the top: "Phase labels in entries below reflect the phase in which the decision was made."

---

## Phase 3.3 Exit Validation

After all fixes from Phases 3.3a and 3.3b are applied:

### Acceptance Criteria

| # | Criterion | Measurement | Pass Condition |
|---|-----------|-------------|----------------|
| E1 | All 289 tests pass | `make test-all` | 0 failures |
| E2 | Φ-IQ overall ≥ 0.60 | `python scripts/benchmark.py --quick` (Level 0, 20 cy) | L0 Φ-IQ ≥ 0.60 |
| E3 | Energy violations detectable | Artificially low B_energy, run 1 cycle | RBTA returns ENERGY violation |
| E4 | No dead code paths | `grep -r "SkillLibrary\|pareto =.*pareto_front\|TSPL-E\|TSPL-S"` | No references to removed code |
| E5 | File logging active | Run benchmark, check `logs/phca.log` | File exists with content |
| E6 | D5 drive can fire | Inject high energy cost (10.0), run MDIM | D5 deficit > 0 |
| E7 | No double consolidation | Run 100 cycles, count `consolidation.complete` logs | Count = 10 (every 10 cycles) |
| E8 | All DECISIONS.md entries recorded | Append after each fix | ~6-9 new entries |

### Benchmark Suite to Run

```bash
# Quick check after each fix
make test-all

# After all fixes:
python scripts/benchmark.py --levels=0,1 --cycles=100 --use-mlp --output=logs/benchmark_phase3.3_final.json
python scripts/benchmark.py --levels=2,3 --cycles=200 --use-mlp --output=logs/benchmark_phase3.3_final_l23.json

# Monitoring verification (text mode to avoid curses requirement)
python scripts/phca-monitor.py --cycles=20 --text
python scripts/phca-logs.py --tail=5
```

### Go / No-Go Decision

| Condition | Status | Authority |
|-----------|--------|-----------|
| All P0/P1 fixes applied | Required | Chief Architect |
| 289 tests pass | Required | CI |
| No energy key mismatch | Required | Trace audit |
| File logging active | Required | Manual check |
| Level 0 Φ-IQ ≥ 0.60 | Aspirational | Benchmark |

---

## Execution Order Summary

```
Phase 3.3a (Critical — must fix before release):
├── A-001: Energy key mismatch in RBTA        [30min] ← HIGHEST PRIORITY
├── A-002: Wire monitoring into scripts        [45min]
├── A-003: Orphaned audit report              [1h]
├── A-004: Subtree energy key mismatch         [15min] (combined with A-001)
├── A-005: Remove dead pareto call             [5min]
├── A-006: Remove SkillLibrary                 [30min]
├── A-007: Energy estimation TODO              [5min]
├── A-008: Remove redundant sleep cycle        [10min]
└── A-009: Fix D5 energy_cost hardcode         [5min]
                                    Total: ~3.5h

Phase 3.3b (Structural — fix before final release):
├── B-001: Remove TSPL-E/TSPL-S from logs     [5min]
├── B-002: Remove unused attention_focus       [5min]
├── B-003: Fix D6 empowerment blend            [2min]
├── B-004: Move environments/ into phca/       [30min]
├── B-005: Fix Makefile benchmarks             [5min]
└── B-006: Consolidate requirements files      [10min]
                                    Total: ~1h

Phase 3.3c (Cleanup — fix as time permits):
├── C-001 through C-008                        [~2h]
                                    Total: ~2h
```

**Grand total:** ~6.5 engineering hours (approximately 1-2 days)

---

*End of Gap Closure Plan*
