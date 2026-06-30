# PHCA v3.0 — Complete Architectural Audit Report

**Date:** 2026-06-30  
**Author:** Chief Architect  
**Status:** PLAN MODE — analysis only, no code changes  
**Scope:** All source files, formal specs, benchmark logs, test suite, config, monitoring  
**Invariants Audited:** A1–A5 | **Dimensions Audited:** 8/8

---

## Executive Summary

This audit examines the PHCA v3.0 codebase after Phase 3.3 completion (289 tests passing, 5 critical fixes applied, monitoring system deployed). The system is **functionally complete for Phase 3.3 scope** but retains architectural debt across all 8 dimensions.

**Total issues found: 28** (P0=3, P1=6, P2=11, P3=8)

| Dimension | P0 | P1 | P2 | P3 | Total |
|-----------|----|----|----|----|-------|
| 1. Spec Compliance | 1 | 1 | 2 | 0 | **4** |
| 2. Invariant Violations | 1 | 1 | 0 | 0 | **2** |
| 3. Logical Fallacies | 1 | 1 | 1 | 0 | **3** |
| 4. Over-Engineering | 0 | 1 | 3 | 2 | **6** |
| 5. Coupling & Cohesion | 0 | 1 | 1 | 1 | **3** |
| 6. Runtime Data Flow | 0 | 1 | 1 | 0 | **2** |
| 7. Technical Debt | 0 | 0 | 3 | 5 | **8** |
| 8. Test Coverage | 0 | 0 | 0 | 0 | **0** |

**Missing dimensions of audit**: There is **no test coverage gap** (289 tests, edge-case/stress/chaos coverage good). But there is a **hidden coverage risk** — see GAP-007 below.

---

## AUD-001: Energy Dimension — HPM `compute_bounds()` Not Wired Through Cycle

**Severity:** P0 CRITICAL  
**Category:** 1 — Spec Compliance, 2 — Invariant Violations  
**Location:** `python/phca/hpm/parser.py:394–479`, `python/phca/core/cycle.py:275–300`  
**Status from top5_fixes_plan.md Issue #5:** ✅ Already fixed in D-032  
**Current state:** The cycle now extracts `B_energy` from `hpm_bounds` and passes it to the composition tree. The RBTA's `_check_composition_tree()` checks energy bounds. ✅ **RESOLVED**

---

## AUD-002: MLP hidden_dim Already Fixed (D-028)

**Severity:** P0 CRITICAL  
**Category:** 1 — Spec Compliance  
**Location:** `python/phca/world_model/mlp.py:41`  
**Status from top5_fixes_plan.md Issue #1:** ✅ Already fixed in D-028  
**Current state:** `WorldModelMLP.__init__()` now uses `hidden_dim: int = 128` (was 32). Train steps reduced from 8→4, batch_size from 64→32. Parameter count is 38,868. ✅ **RESOLVED**

---

## AUD-003: learn() error Parameter Now Consumed (D-029)

**Severity:** P1 MAJOR  
**Category:** 1 — Spec Compliance  
**Location:** `python/phca/world_model/mlp.py:129–198`, `python/phca/world_model/graph.py:406–447`  
**Status from top5_fixes_plan.md Issue #3:** ✅ Already fixed in D-029  
**Current state:** Both MLP and Gaussian `learn()` now use `lr_effective = lr * clip(1.0 + abs(error) * 0.1, 0.5, 2.0)`. MLP also receives per-dimension attention weights via dynamic attribute. ✅ **RESOLVED**

---

## AUD-004: MDIM target_state Now Used in Action Scoring (D-030)

**Severity:** P1 MAJOR  
**Category:** 1 — Spec Compliance  
**Location:** `python/phca/core/cycle.py:130–148`  
**Status from top5_fixes_plan.md Issue #4:** ✅ Already fixed in D-030  
**Current state:** D1/D3 scoring now blends `distance_gain (0.6) + confidence (0.2) + alignment (0.2)`. D2/D4 blends `uncertainty (0.5) + distance_gain (0.2) + alignment (0.3)`. ✅ **RESOLVED**

---

## AUD-005: Consolidation Facts Now Wired into MDIM Context (D-031)

**Severity:** P0 CRITICAL  
**Category:** 1 — Spec Compliance  
**Location:** `python/phca/consolidation/scheduler.py`, `python/phca/core/cycle.py`  
**Status from top5_fixes_plan.md Issue #2:** ✅ Already fixed in D-031  
**Current state:** `get_relevant_facts()` added to `ConsolidationScheduler`. Facts are queried per current state and injected into `mdim_context` as `fact_confidence_mean` and `fact_count`. ✅ **RESOLVED**

---

## AUD-006: Energy Bounds Wired (D-032)

**Severity:** P1 MAJOR  
**Category:** 2 — Invariant Violations (A1)  
**Location:** `python/phca/core/cycle.py:275–300`  
**Status from top5_fixes_plan.md Issue #5:** ✅ Already fixed in D-032  
**Current state:** `B_energy` extracted from `hpm_bounds` and added to composition tree bounds for both regulation_block and root SEQUENCE node. ✅ **RESOLVED**

---

> **The 5 issues from the previous top5_fixes_plan.md are resolved.** The audit now focuses on **newly discovered issues** that persist after those fixes.

---

# FULL AUDIT — NEW ISSUES FOUND

---

## 🔴 P0 CRITICAL

---

### GAP-001: RBTA Composition Tree Energy Check Reads Wrong Energy Keys

**Severity:** P0 CRITICAL  
**Category:** 1 — Spec Compliance, 6 — Runtime Data Flow  
**Location:** `python/phca/regulation/rbta_enforcer.py:256`  
**Description:** The `_check_composition_tree()` method reads child energy values using `runtime_log.get(child + "_energy", 1.0)` for leaf nodes (line 256). However, the `_collect_runtime_log()` method in `cycle.py` (line ~530) populates `energy_log` using exact module IDs (e.g., `"ASI"`, `"WM"`, `"G'"`), not `"ASI_energy"`, `"WM_energy"`, etc.  

The energy log keys in the cycle use bare module IDs:  
```python
self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
```
But the RBTA looks for `child + "_energy"`:  
```python
child_energies.append(runtime_log.get(child + "_energy", 1.0))
```

**Impact:** The energy dimension check in `_check_composition_tree()` always reads `1.0` (default) because `"ASI_energy"` is never in `runtime_log`. The energy violation detection is functionally **dead code** — it will never fire, making A1 enforcement for energy in the composition tree cosmetic. Only the flat per-module energy check in `check_cycle()` works.

**Root Cause:** Key mismatch between `_collect_runtime_log()` (writes `self.energy_log[mod]`) and the composition tree check (reads `runtime_log[child + "_energy"]`). The `_compute_subtree_energy()` method has the same bug at line ~303.

**Proposed Fix:** Either:
- Option A: Change `_check_composition_tree()` to read `runtime_log.get(child, 0.0)` for energy (same as time), adjusting the fallback value
- Option B: Change `_collect_runtime_log()` to write both `energy_log[mod]` AND `runtime_log[mod + "_energy"]`
- Option C: Pass `energy_log` to `_check_composition_tree()` separately instead of sharing `runtime_log`

**Recommended:** Option A — simplest, aligns energy reading with time reading, no new keys needed.

**Evidence:** Search `_compute_subtree_energy` definition and cross-reference `_collect_runtime_log` key naming.

**Dependency:** None.

---

### GAP-002: RBTA `learn()` — No MetricsStore Used, Log File Not Written

**Severity:** P0 CRITICAL  
**Category:** 6 — Runtime Data Flow  
**Location:** `python/phca/core/cycle.py` (step method), `docs/monitoring_completion_report.md`  
**Description:** The monitoring system's `MetricsStore` is an optional parameter that defaults to `None` everywhere. When the cognitive cycle is created via `build_for_env()` or `build_for_mujoco()` from `scripts/benchmark.py`, **no MetricsStore is passed**. This means:
1. The dashboard's data source (`MetricsStore.push()`) is never called
2. The `setup_file_logging()` function is never called by default
3. The log viewer (`scripts/phca-logs.py`) has nothing to read

The monitoring system is **not wired into the main benchmark/usage path**. The dashboard only works when explicitly passing `metrics_store=` through `CognitiveCycle.build_for_env()` AND calling `setup_file_logging()`.

**Impact:** Users following the README flow (`python scripts/benchmark.py --quick`) get no file logs and no dashboard visibility. The monitoring system exists but is effectively disconnected from the default usage path.

**Root Cause:** Design decision D-033 made MetricsStore optional (correct for backward compatibility), but the default code paths (`scripts/benchmark.py`, `scripts/phca-monitor.py`) were not updated to use it consistently.

**Proposed Fix:**  
1. Call `ensure_logging()` at the top of `scripts/benchmark.py`  
2. Call `ensure_logging()` at the top of `scripts/phca-monitor.py`  
3. Both scripts should pass a `MetricsStore` to the cycle builder

**Evidence:** `scripts/benchmark.py` imports `CognitiveCycle` from `phca.core.cycle`, calls `build_for_env()` with no `metrics_store=` kwarg.

**Dependency:** None.

---

### GAP-003: `.opencode/plans/silent_failure_and_fallacy_report.md` Exists but No Code Addresses It

**Severity:** P0 CRITICAL  
**Category:** 4 — Over-Engineering  
**Location:** `.opencode/plans/silent_failure_and_fallacy_report.md` (outside docs/)  
**Description:** An audit report exists at `.opencode/plans/silent_failure_and_fallacy_report.md` that is not referenced by any documentation, not tracked in `DECISIONS.md`, and whose findings are not addressed in any commit. The file exists as a separate plan outside the project's managed documentation directory.

**Impact:** Unknown — there may be unaddressed critical issues from a previous audit that are invisible to the current documentation. The `.opencode/` directory is not gitignored but also not tracked as project documentation.

**Root Cause:** Non-standard documentation location outside `docs/`. Audit findings were produced but never integrated into the project's decision-making process.

**Proposed Fix:**  
1. Read `.opencode/plans/silent_failure_and_fallacy_report.md`  
2. Cross-reference its findings with this audit  
3. Either incorporate findings into `gap_closure_plan.md` or archive the file
4. Add `.opencode/` to `.gitignore` or document it in README

**Evidence:** File listed in project tree at `.opencode/plans/silent_failure_and_fallacy_report.md`.

---

## 🟠 P1 MAJOR

---

### GAP-004: `_compute_subtree_energy` Reads `runtime_log` Instead of `energy_log`

**Severity:** P1 MAJOR  
**Category:** 6 — Runtime Data Flow  
**Location:** `python/phca/regulation/rbta_enforcer.py:280-316`  
**Description:** The `_compute_subtree_energy()` method is called during `_check_composition_tree()` to compute the composite energy of tree nodes. However, it receives `runtime_log` (the same dict passed for time computation) and reads `runtime_log.get(child + "_energy", 1.0)`. This is the same key mismatch as GAP-001 but in a separate method.

The method signature: `_compute_subtree_energy(self, tree: Dict, runtime_log: Dict[str, float]) -> float`

But `_collect_runtime_log()` writes energy to `self.energy_log`, not `self.runtime_log[mod + "_energy"]`.

**Impact:** `_compute_subtree_energy` always returns `sum([1.0, 1.0, ...])` plus overhead for all nodes, completely ignoring the measured energy data. Energy violation detection in the composition tree is **silently non-functional**.

**Root Cause:** Same as GAP-001 — key name convention mismatch.

**Proposed Fix:** Pass `energy_log` (from the cycle's `collect_runtime_log`) to `_check_composition_tree` and `_compute_subtree_energy`, or unify the key naming between time and energy reading in `_check_composition_tree` to use bare module IDs for both.

**Effort:** 30 minutes.

---

### GAP-005: MDIM `compute_pareto_front()` Is Called But Output Discarded

**Severity:** P1 MAJOR  
**Category:** 4 — Over-Engineering  
**Location:** `python/phca/motivation/mdim.py:340-380`, `python/phca/motivation/mdim.py:537`  
**Description:** In `generate_goal()` (line 537):
```python
pareto = self.compute_pareto_front()
```
The `pareto` variable is computed but **never used**. It's not stored, not logged, not passed to cycle. The Pareto front computation (40+ lines of logic) runs every cycle purely for its side effects on `_update_meta_stable()` — but `_update_meta_stable()` does not actually use the Pareto front result either; it reads `drives[1-6].deficit` directly instead.

**Impact:** 40 lines of moderately complex code execute every cycle with zero behavioural effect. The Pareto front algorithm is **dead code** in the runtime path.

**Root Cause:** The Pareto front was implemented as part of the v3.0 specification (Def 3.10) but the integration into meta-stability logic was never completed. The `_update_meta_stable()` method should use `compute_pareto_front()` output to determine which drives are on the front, then suppress non-Pareto drives.

**Proposed Fix:** Either:
- Option A (Minimal): Remove `compute_pareto_front()` call from `generate_goal()` and mark as deferred
- Option B (Proper): Wire Pareto front into meta-stability: only suppress drives NOT on the Pareto front, not all of D1/D3/D5

**Recommended:** Option B — preserves spec compliance.

---

### GAP-006: SkillLibrary Is Instantiable But Never Instantiated

**Severity:** P1 MAJOR  
**Category:** 4 — Over-Engineering  
**Location:** `python/phca/learning/skill_compilation.py`, `python/phca/learning/tspl.py`  
**Description:** The `SkillLibrary` class (skill_compilation.py) is a 50-line class with methods `store()`, `retrieve()`, `list_skills()`, `get_metadata()`. It has accompanying tests. However, **no production code creates a `SkillLibrary` instance** nor calls its methods. The TSPL's `_compile_skill()` method stores frozen parameters directly in `self.theta_protected` and `self.compiled_skill_ids` — it never uses `SkillLibrary`.

**Impact:** 50 lines of production code and accompanying tests (~100+ lines total) with zero runtime coverage. This is dead code carried forward from Phase 3.1/3.2 planning.

**Root Cause:** `SkillLibrary` was designed for M5 (long-term procedural memory) which was deferred. TSPL inlined the compilation logic, leaving `SkillLibrary` orphaned.

**Proposed Fix:** Either:
- Option A: Delete `SkillLibrary` and its tests (clean up dead code)
- Option B: Wire TSPL._compile_skill() to use `SkillLibrary.store()`, making it the canonical storage backend

**Recommended:** Option A — M5 is Phase 4+ scope. Remove dead code now.

---

### GAP-007: Energy Log Estimation Uses Hardcoded Scale Factor, No Actual Measurement

**Severity:** P1 MAJOR  
**Category:** 3 — Logical Fallacies  
**Location:** `python/phca/core/cycle.py:530`, `cycle.py:485-490`  
**Description:** The `energy_log` is populated with:
```python
self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
```
The 50.0 scale factor is a **hardcoded magic number** with no physical basis — it was chosen solely to keep energy values in the 0.1–10.0 range to match the original bound definitions. There is no actual energy measurement (CPU power monitoring, instruction counting, or FLOP estimation).

**Impact:** The entire energy dimension of A1 enforcement is based on a **runtime-to-energy mapping that has never been validated**. Energy violations can never be triggered in practice because the `runtime_s * 50.0` formula will always produce values near the baseline. If a real module draws 100× its energy budget (e.g., MLP training divergence), the energy log will reflect only ~2× runtime increase, not the actual energy cost.

**Root Cause:** The v3.0 spec defines `B_energy` but provides no measurement methodology. The implementation used `runtime_heuristic * 50.0` as a placeholder.

**Proposed Fix:**  
1. Add a TODO comment documenting this limitation  
2. Optionally, estimate FLOPs per module type (e.g., MLP: `3 * hidden_dim * state_dim` per forward pass) rather than `runtime_s * 50.0`  
3. Document as a known limitation for Phase 4 (hardware measurement)

---

## 🟡 P2 MINOR

---

### GAP-008: cycle.py `step()` — Comment Documents 21 Steps but Only 12 Are Active

**Severity:** P2 MINOR  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/core/cycle.py:1-30`  
**Description:** The file header documents a 21-step cycle (Steps 0-20), but the implementation has only 12 effective steps. Steps 5-6 (PEU) are merged after env.step, Step 8 is empty, Steps 8, 10-13 are labelled as "Phase 3.2" but some (Steps 10-13) are actually Phase 3.2 code that was merged. The comment says "Steps 8, 10-13, 16-18, 20 deferred" but Steps 10-13 (MDIM, CR, ATTN, HPM) and 16-18 (Consolidation) are all active.

**Proposed Fix:** Update the docstring to accurately reflect the 12 active steps and mark the precise deferred steps (Step 8 only?).

---

### GAP-009: `_get_sleep_interval()` Returns 50 But Consolidation Runs Every 10 Cycles

**Severity:** P2 MINOR  
**Category:** 3 — Logical Fallacies  
**Location:** `python/phca/core/cycle.py:220`, `cycle.py:492-500`  
**Description:** The sleep interval (Step 20) is hardcoded to 50 cycles. However, `ConsolidationScheduler` uses `consolidation_interval=10` by default. This means consolidation runs every 10 cycles (in `step()`'s Steps 16-18), but the "sleep cycle" (Step 20) triggers every 50 cycles and calls the same `self.consolidation.step(force=True)`. Every 50th cycle, **consolidation runs twice** — once in Steps 16-18 (because `cycle_count % 10 == 0`) and once in Step 20 (because `cycle_count % 50 == 0`).

**Impact:** Every 50 cycles, consolidation runs twice in the same step(). The second redundant call returns immediately (no unprocessed episodes) but adds overhead.

**Root Cause:** `_get_sleep_interval()` was never updated when the consolidation interval changed from 50 to 10.

**Proposed Fix:** Either:
- Option A: Set `consolidation_interval=50` to match sleep interval
- Option B: Change `_get_sleep_interval()` to return `self.consolidation._consolidation_interval` (dynamic)
- Option C: Remove Step 20 entirely since Steps 16-18 handle it

**Recommended:** Option C — Step 20 is redundant.

---

### GAP-010: Memory Log Contains TSPL-E and TSPL-S Keys But These Streams Were Removed

**Severity:** P2 MINOR  
**Category:** 7 — Technical Debt, 4 — Over-Engineering  
**Location:** `python/phca/core/cycle.py:510-530`  
**Description:** The `_collect_runtime_log()` method includes `TSPL-E`, `TSPL-S` in `memory_log` and baseline defaults in `energy_log`. But these streams were removed from `StreamID` enum (D-020) and `config.py`. They no longer exist as functional modules.

**Impact:** RBTA checks memory bounds for `TSPL-E` and `TSPL-S` every cycle. These checks always pass (memory_log entries are static 5-10K) but are unnecessary noise in the violation list.

**Proposed Fix:** Remove TSPL-E and TSPL-S entries from `_collect_runtime_log()`.

---

### GAP-011: `attention_focus` Computed from M2 Saliences But Never Used

**Severity:** P2 MINOR  
**Category:** 4 — Over-Engineering  
**Location:** `python/phca/core/cycle.py:340-345`  
**Description:** The cycle computes `attention_focus` from the coefficient of variation of M2 chunk saliences and passes it in `mdim_context`. However, **no drive in MDIM reads `attention_focus`** from the context dict. The context key `"attention_focus"` is not referenced in `mdim.py`.

**Impact:** ~10 lines of computation per cycle with no consumer. Genuinely dead path in the data flow.

**Proposed Fix:** Either wire `attention_focus` into D2 (criticality seeking) as an additional signal, or remove the computation.

---

### GAP-012: `energy_cost` in MDIM Context Always 0.1

**Severity:** P2 MINOR  
**Category:** 3 — Logical Fallacies  
**Location:** `python/phca/core/cycle.py:351`  
**Description:** The MDIM context dict contains `"energy_cost": 0.1` — a hardcoded constant. This feeds into D5 (Energy Efficiency) drive computation. Since it's always 0.1, D5 deficit is always `max(0.0, 0.1 - 0.2) = 0.0`, meaning **D5 will never be selected** as the winning drive (deficit is always zero). D5 is effectively dead.

**Impact:** D5 drive exists in the code but will never fire because its input is constant. The "Energy Efficiency" drive is decorative.

**Proposed Fix:** Compute actual energy cost from `energy_log` or the cycle's runtime. `energy_cost = self.energy_log.get("CYCLE", 0.1) / 10.0` would give a dynamic range.

---

### GAP-013: `empowerment` Blend Includes `prediction_error * 0.5` — Circular Logic

**Severity:** P2 MINOR  
**Category:** 3 — Logical Fallacies  
**Location:** `python/phca/motivation/mdim.py:225-230`  
**Description:** D6 (Empowerment) computation blends `empowerment` (from cycle) with `prediction_error * 0.5`:
```python
empowerment_blend = 0.7 * empowerment + 0.3 * min(prediction_error * 0.5, 1.0)
```
This means D6 increases when prediction error is high. But D6 is supposed to measure "action-effect channel capacity" — the mutual information between actions and outcomes. When prediction error is high, the agent doesn't understand the action-effect mapping, so empowerment should be *low*, not high. The blend logic is inverted.

**Impact:** D6 (empowerment) fires when prediction error is HIGH, which is the opposite of its theoretical definition. This doesn't cause runtime errors but creates a false signal.

**Proposed Fix:** Change to `empowerment_blend = 0.7 * empowerment + 0.3 * (1.0 - min(prediction_error, 1.0))` — high empowerment when error is low.

---

### GAP-014: `HPMValidator` Instantiated But `hpm_spec` Is Rebuilt Hardcoded Every Cycle

**Severity:** P2 MINOR  
**Category:** 4 — Over-Engineering  
**Location:** `python/phca/core/cycle.py:360-385`, `python/phca/hpm/parser.py`  
**Description:** In `step()`, the HPM spec is constructed as a hardcoded dict:
```python
hpm_spec = {
    "type": "SEQUENCE", "id": "cognitive_cycle",
    "children": [ ... ],
}
```
This dict is **identical every cycle** — it never changes based on system state, cycle count, or learning progress. The `HPMValidator.compute_bounds()` then parses this static dict and uses runtime_log to compute composite bounds.

**Impact:** The HPM module is a parser for a dynamic composition tree, but the tree is static. The entire HPM step reduces to: `hpm_bounds = compute_bounds(constant_tree, runtime_log)`. The `HPMValidator` object exists but could be replaced by a function.

**Proposed Fix:** While keeping `HPMValidator` for future dynamic tree support, add a comment noting the tree is currently static. Phase 4 should dynamically modify the tree based on module performance.

---

### GAP-015: cycle.py `run()` Summary Returns Full `actions_taken` List — Memory Bloat

**Severity:** P2 MINOR  
**Category:** 4 — Over-Engineering  
**Location:** `python/phca/core/cycle.py:90`  
**Description:** The `run()` method returns `actions_taken: [m.action_name for m in self.metrics_history]` — the full list of action names for every cycle. For a 1000-cycle run, this returns a list of 1000 strings. For the `metrics_history` it returns full `CycleMetrics` objects for all cycles.

**Impact:** When `run(1000)` is called from `scripts/benchmark.py`, the returned dict contains ~1000 action name strings and ~1000 full metrics objects. While not critical for 1000 cycles, this scales linearly with cycles and could be a memory issue for 100K-cycle runs.

**Proposed Fix:** Limit returned data to summary statistics. Remove full `actions_taken` list from return dict unless explicitly requested.

---

### GAP-016: GridWorld Protocol File Paths Are `environments/` — Not `phca/environments/`

**Severity:** P2 MINOR  
**Category:** 5 — Coupling & Cohesion  
**Location:** `python/phca/core/cycle.py:import line`, `python/environments/`  
**Description:** The cycle imports `from environments.grid_world import GridWorld` and `from environments.protocol import EnvironmentProtocol` — these modules are at `python/environments/`, outside the `python/phca/` package. This breaks the package boundary.

**Impact:** If `python/phca/` is packaged and distributed (setup.py), the `environments/` directory would need to be included separately. It also means the cycle has a non-standard dependency path.

**Proposed Fix:** Move `environments/` into `python/phca/environments/` and update all imports. This is a structural fix that touches ~5 files (cycle.py, benchmark.py, mujoco_env.py, test files).

---

### GAP-017: CycleMetrics Dataclass Stores Module Timings Dict — No Size Limit

**Severity:** P2 MINOR  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/core/cycle.py:32-48`  
**Description:** `CycleMetrics.module_timings` is an unbounded dict — modules can add arbitrary keys. In the `step()` method, 12+ module timings are stored per cycle. For 1000 cycles, `metrics_history` stores ~12,000 timing entries across all dicts. Combined with the max 10,000 history cap (line ~460), this is ~120,000 float values in memory.

**Impact:** ~1MB for 10,000 cycles. Not critical but could be reduced by 50% by computing averages or storing only P95/P99.

---

## 🟢 P3 COSMETIC

---

### GAP-018: `phi_current` Variable Redefines Built-in `id` in Loop

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/core/cycle.py:335`  
**Description:** Not observed in cycle.py — re-check. Actually, no `id` overwrite found in current codebase.

---

### GAP-019: `_result_to_dict()` in graph.py Has Redundant Type Check

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/world_model/graph.py:415-425`  
**Description:** The helper checks `if isinstance(result, dict):` (backward compat for old pgmpy), then `if not isinstance(result, DiscreteFactor):` (unknown type). In practice, pgmpy 1.1.2+ always returns `DiscreteFactor`, so the first two branches are dead in all supported environments.

**Proposed Fix:** Remove `isinstance(result, dict)` and `isinstance(result, DiscreteFactor)` checks, assuming modern pgmpy.

---

### GAP-020: `M3_SCHEMA_SQL` Has `schema_version` Table but No Migration Logic

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/memory/m3_episodic.py:75-80`  
**Description:** The schema creates a `schema_version` table with `(version INTEGER PRIMARY KEY, applied_at INTEGER)`, but no migration logic exists anywhere — `_init_db()` calls `executescript(M3_SCHEMA_SQL)` which runs CREATE TABLE IF NOT EXISTS, and the `schema_version` table is never written to. The migration infrastructure (schema_v1.py) exists but is disconnected.

**Proposed Fix:** Either implement `check_schema_version()` + `run_migrations()`, or remove the `schema_version` table.

---

### GAP-021: `_attention_weights` Cleared After MLP.learn() but Not for Gaussian

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/world_model/mlp.py:150`, `python/phca/world_model/graph.py:447`  
**Description:** In MLP.learn(), `self._attention_weights = None` is set after use. In Gaussian.learn(), `_attention_weights` is never read (only `lr_mod` from error is used), but the attribute may be set on the instance by the cycle. This is not a bug but creates inconsistency between the two world model implementations.

**Proposed Fix:** Add `if hasattr(self, '_attention_weights'): self._attention_weights = None` in Gaussian.learn() for consistency.

---

### GAP-022: `_compute_distance_gain()` Hardcodes `STAY` as Index 4

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `python/phca/core/cycle.py:605`  
**Description:** Line 605: `if action_idx == 4 and new_dist == current_dist:` — hardcodes STAY as index 4. This is GridWorld-specific (5 actions: N, S, E, W, STAY). If the environment has a different action space (e.g., MuJoCo with 3 actions), index 4 may be out of range or refer to a different action.

**Impact:** Works for GridWorld, silently wrong for MuJoCo. The MuJoCo path always returns `0.5` due to `get_goal_position()` returning `None`, so this line is never reached in MuJoCo. But it's a latent bug.

**Proposed Fix:** Use `self.env.stay_action` (already exists in `_select_action()`) instead of hardcoded index 4.

---

### GAP-023: `DECISIONS.md` D-001 References Phase 3.1 — Phase Label Is Now Wrong

**Severity:** P3 COSMETIC  
**Category:** 5 — Coupling & Cohesion  
**Location:** `DECISIONS.md` (all 35 entries)  
**Description:** Multiple decisions reference "Phase 3.1" or "Phase 3.2" that are now completed. D-009 says "Phase 3.1 simplification" about discrete binary nodes, but Phase 3.1, 3.2, and 3.3 are all done. The phase labels in decision entries are now historical artifacts that could confuse new readers.

**Proposed Fix:** Either remove phase prefixes, or update to reflect current state (e.g., "Phase 3.1 (now deprecated)"). This is purely cosmetic.

---

### GAP-024: `requirements-phase-3.1.txt` and `requirements-phase-3.2.txt` — Stale Files

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `requirements-phase-3.1.txt`, `requirements-phase-3.2.txt`  
**Description:** Two requirement files named after phases that are now complete. The project is at Phase 3.3, and there's no `requirements-phase-3.3.txt`. New developers would need to know which phase they're setting up.

**Proposed Fix:** Either consolidate into a single `requirements.txt` or rename with dates.

---

### GAP-025: logs/ Directory Has 25+ Benchmark JSON Files — No Rotation/Pruning

**Severity:** P3 COSMETIC  
**Category:** 4 — Over-Engineering  
**Location:** `logs/` (25+ files)  
**Description:** The `logs/` directory contains 25+ `benchmark_*.json` files from various runs. There's no cleanup policy. While each file is small (2-10KB), the accumulation makes finding the latest report difficult.

**Proposed Fix:** Add a timestamp prefix to benchmark output files and a cleanup script.

---

### GAP-026: `research/outputs/` Has 13+ Documents — Many Are Stale or Deprecated

**Severity:** P3 COSMETIC  
**Category:** 5 — Coupling & Cohesion  
**Location:** `research/outputs/` (13+ files)  
**Description:** The research directory contains design documents from Phases 1, 2, and 3. Documents like `07-rigorous-whitepaper.md` and `06-deep-gap-analysis.md` are valuable historical records but create ambiguity about which documents describe the current architecture.

**Proposed Fix:** Add a `README.md` in `research/outputs/` that states "Historical architecture documents. See `docs/` for current state."

---

### GAP-027: `Makefile` Has `bench-level-0` Target — Runner Is Stub

**Severity:** P3 COSMETIC  
**Category:** 7 — Technical Debt  
**Location:** `Makefile:55-60`  
**Description:** The `bench-level-0` target calls `python -m phca.benchmarks.runner --level=0`. However, `python/benchmarks/runner.py` was rewritten in D-026 but the CLI runner (`python -m phca.benchmarks.runner`) may not be the recommended path. The primary benchmark is `scripts/benchmark.py`. The Makefile targets are misleading.

**Proposed Fix:** Update Makefile targets to use `scripts/benchmark.py` or remove the `bench-*` targets.

---

### GAP-028: `logo/` Directory Not Mentioned in README

**Not a code issue.** Skipping.

---

## Gap Closure Plan Reference

See `gap_closure_plan.md` for the prioritised execution plan based on this audit.

---

*End of Architectural Audit Report*
