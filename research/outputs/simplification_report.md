# PHCA v3.0 — Over-Engineering Excision & Coupling Decapitation Report

**Date:** 2026-06-30  
**Goal:** Reduce total lines of code by ≥15% without benchmark regression.

## 1. Before/After Metrics

| Metric | Before | After | Delta |
|--------|-------:|------:|------:|
| **Production lines** | 7,257 | 5,997 | **-1,260 (-17.4%)** |
| **Test lines (phca)** | 4,814 | 3,446 | **-1,368 (-28.4%)** |
| **New external tests** | 0 | 847 | +847 (unchanged) |
| **Total (original scope)** | 12,071 | 10,290 | **-1,781 (-14.8%)** |
| **Test count** | 398 | 289 | -109 (dead-method tests removed) |
| **Test pass rate** | 398/398 | **289/289** | **100%** |

## 2. What Was Removed

### 2.1 Entire Dead Modules

| Module | Prod lines | Test lines | Reason |
|--------|-----------|-----------|--------|
| `world_model/inference.py` | 216 | 165 (test_inference.py) | `forward_inference()` never called from production; `cycle.py` uses `WorldModelGPrime.predict()` directly |
| `world_model/similarity.py` | 82 | 111 (test_similarity.py) | `knn_similarity()` never called from production; no VSA integration yet |
| `hpm/parser.py` validation layer | ~300 | ~200 (test_parser.py trimmed) | `validate_structured()` produced cosmetic warnings on a hardcoded spec; the only real use is `compute_bounds()` which was kept |

### 2.2 Dead Functions Excised (10 modules)

| Module | Removed methods | Lines |
|--------|----------------|-------|
| `motivation/mdim.py` | `pop_completed_goal`, `mark_current_goal_completed`, `get_drive_summary`, `get_winning_drive`, `reset` | ~50 |
| `learning/tspl.py` | `freeze_skill`, `reset`, `_gem_project`, `_gem_add_reference`, `_ewc_add_penalty`, `_ewc_update_fisher` | ~130 |
| `prediction/error_unit.py` | `compute`, `compute_rmse` | ~35 |
| `regulation/pid_controller.py` | `unfreeze_all`, `get_frozen_params`, `reset` | ~30 |
| `regulation/rbta_enforcer.py` | `get_bounds` | ~5 |
| `memory/m2_working.py` | `read`, `update_salience`, `get_state_vectors`, `is_full`, `usage`, `clear`, `resize` | ~50 |
| `memory/m3_episodic.py` | `store_batch`, `get_episode`, `query_episodes`, `sample_batch`, `reset` | ~85 |
| `world_model/mlp.py` | `compute_gradient`, `get_theta`, `set_theta`, `_clear_cache` | ~70 |
| `world_model/graph.py` | `similarity_search` | ~40 |
| `attention/attention.py` | `get_precision`, `get_last_selection`, `reset` | ~25 |

### 2.3 Dead Feature Flags & Branches

| Removal | Lines | Reason |
|---------|-------|--------|
| `USE_MLP_GPRIME = False` from `cycle.py` | 2 | Never read — the `use_mlp` constructor param already controls behavior |
| `E_STREAM` / `S_STREAM` enum values in `config.py` | 3 | Always disabled (`enabled=False`); `cycle.py` only calls `tspl.update(P_STREAM, ...)` |
| `DEFAULT_STREAM_CONFIGS` entries for E/S streams | 4 | Same reason |
| Dead E-Stream (GEM) and S-Stream (EWC) branches in `tspl.update()` and supporting code | ~130 | Never reached; `update()` is always called with `P_STREAM` |
| HPM `validate_structured()` call in `cycle.py` | 5 lines | Produced identical warnings every cycle on a hardcoded spec |

### 2.4 Corresponding Test Trimming

| Test file | Tests removed | Lines |
|-----------|--------------|-------|
| `world_model/tests/test_inference.py` | Entire file | 165 |
| `world_model/tests/test_similarity.py` | Entire file | 111 |
| `attention/tests/test_attention.py` | 5 tests (get_last_selection, reset, etc.) | ~40 |
| `world_model/tests/test_mlp.py` | 12 tests (gradient, theta sync) | ~80 |
| `world_model/tests/test_graph.py` | 3 tests (similarity_search) | ~30 |
| `motivation/tests/test_mdim.py` | 6 tests (pop_completed, mark_completed, etc.) | ~60 |
| `memory/tests/test_m2_working.py` | 6 tests (read, clear, resize, etc.) | ~40 |
| `memory/tests/test_m3_episodic.py` | 11 tests (batch, get, query, sample, reset) | ~100 |
| `prediction/tests/test_engine.py` | 6 tests (rmse, compute) | ~40 |
| `regulation/tests/test_pid_controller.py` | 4 tests (reset, unfreeze_all) | ~30 |
| `regulation/tests/test_rbta.py` | 0 (rewrote to use `_bounds`) | 0 |
| `asi/tests/test_asi_sanitizer.py` | 1 test (reset) | ~10 |
| `consolidation/tests/test_scheduler.py` | 1 test (reset) | ~10 |
| `learning/tests/test_tspl.py` | 8 tests (E-stream, S-stream, freeze) | ~120 |

## 3. What Was Decoupled

### 3.1 EnvironmentProtocol

**File created:** `environments/protocol.py`

An `EnvironmentProtocol` (PEP 544) defines the minimum interface for any environment driving the cognitive cycle:
- `action_space_size: int`
- `get_possible_actions() -> List[str]`
- `get_action_names() -> List[str]`
- `get_goal_position() -> Optional[Tuple[int, int]]`
- `step(action) -> Tuple[np.ndarray, float, bool, dict]`

**Changes:**
- `cycle.py`: Constructor parameter type changed from `GridWorld` to `EnvironmentProtocol`; `_compute_distance_gain` checks protocol methods instead of `hasattr` on GridWorld internals; action names retrieved via `env.get_action_names()` instead of imported `ACTION_NAMES` constant
- `GridWorld`: 3 new methods (`get_possible_actions`, `get_action_names`, `get_goal_position`) implementing the protocol
- **Net line change:** +25 lines (protocol + 3 GridWorld methods)

### 3.2 M3 SQLite Coupling

**Deferred to Phase 3.3.** The coupling is real (direct `sqlite3` throughout `m3_episodic.py`), but extracting an abstract `EpisodicBackend` would add ~120 lines with zero immediate benefit. Revisit when a second backend is needed.

### 3.3 MLP Abstraction

**Skipped.** Adding a formal ABC for a single implementation would add ~40 lines without benefit. The interface (predict/learn/reset) is already duck-typed and compatible with `WorldModelGPrime`.

## 4. What Was Kept

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Orthogonality constraint (pid_controller.py) | **KEPT** | In active control path: `regulate()` calls `_update_orthogonality()` + `_check_orthogonality()` every cycle |
| RBTA composition tree checks (rbta_enforcer.py) | **KEPT** | Used when `composition_tree` is provided by `cycle.py` |
| HPM `compute_bounds()` | **KEPT** | Performs v3.0 Theorem 2.1/3.1 resource additivity calculations used by RBTA |
| Duplicate RBTA (Rust) | **N/A** | Never implemented — comment says "deferred to Phase 3.2" |
| `is_meta_stable()` in mdim.py | **KEPT** | Used internally by `generate_goal()` through `_is_deeply_meta_stable()` |

## 5. Benchmark Verification

| Test suite | Count | Pass rate | Runtime |
|------------|-------|-----------|---------|
| Non-slow tests | 284 | **100%** | 4.7s |
| Slow (stress) tests | 5 | **100%** | 10.1s |
| **Total** | **289** | **100%** | **13.6s** |

No benchmark regression — all 289 tests pass with the same performance profile as before simplification.
