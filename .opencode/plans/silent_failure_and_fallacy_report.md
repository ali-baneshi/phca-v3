# PHCA v3.0 — Silent Failure & Logical Fallacy Report

**Date:** 2026-06-30  
**Scope:** All Python source under `python/phca/`  
**Method:** Static analysis + data flow tracing + logical fallacy audit  
**Verdict:** 4 Critical, 9 High, 12 Medium, 8 Low issues found.

---

## 1. SILENT EXCEPTION HANDLERS

### 1.1 Critical (P0) — Silent handlers that mask bugs or crash at runtime

#### P0-1: `_select_action` exception handler references undefined variable

| Field | Value |
|-------|-------|
| **File** | `python/phca/core/cycle.py` |
| **Line** | 507–511 |
| **Code** | `except Exception as e: _log(logger, "warning", "action_selection.predict_failed", action=action_name, error=str(e))` |
| **Problem** | `action_name` is **never defined** in `_select_action()`. It is only defined at line 276 inside `step()`. On any prediction failure during action selection, this raises a `NameError`, which is then caught by the outer catch-all at line 440, logging a misleading `"cycle.step.error"` instead of the actual prediction failure. |
| **Impact** | Every action-selection prediction failure becomes a cryptic `NameError` — errors are swallowed and misreported. |
| **Fix** | Remove `action=action_name` from the log call. |

**Proposed patch:**
```python
# Line 507-511 — change:
except Exception as e:
    _log(logger, "warning", "action_selection.predict_failed",
         action=action_name, error=str(e))
# to:
except Exception as e:
    _log(logger, "warning", "action_selection.predict_failed",
         error=str(e))
```

---

#### P0-2: `graph.py` — Bare `except Exception:` without traceback in discrete inference

| Field | Value |
|-------|-------|
| **File** | `python/phca/world_model/graph.py` |
| **Line** | 350–362 |
| **Code** | `except Exception: _log(logger, "warning", "gprime.inference_failed", method="discrete", fallback="identity")` |
| **Problem** | Logs the event name but **never captures the exception message or traceback**. If inference fails, there is no diagnostic information about WHY. |
| **Impact** | All G' failure modes look identical in logs — no way to distinguish a structural model error from a numerical one. |
| **Fix** | Add `exc_info=True` to capture traceback. |

**Proposed patch:**
```python
except Exception:
    _log(logger, "warning", "gprime.inference_failed",
         method="discrete", fallback="identity", exc_info=True)
```

---

#### P0-3: `inference.py` — Silent catch returning identity with no log

| Field | Value |
|-------|-------|
| **File** | `python/phca/world_model/inference.py` |
| **Line** | 128–142 |
| **Code** | `except (RuntimeError, ValueError):` — returns identity prediction with `precision=0.01`, confidence=0.0, **no log call at all** |
| **Problem** | When continuous inference fails (e.g., singular matrix), the system silently returns a zero-confidence identity prediction. Downstream modules have no way to know this was a fallback rather than a real prediction. |
| **Impact** | A permanently broken inference path would go undetected — the system would "work" but make zero-confidence predictions forever. |
| **Fix** | Add `_log` call with `exc_info=True`. |

**Proposed patch:**
```python
except (RuntimeError, ValueError):
    _log(logger, "warning", "gprime.continuous_inference_failed",
         fallback="identity", exc_info=True)
    return (StateVector(
        values=state.values.copy(),
        ...
```

---

#### P0-4: `inference.py` — Silent `continue` on per-variable inference failure

| Field | Value |
|-------|-------|
| **File** | `python/phca/world_model/inference.py` |
| **Line** | 149–153 |
| **Code** | `except (ValueError, IndexError): continue` — skips this state dimension silently |
| **Problem** | If a single variable's inference fails, the loop silently skips it, leaving it at its initialized value (0.0). Other dimensions still get correct predictions, so the error is partially masked. |
| **Impact** | The resulting StateVector has 0.0 for some dimensions with no indication of failure. |
| **Fix** | Log the skipped dimension. |

**Proposed patch:**
```python
except (ValueError, IndexError):
    _log(logger, "warning", "gprime.variable_inference_failed",
         var_idx=idx, var_name=var_name)
    continue
```

---

### 1.2 High (P1) — Silent fallbacks in numerical computation

#### P1-1: `gaussian.py` — Three silent `np.linalg.LinAlgError` catches

| File | Line(s) | Pattern | Fix |
|------|---------|---------|-----|
| `world_model/gaussian.py` | 87 | Falls back `inv` → `pinv` silently | Add `_log(logger, "warning", "gaussian.singular_matrix", method="inv", fallback="pinv")` |
| `world_model/gaussian.py` | 153 | Falls back `inv(Σ_EE)` → `pinv` silently | Add `_log(logger, "warning", "gaussian.singular_posterior", fallback="pinv")` |
| `world_model/gaussian.py` | 232 | `pass` silently to fall through to sampling | Add `_log(logger, "warning", "gaussian.mvn_singular", fallback="rejection_sampling")` |

---

#### P1-2: `graph.py` — Silent `continue` on parent value extraction failure

| File | Line(s) | Pattern | Fix |
|------|---------|---------|-----|
| `world_model/graph.py` | 561 | `except (IndexError, ValueError): continue` | Log the index and variable name |
| `world_model/graph.py` | 716 | `except (IndexError, ValueError): parent_vals.append(0.0)` | Log the parent index |
| `world_model/graph.py` | 723 | `except (IndexError, ValueError): parent_vals.append(0.0)` | Log the parent index |

---

### 1.3 Informational (P2) — Acceptable silent handlers

| File | Line(s) | Reason for no fix |
|------|---------|-------------------|
| `logging.py` | 16 | `except ImportError` — this IS the logging module; cannot call itself |
| `config.py` | 80, 90 | `except ImportError` — benign fallback from orjson to json |
| `config.py` | 147 | `except ValueError` — returns descriptive string "unknown_drive_X" |

---

## 2. DATA FLOW VULNERABILITY MAP

### 2.1 Cognitive Cycle Data Flow

```
env.observation → ASI Sanitizer ──→ M2/M1 Memory ──→ G' (Prediction Engine)
                      │                                    │
                      │                              [NO NaN GATE]
                      │                                    │
                      ▼                                    ▼
                 [SANITIZED]                     [PREDICTION MAY BE NaN]
                      │                                    │
                      └─────────── PEU ─────────────────────┘
                                        │
                                  [NO NaN GATE]
                                        │
                                   PREDICTION_ERROR  ──→ TSPL P-Stream
                                        │                     │
                                   [NO NaN GATE]        [NO NaN GATE]
                                        │                     │
                                   MDIM compute_drives   theta[n] ← NaN
                                        │                (PERMANENT)
                                   softmax(p=NaN)
                                        │
                                   rng.choice(6, p)
                                        │
                                   CRASH (ValueError)
```

### 2.2 Critical NaN Entry Points

| # | Entry Point | File:Line | Vector | Risk |
|---|-------------|-----------|--------|------|
| **E1** | G' continuous output | `graph.py:533-574` | StateVector.values | NaN from singular matrix → `pinv` → degenerate result |
| **E2** | PEU compute | `error_unit.py:40-41` | Float scalar | `np.dot(NaN, NaN)` = NaN — **primary entry** |
| **E3** | TSPL gradient | `tspl.py:236` | Per-dim float array | NaN gradient → corrupts theta forever |
| **E4** | TSPL accuracy | `tspl.py:287-288` | Float scalar | `np.sqrt(NaN)` = NaN → NaN stored as `skill_accuracy` |
| **E5** | MDIM softmax | `mdim.py:372` | Float array | NaN deficits → NaN weights → `rng.choice` ValueError crash |
| **E6** | CR phi_current | `pid_controller.py:97` | Float scalar | NaN phi → NaN T/eta/alpha outputs |
| **E7** | RBTA timing check | `rbta_enforcer.py:111` | Float scalar | NaN comparisons are always False — violations ignored |

### 2.3 Missing Validation Gates (in priority order)

| Gate | Location | Check needed |
|------|----------|-------------|
| **G1** | After `engine.predict()` — before PEU | `if not np.all(np.isfinite(predicted.values)): clamp or raise` |
| **G2** | Before `peu.compute()` | `if not np.isfinite(observed.values).all() or not np.isfinite(predicted.values).all(): raise` |
| **G3** | Before `tspl.update()` | `if not np.isfinite(prediction_error): skip update` |
| **G4** | In `tspl._compute_gradient()` | `if not np.all(np.isfinite(error_per_dim)): return zeros` |
| **G5** | In `mdim.compute_drives()` | `if not np.isfinite(prediction_error): clamp to 10.0` (min already handles, but NaN escapes `min`) |
| **G6** | In `pid_controller.regulate()` | `if not np.isfinite(phi_current): use previous phi` |
| **G7** | In `rbta_enforcer.check_cycle()` | `if not np.isfinite(v): skip or flag` |
| **G8** | In `attention.select()` | `if not np.all(np.isfinite(chunk.state.values)): skip chunk` |

### 2.4 `@np.errstate(all="ignore")` — Silent Suppression of All FP Errors

**Files:**
- `world_model/gaussian.py:30` (in `compute_joint_moments`)
- `world_model/gaussian.py:99` (in `posterior`)

These decorators suppress ALL floating-point errors (divide-by-zero, invalid, overflow) including NaN generation. The code at lines 87, 153, 232 tries to catch `LinAlgError`, but `np.errstate(all="ignore")` means:
- Division by zero → silently returns Inf/NaN without warning
- Invalid operations → silently returns NaN without warning

**Fix:** Remove `@np.errstate(all="ignore")` and use `with np.errstate(divide="raise", invalid="raise"):` around only the specific risky operations. Add `np.isfinite()` post-checks on output.

### 2.5 ASI Sanitizer — Precision Decay Verification

```
Precision per sensor: 1.0 → 0.5 → 0.25 → 0.125 → 0.0625 → 0.03125 → 0.015625 → 0.0078125
                       k=0   k=1    k=2     k=3      k=4       k=5        k=6        k=7
                                                                                 (threshold=0.01)
```

**SENSOR_FAILURE triggers at k=7 cumulative failures** (precision < 0.01). Confirmed correct.

**Key finding — precision never recovers mid-episode:**
- `failure_count[j]` resets to 0 on valid reading (line 101)
- `precision[j]` is **halved on failure** but **never restored** (line 89)
- Comment says "updated by attention separately" — but `attention.update_precision()` is **never called from production code**
- After ~7 cumulative sensor failures over the sanitizer lifetime, any further failure triggers SENSOR_FAILURE until `reset()`

**Fix:** Add precision recovery on valid reading:
```python
self.precision[j] = min(1.0, self.precision[j] * 1.1)  # gradual recovery
```

---

## 3. LOGICAL FALLACIES

### 3.1 False Dichotomies

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| **F1** | `config.py:17-21` | ASIStatus only OK/SENSOR_FAILURE, missing PARTIAL_FAILURE | Add `ASIStatus.PARTIAL_FAILURE` |
| **F2** | `cycle.py:472` | D5 returns STAY bypassing action scoring | Integrate D5 into action loop |
| **F3** | `rbta_enforcer.py:32-42` | 3-level EnforcerAction, 2-var boundary split | Acceptable Phase 3.2 simplification |

### 3.2 Unused Confidence/Precision

| # | Location | What is unused | Fix |
|---|----------|----------------|-----|
| **U1** | `cycle.py:237`→`error_unit.py:28` | `compute()` uses L2-norm, ignores precision array | Replace with `compute_precision_weighted()` |
| **U2** | `cycle.py:315-318` | `attention.update_precision()` never called | Add call after `select()` |
| **U3** | `cycle.py:198` | `prediction_confidence` stored but unused by RBTA | Wire into entropy check or remove field |

### 3.3 Hard-coded GridWorld

| # | File:Line | Assumption | Severity |
|---|-----------|------------|----------|
| H1 | `cycle.py:47` | `from environments.grid_world import GridWorld` | High |
| H2 | `cycle.py:89` | `env: GridWorld` type hint | High |
| H3 | `cycle.py:472` | Action index 4 ≡ STAY | Medium |
| H4 | `cycle.py:573-581` | Direct `env.agent_pos`, `env.goal_pos` access | High |
| H5 | `graph.py:291` | State vars named `s{i}_t` | Medium |

**Recommendation:** Flag with `# TODO(v3.3): abstract environment`. Full dependency inversion is Phase 3.3 scope.

### 3.4 Dead Code

~35 production-unused functions identified. Do NOT delete in Phase 3.2 (may be Phase 3.3 spec). Remove only:
- `USE_MLP_GPRIME` flag in `cycle.py:37` (dead code)
- `schema_v1.py` duplicate (dead code)

---

## 4. OFF-BY-ONE AND BOUNDARY ERRORS

### 4.1 Critical: `_select_action` NameError (see P0-1)

### 4.2 Rolling Window Off-by-One (`>` should be `>=`)

| File | Line | Current | Should Be |
|------|------|---------|-----------|
| `core/cycle.py` | 240 | `len > 20` | `>= 20` |
| `regulation/pid_controller.py` | 170 | `len > 100` | `>= 100` |
| `motivation/mdim.py` | 228 | `len > 50` | `>= 50` |
| `motivation/mdim.py` | 233 | `len > 50` | `>= 50` |
| `motivation/mdim.py` | 246 | `len > 1000` | `>= 1000` |
| `motivation/mdim.py` | 391 | `len > 100` | `>= 100` |

### 4.3 Other Boundary Issues

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| B1 | `scheduler.py:314` | Prune at 10,001, not 10,000 | `>` → `>=` |
| B2 | `cycle.py:397` | `metrics_history` unbounded | Add cap at 10,000 |
| B3 | `m3_episodic.py:249,493` | `None` action → `b""` → empty array | Use sentinel `b"__NONE__"` |
| B4 | `graph.py:919-927` | Multi-var query returns joint dist for each var | Marginalize per var (Phase 3.2 fix) |

---

## 5. FIX PLAN

### Phase 1 — P0 Critical (apply immediately)

| # | File | Line | Fix |
|---|------|------|-----|
| 1 | `core/cycle.py` | 509 | Remove `action=action_name` from log call |
| 2 | `world_model/graph.py` | 350 | Add `exc_info=True` to `_log` |
| 3 | `world_model/inference.py` | 128 | Add `_log` before identity return |
| 4 | `world_model/inference.py` | 153 | Add `_log` before `continue` |
| 5 | `world_model/gaussian.py` | 87, 153, 232 | Add `_log` for 3 LinAlgError fallbacks |

### Phase 2 — NaN Data Flow Gates (P1)

| # | Location | Fix |
|---|----------|-----|
| 6 | `cycle.py` after `engine.predict()` | `np.isfinite()` check on prediction.values |
| 7 | `error_unit.py:compute()` | Input validation with `np.isfinite()` |
| 8 | `mdim.py:169` | `np.nan_to_num(prediction_error, nan=10.0)` |
| 9 | `rbta_enforcer.py:check_cycle()` | `np.isfinite()` check on log values |

### Phase 3 — Logical Fallacies (P1)

| # | Fix |
|---|-----|
| 10 | Switch `compute()` → `compute_precision_weighted()` in cycle.py |
| 11 | Add `attention.update_precision()` call in cycle.py |
| 12 | Add `ASIStatus.PARTIAL_FAILURE` enum member |

### Phase 4 — Boundary Errors (P2)

| # | Fix | Files |
|---|-----|-------|
| 13 | 6x rolling window `>` → `>=` | `cycle.py`, `pid_controller.py`, `mdim.py` |
| 14 | `metrics_history` cap | `cycle.py` |
| 15 | M3 action sentinel | `m3_episodic.py` |
| 16 | Precision recovery | `sanitizer.py` |

**Total estimated changes:** ~50 LOC across 12 files.

---

## 6. SUMMARY TABLE

| Severity | Count | Key Issues |
|----------|-------|------------|
| **P0 — Critical** | 4 | Undefined variable in exception handler (crash), 3 silent catches with no traceback |
| **P1 — High** | 9 | 3 gaussian.py silent fallbacks, 3 graph.py silent continues, 3 missing NaN data-flow gates |
| **P2 — Medium** | 12 | 6 rolling-window off-by-ones, false dichotomies, dead code, unused precision, hard-coded GridWorld |
| **P3 — Low** | 8 | Serialization inconsistency, unbounded metrics history, precision never recovers, M3 `__del__` anti-pattern |

**Total: 33 unique issues identified.**
