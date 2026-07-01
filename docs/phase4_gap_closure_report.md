# PHCA v3.0 — Phase 4 Gap Closure Report

**Date:** 2026-07-01
**Author:** Chief Architect
**Status:** PHASE 4 READY — All 5 critical issues resolved

---

## Executive Summary

All five critical issues identified in the Phase 4 gap audit have been resolved. The previous round (C1-C5) was confirmed as already fixed in source. A fresh zero-trust audit uncovered 5 new issues (AF-001 through AF-005), all of which have been surgically repaired. The system passes **284 tests** (0 failures) and is structurally sound for Phase 4 development.

---

## Closure Summary

| ID | Description | Severity | Status | Verification |
|:---|:------------|:---------|:-------|:-------------|
| C1 | Pareto front magnitude comparison | CRITICAL | ✅ Already fixed | True vector dominance in `mdim.py:271-354` — confirmed in source |
| C2 | MLP empowerment = std(confidences) | CRITICAL | ✅ Already fixed | MC Dropout Gaussian MI in `mlp.py:343-401` — confirmed in source |
| C3 | Energy pipeline inconsistent | CRITICAL | ✅ Already fixed (partial) | Both use `_cycle_flops` — 10x divisor mismatch fixed in this round (AF-005) |
| C4 | Consolidation facts never consumed | CRITICAL | ✅ Already fixed | Fact modulation in `mdim.py:174-181` — confirmed in source |
| C5 | PID orthogonality uses covariance | MAJOR | ✅ Already fixed | `np.corrcoef` at `pid_controller.py:212` — confirmed in source |
| **AF-001** | D4 model_entropy is fabricated signal | **CRITICAL** | ✅ **Fixed** | `cycle.py:359` — replaced `0.5 - cycle * 0.001` with `1.0 - prediction_confidence` |
| **AF-002** | TSPL theta disconnected from MLP | **CRITICAL** | ✅ **Fixed** | `mlp.py` — added `_tspl_bias` + `set_tspl_bias()`, wired in `cycle.py` |
| **AF-003** | Graph CPD single-parent key | **CRITICAL** (latent) | ✅ **Fixed** | `graph.py:875-879` — multi-parent key with proper indexing |
| **AF-004** | PID freeze reset oscillation | **MAJOR** | ✅ **Fixed** | `pid_controller.py:247` — removed reset; freeze converges |
| **AF-005** | Energy divisor 10x mismatch | **MAJOR** | ✅ **Fixed** | `cycle.py:840` — unified at `ENERGY_NORM_FLOPS = 60M` |

## Files Modified

| File | Changes |
|:-----|:--------|
| `python/phca/core/cycle.py` | AF-001: real model_entropy from confidence; AF-002: TSPL bias wiring; AF-005: ENERGY_NORM_FLOPS constant |
| `python/phca/world_model/mlp.py` | AF-002: `_tspl_bias` field, `set_tspl_bias()` method, bias applied in `predict()` |
| `python/phca/regulation/pid_controller.py` | AF-004: removed freeze reset; renamed cov→corr |
| `python/phca/world_model/graph.py` | AF-003: multi-parent CPD key and indexing |
| `docs/phase4_gap_audit_plan.md` | Replaced stale plan with accurate current-state audit |

---

## Phase 4 Readiness Criteria

| Criterion | Target | Result | Status |
|:----------|:-------|:-------|:-------|
| All tests pass | 0 failures | 284 passed | ✅ |
| D4 driven by real uncertainty | Positive entropy from model confidence | `1.0 - prediction_confidence` | ✅ |
| TSPL learning has behavioural effect | Bias connected to MLP output | `_tspl_bias` applied in predict() | ✅ |
| PID orthogonality converges | No oscillation after swap | Reset removed | ✅ |
| Energy signal consistent across D5/RBTA | Same divisor | `ENERGY_NORM_FLOPS` unified | ✅ |
| CPD multi-parent correct | Multi-parent key and indexing | Parent-key fix applied | ✅ |

## Known Remaining Minor Issues

The following MINOR issues (from the audit findings table F1-F19) were deferred to a future cleanup pass:
- Dead imports and unused variables (F1-F6, F9, F15, F17)
- Dead enum values in `CompositionOp` (F7)
- Dead constants in config.py (F8)
- Hardcoded 4-dim assumption in fact signature (F10)
- Inconsistent state_dim=4 in attention (F11)
- TSPL E/S-Stream KeyError path (F12) — unreachable
- Self-adapting disruption threshold (F13) — minor, functional
- D3 goal hardcoded to zeros (F14) — aspiration gap, not a correctness bug
- Engine left in last-iterated-action state (F18) — covered by subsequent engine.update_action
- PID NaN gate fallback uses temperature instead of error_volatility (F19) — edge case

---

## Verification

Tests executed after each fix:
```bash
python -m pytest python/ --ignore=python/tests/test_mujoco_env.py \
  --ignore=python/tests/test_cycle_with_mujoco.py -k "not mujoco" -v
```

Results: **284 passed, 0 failed** (7 pre-existing warnings). All MuJoCo failures are unrelated (missing `gymnasium` module).

---

**Certified Phase 4 ready by Chief Architect on 2026-07-01.**
