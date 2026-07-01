# PHCA v3.0 — Phase 4 Gap Audit & 5-Step Surgical Plan

**Author:** Chief Architect & Principal Engineer  
**Date:** 2026-07-01  
**Status:** BUILD MODE — 3 fixes remaining (2 already resolved in prior gap closure)  
**Verdict:** CONDITIONAL — 2 critical measurement proxies (confidence, empowerment) and 1 energy estimation remain unvalidated. The online/replay conflict (AF-003) and TSPL theta drift (AF-004) were already resolved in D-055 and D-053 respectively.

---

## 1. Executive Summary

The codebase has undergone extensive gap closure (D-045 through D-056), resolving 12 of the 26 findings from the Phase 4 gap audit. However, **the two most architecturally critical flaws remain**: MLP confidence = `exp(-MSE)` (AF-001) and empowerment = `std(confidences)` (AF-002). These are not bugs — they are mathematically unvalidated proxies that every downstream decision (action selection, MDIM drive computation, D6 empowerment) depends on. A 10% error in either proxy silently degrades all adaptive behaviour. The energy formula (AF-005) was partially addressed (FLOP logging added, energy_cost made dynamic) but `runtime × 50.0` remains the primary energy signal with no physical basis. This plan addresses the 3 remaining issues and verifies the 2 already-fixed ones.

---

## 2. Audit Findings — Current State

| ID | Issue | Severity | Status | Location |
|----|-------|----------|--------|----------|
| **AF-001** | MLP Confidence = `exp(-MSE)` is not a valid confidence | CRITICAL | **UNRESOLVED** | `mlp.py:290-308` |
| **AF-002** | Empowerment = `std(confidences)` is not mutual information | CRITICAL | **UNRESOLVED** | `cycle.py:685-707`, `mdim.py:220-250` |
| **AF-003** | Online vs Replay learning conflict | CRITICAL | ✅ **FIXED** (D-055) | `mlp.py:116-198` |
| **AF-004** | TSPL theta drift from MLP weights | MAJOR | ✅ **FIXED** (D-053) | `tspl.py:91-276`, `cycle.py:271-285` |
| **AF-005** | Energy = `runtime × 50.0` has no physical basis | MAJOR | ⚠️ **PARTIAL** — energy_cost dynamic (D-041), FLOP logging added (D-054), but primary signal still magic constant | `cycle.py:738-820` |

---

## 3. The 3 Remaining Critical Issues

### AF-001 [CRITICAL] MLP Confidence = `exp(-MSE)` Is Not a Valid Confidence Measure

| Field | Value |
|-------|-------|
| **File** | `python/phca/world_model/mlp.py:290-308` |
| **Problem** | `_compute_confidence()` returns `exp(-0.5 × mean((pred - target)²))`. A perfectly wrong model shows confidence ~0.14. No Bayesian treatment, no dropout, no ensemble. Consumed by action selection (cycle.py:566-578), MDIM drives (mdim.py:147-250), and empowerment (cycle.py:699). |
| **Fix** | Add MC Dropout: N=20 forward passes with dropout → mean prediction + variance → confidence = inverse-variance. |
| **Effort** | 3-4 hours |
| **Dependencies** | None |

### AF-002 [CRITICAL] Empowerment = `std(confidences)` Is Not Mutual Information

| Field | Value |
|-------|-------|
| **File** | `python/phca/core/cycle.py:685-707` |
| **Problem** | `_estimate_empowerment()` returns `np.std(confidences)` across actions. This is not `I(S'; A \| s)`. D6 drive fires on a mathematically unrelated quantity. |
| **Fix** | For Gaussian G': compute closed-form differential entropy: `I = H(S'|s) - Σ_a p(a|s)·H(S'|s,a)`. For MLP: keep std(confidences) as documented fallback. |
| **Effort** | 3-4 hours |
| **Dependencies** | AF-001 (or implement Gaussian path independently) |

### AF-005 [MAJOR] Energy = `runtime × 50.0` Has No Physical Basis

| Field | Value |
|-------|-------|
| **File** | `python/phca/core/cycle.py:738-820` |
| **Problem** | Primary energy signal is `max(0.1, min(10.0, runtime_s × 50.0))`. Factor 50.0 chosen to match bound definitions. D5 drive behaviour relative to actual energy is unknown. |
| **Fix** | Make FLOP-based energy the primary signal. Normalize: `energy = clip(flops / baseline_flops, 0.1, 10.0)`. |
| **Effort** | 2-3 hours |
| **Dependencies** | None |

---

## 4. Surgical Repair Plan

| Order | ID | Fix | Effort | Dependencies | Current Status |
|-------|----|-----|--------|-------------|----------------|
| 1 | AF-003 | Verify replay-only learning | 10min | None | ✅ Already fixed (D-055) |
| 2 | AF-004 | Verify TSPL accuracy sync | 10min | None | ✅ Already fixed (D-053) |
| 3 | AF-005 | Replace runtime×50.0 with FLOP-based energy | 2-3h | None | ⚠️ Partial |
| 4 | AF-001 | MC Dropout for MLP confidence | 3-4h | None | ❌ Pending |
| 5 | AF-002 | True empowerment MI for Gaussian | 3-4h | AF-001 or independent | ❌ Pending |

---

## 5. Acceptance Criteria

1. **AF-001**: MLP confidence ≤ 0.05 for wrong predictions, ≥ 0.95 for correct. Pearson r > 0.8 with actual accuracy.
2. **AF-002**: Gaussian MI closed-form produces empowerment that differs from std(confidences) by > 10% on at least 50% of cycles.
3. **AF-005**: Energy values reflect actual computation (more compute = higher energy). D5 deficit varies dynamically.
4. **All tests pass** — full suite.
5. **Φ-IQ ≥ 0.6** — no regression.

---

