# PHCA v3.0 — Phase 4 Gap Audit & 5-Step Surgical Plan

**Date:** 2026-07-01
**Auditor:** Chief Architect & Principal Engineer
**Status:** NOT READY — 2 fractured core systems (D4 guidance, TSPL gradient) must be fixed before proceeding

---

## Executive Summary

PHCA v3.0 passes 289 tests, achieves Φ-IQ = 0.626 overall (0.320 at Level 2), runs at ~50ms latency with 3.4% failure rate. The Phase 4 gap closure report correctly claimed C1, C2, C4, C5 were fixed — true vector-dominance Pareto, MC Dropout Gaussian MI, fact consumption in MDIM, and PID correlation coefficient are all confirmed in source. However, a fresh zero-trust audit reveals **two new critical fractures**: D4 (epistemic curiosity) is driven by a fabricated linear decay (`0.5 - cycle * 0.001`) instead of actual model uncertainty — the real G' posterior entropy is computed but never forwarded, violating A3. Additionally, TSPL gradient computation degenerates to flat mean-abs-error for every MLP parameter except output bias — the P-Stream learning channel is effectively a placebo. A latent CPD corruption bug in the graph model and two functional design flaws (PID freeze oscillation, energy scaling mismatch) complete the top 5. **Verdict: NOT READY.** These five issues must be resolved before Phase 4 proceeds.

---

## Audit Findings (All Dimensions)

| ID | Dimension | Severity | Location | Description |
|:---|:----------|:---------|:---------|:------------|
| **AF-001** | Hidden Assumption / AI Hallucination | **CRITICAL** | `cycle.py:359` | `model_entropy = 0.5 - self.cycle_count * 0.001` — D4's primary input is a fabricated linear decay, not G' posterior entropy; goes negative after cycle 500. Real entropy computed at `cycle.py:849` but never forwarded to MDIM. Violates A3 (Incomplete Knowledge). |
| **AF-002** | Logical Fallacy | **CRITICAL** | `tspl.py:224-245` | `_compute_gradient()` falls through to flat `mean_abs_error` fallback for every MLP parameter except output bias b3. Weight matrices `(89,128)`, `(128,128)`, `(128,84)` and hidden biases `(128,)` all receive identical gradient per element — no per-dimension structure. P-Stream learning is a shape-mismatch noise channel. |
| **AF-003** | AI Hallucination | **CRITICAL** | `graph.py:875,879,834` | CPD normalization uses `f"{parents[0]}->{node_name}"` as key — only captures first parent. Indexing `parent_idx % card` uses child cardinality instead of parent cardinality. Latent bug: not triggered with single-parent GridWorld topology, but structurally wrong and will corrupt multi-parent graphs. |
| **AF-004** | Over-Simplification | **MAJOR** | `pid_controller.py:244-247` | After 100 high-cov cycles, freeze swap resets `_high_cov_cycles = 0`, causing oscillation: freeze parameter A for 100 cycles → swap to B for 100 cycles → swap back. Never converges to a stable freeze set. Also: variable `cov` actually holds `corrcoef` (cosmetic). |
| **AF-005** | Hidden Assumption | **MAJOR** | `cycle.py:352 vs 840` | D5 divides FLOPs by 60M (`[0.01, 1.0]` range), RBTA divides by 6M (`[0.1, 10.0]` range) — same FLOP count, 10× different scaled values. The "unified energy" claim is misleading. Both use FLOP source but with inconsistent calibration. |
| F1 | Dead Code | MINOR | `scheduler.py:256` | `rng = np.random.RandomState(42)` in `_extract_facts()` — created but never used. |
| F2 | Dead Code | MINOR | `scheduler.py:324` | `n_new = 0` immediately overwritten at line 327. |
| F3 | Dead Code | MINOR | `pid_controller.py:96-98` | `self._param_history` dict initialized but never updated or read. |
| F4 | Dead Code | MINOR | `attention/attention.py:62` | `self.state_dim` stored in constructor but never read — and value `4` is inconsistent with GridWorld's 84. |
| F5 | Dead Code | MINOR | `graph.py:105` | `self.state_history` written and trimmed but never read by production code (test-only). |
| F6 | Dead Code | MINOR | Various | `GoalVector.drive_name` property, `ensure_logging()` function, `M1SensoryBuffer.read_latest()`/`is_full`, `GridWorld.render()` — defined but never called from production. |
| F7 | Dead Code | MINOR | `hpm/parser.py:18-27` | `CompositionOp` — `HIERARCHY`, `RECURSE`, `INTERLEAVE`, `TEMPORAL_INVARIANT`, `REACTIVE` — 5 of 8 enum values never referenced. |
| F8 | Dead Code | MINOR | `config.py:183-185` | `CYCLE_TARGET`, `T_COMP`, `T_SYNC` — defined but never imported or referenced. |
| F9 | Dead Code | MINOR | `mdim.py:53,584` | `GoalStackEntry.completed` — declared, never set or checked. `MetaStableState.drive_values` — set but never read. |
| F10 | Over-Simplification | MINOR | `scheduler.py:264-267` | Fact signature truncates `state_after.values[:4]` — hardcoded 4-dim assumption. |
| F11 | Over-Simplification | MINOR | `attention/attention.py:40-45` | `state_dim=4` hardcoded — inconsistent with MLP/GridWorld's 84-dim states. |
| F12 | Over-Simplification | MINOR | `tspl.py:131` | E-Stream / S-Stream path causes KeyError before disabled-stream guard executes — broken but unreachable. |
| F13 | Hidden Assumption | MINOR | `mdim.py:254-258 vs 566-575` | Disruption detection baselines computed from same rolling window containing current signal — threshold self-adapts to the signal it detects. |
| F14 | Hidden Assumption | MINOR | `mdim.py:469-470` | D3 goal target hardcoded to zeros ("seek familiar states") — not near current state as claimed. |
| F15 | Dead Code | MINOR | `cycle.py:605` | `self._cached_confidences` assigned but never read. |
| F16 | Dead Code | MINOR | `graph.py:875` | Single-parent key in CPD normalization (covered under AF-003 as the core bug manifestation). |
| F17 | Over-Simplification | MINOR | `mlp.py:92-93` | `self._last_input` and `self._last_activations` — cache set but never read by production code. |
| F18 | Over-Simplification | MINOR | `cycle.py:564-566` | `_select_action` iterates all actions calling `engine.update_action()` — engine left in state of last iterated action, not the selected `best_action`. |
| F19 | Over-Simplification | MINOR | `pid_controller.py:135` | NaN gate fallback uses previous temperature value (`self._prev_output[0]`), not previous error_volatility — fallback returns unrelated parameter. |

---

## The 5 Critical Issues (Detailed)

### AF-001: D4 Epistemic Curiosity Driven by Fabricated Signal

| Field | Value |
|:------|:------|
| **Dimension** | Hidden Assumption / AI Hallucination |
| **Severity** | CRITICAL |
| **Location** | `python/phca/core/cycle.py:359` |
| **Description** | `model_entropy` in the MDIM context is `"model_entropy": 0.5 - self.cycle_count * 0.001`. This is a linear decay from 0.5 that goes negative after cycle 500. D4 (epistemic curiosity) uses this as its primary drive value: `d4_value = model_entropy` (mdim.py:215). The real G' posterior entropy IS computed at `cycle.py:843-849` and stored in `self.belief_entropies["G'"]` but is **never forwarded to MDIM**. The entire exploration system is operating on a fake clock signal, not on actual model uncertainty. This directly violates Invariant A3 (Incomplete Knowledge → all beliefs carry uncertainty; entropy floor is enforced). Without real uncertainty, the agent cannot know what it doesn't know. |
| **Root Cause** | A placeholder value was left in place when the entropy computation pipeline was not yet connected to MDIM. The entropy is computed in `_collect_runtime_log` (late in the cycle) while MDIM context is built earlier. The two were never wired together. |
| **Proposed Fix** | (1) Move or duplicate the G' entropy computation earlier in `step()` so it's available before MDIM context construction. (2) Replace `"model_entropy": 0.5 - self.cycle_count * 0.001` with `"model_entropy": self.belief_entropies.get("G'", 0.5)` after ensuring `belief_entropies` is populated. (3) Remove the decay formula entirely. |
| **Effort Estimate** | 2 hours |
| **Dependencies** | None |
| **Acceptance Criteria** | Unit test: after `_collect_runtime_log`, verify `belief_entropies["G'"] > 0`. Integration test: run 600 cycles; verify D4 value after cycle 500 is positive (not negative). |

### AF-002: TSPL Gradient Computation Degenerates to Flat Noise for MLP

| Field | Value |
|:------|:------|
| **Dimension** | Logical Fallacy |
| **Severity** | CRITICAL |
| **Location** | `python/phca/learning/tspl.py:224-245` |
| **Description** | `_compute_gradient()` tries to match per-dimension error signals to parameter shapes. For MLP parameters: `w1: (89,128)`, `b1: (128,)`, `w2: (128,128)`, `b2: (128,)`, `w3: (128,84)`, `b3: (84,)`. The per-dimension `scaled_error` has shape `(84,)`. The matching logic: `ndim==1, shape[0]==84` → matches only `b3`. `ndim==2, shape[0]==84` → matches nothing (first dims are 89, 128, 128). `size == 84` → matches nothing (sizes are 11392, 128, 16384, 128, 10752, 84). All weight matrices and hidden biases fall through to the fallback at line 244: `mean_abs_error` replicated into every element. This means w1, b1, w2, b2, w3 all receive a flat gradient — no per-dimension structure, no directional signal. The P-Stream learning channel, which is supposed to bias parameters toward better predictions, is effectively a noise process for 5 of 6 MLP parameter groups. Only b3 (output bias) receives a proper gradient. |
| **Root Cause** | The gradient shaping logic was designed for a different parameter structure (likely a linear or single-layer model) and was never adapted to the MLP's 3-layer architecture. The `scaled_error` shape is `(state_dim,)` = `(84,)`, but MLP weight matrices have first dimensions of 89, 128, 128 — all mismatched. |
| **Proposed Fix** | Replace the shape-matching approach with proper backpropagation: project the `(state_dim,)` error signal backward through each layer's transpose to compute per-parameter gradients. Specifically: (1) Cache the MLP's forward activations (z1, a1, z2, a2, out) during predict. (2) In `_compute_gradient`, use these cached activations to compute `d_out`, then propagate back through w3.T, ReLU, w2.T, ReLU, w1.T. (3) This gives dimensionally correct gradients for every parameter. Alternatively, expose the MLP's `_backward()` method via a new public `compute_gradient(state, action, target)` that TSPL can call. |
| **Effort Estimate** | 6 hours |
| **Dependencies** | MLP must cache activations (already done in `self._last_activations` — currently dead, used here). |
| **Acceptance Criteria** | Unit test: for a random MLP state, compute gradient via TSPL and via MLP._backward; verify Spearman ρ > 0.9 between gradient vectors. Integration test: after 100 learning steps, MLP prediction error must decrease (was not decreasing before, since gradient was flat noise). |

### AF-003: Graph CPD Normalization Uses Single-Parent Key Only

| Field | Value |
|:------|:------|
| **Dimension** | AI Hallucination |
| **Severity** | CRITICAL (latent) |
| **Location** | `python/phca/world_model/graph.py:875,879` (normalization), `834` (learning) |
| **Description** | In `_normalize_cpds()`, the CPD parameter key is `f"{parents[0]}->{node_name}"` — only the **first parent** is used. For a node with multiple parents, all parent combinations map to the same key, sharing the same transition counts. The indexing `parent_idx % card` uses the child node's cardinality instead of the correct parent combination index, collapsing distinct parent states. In `learn()`, the key at line 833 is `f"{s_name}->{t1_name}"` which is correct for single-parent temporal edges but the normalization side at line 875 does not handle multi-parent nodes. Currently latent: GridWorld's graph has each state dimension node with exactly one parent (itself at t-1), so multi-parent code paths never execute. However, the code explicitly handles `n_parent_combos > 1` (line 867, 872), implying multi-parent was intended — and the implementation is broken. |
| **Root Cause** | The CPD normalization code was written assuming each node has exactly one parent. The loop over `parent_idx in range(n_parent_combos)` suggests multi-parent was considered but not correctly implemented — the key should include all parent IDs, and the row index should use the full parent combination index, not `parent_idx % card`. |
| **Proposed Fix** | (1) Replace key `f"{parents[0]}->{node_name}"` with a key that includes all parents, e.g., `"&".join(parents) + "->" + node_name`. (2) Replace `parent_idx % card` with the correct indexing into the CPD matrix: encode the parent combination index as the flattened product of parent cardinalities. (3) Update the learning side (line 833) to use the same multi-parent key scheme if multi-parent nodes are present. |
| **Effort Estimate** | 4 hours |
| **Dependencies** | None — bug is latent but structurally wrong. |
| **Acceptance Criteria** | Unit test: construct a 2-parent discrete node, run 100 learn steps, verify CPD params show distinct distributions for each parent combination. Not just `parent[0]`'s distribution repeated. |

### AF-004: PID Freeze Cycle Reset Causes Oscillation

| Field | Value |
|:------|:------|
| **Dimension** | Over-Simplification |
| **Severity** | MAJOR |
| **Location** | `python/phca/regulation/pid_controller.py:244-247` |
| **Description** | The orthogonality freeze mechanism swaps which parameter is frozen after 100 consecutive high-covariance cycles. However, line 247 resets `self._high_cov_cycles = 0` after the swap. This means: freeze T for 100 cycles → swap to freeze eta → `_high_cov_cycles` resets → 100 more cycles → swap back to T → repeat indefinitely. The system never converges to a stable freeze state — it oscillates between freezing one parameter for 100 cycles then the other. The variable name `cov` throughout (lines 212, 215-216, 220-222, 225) actually holds the output of `np.corrcoef`, which is a **correlation** matrix, not covariance. Cosmetic but misleading. |
| **Root Cause** | The reset at line 247 treats the swap as a resolution rather than a symptom. If high covariance persists, it means the root cause (redundant parameter dynamics) hasn't changed — swapping which parameter is frozen is a band-aid that needs to be sustained, not reset. |
| **Proposed Fix** | Remove the reset at line 247. Instead, after the first swap, keep `_high_cov_cycles` incrementing so the system maintains the freeze. Only swap back if covariance drops and then rises again (cold restart). Alternatively, make both parameters flagged as "one must be frozen" and let the swap be permanent until covariance drops below threshold naturally. |
| **Effort Estimate** | 1 hour |
| **Dependencies** | None |
| **Acceptance Criteria** | Unit test: simulate 250 cycles with constant high covariance between T and eta → verify after cycle 100, the freeze swaps and stays swapped (doesn't swap back at cycle 200). |

### AF-005: Energy Pipeline 10× Scaling Mismatch

| Field | Value |
|:------|:------|
| **Dimension** | Hidden Assumption |
| **Severity** | MAJOR |
| **Location** | `python/phca/core/cycle.py:352` (D5) vs `line 840` (RBTA) |
| **Description** | Both D5 and RBTA now use the same `_cycle_flops` source for G' energy (C3 fix applied). However: D5 normalises by 60M → range `[0.01, 1.0]` (line 352), while RBTA normalises by 6M → range `[0.1, 10.0]` (line 840). Same FLOP count yields 10× different values. The two subsystems receive inconsistent energy signals from the same underlying computation. Additionally, non-G' modules still use `runtime_s * 50.0` (line 832), which is a wall-clock proxy and has no consistent relationship to FLOP-based values. The unification is partial: only G' is unified, and with a 10× scaling discrepancy. |
| **Root Cause** | The two divisors were calibrated independently — 60M for D5 to keep `energy_cost` in `[0,1]` drive range, 6M for RBTA to match pre-existing bound ranges. No cross-calibration was performed. |
| **Proposed Fix** | (1) Move the divisor to a named constant: `ENERGY_NORMALISATION_FLOPS = 60_000_000.0`. (2) Change line 840 to use the same constant: `self._cycle_flops / 60_000_000.0`. (3) Keep the RBTA clamp at `[0.1, 10.0]` — this widening is fine for a constraint check based on the same normalised value. (4) Document the scaling factor rationale (6e7 FLOPs ≈ 1.0 on drive scale, derived from MLP forward pass at h=128, bs=32, ts=4). |
| **Effort Estimate** | 1 hour |
| **Dependencies** | None |
| **Acceptance Criteria** | Unit test: mock `_cycle_flops = 30_000_000.0` → verify `energy_cost == 0.5` and `energy_log["G'"] == 0.5`. |

---

## Surgical Repair Plan

### Execution Order (Sequential — single engineer)

```
Step 1: AF-001 (2h) — Wire real model_entropy      ← MOST CRITICAL, no deps
Step 2: AF-002 (6h) — Fix TSPL gradient for MLP     ← Depends on MLP cache (already exists, dead)
Step 3: AF-004 (1h) — Fix PID freeze reset          ← Simple fix, no deps
Step 4: AF-005 (1h) — Unify energy divisor          ← Simple fix, no deps
Step 5: AF-003 (4h) — Fix CPD multi-parent key      ← Latent bug, deferrable
```

### Detailed Steps

#### Step 1: AF-001 — Wire Real Model Entropy Into MDIM (2h)

**Files:** `python/phca/core/cycle.py`

1. In `step()`, compute `model_entropy` from `self.belief_entropies` after `_collect_runtime_log()` populates it. The current ordering has MDIM context built at line 355 and `_collect_runtime_log` at line 785. Move the entropy computation earlier, or compute it inline from `self.engine`'s prediction confidence before MDIM context.
2. Replace `"model_entropy": 0.5 - self.cycle_count * 0.001` with a call to `self._compute_model_entropy()`.
3. Implement `_compute_model_entropy()` that reads G' posterior entropy from the engine (or from cached prediction confidence variance).
4. Verify the baseline disruption detection (`_baseline_wm_entropy`) receives real entropy values.

**Difficulty:** Requires understanding ordering of `step()` — MDIM context is built early (line 355), but `belief_entropies` is computed late (line 843). Need to compute entropy earlier or pass it directly from the prediction phase.

**Chief Architect Review Questions:**
- Does this preserve A1 (no new infinite loops)? Yes — single computation per cycle.
- Does this preserve A3 (entropy floor)? Yes — now entropy is real.
- Does this add unnecessary complexity? No — replaces a fake value with a real one.
- New silent failure modes? If `belief_entropies` is empty, fallback to 0.5.

#### Step 2: AF-002 — Fix TSPL Gradient for MLP (6h)

**Files:** `python/phca/learning/tspl.py`, `python/phca/world_model/mlp.py`

**Option A (preferred):** Expose MLP backward computation to TSPL.
1. Add method `WorldModelMLP.compute_gradient(state, action, target) -> Dict` that calls `_forward` + `_backward` and returns per-parameter gradients.
2. In `cycle.py`, when TSPL updates, pass `gprime.compute_gradient(state, action, next_state)` instead of the current error-only gradient.
3. TSPL receives dimensionally correct gradients for all MLP parameters.

**Option B (simpler):** Fix shape matching in `tspl.py:_compute_gradient()`.
1. For 2D params where `param.shape[0] != scaled_error.shape[0]`, compute `error_projected = scaled_error @ some_projection` — but this requires knowing the parameter structure, which TSPL doesn't.

**Recommendation:** Option A — it reuses existing tested backward code.

**Chief Architect Review Questions:**
- Does this preserve A4 (Prediction as Primary)? Yes — now gradient actually drives learning.
- Does this add unnecessary coupling? Minimal — MLP already exposes `get_theta()`. Adding `compute_gradient()` is consistent.
- New silent failure modes? If `compute_gradient` is called before any predict, cached activations may be stale. Guard with a version counter.

#### Step 3: AF-004 — Fix PID Freeze Reset (1h)

**Files:** `python/phca/regulation/pid_controller.py`

1. Remove `self._high_cov_cycles = 0` at line 247.
2. Let `_high_cov_cycles` continue incrementing after swap.
3. Only swap back if covariance drops below threshold and then rises again (new cold detection window). Simplest: leave the freeze in place once swapped.

**Chief Architect Review Questions:**
- Does this preserve A1? Yes — no new loops.
- Does this preserve A5 (Feedback-Driven Adaptation)? Yes — now freeze converges.
- New silent failure modes? Once swapped, T might never be frozen again even if circumstances change. Mitigation: add a decay mechanism that periodically re-evaluates the freeze after N cycles of low covariance.

#### Step 4: AF-005 — Energy Divisor Unification (1h)

**Files:** `python/phca/core/cycle.py`

1. Define `ENERGY_NORM_FLOPS = 60_000_000.0` as a module-level constant.
2. Change line 840 divisor from `6_000_000.0` to `ENERGY_NORM_FLOPS`.
3. Keep RBTA clamp as `[0.1, 10.0]` — different clamp ranges are valid for different subsystems (drive vs constraint), but the base normalised value must match.

**Chief Architect Review Questions:**
- Does this preserve any invariant? Yes — now D5 and RBTA agree on what "1.0 energy" means.
- New coupling? No — just constant unification.

#### Step 5: AF-003 — Fix CPD Multi-Parent Key (4h)

**Files:** `python/phca/world_model/graph.py`

1. In `_normalize_cpds()`, build the key from all parents: `"&".join(sorted(parents)) + "->" + node_name`.
2. Replace `parent_idx % card` with proper multi-dimensional indexing: decode `parent_idx` into individual parent values using the product of cardinalities, then use those as indices into the CPD matrix.
3. In `learn()`, update the key construction to include all parents if multi-parent.

**Chief Architect Review Questions:**
- Does this add complexity? Yes — but the alternative (broken multi-parent) is worse.
- Latent bug — is it worth fixing now? Yes — it's a correctness issue that will bite anyone extending the graph.

### Timeline

| Step | Issue | Effort |
|:-----|:------|:-------|
| 1 | AF-001 — Real model_entropy | 2h |
| 2 | AF-002 — TSPL gradient | 6h |
| 3 | AF-004 — PID freeze reset | 1h |
| 4 | AF-005 — Energy divisor | 1h |
| 5 | AF-003 — CPD multi-parent key | 4h |
| **Total** | | **14 engineering hours** |

With 1 engineer: ~2 wall-clock days.

---

## Assumption Validation Plan

| Assumption | Experiment | Success Criterion |
|:-----------|:-----------|:-----------------|
| Real model entropy improves exploration | Run 2 × 500-cycle benchmarks: one with fake entropy (current), one with real entropy. Compare D4 deficit variance and state coverage | Real entropy benchmark covers ≥ 20% more distinct states |
| Fixed TSPL gradient reduces prediction error | Run 2 × 500-cycle benchmarks: one with old flat gradient, one with backprop gradient. Compare MLP prediction error at cycle 500 | New gradient: error decreases ≥ 20% relative to old gradient |
| PID freeze converges after fix | Run 500-cycle simulation with constant high T-eta covariance. Log frozen params every cycle | After cycle 200, no further freeze swaps occur |
| Unified energy gives consistent D5/RBTA signals | Mock 100 cycles with known FLOP counts. Log both D5 energy_cost and RBTA energy_log["G'"] | Spearman ρ = 1.0 between the two signals |
| CPD multi-parent fix is correct | Create 2-parent node with known transition probabilities. Run 1000 learn steps. Compare learned vs true CPD | KL divergence < 0.05 between learned and true CPD |

---

## Open Questions

1. **Should the TSPL gradient fix use Option A (MLP.compute_gradient) or Option B (fix shape matching)?** Option A reuses tested backprop code but adds a new cross-module API. Option B keeps TSPL self-contained but requires engineering the shape-matching logic. Option A is recommended unless the MLP backward path is unreliable.

2. **Can model entropy be computed earlier in the cycle?** Currently `_collect_runtime_log()` runs at line 785, after MDIM context at line 355. The G' prediction is done at lines ~210-260. The prediction confidence/variance is available immediately after predict — move the entropy computation inline after the prediction phase rather than in `_collect_runtime_log`.

3. **Should dead code (F1–F19) be cleaned up in this pass or deferred?** Recommend deferring all MINOR dead-code items to a separate cleanup pass. Focus on the 5 critical/MAJOR issues only.

4. **The graph.py CPD learning-only path (line 833-837) uses single-parent keys — is this also wrong for multi-parent?** For GridWorld, each state dimension node has exactly one parent (itself at t-1), so `s_name->t1_name` is correct. The multi-parent bug only manifests in `_normalize_cpds`. However, the learning path would need identical multi-parent keying if extended.

5. **Is the PID `cov` variable rename worth doing?** Cosmetic and low-risk. Recommend fixing as part of the AF-004 change since it's in the same function.

---

## Appendices

### A. Files to Modify

| File | Issue | Change Summary |
|:-----|:------|:---------------|
| `python/phca/core/cycle.py` | AF-001 | Replace fake `model_entropy` with real G' posterior entropy |
| `python/phca/core/cycle.py` | AF-005 | Unify energy divisor (60M for both D5 and RBTA) |
| `python/phca/learning/tspl.py` | AF-002 | Fix gradient computation — use proper backprop for MLP params |
| `python/phca/world_model/mlp.py` | AF-002 | Add `compute_gradient()` public method |
| `python/phca/regulation/pid_controller.py` | AF-004 | Remove freeze reset at line 247; rename `cov` → `corr` |
| `python/phca/world_model/graph.py` | AF-003 | Fix multi-parent key in CPD normalization |

### B. Test Suite

```bash
cd <repo-root>
python -m pytest tests/ -x -v --timeout=120
```

### C. Final Validation Benchmark

```bash
cd <repo-root>
python -m phca.benchmark --levels 0 1 2 3 --cycles 500 --output-phase4-report
```

Expected minimums after all fixes:
- Φ-IQ overall ≥ 0.7
- Φ-IQ Level 2 ≥ 0.5
- Failure rate < 10%
- Mean latency < 25ms
- P95 latency < 50ms
- MLP prediction error decreasing over 500 cycles (AF-002 validation)
- D4 value positive and driven by real uncertainty (AF-001 validation)
