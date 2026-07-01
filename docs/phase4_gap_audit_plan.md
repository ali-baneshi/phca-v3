# PHCA v3.0 — Phase 4 Gap Audit & 5-Step Surgical Plan

**Date:** 2026-07-01
**Auditor:** Chief Architect & Principal Engineer
**Status:** NOT READY — 2 mathematically incorrect algorithms must be fixed before proceeding

---

## Executive Summary

PHCA v3.0 passes 289 tests, achieves Φ-IQ = 0.626 overall (0.320 at Level 2), runs at ~50ms latency with 3.4% failure rate. Superficially, the system works. Architecturally, it contains two mathematically incorrect algorithms at its core — the Pareto front (compares drive deficits with different units as if they were the same quantity) and the MLP empowerment proxy (`std(confidences)` has no guaranteed relationship to mutual information `I(S';A|S)`). A third issue makes the energy optimisation loop internally inconsistent (D5 minimises wall-clock time while RBTA enforces FLOP-based energy). A fourth means the consolidation pipeline (~5ms/10 cycles) has zero behavioural effect. A fifth makes the PID orthogonality freeze fire based on parameter scale, not genuine redundancy. **Verdict: NOT READY.** All five must be resolved before Phase 4 proceeds.

---

## Audit Findings (All Dimensions)

| ID | Dimension | Severity | Location | Description |
|:---|:----------|:---------|:---------|:------------|
| C1 | Over-Simplification / AI Hallucination | CRITICAL | `mdim.py:259-327` | Pareto front compares deficit magnitudes across different drives — not true Pareto dominance |
| C2 | AI Hallucination / Over-Simplification | CRITICAL | `cycle.py:685-707` | MLP empowerment = `std(confidences)` — not mutual information; mathematically unrelated |
| C3 | Hidden Assumption | CRITICAL | `cycle.py:351,807-833` | Energy pipeline inconsistent: D5 uses wall-clock, RBTA uses FLOP-based; contradictory signals |
| C4 | Dead Code / Logical Fallacy | CRITICAL | `cycle.py:336-358`, `mdim.py:147-255` | Consolidation facts injected into MDIM context but never consumed — A5 violation |
| C5 | Over-Simplification | MAJOR | `pid_controller.py:210-212` | Orthogonality uses `np.cov()` — scale-dependent; T always frozen first |
| F1 | Dead Code | MINOR | `hpm/parser.py:18-27` | `CompositionOp.INTERLEAVE` and `TEMPORAL_INVARIANT` — dead enum values |
| F2 | Dead Code | MINOR | `mdim.py:325-327` | Pareto front results commented out — called but never used |
| F3 | Hidden Assumption | MAJOR | `monitoring/metrics_store.py` | All timeseries data stored in memory — no persistence or eviction policy |
| F4 | Over-Simplification | MAJOR | `world_model/mlp.py` | 3-layer MLP (89→128→128→84) used for all environments — no architecture search |
| F5 | Over-Simplification | MINOR | `environments/grid_world.py` | Grid distances and action effects hardcoded; not data-driven |
| F6 | Dead Code | MINOR | `asi/sanitizer.py:77-80` | VSA stubs referenced but `vsa` module does not exist |
| F7 | Hidden Assumption | MINOR | `cycle.py:500-502` | D5 `elapsed_time * 2.0` magic number — origin unknown |
| F8 | AI Hallucination | MAJOR | `cycle.py:metric_logging` | Φ proxy = `std(error)/mean(error)` — not Integrated Information |
| F9 | Dead Code | MINOR | `core/config.py` | Several configuration parameters defined but never read |
| F10 | Over-Simplification | MINOR | `regulation/pid_controller.py` | Simple PID for "edge-of-chaos" dynamics — no chaos theory basis |
| F11 | Hidden Assumption | MAJOR | `world_model/graph.py:578-638` | Gaussian empowerment assumes Gaussian posterior — may not hold for multimodal state dist |
| F12 | Dead Code | MINOR | `learning/tspl.py` | E-Stream / S-Stream remnants in type annotations |
| F13 | Logical Fallacy | MINOR | `environments/grid_world.py` | "Ground truth" derived from agent's internal state, not environment |
| F14 | Over-Simplification | MINOR | `consolidation/scheduler.py` | Cosine similarity for semantic fact extraction — statistical, not semantic |

---

## The 5 Critical Issues (Detailed)

### C1: Pareto Front Algorithm is Mathematically Incorrect

| Field | Value |
|:------|:------|
| **Dimension** | Over-Simplification / AI Hallucination |
| **Severity** | CRITICAL |
| **Location** | `python/phca/motivation/mdim.py:259-327` |
| **Description** | The `_compute_pareto_front()` function compares drive deficits via magnitude comparison: `j_deficit > i_deficit + 0.01`. This is mathematically **not** Pareto dominance. A point `a` dominates `b` iff `a_k <= b_k` for all objectives `k` and `a_m < b_m` for at least one `m`. Comparing deficits across different drives (D1 error O(1), D3 competence O(0.1), D5 energy O(1)) is a category error — they have different units, scales, and meanings. The `0.01` threshold is an arbitrary magic number. The function only evaluates a single configuration against itself, not a set of candidates — a Pareto front requires multiple candidates to compare. Downstream, `mdim.py:372-379` uses this incorrect front for meta-stable state suppression, producing unpredictable behaviour. |
| **Root Cause** | The concept of Pareto dominance was understood superficially. The implementer treated "better/worse" as a magnitude comparison across objectives rather than a vector dominance relation. This is likely an AI hallucination where the output "looks like" a Pareto front but isn't one. |
| **Proposed Fix** | Rewrite `_compute_pareto_front()` to: (1) accept an array of candidate configuration vectors, (2) normalise each objective dimension to [0,1] using min-max over candidates, (3) apply true vector-dominance check: `a` dominates `b` iff `all(a <= b) and any(a < b)`, (4) return set of Pareto-optimal indices. Replace the single-configuration call site with a sampled neighbourhood (e.g., ±10% perturbation of each deficit, 10 candidates). |
| **Effort Estimate** | 5 hours |
| **Dependencies** | None |
| **Acceptance Criteria** | Property-based test: for any set of random normalised vectors, `_compute_pareto_front()` returns only non-dominated points. Integration test: with fixed deficits `[0.5, 0.2, 0.8]` and candidate deficits `[[0.4, 0.1, 0.7], [0.5, 0.3, 0.6], [0.6, 0.2, 0.9]]`, verify [0.4, 0.1, 0.7] is the only Pareto-optimal point. |

### C2: MLP Empowerment = std(confidences) is Not Mutual Information

| Field | Value |
|:------|:------|
| **Dimension** | AI Hallucination / Over-Simplification |
| **Severity** | CRITICAL |
| **Location** | `python/phca/core/cycle.py:685-707` (MLP path), `python/phca/motivation/mdim.py:223-229` (D6 consumer) |
| **Description** | `I(S';A|S)` — empowerment — measures the causal influence of an action on the outcome state. The Gaussian path computes it correctly via closed-form mutual information (`graph.py:578-638`). The MLP path computes `empowerment = np.std(confidences)` where `confidences` is `[mean(1/(1+var))]` per action. Counterexample: an agent in a deterministic environment where actions 1-4 lead to different outcomes with equal confidence → `std(confidences) = 0` → empowerment = 0, but true MI is maximal (actions cause different outcomes). Conversely, actions with wildly varying confidences but identical outcome distributions produce non-zero "empowerment" with zero true MI. D6 drives exploration toward states with high `std(confidences)`, which may be completely unrelated to states with high action→outcome causality. This is an architectural hallucination: `std` of confidence is not an approximation of mutual information; it is a different concept entirely. |
| **Root Cause** | The developer knew empowerment ≈ mutual information but did not know how to compute it from an MLP. They substituted a quantity they could compute (`std(confidences)`) without verifying the mathematical relationship. The Gaussian path shows the correct approach already exists in the codebase. |
| **Proposed Fix** | For n ≤ 5 discrete actions, run MC Dropout (20 forward passes) per action to estimate `p(s'|s,a)` as a Gaussian Mixture Model. Compute `I(S';A|s) ≈ H(Σ p(a) * p(s'|s,a)) - Σ p(a) * H(p(s'|s,a))` where `H` is differential entropy of a GMM (approximate via sampling). Fall back to variance-based heuristic only when n > 5. |
| **Effort Estimate** | 7 hours |
| **Dependencies** | None architecturally, but the MLP must support MC Dropout at inference time (already does — `confidences` uses MC Dropout). |
| **Acceptance Criteria** | Unit test: construct a toy MLP with known input/output where empowerment can be computed analytically (e.g., 2 actions, 2 possible outcomes). Verify MLP empowerment ≥ 0.8 of the analytic value. Integration test: for a random state in GridWorld, D6 values from MLP path must correlate (r > 0.5) with D6 values from Gaussian path for same state. |

### C3: Energy Estimation Pipeline is Internally Inconsistent

| Field | Value |
|:------|:------|
| **Dimension** | Hidden Assumption |
| **Severity** | CRITICAL |
| **Location** | `python/phca/core/cycle.py:351` (D5), `python/phca/core/cycle.py:807-833` (RBTA) |
| **Description** | D5 (energy efficiency drive) receives wall-clock-based energy: `energy_cost = elapsed_time * 2.0` clamped to [0.01, 1.0] (`cycle.py:500-502`). RBTA receives FLOP-based energy: `runtime_s * 50.0` or FLOP estimate clamped to [0.1, 10.0] (`cycle.py:807-833`). These use different units, different scales, and different clamping ranges. A module with low FLOP cost but high latency (e.g., waiting for SQLite I/O) would appear cheap to RBTA but expensive to D5 — the agent receives contradictory optimisation signals. The correct signal depends on what the architect intended to minimise (computation vs. wall time), but the code uses both with no unifying principle. |
| **Root Cause** | Energy estimation was implemented by two different developers (or one developer at different times) without unifying the definition. D5 inherited an earlier wall-clock heuristic; RBTA introduced a later FLOP-based estimate. |
| **Proposed Fix** | Unify both to use the same FLOP-based estimate: `energy_cost = clip(total_flops / baseline_flops, 0.01, 1.0)`. Calibrate `baseline_flops` from a 100-cycle warmup period at startup. Route the same FLOP count to both D5 and RBTA. Remove the `elapsed_time * 2.0` heuristic. |
| **Effort Estimate** | 5 hours |
| **Dependencies** | None |
| **Acceptance Criteria** | After warmup, D5 `energy_cost` and RBTA `energy_ratio` must match exactly for any given step. Unit test: mock module that reports known FLOP counts; verify both D5 and RBTA receive identical energy values. |

### C4: Consolidation Facts Collected But Never Consumed by MDIM

| Field | Value |
|:------|:------|
| **Dimension** | Dead Code / Logical Fallacy |
| **Severity** | CRITICAL |
| **Location** | `python/phca/core/cycle.py:336-358`, `python/phca/motivation/mdim.py:147-255` |
| **Description** | Every 10 cycles, the cycle reads M3 → `get_relevant_facts()` computes cosine similarity against all committed facts, acquires M4 lock, extracts `fact_confidence_mean` and `fact_count`, injects these into `mdim_context` (`cycle.py:350-358`). However, `MDIM.compute_drives()` (`mdim.py:147-255`) only reads 6 keys from context: `prediction_error`, `volatility`, `empowerment`, `energy_cost`, `curiosity`, `competence`. The three fact keys are injected with correct names `fact_confidence_mean`, `fact_count`, `attention_confidence` but are **never read**. This violates Invariant A5 (Feedback-Driven Adaptation): the accumulated knowledge base has zero influence on goal generation. The entire consolidation pipeline (~2-5ms every 10 cycles) is pure computational overhead with no behavioural effect. As M3 grows, this overhead increases without bound. |
| **Root Cause** | The consolidation→MDIM connection was designed as a planned feature but the MDIM `compute_drives()` function was never updated to consume the fact keys. The injection code was written but the consumer was not. |
| **Proposed Fix** | Wire `fact_confidence_mean` into D3 (competence) modulation: high fact confidence → competence deficit decreases (agent has already learned this area). Wire `fact_count` into D4 (curiosity) modulation: many facts → curiosity target decreases (area is well-explored). Specifically: `mdim.py` in `compute_drives()`, after computing base deficits, read `fact_confidence_mean` and `fact_count` from context; apply `competence_target *= (1 - 0.3 * fact_confidence_mean)` and `curiosity_target *= (1 - 0.2 * clip(fact_count / max_facts, 0, 1))`. Remove the 10-cycle gating from the injection in `cycle.py` — let facts flow every cycle for smoother modulation. |
| **Effort Estimate** | 3 hours |
| **Dependencies** | None |
| **Acceptance Criteria** | Integration test: seed M3 with 10 episodes → run consolidation → verify D3 deficit decreases when `fact_confidence_mean` is high. Unit test: mock context with varying `fact_confidence_mean` values → verify D3 competence target changes. |

### C5: PID Orthogonality Constraint Uses Covariance, Not Correlation

| Field | Value |
|:------|:------|
| **Dimension** | Over-Simplification |
| **Severity** | MAJOR |
| **Location** | `python/phca/regulation/pid_controller.py:210-212` |
| **Description** | The orthogonality freeze mechanism checks whether parameter updates are redundant by computing `np.cov(update_trajectory)`. Covariance is scale-dependent: if parameter A has range [0.1, 2.0] and parameter B has range [0.001, 0.2], A's variance dominates the covariance matrix. Parameter T (temperature, range ~2.0) has ~1000× the variance of eta (learning rate, range ~0.2). Consequently, `abs(cov(T, eta))` almost always exceeds the threshold of 0.8 regardless of whether T and eta are actually correlated. T gets frozen in nearly every check, regardless of whether freezing is justified. Conversely, if eta and alpha were perfectly correlated (r=1.0) but both have small scales, `cov(eta, alpha)` would be well below 0.8 and they would never be frozen — even though they are perfectly redundant. |
| **Root Cause** | The implementer used `np.cov()` (covariance) instead of `np.corrcoef()` (correlation/Pearson's r). Both measure association, but covariance is unit-dependent and scale-dependent; correlation is unitless and scale-invariant. The intent was clearly to detect redundant parameters, which requires scale-invariant measurement. |
| **Proposed Fix** | Replace `np.cov()` with `np.corrcoef()` at `pid_controller.py:210`. Change the threshold from `0.8` to `abs(r) > 0.8` (already absolute-value semantics). No other changes needed. |
| **Effort Estimate** | 2 hours (trivial change) |
| **Dependencies** | None |
| **Acceptance Criteria** | Unit test: construct update trajectories where T varies 10× more than eta but they are uncorrelated (r ≈ 0). Verify `abs(r) < 0.8` → no freeze. Construct trajectories where eta and alpha are perfectly correlated (r = 1.0) but small scale → verify `abs(r) > 0.8` → freeze triggers. |

---

## Surgical Repair Plan

### Execution Order (with Parallel Tracks)

```
Week 1 (Track A - Engineer 1)            Week 1 (Track B - Engineer 2)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ C5: PID corrcoef fix (2h)   │          │ C1: Pareto rewrite (5h)     │
├─────────────────────────────┤          ├─────────────────────────────┤
│ C3: Energy unify (5h)       │          │ C4: Wire facts (3h)         │
├─────────────────────────────┤          └─────────────────────────────┘
│ C2: MLP MI (7h) *           │
└─────────────────────────────┘
```

*\* C2 can be deferred to Track B after C1/C4 if Track A finishes early.*

### Detailed Steps

#### Step 1: C5 — PID Correlation Fix (2h)
**Files:** `python/phca/regulation/pid_controller.py`
- Change `np.cov()` → `np.corrcoef()` at line 210.
- Verify threshold `0.8` applies to `abs(r)` correctly.
- Run full test suite.
- Chief Architect review.

#### Step 2: C3 — Energy Unification (5h)
**Files:** `python/phca/core/cycle.py`, possibly `python/phca/core/config.py`
- Add FLOP counter to `CognitiveCycle` (or reuse cycle timing with MFLOPS calibration).
- Implement 100-cycle warmup for baseline FLOP estimation.
- Route same FLOP count to both D5 (via `mdim_context`) and RBTA.
- Remove `elapsed_time * 2.0` heuristic.
- Run full test suite.
- Chief Architect review.

#### Step 3: C1 — Pareto Front Rewrite (5h)
**Files:** `python/phca/motivation/mdim.py`
- Replace `_compute_pareto_front()` with true vector-dominance.
- Add `_normalise_objectives()` helper.
- Replace single-configuration call with sampled neighbourhood.
- Update meta-stable suppression to use correct Pareto-optimal set.
- Run full test suite.
- Chief Architect review.

#### Step 4: C4 — Wire Facts into MDIM (3h)
**Files:** `python/phca/motivation/mdim.py`, `python/phca/core/cycle.py`
- In `MDIM.compute_drives()`, add reads for `fact_confidence_mean`, `fact_count` from context.
- Modulate D3 competence target by fact confidence.
- Modulate D4 curiosity target by fact count.
- Remove 10-cycle gating from injection in `cycle.py`.
- Run full test suite.
- Chief Architect review.

#### Step 5: C2 — MLP Empowerment MI (7h)
**Files:** `python/phca/core/cycle.py`, `python/phca/world_model/mlp.py`
- For n ≤ 5 actions: run MC Dropout (20 passes) per action.
- Estimate `p(s'|s,a)` as GMM from MC samples.
- Compute differential entropy of GMM via sampling approximation.
- Compute `I(S';A|s) = H(Σ p(a)p(s'|s,a)) - Σ p(a)·H(p(s'|s,a))`.
- Set uniform `p(a) = 1/n` (no action prior).
- For n > 5: fall back to variance-based heuristic.
- Run full test suite.
- Chief Architect review.

### Timeline
- **Total effort:** 22 engineering hours
- **With 2 engineers in parallel:** ~1.5 wall-clock days
- **With 1 engineer sequentially:** ~3 wall-clock days

---

## Assumption Validation Plan

For each critical assumption, here is a simple experiment to validate it:

| Assumption | Experiment | Success Criterion |
|:-----------|:-----------|:-----------------|
| Pareto dominance is correct after fix | Generate 1,000 random normalised deficit vectors; check that all returned points are non-dominated | ≥ 99.9% pass rate over 10 runs |
| MLP empowerment correlates with mutual information | On 100 random GridWorld states, compute both MLP MI (new) and Gaussian MI (existing); compute Spearman correlation | ρ > 0.7 |
| Unified energy gives same behaviour in simple cases | Run 10 GridWorld episodes with heat-seeking goal (D5 dominant) using old and new energy; compare trajectory similarity | Trajectory edit distance < 20% of max |
| Fact consumption actually changes behaviour | Run 2 × 200-cycle benchmarks: one with fact modulation, one without (fact keys = 0). Compare D3/D4 deficits after cycle 50 | Deficit divergence > 0.05 after cycle 50 |
| PID orthogonality counts only true redundancy | Random uncorrelated PID trajectories with 10× scale difference → never freeze. Perfectly correlated small-scale trajectories → always freeze | 100% pass on 100 random trials |
| SQLite can handle 10,000+ episodes at 50ms cycle | Insert 10,000 episodes of realistic size (84 floats + metadata); benchmark query time | p95 query time < 10ms |
| k-WTA attention (k=3) is sufficient | Run ablation with k=1,3,5,all; compare Φ-IQ at Level 2 | k=3 ≥ 90% of "all" performance |
| MC Dropout (20 samples) is sufficient | Run ablation with N=5,10,20,50; compare MI estimate stability | N=20 gives σ < 0.05 of N=50 estimate |

---

## Open Questions

1. **Should Pareto operate over all 6 drives or just D1/D3/D5?** D2 (volatility) is derived from D1 (error) — they are not independent objectives. Including both creates a weighted duplicata. Recommend excluding D2 from Pareto (it is a meta-drive).

2. **Is 20 MC samples × 5 actions = 100 forward passes computationally feasible at higher observation dims?** Ant-v2 has 111-dim observation. 100 forward passes of a 89→128→128→84 MLP ≈ 1.2M FLOPs. At 50ms cycle target, this adds ~5ms — acceptable but tight. Consider reducing to 10 samples if benchmarks show insufficient headroom.

3. **Should `fact_confidence_mean` *reduce* D4 (don't need to explore known areas) or *increase* D3 (push beyond what we know)?** Two plausible signals. Recommend reducing both D3 and D4 (less error-avoidance, less curiosity when area is well-understood). This is the conservative choice.

4. **Should the FLOP baseline be a static calibration or running EMA?** Static calibration (100-cycle warmup) is simpler and deterministic. Running EMA adapts to workload changes but introduces coupling between cycles. Recommend static baseline for Phase 4; revisit if deployment shows workload drift.

5. **Are 6 of 8 `CompositionOp` enum values dead weight?** Only `SEQUENCE`/`PARALLEL` are used in `__call__` dispatch. `INTERLEAVE` and `TEMPORAL_INVARIANT` are defined but never referenced in any production code. Recommend removing dead enum values in a separate cleanup pass.

6. **Should `grounding_level` be removed or implemented?** It is defined in `StateVector`, set to 1 (feature level), stripped on deserialization, never checked. If A3 (Incomplete Knowledge) truly requires it, implement the full pipeline. If not, remove it to eliminate dead code.

---

## Appendices

### A. Files to Modify (Complete List)

| File | Issue | Change Summary |
|:-----|:------|:---------------|
| `python/phca/regulation/pid_controller.py` | C5 | `np.cov()` → `np.corrcoef()` at line 210 |
| `python/phca/core/cycle.py` | C3 | Add FLOP counter, warmup, unify D5/RBTA energy |
| `python/phca/core/cycle.py` | C4 | Remove 10-cycle gating for fact injection |
| `python/phca/motivation/mdim.py` | C1 | Rewrite `_compute_pareto_front()` with true vector dominance |
| `python/phca/motivation/mdim.py` | C4 | Read fact keys from context; modulate D3/D4 targets |
| `python/phca/core/cycle.py` | C2 | Replace `std(confidences)` with MC-Dropout MI for n ≤ 5 |
| `python/phca/world_model/mlp.py` | C2 | (If needed) Add GMM entropy computation helper |
| `DECISIONS.md` | All | Append decision entry per fix |

### B. Test Suite to Run After Each Fix

```bash
cd /home/<username>/Pictures/Autonomous-AI-june2026/new-ai
python -m pytest tests/ -x -v --timeout=120
```

### C. Benchmark to Run for Final Validation

```bash
cd /home/<username>/Pictures/Autonomous-AI-june2026/new-ai
python -m phca.benchmark --levels 0 1 2 3 --cycles 500 --output-phase4-report
```

Expected minimums after all fixes:
- Φ-IQ overall ≥ 0.7
- Φ-IQ Level 2 ≥ 0.5
- Failure rate < 10%
- Mean latency < 25ms
- P95 latency < 50ms
- Consolidation producing ≥ 10 facts per 100 cycles
