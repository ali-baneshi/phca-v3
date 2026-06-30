# PHCA v3.0 — Specification Compliance Audit & Architectural Gap Analysis

**Auditor:** Chief Architect (Automated)  
**Date:** June 30, 2026  
**Specification Baseline:** `09-phca-v3-patch.md`  
**Codebase:** `python/phca/` (primary), `rust/` (deferred — empty sources per workspace decision)  
**Status:** AUDIT  

---

## 1. Violation Table

### CRITICAL Violations

| # | Severity | Component | File | Lines | Spec Reference | Finding |
|---|----------|-----------|------|-------|----------------|---------|
| C1 | CRITICAL | MDIM Meta-Stable | `mdim.py` | 482–514 | §2.4.1 Def 3.11(4) | **Missing disruption exit condition.** The spec requires meta-stable state to persist until a significant external event (`‖δ_t‖ > 3σ_δ` or `H(WM_t) > 3·H(WM_train)`). The code only checks if deficits are below threshold each cycle — there is no mechanism to detect the disruption event and exit meta-stable state. A system stuck in meta-stable state would never react to novel stimuli. |
| C2 | CRITICAL | Consolidation M4 | `scheduler.py` | 279–295 | §2.3.2 (M4 Write-Lock) | **M4 write-lock semantics not implemented.** The spec requires: (1) exclusive write lock acquisition before E→S transfer, (2) single atomic transaction commit, (3) readers see last committed state, (4) lock released after commit, (5) max wait time with skip-on-timeout. The code merely appends to an in-memory list (`_semantic_facts.extend(facts)`). No locking, no atomicity, no transaction semantics. |
| C3 | CRITICAL | CR Orthogonality | `pid_controller.py` | 189–199 | §2.6.1 Def 2.8a | **Freeze priority uses wrong metric.** The spec mandates freeze priority by **PID gain timescale** (T slow → 1st, α medium → 2nd, η fast → last). The code freezes the parameter with **lower variance**, which is a proxy, not the spec-defined timescale. Variance can diverge from timescale under certain conditions (e.g., high-frequency but low-magnitude oscillations in a fast parameter). |
| C4 | CRITICAL | MDIM Pareto Front | `mdim.py` | 229–269 | §2.4.1 Def 3.10 | **Pareto-optimal check operates on deficits, not configuration values (v1, v3, v5).** The spec defines Pareto front over the *actual configuration* `(v1, v3, v5)` = (prediction error, learning progress, energy cost). The code compares deficits across drives, which is a derived quantity. Two different configurations can yield the same deficits, making the Pareto front ambiguous. |

### MAJOR Violations

| # | Severity | Component | File | Lines | Spec Reference | Finding |
|---|----------|-----------|------|-------|----------------|---------|
| M1 | MAJOR | ASI Sanitizer | `sanitizer.py` | 116, 127 | §2.5.1 Def 2.5a | **Grounding level hardcoded to 1.** The sanitizer always sets `grounding_level=1` in returned StateVectors. The spec requires the grounding level to come from sensor metadata parameter ℓ (levels 0, 1, 2). No mechanism exists to propagate ℓ from sensor metadata. |
| M2 | MAJOR | World Model | (all G') | — | §2.5.1 Def 2.5a | **Grounding Level Adapter not implemented as specified.** The spec's `adapt(s_t, ℓ)` function with three explicit branches (level 0: encoder→G', level 1: direct→G', level 2: V.decode bypass) does not exist. Levels 0 and 1 are implicitly handled via passthrough; level 2 raises NotImplementedError. The VSA bypass path (level 2) is absent entirely. |
| M3 | MAJOR | RBTA Enforcer | `rbta_enforcer.py` | 159–161, 183–254 | §2.1.1 Def 3.6 | **Composite bound check only verifies time, not memory or energy.** The spec defines three resource dimensions for SEQUENCE and PARALLEL: `B_time`, `B_mem`, `B_energy`. The composition tree check (`_check_composition_tree`) only computes and verifies `B_time`. Memory composite bounds are computed by HPM but never checked against violations. Energy composite additivity is never verified. |
| M4 | MAJOR | RBTA Enforcer / HPM | `parser.py` | 394–479 | §2.1.1 Def 3.6 | **Energy composition formulas absent from HPM bound computation.** The spec gives explicit formulas: SEQUENCE energy = `B_energy(M1) + B_energy(M2) + ε_overhead`, PARALLEL energy = `B_energy(M1) + B_energy(M2) + ε_comm`. The HPM `_compute_bounds` returns only `B_time` and `B_mem` — energy dimension is never computed or returned. |
| M5 | MAJOR | MDIM Meta-Stable | `mdim.py` | 307–315, 503–514 | §2.4.1 Def 3.11(3) | **D6 (Empowerment) incorrectly suppressed in meta-stable state.** The spec explicitly states: "Only non-conflicting drives (D2: Criticality, D4: Epistemic Curiosity, **D6: Empowerment**) may generate new goals." The code's softmax over drives 1–5 excludes D6 from goal generation in all states. D6 is not even in the softmax choice set (line 318: `range(1, 6)`). |
| M6 | MAJOR | Consolidation M4 | `scheduler.py` | 279–295 | §2.3.2 C.5 | **Fact storage has no write-lock acquisition timeout.** The spec requires: "Lock acquisition has a maximum wait time: if the lock cannot be acquired within τ_lock_wait, the consolidation cycle is skipped." No locking exists at all (see C2). |

### MINOR Violations

| # | Severity | Component | File | Lines | Spec Reference | Finding |
|---|----------|-----------|------|-------|----------------|---------|
| N1 | MINOR | MDIM Meta-Stable | `mdim.py` | 318–319 | §2.4.1 Def 3.11(1) | **Meta-stable T lock uses CR's raw T, not the spec's "current value" freeze.** In meta-stable state, the spec says temperature should be locked. The MDIM uses `self.temperature` which is set externally by CR each cycle (cycle.py:320). No explicit "lock" mechanism exists — if CR changes T, MDIM follows. However, CR's PID would naturally produce stable T near setpoint, so this is a hardening gap, not a functional one. |
| N2 | MINOR | Attention | `attention.py` | 40–42 | (implicit) | **Gumbel temperature is controlled by CR as `alpha * 0.5` (cycle.py:322), not directly.** The CR output dimension `alpha` has spec-defined semantics as "attention spread." Using `alpha * 0.5` as Gumbel temperature is an undocumented scaling factor without spec justification. |
| N3 | MINOR | HPM Validator | `parser.py` | 350–374 | §3.2 Def 3.4-3.7 | **Type compatibility checks produce warnings, not errors.** The `_check_sequence_types` method (line 350) only appends to warnings list for type mismatches. The spec's grammar implies type safety should be enforced at validation time. Mismatched types should produce validation failures. |
| N4 | MINOR | ASI Sanitizer | `sanitizer.py` | 100–106 | §2.2.3 | **Global failure check uses stale failure count from previous cycles.** Line 101: `total_failed = int(np.sum(self.failure_count > 0))` — this counts sensors with *any historical* failure, not *current cycle* failures. The spec's `ASI_FAILURE_LIMIT` check (line 149 of the spec) uses `∑_j 𝟙[¬isvalid(v_j^(t))]` — only current cycle. The code could trigger global failure recovery based on stale non-zero counts. |
| N5 | MINOR | TSPL | `tspl.py` | 190–196 | §3.1 Def 3.3.3 | **Skill compilation uses theta mirror of G' params, not actual learned skill.** The accuracy estimate is derived from prediction error on G', which doesn't directly correspond to the actual parameter convergence for skill mastery. Phase 3.3 will address this when theta→G' is wired. |
| N6 | MINOR | Cycle Orchestrator | `cycle.py` | 278–280 | §2.2.1 | **Terminal state resets G' and env but does not clear sanitizer state.** When env hits terminal, `gprime.reset()` is called but `sanitizer.reset()` is not. The sanitizer retains stale precision and last_valid values across episode boundaries, which could cause cross-episode contamination. |

---

## 2. Invariant Violation Scan

### A1: Resource Boundedness
| Check | Result | Notes |
|-------|--------|-------|
| Unbounded loops | PASS | All loops bounded by sensor_dim, horizon, action_count, train_steps |
| MLP training time cap | WARN | MLP `learn()` does 8×64 forward/backward passes. RBTA check occurs *after* this completes. No mid-execution preemption cap. See Violation M3. |
| Precision recovery loop bound | PASS | Sanitizer's exponential decay reaches ε_confidence in ≤7 cycles per Theorem 3.2 |

### A2: Temporal Causality
| Check | Result | Notes |
|-------|--------|-------|
| Feedback delay compensation | PASS | PEU uses `corrected_prediction` conditioned on action_vec, not stale last_action |
| Cycle ordering | PASS | sanitize → WM → predict → act → PEU → TSPL → RBTA → log |
| Sanitizer uses last valid | PASS | `v_j^(t) ← v_j^(t-1)` correctly |

### A3: Incomplete Knowledge
| Check | Result | Notes |
|-------|--------|-------|
| Precision carried everywhere | PASS | StateVector always carries precision array |
| Entropy floor enforcement | PASS | RBTA checks entropy_floor |
| Confidence propagation | PASS | Prediction confidence decays with horizon (1/h heuristic) |

### A4: Prediction as Primary
| Check | Result | Notes |
|-------|--------|-------|
| Core loop: predict → compare → learn | PASS | Cycle steps 2-7 implement this |
| Action selection uses prediction | PASS | `_select_action` calls `engine.predict` for each candidate |
| No action without prediction | PASS | Every action is preceded by G' prediction |

### A5: Feedback-Driven Adaptation
| Check | Result | Notes |
|-------|--------|-------|
| Closed-loop sensorimotor | PASS | Cycle: observe → predict → act → observe → compare |
| Learning from prediction error | PASS | TSPL update uses PEU error signal |
| RBTA enforces bounds per cycle | PASS | check_cycle called every step |

---

## 3. Orphaned Specification Features

### Missing Spec Features

| # | Feature | Spec Section | Implementation Status |
|---|---------|--------------|----------------------|
| F1 | **Grounding Level Adapter** (`adapt(s_t, ℓ)` with 3-case match) | §2.5.1 Def 2.5a | **NOT IMPLEMENTED** — See Violation M2. `adapt()` function does not exist. |
| F2 | **M4 Write-Lock with atomic transaction** | §2.3.2 | **NOT IMPLEMENTED** — See Violation C2. Fact storage is a plain list append. |
| F3 | **Meta-Stable Disruption Detection** | §2.4.1 Def 3.11(4) | **NOT IMPLEMENTED** — See Violation C1. No `‖δ_t‖ > 3σ_δ` or `H(WM) > 3·H(WM_train)` check exists. |
| F4 | **Energy Bounds in RBTA Composition Tree** | §2.1.1 Def 3.6 | **NOT IMPLEMENTED** — See Violation M4. Composite bounds return only B_time and B_mem. |
| F5 | **RBTA Mid-Execution Preemption** | §2.1 Def 2.3 (INTERRUPT/TERMINATE) | **PARTIAL** — RBTA runs *after* all modules execute. No mechanism to preempt a module mid-cycle based on runtime guard. Spec implies enforcement could be real-time. |
| F6 | **ASI Sensor Metadata Grounding Level ℓ Input** | §2.5.1 | **NOT IMPLEMENTED** — No sensor metadata channel exists for ℓ. Hardcoded to 1. |
| F7 | **Sanitizer Reset on Episode Boundary** | §2.2 | **NOT IMPLEMENTED** — See Violation N6. Terminal state does not reset sanitizer. |

### Drift: Features Present but Not in Spec

| # | Feature | File | Notes |
|---|---------|------|-------|
| D1 | **MLP World Model** (`WorldModelMLP`) | `world_model/mlp.py` | Introduced in Phase 3.3b as a G' replacement. Not in v3.0 patch spec, but is a planned extension. Acceptable per phasing plan. |
| D2 | **GEM Projection (E-Stream)** | `tspl.py` | GEM implementation in TSPL for E-Stream. Spec only defines E-Stream as "episodic" without GEM details. Minor over-engineering. |
| D3 | **EWC Penalty (S-Stream)** | `tspl.py` | EWC implementation in TSPL for S-Stream. Similar over-engineering. |
| D4 | **SQLite-backed M3** | `memory/m3_episodic.py` | Spec defines M3 logically (MVCC), not storage backend. SQLite is an implementation choice not mandated. |

---

## 4. Benchmark Requirement Check

### Spec Requirements (from prompt)
| Metric | Spec Threshold | Reported Value | Verdict |
|--------|---------------|----------------|---------|
| Overall Φ-IQ | ≥ 0.5 | 0.65* | **PASS** (if verified) |
| Level 2 Goal Success Rate | ≥ 50% | 72%* | **PASS** (if verified) |

** However:** The benchmark runner (`python/benchmarks/runner.py`) returns `"status": "not_implemented"` for all levels. No actual benchmark results exist in the codebase. The reported values (0.65, 72%) cannot be independently verified from the code. **Benchmark infrastructure is not implemented.** This is a GATE-blocking issue if Gate 1 verification requires empirical metric validation.

| Benchmark Level | Implemented | Notes |
|-----------------|-------------|-------|
| Level 0 (Stationary Prediction) | ❌ | `runner.py` returns `"not_implemented"` |
| Level 1 (Reactive Control) | ❌ | Not implemented |
| Level 2 (Goal-Directed) | ❌ | Not implemented |
| Level 3 (Compositional) | ❌ | Deferred to Phase 3.3 |
| Level 4 (Multi-Agent) | ❌ | Deferred to Phase 3.4 |
| Level 5 (Meta-Learning) | ❌ | Deferred to Phase 3.4 |

---

## 5. Missing Features — Implementation Plan

For each orphaned spec feature, here is a concrete plan to bring the codebase into compliance:

### Plan F1: Grounding Level Adapter
**File:** New `python/phca/world_model/adapter.py`  
**Implementation:**
```
Create adapt(s_t: StateVector, ℓ: int) -> tuple:
  ℓ=0: encoder(s_t) → project_to_state_vars(features) → return (state, None)
  ℓ=1: direct_to_state_vars(s_t) → return (state, None)
  ℓ=2: V.decode(s_t) → V_retrieval_to_prior(concept) → return (None, state)
```
Integrate into `PredictionEngine.predict()` as the first step before G' inference. Wire the `grounding_level` from StateVector metadata (currently hardcoded to 1 in sanitizer — fix that first).

### Plan F2: M4 Write-Lock Semantics
**File:** `python/phca/consolidation/scheduler.py`  
**Implementation:**
- Add `threading.Lock` for M4 write access in `ConsolidationScheduler`
- Wrap `_store_facts()` in lock acquisition with timeout (spec: `τ_lock_wait`)
- Implement atomic swap: build new fact set in staging buffer, then atomically replace `_semantic_facts`
- Ensure reads (via `get_semantic_facts()`) see last committed state (read-copy or versioned pointer)

### Plan F3: Meta-Stable Disruption Detection
**File:** `python/phca/motivation/mdim.py`  
**Implementation:**
- Add tracking for `‖δ_t‖` (prediction error magnitude) rolling window (σ_δ)
- Add tracking for `H(WM_t)` (working memory entropy) rolling window (H_train baseline)
- In `_update_meta_stable()`, add disruption check:
  ```python
  if state.prediction_error > 3 * self.baseline_error_std or \
     wm_entropy > 3 * self.baseline_wm_entropy:
      self.meta_stable.is_meta_stable = False
  ```

### Plan F4: Energy Bounds in RBTA Composition Tree
**File:** `python/phca/regulation/rbta_enforcer.py` and `python/phca/hpm/parser.py`  
**Implementation:**
- Add `B_energy` to HPM `_compute_bounds()` return dict
- SEQUENCE: `B_energy = sum(child_energy) + ε_overhead`
- PARALLEL: `B_energy = sum(child_energy) + ε_comm`
- Add energy composite check to `_check_composition_tree()`
- Define `EPSILON_OVERHEAD = 0.001` and `EPSILON_COMM = 0.002`

### Plan F5: Grounding Level from Sensor Metadata
**File:** `python/phca/asi/sanitizer.py`  
**Implementation:**
- Change `sanitize()` signature to accept optional `grounding_level: int = 1`
- Add metadata parameter ℓ to the method
- Pass ℓ through to returned StateVector instead of hardcoded 1
- Wire ℓ from ASI input source (future sensor interface)

### Plan F6: Sanitizer Reset on Episode Boundary
**File:** `python/phca/core/cycle.py` line 280  
**Implementation:**
- Add `self.sanitizer.reset()` alongside `self.gprime.reset()` in the terminal branch

### Plan F7: RBTA Mid-Execution Preemption (Phase 3.2+)
**File:** `python/phca/core/cycle.py`  
**Implementation:**
- Add per-module time guards using `signal.SIGALRM` or `threading.Timer` with `B_time` as deadline
- If a module exceeds its bound mid-execution, raise `RBTAInterrupt` and skip to next cycle
- This requires careful exception handling to avoid leaving system in inconsistent state

---

## 6. Overall Verdict

### VERDICT: **Conditional Pass**

The codebase implements the vast majority of the PHCA v3.0 specification correctly. The core cognitive cycle, ASI sanitization, RBTA enforcer, MDIM drive computation, criticality regulator, HPM grammar, TSPL learning, and consolidation scheduler all functionally match their formal definitions.

However, **4 CRITICAL** deviations and **6 MAJOR** deviations must be resolved before an unconditional "Pass" can be issued:

### Required Fixes for Unconditional Pass

| Priority | Fix | Target Violation | Effort Estimate |
|----------|-----|-----------------|-----------------|
| P0 | Implement M4 write-lock with atomic transaction semantics | C2 | 1-2 days |
| P0 | Implement meta-stable disruption detection (‖δ_t‖ > 3σ_δ, H(WM) check) | C1 | 0.5 day |
| P1 | Fix CR orthogonality freeze priority to use PID gain timescale instead of variance | C3 | 0.5 day |
| P1 | Fix MDIM Pareto front to evaluate on (v1, v3, v5) configurations, not deficits | C4 | 1 day |
| P1 | Add energy dimension to HPM/RBTA composite bound computation | M3, M4 | 0.5 day |
| P2 | Implement Grounding Level Adapter or explicit ℓ propagation | M1, M2, F1 | 1-2 days |
| P2 | Fix D6 exclusion from meta-stable goal generation | M5 | 0.25 day |
| P3 | Fix sanitizer global failure check to use per-cycle failure mask | N4 | 0.25 day |
| P3 | Add sanitizer reset on episode boundary | N6 | 0.1 day |
| P3 | Implement benchmark suite to validate Level 2 Φ-IQ ≥ 0.5, success ≥ 50% | (Benchmark) | 2-3 days |

### Summary Metrics

| Category | Count |
|----------|-------|
| **CRITICAL violations** | 4 |
| **MAJOR violations** | 6 |
| **MINOR violations** | 6 |
| **Missing spec features (F1-F7)** | 7 |
| **Drift items (D1-D4)** | 4 (all acceptable extensions) |
| **A1-A5 invariants** | All PASS (with 1 WARN on MLP time cap) |
| **Benchmark infrastructure** | NOT IMPLEMENTED |
| **Verdict** | **Conditional Pass** |

### Key Risk Areas

1. **Consolidation M4 write-lock** (C2) is the highest-severity gap — without it, concurrent reads during consolidation could observe inconsistent state, violating Theorem 3.3 (Consolidation Atomicity).

2. **Meta-stable disruption detection** (C1) is essential for system responsiveness — without it, a system in meta-stable state will ignore novel stimuli, violating A1 (boundedness of exploration).

3. **Benchmark infrastructure** is entirely absent — the spec-required Φ-IQ ≥ 0.5 and Level 2 success rate ≥ 50% cannot be empirically validated. This is a Gate 1 blocking issue.

---

*End of Audit — 30 June 2026*
