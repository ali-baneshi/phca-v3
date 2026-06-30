# PHCA v3.0 — Phase 3.2 Execution Plan: From Gate 1 to Gate 2

**Document Type:** Execution Plan  
**Status:** DRAFT  
**Governing Documents:** v3.0 Patch (`09-phca-v3-patch.md`), Phase 2 Architecture (`05-phase2-architecture.md`), Phase 3.1 Execution Plan (`13-phase-3.1-execution-plan.md`)  
**Date:** June 30, 2026  
**Audience:** Development Team (3-4 engineers)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Where We Are

**Phase 3.1 completed:** 140 Python tests, all Gate 1 criteria verified. A working 15-step cognitive cycle runs on 5x5 GridWorld with 0.25ms median latency — 2000x under the 500ms budget.

The Phase 3.1 implementation established the core pipeline (ASI -> M2 -> G' -> PE -> PEU -> TSPL -> action -> RBTA) but left several components as stubs: MDIM returns only D1 defaults, Attention is a pass-through, Criticality Regulator returns fixed parameters, E-Stream and S-Stream are no-ops, and the G' graph uses simplified binary nodes.

### 1.2 Where We're Going (Weeks 13-24)

| Week | Focus | Milestone |
| :--- | :--- | :--- |
| **13-14** | Continuous G' CPDs + inference | G' predicts grid-world with actual state vectors |
| **15-16** | E-Stream TSPL (episodic) + M3 SQLite | Episodic memory stores and retrieves experiences |
| **17** | S-Stream TSPL (semantic) + EWC | Semantic abstraction from episodes |
| **18-19** | MDIM full drives D1-D6 | Autonomous goal generation |
| **20** | Criticality Regulator PID | Self-tuning to edge of chaos |
| **21** | Precision-Weighted Attention | k-WTA sparse selection |
| **22** | HPM Grammar type-checking | Composition validation |
| **23** | Full 21-step cycle + consolidation | End-to-end Phase 3.2 |
| **24** | Gate 2 review | **Phase 3.2 -> 3.3 transition** |

### 1.3 Key Changes from Phase 3.1

| Change | Phase 3.1 | Phase 3.2 | Rationale |
| :--- | :--- | :--- | :--- |
| **G' CPD type** | Binary discrete (cardinality=2) | Continuous Gaussian + Conditional Gaussian | Enable actual state vector prediction |
| **Inference** | pgmpy VariableElimination (exact) | pgmpy for small, pyro sampling for large | Scale beyond 100 nodes |
| **TSPL streams** | P-Stream only (E/S stubs) | All 3 streams active | Three-stream predictive learning |
| **Goal generation** | Fixed D1 (prediction error) | MDIM D1-D6 Pareto front | Autonomous motivation |
| **Attention** | Pass-through (all chunks) | k-WTA precision-weighted sparse | Cognitive bottleneck |
| **Criticality Regulator** | Fixed (T=1.0, eta=0.1, alpha=0.5) | PID controller with orthogonality constraint | Self-tuning meta-stability |
| **HPM Grammar** | Always-valid stub | Type-checking module composition | Formal verification |
| **Cycle steps** | 15 steps (subset) | 21 steps (full) | Complete cognitive architecture |
| **Memory** | M1 + M2 only | M1, M2, M3 (SQLite), M4, M5 | Full memory hierarchy |
| **RBTA** | Python only | Rust FFI (re-activate) | Performance + deferred V-1 fix |

### 1.4 Critical Path

```
Week 13-14    Week 15-17    Week 18-20    Week 21-23    Week 24
[G' cont.]---[Memories]---[Motivation]---[Full Cycle]---[Gate 2]
    |             |             |             |
    ├─ Gaussian  ├─ M3 SQLite  ├─ MDIM D1-D6 ├─ HPM
    ├─ Inference ├─ E-Stream   ├─ CR PID     ├─ Consolidation
    └─ Sampling  └─ S-Stream   └─ Attention  └─ Latency opt.
```

---

## 2. PHASE 3.2 TICKETS

### 2.1 PHCA-3.2-001 — Continuous G' CPDs + Sampling Inference

**Priority:** P0 (blocks all prediction/learning)  
**v3.0 Reference:** §2.2 Definition 2.4b, Phase 2 Architecture §3.1  
**Files:** `python/phca/world_model/graph.py`, `inference.py`  
**Effort:** 2 weeks  
**Dependencies:** Phase 3.1 G' graph (PHCA-3.1-007)

#### Required Changes

1. **Gaussian CPD support:** `StateNode.cpd_type` already supports `"gaussian"` and `"conditional_gaussian"`. Implement proper CPD creation for these types in `_build_graph()`.

2. **State vector discretization:** Replace binary nodes with continuous state mapping. Each node represents a state dimension with Gaussian CPD. Mean = predicted value, variance = prediction uncertainty.

3. **Grid-world graph structure:** For a 5x5 grid (84-dim state), create 84 Gaussian nodes, each with parent edges from action nodes and temporal self-edges.

4. **Sampling inference for |V| > 100:** When graph exceeds pgmpy's exact inference limit, fall back to pyro importance sampling.

5. **Confidence from variance:** Confidence = 1 / (1 + variance) — max when variance = 0.

#### Acceptance Criteria

- [ ] G' predict() returns StateVector with non-zero prediction error on GridWorld
- [ ] Confidence < 1.0 for stochastic transitions
- [ ] Sampling inference works for |V| = 200
- [ ] Inference completes < 50ms for |V| = 100 (pgmpy exact)
- [ ] Inference completes < 200ms for |V| = 200 (pyro sampling)
- [ ] All existing G' tests still pass

#### v3.0 Traceability

| Code Element | v3.0 Reference |
| :--- | :--- |
| Gaussian CPD | §2.2 Def 2.4b — probabilistic graph with continuous vars |
| Sampling inference | §2.2 Def 2.5 — Phase 3.2+ importance sampling |
| Confidence from variance | §2.2 Def 2.5 — confidence = 1 - entropy/max_entropy |

---

### 2.2 PHCA-3.2-002 — E-Stream TSPL + M3 Episodic Memory

**Priority:** P0 (blocks multi-stream learning)  
**v3.0 Reference:** §3.1 Definition 3.2 (E-Stream), v3.0 Patch §2.3 (M3 MVCC)  
**Files:** `python/phca/learning/tspl.py`, `python/phca/memory/m3_episodic.py`  
**Effort:** 2 weeks  
**Dependencies:** PHCA-3.2-001 (continuous G')

#### Required Changes

1. **M3 Episodic Memory (SQLite):** Create `python/phca/memory/m3_episodic.py` with SQLite-backed storage using MVCC concurrency (v3.0 Patch §2.3.2). Store (state_before, action, state_after, prediction_error, timestamp) per episode.

2. **E-Stream activation:** Replace the stub `update()` in `TSPL` with real implementation using GEM (Gradient Episodic Memory) projection to prevent catastrophic forgetting.

3. **GEM projection:** When updating E-Stream parameters, project gradients onto the feasible region defined by previous task gradients. Use the quadratic programming formulation from Lopez-Paz & Ranzato (2017).

4. **Experience replay:** Sample random batches from M3 for interleaved learning during consolidation.

#### Acceptance Criteria

- [ ] M3 stores episodes to SQLite and retrieves by timestamp/context
- [ ] MVCC: consolidation reads snapshot, new writes go to active version
- [ ] E-Stream.update() returns modified theta (no longer a stub)
- [ ] GEM projection prevents catastrophic forgetting on 2-task sequence
- [ ] M3 supports at least 10^4 episodes without performance degradation

#### Performance Budget

- M3 write: < 1ms (SQLite insert)
- M3 read (batch of 32): < 5ms
- GEM projection (< 10K params): < 10ms
- Experience replay batch update: < 10ms

---

### 2.3 PHCA-3.2-003 — S-Stream TSPL + EWC Consolidation

**Priority:** P0 (blocks semantic learning)  
**v3.0 Reference:** §3.1 Definition 3.2 (S-Stream), v3.0 Patch §2.3.2 (M4 write-lock)  
**Files:** `python/phca/learning/tspl.py` (update), `python/phca/memory/m4_semantic.py`  
**Effort:** 1 week  
**Dependencies:** PHCA-3.2-002 (M3 provides episodes for abstraction)

#### Required Changes

1. **M4 Semantic Memory:** Create `python/phca/memory/m4_semantic.py` — in-memory fact store with write-lock semantics. Stores abstracted patterns: (pattern_id, feature_vector, confidence, creation_timestamp).

2. **S-Stream activation:** Replace the stub `update()` with real implementation using EWC (Elastic Weight Consolidation) penalty. The penalty term `lambda_ * (theta - theta_protected)` is already implemented — activate it for S-Stream.

3. **E -> S abstraction:** During consolidation (PHCA-3.2-010), extract patterns from M3 episodes and store as semantic facts in M4.

#### Acceptance Criteria

- [ ] S-Stream.update() returns modified theta (no longer a stub)
- [ ] EWC penalty prevents catastrophic forgetting on sequential tasks
- [ ] M4 stores semantic facts with write-lock atomicity
- [ ] E -> S abstraction extracts meaningful patterns from episodes

---

### 2.4 PHCA-3.2-004 — MDIM Full Drives (D1-D6)

**Priority:** P0 (blocks autonomous goal generation)  
**v3.0 Reference:** §3.3 Definition 3.5, §3.3 Definition 3.6, v3.0 Patch §2.4 (Pareto front + meta-stable state)  
**Files:** `python/phca/motivation/mdim.py` (replace stub)  
**Effort:** 2 weeks  
**Dependencies:** PHCA-3.2-001 (prediction error for D1), PHCA-3.2-002 (episodic retrieval for D4), PHCA-3.2-005 (criticality for D2)

#### Required Changes

Replace the MDIM stub with a full implementation:

1. **D1 — Prediction Error Minimization:** Drive = current prediction error. Goal = explore uncertain regions (where confidence < threshold).

2. **D2 — Complexity Seeking (Criticality):** Drive = |Phi_current - Phi_critical|. Goal = adjust exploration noise to maintain edge of chaos.

3. **D3 — Competence Acquisition:** Drive = 1 - skill_accuracy. Goal = practice tasks where accuracy < 0.95.

4. **D4 — Epistemic Curiosity:** Drive = model uncertainty (entropy of G' posterior). Goal = seek states with high information gain.

5. **D5 — Energy Efficiency:** Drive = computational cost / budget. Goal = optimize routines, batch operations.

6. **D6 — Empowerment (deferred):** Drive = action-effect capacity. Goal = seek states with many action outcomes. Phase 3.3.

7. **Pareto front computation (v3.0 Patch §2.4.1):** Compute D1/D3/D5 Pareto front. Enter meta-stable state when all drives are within tolerance.

8. **Goal generation:** Weight drives by softmax deficit. Generate concrete goal from winning drive.

#### Acceptance Criteria

- [ ] All 5 active drives (D1-D5) generate distinct goals
- [ ] D1 drives exploration when prediction error is low (uncertain regions)
- [ ] D3 drives skill practice when accuracy < 0.95
- [ ] D5 drives energy optimization when cost exceeds budget
- [ ] D1/D3/D5 Pareto front correctly identifies meta-stable configurations
- [ ] Goal stack supports goal decomposition and pruning
- [ ] Existing cycle tests still pass (MDIM is called but results may differ)

#### Performance Budget

- Drive computation (5 drives): < 1ms
- Pareto front computation: < 1ms (3 drives, constant time)
- Goal generation + stacking: < 1ms
- Total: < 3ms per cycle

---

### 2.5 PHCA-3.2-005 — Criticality Regulator PID

**Priority:** P1 (blocks self-tuning)  
**v3.0 Reference:** §3.4 Definition 3.7, v3.0 Patch §2.6 (orthogonality constraint)  
**Files:** `python/phca/regulation/pid_controller.py` (replace stub)  
**Effort:** 1 week  
**Dependencies:** PHCA-3.2-001 (G' provides model entropy)

#### Required Changes

Replace the CriticalityRegulator stub with a PID controller:

1. **Phi estimation:** Estimate integrated information Phi from module covariance. For Phase 3.2, approximate Phi as negative entropy of the module state distribution: Phi ≈ -H(module_states).

2. **PID control:** temperature = T0 + k_p * Delta + k_i * integral(Delta) + k_d * d(Delta)/dt, where Delta = Phi_critical - Phi_current.

3. **Orthogonality constraint (v3.0 Patch §2.6.1):** Monitor covariance between T, eta, and alpha. Freeze slowest parameter if covariance exceeds threshold.

4. **Regulate() return values:** Return (temperature, exploration_noise, attention_spread) which are consumed by MDIM, TSPL, and Attention respectively.

#### Acceptance Criteria

- [ ] Regulate() returns non-default values that change with Phi estimates
- [ ] PID correctly reduces Phi error over time (Delta -> 0)
- [ ] Orthogonality constraint prevents parameter covariance from exceeding threshold
- [ ] Parameter freeze/unfreeze logic works correctly

---

### 2.6 PHCA-3.2-006 — Precision-Weighted Sparse Attention

**Priority:** P1 (cognitive bottleneck)  
**v3.0 Reference:** §3.2 Definition 3.4, Phase 2 Architecture §3.4  
**Files:** `python/phca/attention/attention.py` (replace stub)  
**Effort:** 1 week  
**Dependencies:** PHCA-3.2-004 (MDIM provides goal for top-down salience)

#### Required Changes

Replace the Attention stub with k-WTA sparse selection:

1. **Bottom-up salience:** S_bu(chunk) = |chunk.state - prediction| — unexpected chunks are salient.

2. **Top-down relevance:** S_td(chunk) = similarity(chunk.state, current_goal.target_state) — goal-relevant chunks are salient.

3. **Precision weighting:** S(chunk) = (alpha * S_bu + beta * S_td) * precision, where precision comes from the chunk's StateVector.

4. **k-WTA selection:** Keep top-k chunks by composite salience. Add Gumbel noise for stochasticity.

5. **Precision learning:** Update precision per chunk: p(t+1) = p(t) + eta * (|delta| - p(t)).

#### Acceptance Criteria

- [ ] Attention.select() returns <= k chunks (sparse)
- [ ] Unexpected chunks (high S_bu) are selected over predictable ones
- [ ] Goal-relevant chunks (high S_td) are selected over irrelevant ones
- [ ] Low-precision chunks are down-weighted
- [ ] k is configurable (default: 3 for 7±2 WM chunks)

---

### 2.7 PHCA-3.2-007 — HPM Grammar Type-Checking

**Priority:** P2 (blocks formal composition)  
**v3.0 Reference:** §2.1 Definition 2.1, §4, Phase 2 Architecture §3.2  
**Files:** `python/phca/hpm/parser.py` (replace stub)  
**Effort:** 1 week  
**Dependencies:** None (can be parallelized)

#### Required Changes

Replace the HPMValidator stub with type-checking:

1. **Type system:** Define module types (Sensor, Predictor, Controller, Composite) with input/output type signatures.

2. **Composition validation:** Validate SEQUENCE/PARALLEL/pipeline compositions against the type grammar.

3. **Resource bound computation:** Compute B_time composite bound as sequence (sum) or parallel (max) per v3.0 Theorem 2.1.

4. **Uncertainty propagation:** Compute H(output) >= H(input) - I(input; model) for each composition.

#### Acceptance Criteria

- [ ] validate() correctly rejects type-mismatched compositions
- [ ] Resource bounds are computed correctly for nested compositions
- [ ] Uncertainty propagation bound is computed
- [ ] All previously-all-valid compositions still pass (Phase 3.1 compatibility)

---

### 2.8 PHCA-3.2-008 — Full 21-Step Cognitive Cycle

**Priority:** P0 (integration point)  
**v3.0 Reference:** Blueprint §B (full 21-step cycle), Phase 2 Architecture §3.2  
**Files:** `python/phca/core/cycle.py` (update), `python/phca/core/tests/`  
**Effort:** 2 weeks  
**Dependencies:** PHCA-3.2-001 through PHCA-3.2-007

#### Required Changes

Update `CognitiveCycle` to add the 6 missing Phase 3.2 steps:

| Step | Phase 3.2 Component | What It Does |
| :--- | :--- | :--- |
| **8** | Attention | `attended = attention.select(wm_chunks, goal)` — replaces pass-through |
| **10** | MDIM | `goal = mdim.generate_goal(context)` — replaces fixed goal |
| **11** | MDIM goal stack | Push goal, decompose subgoals |
| **12** | MDIM drive update | Recompute drive deficits after action |
| **13** | CR regulation | `(T, eta, alpha) = cr.regulate(phi)` — replaces fixed params |
| **16-18** | HPM composition | Validate module compositions |
| **20** | Consolidation | Sleep-cycle analogue: E -> S transfer |

The step ordering becomes:

```
Step  0: ASI sanitize(raw_sensor) -> clean_vector
Step  1: clean_vector -> M2.write()
Step  2: WM -> G': current state s_t
Step  3: G' -> Prediction Engine: P(s_{t+1} | s_t, a_{t-1})
Step  4: Prediction Engine -> PEU: predicted s_{t+1}
Step  5: Environment -> ASI: observe s_{t+1}
Step  6: PEU: delta_t = predicted - observed
Step  7: P-Stream -> TSPL: update theta via delta_t
Step  8: Attention: k-WTA selection of WM chunks        [NEW]
Step  9: Action Selection: a_t = argmax U(action)
Steps 10-11: MDIM: generate goal + push to goal stack    [NEW]
Step 12: MDIM: recompute drive deficits                  [NEW]
Step 13: CR: regulate(T, eta, alpha)                     [NEW]
Step 14: RBTA Enforcement: check(time, mem, energy)
Step 15: Logging: log cycle metrics
Steps 16-18: HPM: validate + compose modules              [NEW]
Step 19: Cycle Counter: t += 1
Step 20: Consolidation: E -> S transfer                  [NEW]
```

#### Acceptance Criteria

- [ ] Full 21-step cycle executes without error
- [ ] Cycle latency < 500ms median over 1000 cycles (AT-1)
- [ ] Attention returns fewer chunks than WM has (sparse)
- [ ] MDIM generates different goals in different contexts
- [ ] CR parameters vary with system state
- [ ] HPM validates compositions
- [ ] All 15 Phase 3.1 cycle tests still pass

---

### 2.9 PHCA-3.2-009 — M3 Schema + Migration

**Priority:** P0 (blocks M3 storage)  
**v3.0 Reference:** v3.0 Patch §2.3 (Memory concurrency), Phase 3.1 exec plan §6.3  
**Files:** `python/phca/memory/m3_episodic.py`, `python/phca/memory/migrations/`  
**Effort:** 1 week  
**Dependencies:** None (infrastructure)

#### SQLite Schema

```sql
-- Episodic memory store (M3)
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL DEFAULT 1,
    state_before BLOB NOT NULL,        -- StateVector.to_bytes()
    action_taken BLOB NOT NULL,         -- np.ndarray.tobytes()
    state_after BLOB NOT NULL,          -- StateVector.to_bytes()
    prediction_error FLOAT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    timestamp INTEGER NOT NULL,         -- monotonic cycle count
    consolidated INTEGER DEFAULT 0,     -- 0 = pending, 1 = consolidated
    drive_id INTEGER DEFAULT NULL       -- which MDIM drive generated the goal
);

-- Index for consolidation scan
CREATE INDEX idx_episodes_consolidated ON episodes(consolidated, timestamp);

-- Consolidation log (for idempotent processing)
CREATE TABLE IF NOT EXISTS consolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_version INTEGER NOT NULL,
    episodes_processed INTEGER NOT NULL,
    facts_generated INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT DEFAULT 'in_progress'
);
```

---

### 2.10 PHCA-3.2-010 — Consolidation Scheduler

**Priority:** P1 (blocks E -> S transfer)  
**v3.0 Reference:** v3.0 Patch §2.3.2 (MVCC + write-lock), Phase 2 Architecture §3.3  
**Files:** `python/phca/consolidation/scheduler.py`  
**Effort:** 1 week  
**Dependencies:** PHCA-3.2-002 (M3), PHCA-3.2-003 (M4), PHCA-3.2-009 (M3 schema)

#### Required Changes

1. **Sleep-cycle analogue:** Run consolidation every N cycles (default: 100 cycles ~ 10 seconds at 100ms/cycle).

2. **M3 snapshot:** Create MVCC snapshot of M3 at start of consolidation cycle. Only process episodes with version <= snapshot.

3. **E -> S abstraction:** Extract patterns from unconsolidated episodes. For each pattern: (a) cluster similar episodes, (b) extract prototypical feature vector, (c) compute confidence from cluster cohesion, (d) store in M4.

4. **M4 write-lock:** Acquire exclusive write lock on M4, transfer new facts atomically, release lock.

5. **Episode marking:** Mark consolidated episodes in M3.

---

### 2.11 PHCA-3.2-011 — Navigation Learning

**Priority:** P1 (Gate 2 criterion)  
**v3.0 Reference:** §1.3 (criterion 1)  
**Files:** `python/tests/test_phase_3_2.py`, `python/tests/test_acceptance.py`  
**Effort:** 1 week  
**Dependencies:** PHCA-3.2-008 (full cycle)

#### Acceptance Criteria (Gate 2)

- [ ] Agent learns 5x5 grid-world navigation with > 80% success in 1000 episodes
- [ ] Prediction error decreases over time (learning curve)
- [ ] Skill compilation after ≥ 95% accuracy sustained over 100 episodes
- [ ] MDIM generates D3 (competence) goal when prediction error is low but navigation fails

---

### 2.12 PHCA-3.2-012 — Gate 2 Verification

**Priority:** P0 (gate)  
**v3.0 Reference:** §1.3 (criterion 1), Phase 2 Architecture §2.4 (Phi-IQ metric)  
**Files:** `scripts/gate2_verification.py`, `Makefile` (update)  
**Effort:** 1 week  
**Dependencies:** All Phase 3.2 tickets

#### Gate 2 Checklist

| # | Condition | How to Verify |
| :--- | :--- | :--- |
| **1** | All Phase 3.2 tickets implemented | Ticket board — all in "Done" |
| **2** | Full 21-step cycle latency < 500ms median | `python scripts/profile_cycle.py --cycles=1000` |
| **3** | Navigation learning > 80% success | `pytest tests/test_phase_3_2.py::test_navigation_learning` |
| **4** | MDIM generates distinct D1-D5 goals | `pytest tests/test_phase_3_2.py::test_mdim_drive_diversity` |
| **5** | Attention sparsity (k < WM capacity) | `pytest tests/test_phase_3_2.py::test_attention_sparsity` |
| **6** | CR PID maintains criticality (Delta -> 0) | `pytest tests/test_phase_3_2.py::test_cr_criticality` |
| **7** | HPM type-checks compositions | `pytest tests/test_phase_3_2.py::test_hpm_type_checking` |
| **8** | EWC prevents catastrophic forgetting | `pytest tests/test_acceptance.py::test_no_forgetting` |
| **9** | Python ruff no errors | `ruff check python/ --no-cache` |
| **10** | All Phase 3.1 tests still pass | Regression: 140 tests pass |

---

## 3. DEPENDENCIES

### 3.1 New Python Packages

```bash
# Phase 3.2 adds (from requirements-phase-3.2.txt):
pip install torch torchvision scikit-learn pyro-ppl pandas orjson matplotlib
```

Total additional: ~1.1GB (mostly torch). Phase 3.1 dependencies remain.

### 3.2 Phase Boundary Checklist

When transitioning from Phase 3.1 -> 3.2:

- [x] All Phase 3.1 stubs created (MDIM, Attention, CR, HPM) 
- [ ] Phase 3.2 stubs replaced with real implementations
- [ ] E-Stream and S-Stream TSPL active (alpha_E, alpha_S configured)
- [ ] EWC penalty computation added to S-Stream
- [ ] GEM projection added to E-Stream
- [ ] MDIM Pareto front meta-stable state active
- [ ] Criticality Regulator PID active
- [ ] Precision-Weighted Attention active
- [ ] HPM Grammar type-checking active
- [ ] Consolidation scheduler running on sleep cycles
- [ ] VSA decision recorded (keep or cut)
- [ ] Rust RBTA re-enabled (Phase 3.1 was Python-only)
- [ ] V-1 f64/f32 type mismatch fixed when Rust RBTA re-activated

---

## 4. RISKS

| ID | Risk | Probability | Impact | Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| R1 | Continuous G' CPDs increase latency beyond 500ms | Medium | High | Fall back to discretized Gaussian (histogram bins) |
| R2 | Pyro sampling too slow for real-time | Medium | Medium | Reduce graph size, use pgmpy for |V| <= 100 |
| R3 | GEM projection too slow for real-time | Low | Medium | Approximate with first-order Taylor expansion |
| R4 | MDIM Pareto front hard to compute | Medium | Low | Approximate with simple heuristic (if drives conflict, freeze) |
| R5 | CR PID oscillates before converging | Low | Low | Conservative gains, damping |
| R6 | torch dependency (1.1GB) causes CI issues | Low | Medium | Keep Phase 3.2 deps optional, use fallbacks |
| R7 | Consolidation scheduler uses too many cycles | Low | Low | Run every N cycles, not every cycle |

---

## 5. FILE MANIFEST

### New Files

| File | Purpose | Ticket |
| :--- | :--- | :--- |
| `python/phca/memory/m3_episodic.py` | M3 SQLite-backed episodic memory | 3.2-002, 3.2-009 |
| `python/phca/memory/m4_semantic.py` | M4 in-memory semantic fact store | 3.2-003 |
| `python/phca/memory/migrations/__init__.py` | Schema migration framework | 3.2-009 |
| `python/phca/consolidation/scheduler.py` | Sleep-cycle consolidation | 3.2-010 |
| `python/phca/motivation/goal_stack.py` | Goal decomposition and pruning | 3.2-004 |
| `python/phca/core/tests/test_cycle_full.py` | Full 21-step cycle tests | 3.2-008 |
| `python/tests/test_phase_3_2.py` | Phase 3.2 integration tests | 3.2-011 |
| `scripts/gate2_verification.py` | Gate 2 verification script | 3.2-012 |

### Modified Files

| File | Change | Ticket |
| :--- | :--- | :--- |
| `python/phca/world_model/graph.py` | Gaussian/conditional Gaussian CPD, continuous state mapping | 3.2-001 |
| `python/phca/world_model/inference.py` | Pyro sampling for |V| > 100 | 3.2-001 |
| `python/phca/learning/tspl.py` | E-Stream + GEM, S-Stream + EWC activation | 3.2-002, 3.2-003 |
| `python/phca/motivation/mdim.py` | Replace stub: D1-D6 drives, Pareto front | 3.2-004 |
| `python/phca/regulation/pid_controller.py` | Replace stub: PID control, orthogonality | 3.2-005 |
| `python/phca/attention/attention.py` | Replace stub: k-WTA precision-weighted | 3.2-006 |
| `python/phca/hpm/parser.py` | Replace stub: type-checking, resource bounds | 3.2-007 |
| `python/phca/core/cycle.py` | Full 21-step cycle (6 new steps) | 3.2-008 |
| `python/phca/core/__init__.py` | Export new component types | 3.2-008 |
| `Makefile` | Add Phase 3.2 targets, Gate 2 | 3.2-012 |
| `DECISIONS.md` | Log Phase 3.2 design decisions | All |
| `requirements-phase-3.1.txt` | Add pgmpy + networkx (already present) | None |

---

## 6. SPRINT SCHEDULE

```
Week 13-14: G' Continuous + Inference
  PHCA-3.2-001: Continuous CPDs, Gaussian nodes (2 weeks)

Week 15-16: Memory Systems  
  PHCA-3.2-009: M3 SQLite schema + writers (1 week)
  PHCA-3.2-002: E-Stream TSPL + GEM (1 week)

Week 17: Semantic Learning
  PHCA-3.2-003: S-Stream TSPL + EWC (1 week)

Week 18-19: Motivation + Regulation
  PHCA-3.2-004: MDIM D1-D6 + Pareto front (2 weeks)
  PHCA-3.2-005: CR PID + orthogonality (done in parallel)

Week 20-21: Attention + HPM
  PHCA-3.2-006: Precision-weighted attention (1 week)
  PHCA-3.2-007: HPM type-checking (1 week)

Week 22-23: Integration
  PHCA-3.2-008: Full 21-step cycle (2 weeks)
  PHCA-3.2-010: Consolidation scheduler (parallel)

Week 24: Gate 2
  PHCA-3.2-011: Navigation learning test
  PHCA-3.2-012: Gate 2 verification
```

---

*End of Document — PHCA v3.0 Phase 3.2 Execution Plan*

**Status:** DRAFT — Ready for review  
**Next action:** Review and approve ticket list, begin PHCA-3.2-001  
**Document maintainer:** Lead Implementation Engineer  
**Date:** June 30, 2026
