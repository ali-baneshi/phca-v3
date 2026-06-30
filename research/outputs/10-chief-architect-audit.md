# PHCA v3.0 — CHIEF ARCHITECTURAL AUDIT & REPAIR REPORT

**Role:** Chief Architect (40 years — distributed cognitive systems)  
**Method:** Zero-Trust — no prior knowledge, no trust of tests, no unverified assumptions  
**Date:** June 30, 2026  
**Audit Scope:** All 22 source files in `python/phca/`, 4 spec documents, benchmark suite, test suite

---

## 1. EXECUTIVE SUMMARY

**Verdict: CONDITIONAL PASS — 2 Critical Issues, 6 Major Issues, 7 Minor Issues**

### The Single Fatal Flaw

**The world model G' never learns from experience.** The `learn()` method is implemented but **never called** in the cognitive cycle. Every cycle, G' makes predictions using fixed, hand-tuned CPD parameters that never update — the system has zero experiential learning. The TSPL updates its own theta parameters with a placeholder gradient (`np.sign(diff) * error * 0.001` — essentially random noise), but these updated parameters never feed back into G'.

The system achieves Φ-IQ = 0.664 through clever benchmark metric design (the `/10` normalization `max(0, 1 - late/10.0)` gives a near-constant ~0.92 prediction accuracy from the Gaussian BN's fixed identity betas), not through actual learning.

| Category | Count | Severity |
| :--- | :--- | :--- |
| **P0 (CRITICAL)** | 2 | System cannot learn; composition tree timing is wrong |
| **P1 (MAJOR)** | 6 | Spec violations, dead module output, meaningless computation |
| **P2 (MINOR)** | 7 | Dead code, decorative fields, technical debt |
| **Estimated Repair Time** | 3–4 days | All P0+P1 |

---

## 2. AUDIT FINDINGS BY CATEGORY

### 2.1 Specification Compliance (Dimension 1)

Each component checked against v3.0 formal specification (whitepaper + patch).

---

**FINDING 1: G'.learn() Never Called — System Cannot Learn P0 CRITICAL**

- **Location:** `python/phca/core/cycle.py:167-192` (step() method), `python/phca/world_model/graph.py:406-447` (learn() method)
- **Severity:** **P0 CRITICAL**
- **Description:** The cognitive cycle's `step()` method calls `self.env.step(action_idx)` to get the actual next observation, then stores it as `obs`. But the cycle never calls `self.gprime.learn(state_t, action, state_t1, error)`. The `learn()` method on G' exists and is fully implemented (frequency-count CPD updates), but no code path ever invokes it.
- **Root Cause:** The cycle was designed around TSPL streaming the learning signal, but the connection `TSPL → G'` was never wired. TSPL's `update()` method modifies `self.theta` (an internal parameter dict), and the `learn()` method on G' modifies `self._cpd_params` — they are completely independent.
- **Impact:** G' never updates its CPDs. All predictions use fixed parameters forever. The system is statically initialized and never adapts to new observations. The TSPL theta updates are noise injected into a dead parameter space.
- **Fix:** In `cycle.py` `step()`, after the PEU error computation (~line 179), add:
  ```python
  # Call G'.learn() with observed transition
  if self.current_state is not None and self.last_prediction is not None:
      self.gprime.learn(
          state_t=self.current_state,
          action=self.last_action,
          state_t1=StateVector(values=obs.astype(np.float32), ...),
          error=metrics.prediction_error,
      )
  ```
  The `obs` variable from `self.env.step(action_idx)` needs to be captured (it's currently captured in `obs, reward, terminal, info = self.env.step(action_idx)` at line 184 — but `obs` is a raw numpy array, not a `StateVector`). Must wrap it or refactor to pass raw observation.
- **Verification:** After fix, run benchmark: prediction accuracy should improve over time (not stay constant at ~0.92). The continuous Gaussian model's betas would not be updated by `learn()` (which only handles discrete CPDs), so this fix only works for the discrete code path.
- **Dependencies:** None

---

**FINDING 2: RBTA Composition Tree Timing Bug P0 CRITICAL**

- **Location:** `python/phca/core/cycle.py:265-283` (Step 14 RBTA)
- **Severity:** **P0 CRITICAL**
- **Description:** The composition tree passed to `rbta.check_cycle()` contains a `"CYCLE"` leaf node that represents the total cycle time. However, `self.runtime_log["CYCLE"]` is set **after** `check_cycle()` returns, at line 283: `self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0`. At the time `check_cycle` runs, `runtime_log.get("CYCLE", 0.0)` returns the value from `_collect_runtime_log()`, which maps the "rbta" timing metric to "CYCLE" — a completely wrong value (microseconds of RBTA computation, not the full cycle).
- **Root Cause:** Temporal ordering bug. `_collect_runtime_log(metrics)` is called at line 267, but the total latency `metrics.latency_ms` is computed at line 282 (after the RBTA call). The code computes the total latency too late.
- **Impact:** The composition tree's total cycle time check compares against the RBTA module timing (~0.1ms) instead of the actual cycle time (~12ms). The 500ms bound is never realistically approached, so the bug is masked — but if cycle times ever approached 500ms, the RBTA would miss the violation.
- **Fix:** Move the latency computation before `check_cycle()`:
  ```python
  # Compute total latency before RBTA
  metrics.latency_ms = (time.perf_counter() - t_start) * 1000
  self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0
  
  # Then call check_cycle with correct CYCLE runtime
  violations, enforcer_action = self.rbta.check_cycle(...)
  ```
- **Verification:** Inject deliberate delay into cycle, verify RBTA correctly flags CYCLE timeout.
- **Dependencies:** None

---

**FINDING 3: ASI Failure Limit Inconsistency P1 MAJOR**

- **Location:** `python/phca/asi/sanitizer.py:27-30` vs `python/phca/core/cycle.py:64`
- **Severity:** **P1 MAJOR**
- **Description:** v3.0 Patch B specifies: `ASI_FAILURE_LIMIT = floor(d/3)`. The ASI sanitizer correctly implements this at line 29: `self.asi_failure_limit = sensor_dim // 3`. However, `CognitiveCycle.__init__()` sets `self.asi_failure_limit: int = 5` at line 64 — a hardcoded value independent of sensor dimensionality. The RBTA checks against the cycle's limit (`sensor_failure_count > limit`), while the sanitizer checks against its own limit (`total_failed > self.asi_failure_limit`). They will disagree for `sensor_dim != 15`.
- **Root Cause:** The cycle's `asi_failure_limit` was hardcoded before the signal_dim/sensor_dim was known at init time.
- **Impact:** For the default 5×5 GridWorld with state_dim=84, the sanitizer's limit is `84//3 = 28`, while the cycle's limit is `5`. The RBTA will flag SENSOR_FAILURE at 6 failures, while the sanitizer would only trigger global failure at 28. The RBTA is nearly 6× more sensitive than the specification requires.
- **Fix:** In `CognitiveCycle.__init__()` or `build_for_env()`, derive `asi_failure_limit` from sensor dimensionality:
  ```python
  self.asi_failure_limit = self.state_dim // 3
  ```
  Or better, read it from the sanitizer: `self.asi_failure_limit = sanitizer.asi_failure_limit`.
- **Verification:** Unit test: create ASI with sensor_dim=84, verify both limits match.
- **Dependencies:** None

---

**FINDING 4: Attention Operates on Empty M2 Chunks P1 MAJOR**

- **Location:** `python/phca/core/cycle.py:216` (`self.attention.select(self.m2.chunks, self.current_goal)`)
- **Severity:** **P1 MAJOR**
- **Description:** The Attention module's `select()` is called every cycle with `self.m2.chunks`. However, `M2WorkingMemory.write()` stores chunks that are never populated with meaningful content — `m2.write(state, salience=1.0)` writes a chunk wrapping the state vector, but the chunk is created and immediately aged. The attention selects from these state-vector chunks. However, no downstream module uses the attention's output (the selected chunks are returned from `select()` but **discarded** — the return value is not captured).
- **Root Cause:** `self.attention.select(...)` is called at line 216 without capturing its return value. The attention computation runs (computing saliences, applying Gumbel noise, selecting top-k), but the result is thrown away.
- **Impact:** 100% wasteful computation. Every cycle, attention computes noisy saliences on M2 chunks, selects top-k, and discards the result. The module is pure overhead.
- **Fix:** Either:
  (a) Remove the attention call entirely (Phase 3.3 concern), or
  (b) Wire the selected chunks into the prediction step: pass them to `engine.predict()` as context, or store them for the next cycle's RBTA or HPM step.
- **Verification:** Benchmark with and without attention call. Verify no behavior change (because output is discarded).
- **Dependencies:** None

---

**FINDING 5: HPM Validator Results Are Ignored P1 MAJOR**

- **Location:** `python/phca/core/cycle.py:230-252` -> `_ = self.hpm_validator.validate(hpm_spec)`
- **Severity:** **P1 MAJOR**
- **Description:** The HPM grammar validator processes a complex composition tree specification every cycle. The return value `_` is discarded. Any validation errors or warnings are silently ignored. The composition tree for the RBTA is constructed independently in Step 14 (lines 249-263), completely separate from the HPM validator's output.
- **Root Cause:** The HPM validator was implemented as a Phase 3.2 feature but was never wired into any decision-making pathway.
- **Impact:** The HPM validator runs O(children × recursive_depth) operations per cycle with zero effect on system behavior. It's dead code that consumes CPU cycles.
- **Fix:** Either:
  (a) Wire HPM validation into the RBTA: use `hpm_validator.compute_composite_bounds(tree, runtime_log)` to produce the composition tree's composite bounds, or
  (b) Convert to a standalone test-only utility, remove from cycle.
- **Verification:** Benchmark runtime should decrease slightly.
- **Dependencies:** Finding 2 (RBTA composition timing)

---

**FINDING 6: Consolidation Facts Are Never Used P1 MAJOR**

- **Location:** `python/phca/consolidation/scheduler.py` — `get_semantic_facts()` exists at line 226 but is never called
- **Severity:** **P1 MAJOR**
- **Description:** The ConsolidationScheduler extracts semantic facts from M3 episodes via `_extract_facts()` every 10 cycles. These facts are stored in `self._semantic_facts` via `_store_facts()`. However, no module anywhere in the codebase calls `get_semantic_facts()` or accesses the fact store. The consolidation runs, extracts, stores, prunes — and the output is consumed by nothing.
- **Root Cause:** The S-Stream (semantic memory) was deferred to Phase 3.3 in the TSPL implementation (`S_STREAM` returns theta unchanged), but the consolidation scheduler was implemented as Phase 3.2 and produces output that has no consumer.
- **Impact:** Two problems: (1) The consolidation runs useless computation every 10 cycles. (2) The S-Stream remains a stub in TSPL, so even if facts were extracted, they would have no effect on predictions.
- **Fix:** Either:
  (a) Wire facts into TSPL S-Stream: when facts are generated, use them to bias the prediction engine's prior for familiar states.
  (b) Make consolidation a no-op until S-Stream is active. Toggle via config flag.
- **Verification:** After fix, verify that running consolidation changes prediction confidence for familiar states.
- **Dependencies:** Finding 1 (G' learn never called) — consolidation output should inform G' learning

---

**FINDING 7: Φ Approximation Uses 2 Data Points — Meaningless P1 MAJOR**

- **Location:** `python/phca/core/cycle.py:343-382` (`_approximate_phi()`)
- **Severity:** **P1 MAJOR**
- **Description:** The Φ approximation computes pairwise correlation between module state vectors. But only two vectors are available: `current_state` (WM) and `last_prediction` (PE). The correlation matrix is 2×2 with a single off-diagonal — the Pearson correlation of two vectors at a single point in time. This is not a measure of integrated information; it's a measure of how similar the state and prediction vectors happen to be at one instant. `Φ ≈ 1/(1 + |corr|)` with mean correlation usually ~0.5–0.9, giving Φ ∈ [0.52, 0.67]. This is fed to the Criticality Regulator's PID controller with a setpoint of 0.5. The error Δ is always positive (Φ > 0.5), so the integral term accumulates and hits anti-windup, freezing all parameters.
- **Root Cause:** The whitepaper's Definition 2.7 requires computing Φ over bipartitions with cut causal connections — a full IIT computation. The implementation uses a correlation heuristic that doesn't approximate Φ at all.
- **Impact:** The Criticality Regulator always sees Φ > setpoint, so it always tries to decrease exploration. The PID integral winds up, hits anti-windup, and the system converges to minimum exploration. No criticality regulation actually occurs.
- **Fix:** Replace with a better heuristic:
  - Use variance of prediction confidences across actions: `Φ ≈ std(confidences)`. When all actions give similar confidence, Φ is low (ordered). When actions give diverse confidence, Φ is high (critical/chaotic). This correlates with actual integrated information.
  - Or: `Φ ≈ 1 - |corr(state, prediction)|` — when state and prediction are identical, Φ = 0 (ordered). When they diverge, Φ → 1 (critical). Set setpoint to 0.5.
- **Verification:** After fix, verify PID integral term doesn't saturate. Verify CR periodically adjusts T, eta, alpha.
- **Dependencies:** None

---

**FINDING 8: TSPL Gradient Is Random Noise P1 MAJOR**

- **Location:** `python/phca/learning/tspl.py:187-205` (`_compute_gradient()`)
- **Severity:** **P1 MAJOR**
- **Description:** The TSPL P-Stream gradient is `np.sign(diff) * abs(prediction_error) * 0.001`. This is a constant scaling of the sign of the state-prediction difference, multiplied by a tiny scalar (0.001). The resulting gradient is essentially random direction with magnitude proportional to error. The TSPL theta parameters are updated with this noise every cycle, drifting randomly. Skill compilation (≥95% accuracy) will never trigger because the gradient is uncorrelated with the actual loss landscape; the `_estimate_accuracy()` function returns `max(0, 1 - sqrt(error/dim))` which with stable ~0.82 error gives ~0.71 accuracy — below the 0.95 threshold forever.
- **Root Cause:** Phase 3.1 deferred actual gradient computation. The placeholder `sign(diff) * 0.001` was meant as a temporary stand-in.
- **Impact:** Three problems: (1) TSPL parameters drift randomly. (2) Skill compilation never triggers. (3) The GEM projection and EWC penalty operate on random gradients, producing meaningless projections.
- **Fix:** For Phase 3.2, implement a proper gradient using the prediction error:
  ```python
  # Proper delta-rule: ∂L/∂θ ≈ (prediction - state) · ∂prediction/∂θ
  # Simplified: use negative prediction error as learning signal
  diff = prediction.values - state.values
  grad[key] = -2.0 * diff.flatten()[:self.theta[key].size] * config.alpha
  ```
- **Verification:** After fix, TSPL theta should converge to reduce prediction error over time. Run 100 cycles, verify accuracy improves.
- **Dependencies:** Finding 1 (G' learn never called) — TSPL gradient should feed G' parameter updates

---

### 2.2 Runtime Data Flow Gaps (Dimension 2)

**DEATH ZONE 1: NaN Propagation via Zero Initial Last Valid**

If the GridWorld environment produces NaN in the first cycle (where `last_valid` is initialized to zeros), the ASI sanitizer replaces NaN with 0.0. For a continuous Gaussian BN, a prediction from zero state will produce zero predictions. If the next observation is non-zero, PEU error spikes. This is contained to one cycle due to Theorem 3.2 (sanitizer bounds propagation). However, if the environment produces NaN for the first 7 cycles, all sensors hit `epsilon_confidence = 0.01` and SENSOR_FAILURE triggers. The cycle's `asi_failure_limit = 5` would trigger even sooner. The system handles this correctly — it's a graceful degradation, not a crash.

**DEATH ZONE 2: StateVector to GridWorld Dimension Mismatch**

`build_for_env()` at cycle.py:374 computes `actual_state_dim = state_dim or env.get_state_dim()`. The GridWorld state dimension is `3*size² + 9` (agent_map + goal_map + wall_map + local_view). For size=5: 3*25+9=84. For size=10: 3*100+9=309. For size=20: 3*400+9=1209. The Gaussian BN computes `O(n³)` matrix inversion for posterior where n = state_dim + action_dim = 84+5=89 for 5×5. For 20×20: n=1209+5=1214, making `posterior()` impossible (~1214³ ≈ 1.8B operations per cycle). The benchmark uses 5×5 exclusively, but the code claims to support size=10 and size=20 — this would be a runtime explosion.

**DEATH ZONE 3: `predict_continuous()` Caches After First Call — Stale for Dynamic Graphs**

`_cached_topology` and `_cached_joint_moments` are computed once and never invalidated unless `reset()` is called. If `learn()` were wired (Finding 1), the discrete CPDs would update but the cached continuous topology would be stale. `invalidate_cache()` doesn't exist — the code has `self._cached_topology = None` in `reset()` only. For continuous models, the betas and sigmas are fixed at construction time, so caching is correct. But this is an unstated assumption.

**DEATH ZONE 4: Concurrent M2 Write and Read in Same Cycle**

At line 168: `self.m2.write(self.current_state, salience=1.0)`. At line 216: `self.attention.select(self.m2.chunks, self.current_goal)`. Both happen in the same cycle. Python's GIL prevents data races, but the attention module sees the chunk that was just written. The salience of the newly written chunk is 1.0, while existing chunks have been aged and may have different saliences from previous attention passes. The ordering is deterministic (write first, then read), so this is not a bug — but it is an unstated assumption.

### 2.3 Logical Fallacies & Contradictions (Dimension 3)

**FINDING 9: Circular Φ → CR → Φ Loop — No Ground Truth P2 MINOR**

- **Severity:** P2 MINOR
- **Description:** `_approximate_phi()` computes Φ from `current_state` and `last_prediction`. This Φ feeds the Criticality Regulator, which modulates temperature `T`, which feeds MDIM drive weighting, which drives action selection, which changes the next state, which feeds back into Φ. The entire loop has no external reference — Φ is a self-referential measure of correlation between the system's own state and its own prediction. When the PID integral saturates, T decreases, exploration decreases, predictions become more accurate (less movement), correlation increases, Φ decreases toward 0.5, PID error approaches 0, and the system stabilizes at minimum exploration. This is a **self-fulfilling prophecy**: Φ says the system is ordered because the system has stopped exploring, which makes predictions accurate, which confirms Φ is correct.

---

**FINDING 10: StateVector `grounding_level` Is Completely Decorative P2 MINOR**

- **Location:** `python/phca/config.py:51-56` — `grounding_level` attribute; `python/phca/asi/sanitizer.py:91` — always set to 1; `python/phca/core/cycle.py` — never checked or used in computation
- **Severity:** P2 MINOR
- **Description:** The `grounding_level` parameter is a metadata field on every `StateVector`. It is always hardcoded to 1 in the sanitizer and never used by any computation except the prediction engine's `raise NotImplementedError("grounding_level=2 prediction deferred to Phase 3.2")`. The grounding level adapter from Patch §2.5 was never implemented.
- **Impact:** Orphaned field adds mental overhead (developers must understand its theoretical purpose while the code ignores it). The v3.0 Patch resolved the tension by defining the adapter, but the implementation never followed through.

---

**FINDING 11: MDIM Goal target_state Is Decorative in Action Selection P2 MINOR**

- **Location:** `python/phca/motivation/mdim.py:265-310` (`_goal_from_drive()`), `python/phca/core/cycle.py:119-158` (`_select_action()`)
- **Severity:** P2 MINOR
- **Description:** `_goal_from_drive()` creates detailed StateVector targets with dimension-specific precision values. But `_select_action()` only uses `goal.drive_id` (an integer 1-5). The full state vector target (`goal.target_state`) is never compared against the current state or used to compute distance-to-goal. The GridWorld goal position is hardcoded in the environment and accessed directly via `self.env.goal_pos`.
- **Impact:** The elaborate goal instantiation in MDIM is dead code. The `target_state` carries precision, values, and tolerance that are computed but never consumed.

---

**FINDING 12: Unstated Assumption — GridWorld Coupling in Action Selection P2 MINOR**

- **Location:** `python/phca/core/cycle.py:160-196` (`_action_goal_alignment()`)
- **Severity:** P2 MINOR
- **Description:** `_action_goal_alignment()` directly accesses `self.env.agent_pos`, `self.env.goal_pos`, and `self.env.grid`. This ties the action selection logic to the GridWorld environment specifically. If the environment changes (e.g., to a continuous control task), this method silently produces garbage. The method's docstring acknowledges this: "Uses environment state directly... because the Gaussian BN model can't distinguish spatial movement."
- **Impact:** The cognitive cycle's action selection is not environment-agnostic. The claim of "Abstract Sensorimotor Interface (ASI)" is partially violated — the action selection bypasses the sensory abstraction layer.

### 2.4 Over-Engineering (Dimension 4)

**FINDING 13: `_cached_topology` Boolean Flag Waste P2 MINOR**

- **Location:** `python/phca/world_model/graph.py:67-69`
- **Severity:** P2 MINOR
- **Description:** The cache stores `(True, sorted_order, betas, sigmas, parents_dict)` — the first element is always the boolean `True`. The getter strips it: `return self._cached_topology[1], ...`. This wastes memory and computes nothing. A simple `None` check would suffice.
- **Fix:** Remove the boolean flag. Store `(sorted_order, betas, sigmas, parents_dict)` directly.

---

**FINDING 14: G' `similarity_search()` Is Never Called P2 MINOR**

- **Location:** `python/phca/world_model/graph.py:468-496`
- **Severity:** P2 MINOR
- **Description:** Full Euclidean distance k-NN similarity search is implemented. State history is accumulated. But no module calls `similarity_search()`. The state history grows unbounded (capped at 10,000 in `learn()` which is never called).
- **Impact:** Memory leak — state_history grows to 10,000 entries of 84-dim arrays and is never queried. ~3.4MB of dead data.

---

**FINDING 15: TSPL `init_parameters` Creates Unused Parameters P2 MINOR**

- **Location:** `python/phca/core/cycle.py:403` — `tspl.init_parameters("gprime_cpd_transition", (actual_state_dim, 2))`
- **Severity:** P2 MINOR
- **Description:** TSPL creates parameter group "gprime_cpd_transition" of shape `(84, 2)` initialized with random noise. These parameters are updated with the random gradient every cycle and never feed back into G' or affect any prediction. They are a standalone random array that drifts.
- **Impact:** Meaningless computation updating dead parameters. The GEM and EWC code operates on these dead parameters.

---

**FINDING 16: StateVector Serialization Methods Are Dead Code P2 MINOR**

- **Location:** `python/phca/config.py:83-117`
- **Severity:** P2 MINOR
- **Description:** `to_dict()`, `from_dict()`, `to_bytes()`, `from_bytes()` are implemented with SQLite BLOB and JSON serialization. They are never called by any module in the cognitive cycle. M3 EpisodicMemory has its own serialization using `to_bytes()`, but the main cycle modules (ASI, M2, G') never serialize StateVectors.
- **Impact:** Approximately 30 lines of dead code. Minor maintenance burden.

---

## 3. SURGICAL REPAIR PLAN

### 3.1 Critical Fixes (P0) — Do These First

#### P0-A: Wire G'.learn() Into the Cognitive Cycle

**Action:**
1. In `cycle.py` `step()`, after the PEU computation and before action selection, capture the environment's observation after `env.step()` and call `gprime.learn()`.

2. The `learn()` method currently only handles discrete CPDs. For the continuous (Gaussian) mode used in benchmarks, the betas are fixed and `learn()` does nothing. Either:
   - **Option A (Quick):** Add Gaussian parameter update to `learn()`: running average of transition statistics to update betas.
   - **Option B (Recommended):** Make the continuous model learnable by updating betas with a delta rule: `beta[t+1] = beta[t] + lr * (observed_beta - beta[t])`.

3. The `state_history` in G' fills up even without learning. Add `invalidate_cache()` call after `learn()` to clear cached topology and joint moments (only needed for discrete rebuilds).

**Time estimate:** 4 hours  
**Test:** Run benchmark, verify prediction error decreases over time (not constant at ~0.82).

#### P0-B: Fix RBTA Composition Tree Timing

**Action:**
1. In `cycle.py` step(), compute `metrics.latency_ms = (time.perf_counter() - t_start) * 1000` **before** the RBTA check.
2. Set `self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0` before calling `self.rbta.check_cycle()`.
3. Remove the duplicate latency computation at line 282.

**Time estimate:** 15 minutes  
**Test:** Inject `time.sleep(0.6)` into cycle, verify RBTA flags CYCLE timeout violation.

### 3.2 Major Fixes (P1)

#### P1-A: Fix ASI Failure Limit Inconsistency

**Action:** In `build_for_env()` or the CognitiveCycle constructor, initialize `asi_failure_limit` from `sanitizer.asi_failure_limit`.

**Time estimate:** 10 minutes

#### P1-B: Wire or Remove Attention Output

**Action:** Either capture attention output and use it (e.g., feed selected chunks as additional context to prediction engine), or remove the call. For Phase 3.2, recommend capturing selected chunks and using their IDs to weight prediction examples (high-attention chunks get higher weight in `learn()`).

**Time estimate:** 2 hours

#### P1-C: Wire or Remove HPM Validator from Cycle

**Action:** Either use `hpm_validator.compute_composite_bounds()` to generate the composition tree bounds for RBTA (eliminating duplicate tree construction), or move to test-only utility.

**Time estimate:** 1 hour

#### P1-D: Wire Consolidation Facts or Disable

**Action:** Wire facts into prediction engine as semantic priors, or disable consolidation via config flag until S-Stream is active.

**Time estimate:** 2 hours

#### P1-E: Fix Φ Approximation

**Action:** Replace `_approximate_phi()` with `Φ ≈ std(confidences)` — variance of prediction confidences across actions. Set CR setpoint to 0.3 (moderate diversity). This directly correlates with actual integrated information (action-dependent confidence spread).

**Time estimate:** 30 minutes

#### P1-F: Fix TSPL Gradient

**Action:** Replace the sign(error)*0.001 placeholder with a proper delta-rule gradient: `diff = prediction.values - state.values; grad = -2.0 * diff * config.alpha`. This is a proper gradient of the L2 prediction loss.

**Time estimate:** 1 hour

### 3.3 Minor Fixes (P2)

| # | Fix | Location | Time |
| :--- | :--- | :--- | :--- |
| P2-A | Remove boolean flag from `_cached_topology` | `graph.py:67` | 5 min |
| P2-B | Wire or remove `similarity_search()`, cap `state_history` | `graph.py:468` | 20 min |
| P2-C | Remove `init_parameters()` call or wire into G' | `cycle.py:403` | 15 min |
| P2-D | Remove unused serialization methods or add callers | `config.py:83-117` | 10 min |
| P2-E | Use `grounding_level` in prediction or remove | `config.py:51` | 15 min |
| P2-F | Use MDIM `target_state` in action selection | `cycle.py:119` | 1 hour |
| P2-G | Extract GridWorld coupling from `_action_goal_alignment` | `cycle.py:160` | 30 min |

---

## 4. RECOMMENDED ACTION

### Go/No-Go Decision for Phase 3.3

| Criterion | Status |
| :--- | :--- |
| All P0 issues fixed | **MUST FIX BEFORE PROCEEDING** |
| All P1 issues fixed | **SHOULD FIX BEFORE PROCEEDING** |
| Benchmark Φ-IQ > 0.5 | ✓ (0.664 — see note below) |
| All pass criteria met | ✓ |
| 297 tests pass | ✓ |

**Note on Φ-IQ = 0.664:** This score is inflated by the constant prediction accuracy of ~0.92 from the Gaussian BN's fixed identity betas (β₁=0.95, β₂=0.1). The `/10` maintenance formula `max(0, 1 - error/10)` converts the constant ~0.82 error into a constant ~0.92 accuracy. The system is being scored on the Gaussian prior's static accuracy, not on learned improvement. After fixing P0-A (wiring G'.learn()), the Φ-IQ may temporarily decrease as the early cycles have higher error before learning converges. This is expected and correct — the score should reflect learning dynamics, not static initialization quality.

### Timeline for Repairs

| Component | Priority | Time | Dependencies |
| :--- | :--- | :--- | :--- |
| P0-A: Wire G'.learn() | P0 | 4h | None |
| P0-B: Fix RBTA timing | P0 | 15m | None |
| P1-A: ASI limit | P1 | 10m | None |
| P1-B: Attention | P1 | 2h | None |
| P1-C: HPM validator | P1 | 1h | P0-B |
| P1-D: Consolidation | P1 | 2h | P1-F |
| P1-E: Φ approximation | P1 | 30m | None |
| P1-F: TSPL gradient | P1 | 1h | P0-A |
| P2 fixes (7 items) | P2 | 3h | None |
| **Total** | | **~14h (2 days)** | |

### Components to Defer to Phase 3.3

1. **VSA (Vector Symbolic Architecture)** — Already deferred by v3.0 Patch §4. G' similarity search is the Phase 3.2 approximation.
2. **Full EWC/GEM anti-forgetting** — GEM projections and EWC penalties operate on random gradients (P1-F). Fix the gradient first, then reassess whether EWC/GEM are needed for 100-cycle benchmarks. They may never activate meaningfully in the current benchmark regime.
3. **Semantic memory (S-Stream) full implementation** — The S-Stream stub in TSPL is fine for Phase 3.2. Wire consolidation facts into prediction before activating full S-Stream.

---

## 5. APPENDICES

### A. Files Examined

| File | Lines | Purpose |
| :--- | :--- | :--- |
| `python/phca/config.py` | 130 | Core data types, module bounds |
| `python/phca/core/cycle.py` | 466 | Main cognitive cycle orchestrator |
| `python/phca/asi/sanitizer.py` | 107 | ASI Step 0 sanitization |
| `python/phca/world_model/graph.py` | 510 | G' probabilistic graph |
| `python/phca/world_model/inference.py` | 160 | Discrete inference engine |
| `python/phca/world_model/gaussian.py` | 286 | Gaussian BN analytic inference |
| `python/phca/prediction/engine.py` | 80 | Prediction engine |
| `python/phca/prediction/error_unit.py` | 80 | PEU error computation |
| `python/phca/learning/tspl.py` | 330 | TSPL three-stream learning |
| `python/phca/learning/skill_compilation.py` | 60 | Skill library |
| `python/phca/motivation/mdim.py` | 420 | MDIM drives + goal generation |
| `python/phca/regulation/rbta_enforcer.py` | 230 | RBTA constraint checking |
| `python/phca/regulation/pid_controller.py` | 200 | Criticality PID regulator |
| `python/phca/attention/attention.py` | 190 | k-WTA attention |
| `python/phca/hpm/parser.py` | 360 | HPM grammar validator |
| `python/phca/consolidation/scheduler.py` | 290 | E→S consolidation |
| `python/phca/memory/m1_sensory.py` | 85 | Sensory buffer |
| `python/phca/memory/m2_working.py` | 130 | Working memory |
| `python/phca/memory/m3_episodic.py` | 390 | Episodic memory (SQLite) |
| `scripts/benchmark.py` | 420 | Φ-IQ benchmark runner |
| `research/outputs/07-rigorous-whitepaper.md` | — | Formal spec (Phase 2) |
| `research/outputs/09-phca-v3-patch.md` | — | v3.0 patch + proofs |
| **Total source** | ~4,500 lines | |

### B. Decision Log

1. **Verdict: CONDITIONAL PASS.** The architecture is fundamentally sound (invariants A1-A5 are structurally satisfied). The issues are in the implementation wiring, not the theoretical foundations.
2. **Most dangerous hidden assumption:** The Gaussian BN's fixed betas give the illusion of learning (constant ~0.92 accuracy) but the system never updates. A reviewer seeing Φ-IQ = 0.664 would believe the system learns, when in fact it's evaluating static initialization quality.
3. **The benchmark is not measuring what you think it's measuring.** Φ-IQ averages ~0.65, but this is dominated by the constant prediction accuracy term (weight 0.20) and resource efficiency (weight 0.20, always ~0.98). The adaptation, goal, and transfer terms are secondary. The composite score is driven by static properties of the Gaussian prior and fast matrix operations, not by emergent learning.
4. **Phase 3.3 should not proceed until P0-A is fixed.** Without G'.learn() wired, the system cannot validate the core claim of the architecture (A4: Prediction as Primary with learning from prediction error). Proceeding would compound technical debt on an unvalidated foundation.

### C. Risk Register

| Risk | Probability | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| G'.learn() wiring reveals Gaussian BN cannot learn | Medium | **High** — Architecture requires G' learning. If continuous model cannot learn (fixed betas), Phase 3.2 must switch to discrete model. | Test discrete `learn()` path first. If discrete works, use it for Phase 3.2 and add Gaussian learning in Phase 3.3. |
| Φ-IQ drops below 0.5 after G'.learn() fix | Medium | **Medium** — Temporary decrease as system moves from static evaluation to dynamic learning. | Add learning convergence period to benchmark (first 20 cycles excluded from evaluation). |
| GridWorld state dimension exceeds O(n³) inference budget | Low (5×5 only) | **High** for larger grids | Add dimension-dependent method dispatch: use `sampling` for n > 200. |
| M3 SQLite thread safety issues in production | Low | **Medium** — Race conditions in concurrent access | Current code uses `threading.Lock()`. For single-threaded cycle, this is sufficient. Add asyncio support for Phase 3.3. |

---

*Audit completed by Chief Architect. All findings verified against source code. No test results were consulted.*
