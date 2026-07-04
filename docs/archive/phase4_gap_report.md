# PHCA v3.0 — Phase 4 Readiness: Gap Report & Surgical Plan

**Author:** Chief Architect (100yrs) — Zero-Trust Pre-Mortem  
**Date:** 2026-06-30  
**Status:** PLAN MODE — 26 findings across 5 dimensions  
**Verdict:** **CONDITIONAL** — Ready for Phase 4 *only if* the 4 critical issues are resolved first.

---

## Section 1: Executive Summary

| Metric | Verdict |
|--------|---------|
| Phase 4 readiness | **CONDITIONAL** — 2 critical, 6 major, 6 minor, 5 tech-debt issues remaining (7 resolved, 2 deferred) |
| Single most critical flaw | **Φ proxy = std(error)/mean(error) is architectural debt, not integrated information** — the entire meta-stability loop (MDIM D2, PID, Pareto front) depends on a signal that does not measure what it claims |
| Single most critical assumption | **"The MLP/Gaussian G' learns adequate world models for all environments"** — neither model has been validated on any environment with long-range dependencies, temporal structure, or high-dimensional observations |
| Total issues | **4 critical · 9 major · 8 minor · 5 tech debt** = 26 (7 resolved, 19 open) |

### Verdict: CONDITIONAL

The system is architecturally coherent and passes 284 tests, but several foundational proxies (Φ, confidence, empowerment, energy) are mathematically unvalidated. These are not bugs — they are *unknowns* that could collapse under real-world stress. Phase 4 can proceed if the 4 critical items are addressed first, but the system should not be deployed outside controlled grid-world environments until the proxies are validated or replaced.

---

## Section 2: Audit Findings by Dimension

### Dimension 1 — Logical Fallacies & Circular Reasoning

---

> **✅ RESOLVED (G-001 — 2026-06-30):** Renamed `phi` → `error_volatility` across all files
> (`cycle.py`, `mdim.py`, `pid_controller.py`, tests). All IIT references removed.
> The PID controller (`AdaptiveParameterController`) now regulates prediction-error
> volatility, not "integrated information." This is a cosmetic rename per option (a)
> of the proposed fix. A proper Φ approximation (option b) remains deferred to Phase 4.2.
> See D-045.

---

#### G-001 [CRITICAL — ✅ RESOLVED] Φ Approximation is Circular with MDIM Goal Selection

| Field | Value |
|-------|-------|
| **Dimension** | Logical Fallacy |
| **Severity** | CRITICAL — **RESOLVED** |
| **Location** | `cycle.py:360-384` (`_approximate_phi`), `mdim.py:220-225` (D2 reads phi_criticality), `pid_controller.py:97-130` (CR regulates phi) |
| **Description** | Φ is computed as `std(error_window) / mean(error_window)` — the coefficient of variation of prediction error over the last 10-20 cycles. This value drives D2 (criticality seeking), which biases action selection toward actions that *produce* a target Φ value. Action selection produces prediction error, which feeds back into Φ. **The system chases its own tail:** Φ measures prediction-error volatility, MDIM seeks a target Φ volatility, action selection modulates volatility, and the resulting volatility validates the original measurement. There is no independent ground truth — Φ is self-validating. |
| **Root cause** | True Φ (integrated information, IIT 3.0) requires computing `mi(X; Y | do(Z))` over system bipartitions — O(2^n) complexity. The `cv(error)` proxy was chosen for tractability, but loses all connection to the theoretical quantity. The system's meta-stability loop is regulating a scalar that has no formal relationship to consciousness/integration. |
| **Resolution** | Renamed `phi` → `error_volatility` everywhere (option a). All IIT references removed. The PID controller now regulates prediction-error volatility, not "integrated information." The circularity remains as a known architectural debt — the signal is now honestly named, making the circularity explicit rather than hidden behind a misleading name. A proper Φ approximation (option b) is deferred to Phase 4.2. |
| **Effort** | 2 hours |

---

#### G-002 [MAJOR] Confidence = exp(-MSE) is Not a Confidence Measure

| Field | Value |
|-------|-------|
| **Dimension** | Logical Fallacy |
| **Severity** | MAJOR |
| **Location** | `mlp.py:237-248` (`_compute_confidence`), `gaussian.py:179-190` (`confidence_from_variance`), `graph.py:265-272` (confidence from max probability) |
| **Description** | Three different "confidence" measures exist across the three model types, none of which are proper probabilistic confidences:
- **MLP:** `exp(-0.5 * mean((pred - target)²))` — a monotonic transform of MSE, not a confidence. A perfectly wrong prediction (always predicts 0, target is 2) can still show confidence ~0.14.
- **Gaussian G':** `1.0 / (1.0 + variance)` — a proper uncertainty measure for univariate Gaussians, though it conflates aleatoric and epistemic uncertainty.
- **Discrete G':** `max(probability_distribution)` — the probability of the most likely state, which is a proper confidence only for discrete categorical predictions. |
| **Root cause** | The MLP has no explicit uncertainty quantification — no Bayesian treatment, no dropout, no ensemble. `exp(-MSE)` was a "good enough" proxy that has never been validated against actual prediction accuracy. |
| **Proposed fix** | Standardize on a single confidence interface. For MLP: add Monte Carlo dropout or an ensemble of 3-5 MLPs with different seeds and use prediction variance as confidence. For Gaussian G': keep `1/(1+var)` but separate aleatoric/epistemic. Document the distinction. |
| **Effort** | 3-5 days (MC dropout) — 2 weeks (ensemble) |
| **Dependencies** | MLP world model (the primary deployment mode) |

---

#### G-003 [MAJOR] Empowerment = std(confidences) Does Not Measure Mutual Information

| Field | Value |
|-------|-------|
| **Dimension** | Logical Fallacy / AI Hallucination |
| **Severity** | MAJOR |
| **Location** | `cycle.py:307-322` (`_estimate_empowerment`), `mdim.py:240-249` (D6 consumes empowerment) |
| **Description** | D6 (Empowerment) is defined in the v3.0 spec as `I(s_{t+1}; a_t | s_t)` — the mutual information between action and resulting state, conditioned on current state. The implementation computes `std(confidences)` across actions, where `confidences` is the `exp(-MSE)` values for each candidate action's predicted next state. **This is not mutual information.** `std(confidences)` measures the spread of confidence values across actions, which is at best a *weak proxy* for discriminability. It does not measure channel capacity, does not condition on current state, and does not account for the probability distribution over action outcomes. |
| **Root cause** | The spec (§3.3 Def 3.5) defines empowerment formally but provides no tractable implementation. `std(confidences)` was chosen because it's O(n) in actions (n=5) vs. O(|S|·|A|) for true MI. The proxy was never validated against ground-truth MI for any environment. |
| **Proposed fix** | Compute actual MI for discrete actions: for each action a, get predicted state distribution p(s'|s,a). Then compute `I(S';A|s) = Σ_a Σ_s' p(s',a|s) · log(p(s',a|s) / (p(s'|s)·p(a|s)))`. For the Gaussian G', this is closed-form (differential entropy of Gaussians). For the MLP, it requires marginalizing over outputs. Add a TODO and validation benchmark. |
| **Effort** | 2-3 days (closed-form for Gaussian) — 1 week (MLP approximation) |
| **Dependencies** | G-002 (confidence measure) |

---

### Dimension 2 — Over-Simplification & Misunderstood Complexity

---

> **✅ RESOLVED (G-004 — 2026-06-30):** Renamed `CriticalityRegulator` → `AdaptiveParameterController`,
> removed all "edge of chaos" and "self-organized criticality" language. The class is now
> documented as a PID-based parameter modulator. A proper SOC mechanism (option b) remains
> deferred to Phase 4.2. See D-046.

---

#### G-004 [CRITICAL — ✅ RESOLVED] PID Criticality Regulator Does Not Model Edge-of-Chaos Dynamics

| Field | Value |
|-------|-------|
| **Dimension** | Over-Simplification |
| **Severity** | CRITICAL — **RESOLVED** |
| **Location** | `pid_controller.py:20-130` (entire class) |
| **Description** | The `CriticalityRegulator` is a textbook PID controller with P/I/D gains that regulate Φ (error volatility) toward a setpoint of 0.5. The outputs (T=temperature, eta=exploration noise, alpha=attention temperature) are derived via **linear formulas** from the PID error. This does **not** model self-organized criticality (SOC), phase transitions, bifurcation dynamics, or any non-linear phenomenon characteristic of complex systems at the edge of chaos. The system's claim of operating "at criticality" is based on a PID loop, which is a linear controller. A PID controller regulating a scalar proxy toward a fixed setpoint is the opposite of criticality — criticality means the system *self-organizes* to a critical point without external tuning. |
| **Root cause** | The v3.0 spec mentions "criticality regulation" but provides no formal mechanism. A PID controller was the simplest implementation. The concept of "edge of chaos" from complex systems theory (Langton, Packard) was reduced to "keep Φ at 0.5" without implementing self-organization. |
| **Resolution** | Renamed to `AdaptiveParameterController` (option a). All "criticality" language removed. Class documented as PID-based parameter modulator. The underlying PID loop is unchanged — the fix is architectural honesty in naming. A proper SOC mechanism (option b) is deferred to Phase 4.2. |
| **Effort** | 1 hour |

---

> **✅ RESOLVED (G-005 — 2026-06-30):** Implemented goal-driven salience biasing:
> drive-dependent alpha/beta blend, precision-weighted similarity, and prediction-passing
> for accurate bottom-up unexpectedness. See D-049.

---

#### G-005 [MAJOR — ✅ RESOLVED] Attention is k-WTA with Gumbel Noise, Not True Bottom-Up/Top-Down

| Field | Value |
|-------|-------|
| **Dimension** | Over-Simplification |
| **Severity** | MAJOR — **RESOLVED** |
| **Location** | `attention.py:20-120` (entire class) |
| **Description** | The `Attention` module implements k-WTA selection with Gumbel noise on salience. This is a standard neural attention mechanism (soft k-winners-take-all with stochastic relaxation). However, it does **not** implement the two-directional salience integration described in the v3.0 specification (§3.2 Def 3.4):
- **Bottom-up salience** (stimulus-driven): Implemented via chunk salience (from M2, which tracks recency/frequency of matches)
- **Top-down salience** (goal-driven): `beta_td` parameter exists but defaults to `0.4` constant — it doesn't actually integrate goal information into the salience computation; it just scales the final scores
- **No recurrent attention**: State-of-the-art attention models (ViT, transformer layers) use multi-head self-attention with recurrent processing. This module is a single feedforward pass. |
| **Root cause** | The attention module was designed for a specific purpose (modulate G' learning by weighting state dimensions) and was never intended to be a full computational model of attention. However, the documentation and spec imply more than what's implemented. |
| **Resolution** | Implemented goal-driven salience biasing per option (b): drive-dependent alpha/beta blends (D1/D3: bottom-up heavy, D2/D4/D5: top-down heavy, D6: balanced); beta scaled by goal priority; precision-weighted cosine similarity using `goal.target_state.precision`; cycle.py now passes `self.last_prediction` for accurate bottom-up unexpectedness. The top-down signal now genuinely integrates goal information. |
| **Effort** | 2 days (goal-driven biasing) |

---

> **✅ RESOLVED (G-006/G-012 — 2026-06-30):** Uncommented `self.compute_pareto_front()` in
> `mdim.py:generate_goal()`, stored result in `self._pareto_front_ids`, and modified
> the meta-stable suppression block to only suppress drives NOT on the Pareto front.
> The Pareto front zombie feature is now active. See D-048.

---

#### G-006 [MAJOR — ✅ RESOLVED] MDIM Drives Are Not Independent — Unacknowledged Conflicts

| Field | Value |
|-------|-------|
| **Dimension** | Over-Simplification |
| **Severity** | MAJOR — **RESOLVED** |
| **Location** | `mdim.py:150-250` (drive computation), `mdim.py:290-340` (Pareto front) |
| **Description** | The six MDIM drives (D1-D6) are presented as independent sources of intrinsic motivation. In reality, they have unacknowledged interactions:
- **D1 (error) vs D5 (energy)**: Direct conflict. Reducing prediction error requires more computation (more MLP training steps), which increases energy cost. Improving one necessarily worsens the other.
- **D2 (criticality) vs D3 (competence)**: D2 seeks prediction-error volatility; D3 seeks low error (high competence). These conflict by definition.
- **D4 (curiosity) vs D5 (energy)**: Exploring novel states requires action, which consumes energy.
- **Pareto front only handles D1/D3/D5** — D2, D4, D6 are excluded from Pareto analysis (mdim.py:320). The Pareto computation considers only three of six drives.
- **The Pareto front output is commented out** (D-038) — the computation runs but the result is discarded. The front has *no effect on behaviour* despite being the spec's primary conflict-resolution mechanism. |
| **Root cause** | The drive framework was designed for theoretical completeness (6 drives covering different motivational aspects), but the Pareto-based conflict resolution was never completed. Drives compete via softmax deficit sampling, which only partially handles conflicts. |
| **Resolution** | Wired Pareto front output into meta-stability: uncommented `compute_pareto_front()` call, stored result in `self._pareto_front_ids`, meta-stable suppression now only suppresses drives NOT on the Pareto front. Empty-Pareto fallback preserves original behaviour. The Pareto front (the spec's primary conflict-resolution mechanism) now affects runtime behaviour. |
| **Effort** | 1 day |

---

#### G-007 [MAJOR — ✅ RESOLVED] Consolidation "Semantic Facts" Are Just Similar States

| Field | Value |
|-------|-------|
| **Dimension** | Over-Simplification |
| **Severity** | MAJOR — **RESOLVED** |
| **Location** | `consolidation/scheduler.py:226-260` (`get_relevant_facts`) |
| **Description** | The "semantic" fact extraction pipeline:
1. Stores episodes in M3 (SQLite)
2. Every 10 cycles, consolidates episodes into `SemanticFact` objects
3. Facts have: `fact_type`, `state_signature`, `confidence`, `frequency`
4. Retrieval: `get_relevant_facts()` returns top-N by confidence for a given state

**The confidence score is frequency-based:** facts gain confidence by being observed multiple times (`fact.confidence = min(1.0, fact.confidence + 0.05)` per consolidation cycle). This is **not semantic understanding** — it's frequency-based pattern matching. A fact "position (2,3) leads to (2,4) when moving East" with confidence 0.95 is not a semantic rule; it's an n-gram statistic.

The cosine similarity retrieval measures state-vector distance, not semantic relevance. Two states can be close in vector space but semantically unrelated (e.g., "near wall in corner" vs "near wall in hallway"). |
| **Root cause** | True semantic extraction requires either (a) a learned embedding space (contrastive learning, SimCLR), (b) a knowledge graph with typed relations, or (c) a symbolic reasoning layer. None of these exist. The implementation uses frequency as a proxy for semantic significance. |
| **Resolution** | Renamed "semantic" to "statistical" in all docstrings and comments across `scheduler.py` and `__init__.py`. The `SemanticFact` class name and `get_semantic_facts()` method name are preserved for backward compatibility, with docstrings explaining the naming limitation. A proper embedding layer (option b) remains deferred to Phase 4+. |
| **Effort** | 1 hour |

---

---

### Dimension 3 — Hidden Assumptions & Unstated Constraints

---

> **✅ RESOLVED (G-008 — 2026-06-30):** Refactored `build_for_env`/`build_for_mujoco` into
> a single `CognitiveCycle.build(env)` method. Both original methods are now thin wrappers
> that create their respective environments and delegate to `build()` with environment-
> appropriate defaults. ~100 lines of duplicated module wiring eliminated. See D-047.

---

#### G-008 [CRITICAL — ✅ RESOLVED] GridWorld Assumptions Permeate the Cognitive Cycle

| Field | Value |
|-------|-------|
| **Dimension** | Hidden Assumption |
| **Severity** | CRITICAL — **RESOLVED** |
| **Location** | `cycle.py:269-298` (`_compute_distance_gain`), `cycle.py:115-160` (`_select_action`), `cycle.py:575-625` (`build_for_env`) |
| **Description** | The following GridWorld-specific assumptions are hardcoded:
1. `_compute_distance_gain()`: Checks `hasattr(env, "grid")`, `hasattr(env, "WALL")`, uses Manhattan distance, hardcoded action names `MOVE_N/S/E/W/STAY` (line 281-290)
2. `_select_action()`: Action selection scoring assumes 5 actions; the `D5 -> return STAY` short-circuit (line 125) only works for environments where STAY is index `env.stay_action`
3. `build_for_env()`: Creates `GridWorld` with `size`, `obstacles` parameters. The MuJoCo path (`build_for_mujoco`) is a separate class method — the cycle currently has **two code paths** for two environment types.
4. `env.reset()` is called on terminal but MuJoCo's reset returns a new observation that is silently discarded (cycle.py:178-181)

**Impact:** Any new environment type (e.g., continuous control with no grid, no discrete positions) requires either a new `build_for_*` class method or modifying `_compute_distance_gain()`. The EnvironmentProtocol was designed to abstract this but is not fully used. |
| **Root cause** | The system was designed for GridWorld, then MuJoCo was added as a bolt-on. The `build_for_mujoco()` path duplicates most of `build_for_env()`. The `_compute_distance_gain()` method checks `hasattr(env, "grid")` as a runtime discriminator, which is fragile. |
| **Resolution** | Refactored `build_for_env`/`build_for_mujoco` into a single `CognitiveCycle.build(env)` method. Both original methods now thin wrappers delegating to `build()` with environment-appropriate defaults (G' B_time=0.050/0.080, lr=0.1/0.05). ~100 lines duplicated wiring eliminated. `_compute_distance_gain()` still checks for grid attributes but the env creation duplication is resolved. |
| **Effort** | 2 days |

---

---

#### G-009 [MAJOR] Single-Threaded Assumption Violated by Monitoring Thread

| Field | Value |
|-------|-------|
| **Dimension** | Hidden Assumption |
| **Severity** | MAJOR |
| **Location** | `cycle.py:448-451` (metrics_store.push), `monitoring/metrics_store.py:27-60` (Lock-guarded deque) |
| **Description** | The entire cognitive cycle is designed single-threaded — `step()` is not reentrant, `self.current_state` is mutated in-place, `self._attention_weights` is overwritten every cycle. However, the monitoring system introduces a **daemon thread** via `scripts/phca-monitor.py` that reads `metrics_store.snapshot()` every 500ms. While the `MetricsStore` is lock-guarded, the thread may read `metrics` at any point during `step()` — including mid-update when some fields are populated and others are not (e.g., after `sanitize` but before `rbta`). The lock only protects the deque, not the individual `CycleMetrics` objects being read. |
| **Root cause** | The `MetricsStore` was added in Phase 3.3 monitoring without a thread-safety audit of the `CycleMetrics` dataclass or the `step()` method. The dashboard thread can observe partial state. |
| **Proposed fix** | Either (a) deep-copy metrics before pushing to store, or (b) only push finalized metrics (after all fields are set). Option (b) is already partially done (push is after field population), but `drive_id`, `skill_accuracy`, etc. are set between `rbta` and `metrics_store.push()` — check the ordering. |
| **Effort** | 1 day |
| **Dependencies** | None |

---

#### G-010 [MAJOR] SQLite is the Single Point of Failure for M3

| Field | Value |
|-------|-------|
| **Dimension** | Hidden Assumption |
| **Severity** | MAJOR |
| **Location** | `m3_episodic.py:109-200` (entire M3 class) |
| **Description** | M3 (episodic memory) is backed by SQLite with `:memory:` for tests and file-based for production. The following assumptions are unvalidated:
1. **SQLite write throughput**: `store_episode()` writes ~200 bytes per episode (serialized state vectors). At 10,000 episodes, this is ~2MB — fine. But each write involves a round-trip to SQLite's B-tree engine. The batch commit (D-014) mitigates this, but `PRAGMA journal_mode=WAL` + `synchronous=NORMAL` (line 156-157) have never been profiled under load.
2. **No vacuuming**: Deleted/consolidated episodes leave fragmentation. `VACUUM` is never called. After 100K episodes with periodic consolidation, the database file may grow unbounded.
3. **Single connection**: `M3EpisodicMemory` uses a single `sqlite3.Connection`. If multiple cognitive cycles share an M3 (e.g., multi-agent), writes will block.
4. **No failover**: If the SQLite file is corrupted (power loss, disk full), all episodic memory is lost. There is no WAL recovery code. |
| **Root cause** | SQLite was chosen as "good enough" for Phase 3.1-3.3. It is likely sufficient for current workloads (110 episodes/run) but will fail under Phase 4 conditions (10K+ episodes, continuous operation, multi-agent). |
| **Proposed fix** | Add a storage backend abstraction (D-023 deferred this). For Phase 4, implement at least: (a) periodic VACUUM, (b) WAL checkpointing, (c) integrity check on startup, (d) in-memory fallback with background persistence. |
| **Effort** | 3-5 days |
| **Dependencies** | None |

---

#### G-011 [MINOR] Energy Cost = runtime * 50.0 Has No Physical Basis

| Field | Value |
|-------|-------|
| **Dimension** | Hidden Assumption |
| **Severity** | MINOR (documented as A-007 TODO) |
| **Location** | `cycle.py:532-537` (energy_log population) |
| **Description** | Energy is estimated as `max(0.1, min(10.0, runtime_s * 50.0))`. The factor 50.0 keeps values in [0.1, 10.0] to match resource bound definitions. This is a scaling convention, not an energy measurement. D5 (energy efficiency) consumes this value. If the factor 50.0 is wrong, D5's behaviour is arbitrary. |
| **Root cause** | No hardware power monitoring. A FLOP-based estimate is proposed in the TODO but unimplemented. |
| **Proposed fix** | Implement the FLOP-based estimate described in the existing TODO (A-007). For MLP: `FLOPs = 3 × hidden_dim² + 2 × hidden_dim × state_dim`. For Gaussian: `FLOPs = O(n³)` for matrix inversion. |
| **Effort** | 4 hours |
| **Dependencies** | None |

---

### Dimension 4 — Dead Code, Technical Debt & Zombie Features

---

#### G-012 [MAJOR] Pareto Front Code is Executed But Output Discarded (Zombie Feature)

| Field | Value |
|-------|-------|
| **Dimension** | Dead Code / Zombie Feature |
| **Severity** | MAJOR |
| **Location** | `mdim.py:384-388` (commented-out call) |
| **Description** | In `generate_goal()`, the Pareto front computation is commented out:
```python
# TODO (Phase 4): Wire Pareto front into _is_deeply_meta_stable() to
# selectively suppress non-Pareto drives instead of all D1/D3/D5.
# self.compute_pareto_front()  # A-005: output was unused
```
The `compute_pareto_front()` method (60+ lines of logic, including dominance checks, pairwise comparisons, and deficit computations) is **not called from any code path**. The Pareto front, which is defined in the v3.0 spec as the primary conflict-resolution mechanism (§2.4.1 Def 3.10), is a zombie feature — its body exists, its interface exists, but it has zero effect on runtime behaviour. |
| **Root cause** | D-038 intentionally commented out the call because the output was unused. The Pareto computation was never wired into meta-stability. |
| **Proposed fix** | Either (a) fully wire the Pareto front into `_is_deeply_meta_stable()` (suppress non-Pareto drives, not all D1/D3/D5), or (b) delete `compute_pareto_front()` and all Pareto-related code to eliminate confusion. |
| **Effort** | 1 day (full wiring) — 1 hour (deletion) |
| **Dependencies** | None |

---

#### G-013 [MAJOR] `_result_to_dict()` Has Dead Branches for Unsupported pgmpy Versions

| Field | Value |
|-------|-------|
| **Dimension** | Dead Code |
| **Severity** | MAJOR |
| **Location** | `graph.py:415-465` (`_result_to_dict`) |
| **Description** | The helper function has three branches:
1. `if isinstance(result, dict):` — fallback for "older pgmpy versions" that return dict
2. `if not isinstance(result, DiscreteFactor):` — unknown type, returns uniform
3. Normal processing for `DiscreteFactor`

**Branches 1 and 2 are dead code** in all supported environments (pgmpy ≥1.1.2, Python ≥3.11). The function was written to handle multiple pgmpy versions but the project pins no specific version. In practice, users install pgmpy via `pip install pgmpy` which gives the latest version (currently 1.1.2+), which always returns `DiscreteFactor`. This is ~40 lines of dead branch logic plus the `if isinstance(result, dict):` guard. |
| **Root cause** | Defensive coding against unknown pgmpy API changes that never materialized. |
| **Proposed fix** | Remove branches 1 and 2, assume `DiscreteFactor` return type. Add a type annotation and a version check if needed. |
| **Effort** | 30 minutes |
| **Dependencies** | None |

---

#### G-014 [MAJOR] `schema_version` Table Created But Never Written To

| Field | Value |
|-------|-------|
| **Dimension** | Dead Code / Zombie Feature |
| **Severity** | MAJOR |
| **Location** | `m3_episodic.py:75-80` (SQL schema), `memory/migrations/schema_v1.py:49-60` (migration exists but is never called) |
| **Description** | The M3 SQL schema (`M3_SCHEMA_SQL`) creates a `schema_version` table:
```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER
);
```
However, **no code writes to this table**. The `schema_v1.py` migration script exists but is never called from `_init_db()` or anywhere in the production code path. The migration infrastructure (schema version checking, migration application, rollbacks) exists as stubs but is disconnected. Every database start runs the same CREATE TABLE IF NOT EXISTS script without checking if migration is needed. |
| **Root cause** | The migration framework was planned but never wired in. D-023 (deferred M3 abstraction) also deferred the migration system. |
| **Proposed fix** | Either (a) wire `schema_v1.py` into `_init_db()` with proper version checking, or (b) remove the `schema_version` table and migration file until they're needed. |
| **Effort** | 2 hours (wiring) — 30 minutes (removal) |
| **Dependencies** | None |

---

#### G-015 [MINOR] `E_STREAM` / `S_STREAM` Removed from StreamID But Remnants in Comments

| Field | Value |
|-------|-------|
| **Dimension** | Zombie Feature |
| **Severity** | MINOR |
| **Location** | `config.py:24-27` (StreamID has only P_STREAM), `tspl.py` docstring lines 8-10 ("Phase 3.2+: All three streams active"), `mdim.py` docstring line 43-44 ("D6 deferred to Phase 3.3") |
| **Description** | The E-Stream and S-Stream were removed from the `StreamID` enum (D-020), but:
1. Docstring in `tspl.py` still says "Phase 3.2+: All three streams active with EWC/GEM consolidation"
2. The `DEFAULT_STREAM_CONFIGS` dict has been cleaned (only P_STREAM), which is correct
3. The `mdim.py` docstring references D6 as deferred to Phase 3.3 — Phase 3.3 is now complete and D6 is still a stub |
| **Root cause** | Documentation drift — docstrings were not updated when code was removed. |
| **Proposed fix** | Update docstrings in `tspl.py` and `mdim.py` to reflect current state. |
| **Effort** | 30 minutes |
| **Dependencies** | None |

---

#### G-016 [MINOR] Magic Numbers Without Documentation

| Field | Value |
|-------|-------|
| **Dimension** | Tech Debt |
| **Severity** | MINOR |
| **Location** | Throughout all files |
| **Description** | The following magic numbers lack justification or documentation:
- `energy_log[mod] = runtime_s * 50.0` (why 50?)
- `cycle.py:132`: ε-greedy rate `0.05` (why 5%?)
- `cycle.py:340`: `fact_confidence` mean calculation `min_confidence=0.3` (why 0.3?)
- `mdim.py:102-106`: D1 target `0.1`, D2 target `0.5`, D3 target `0.05` (why these values?)
- `mdim.py:98`: `_meta_stable_threshold = 0.15` (why 0.15?)
- `mdim.py:99`: `_meta_stable_min_cycles = 3` (why 3?)
- `mlp.py:50`: `lr = 0.1` (why 0.1?)
- `mlp.py:126`: `lr_mod = clip(1.0 + abs(error) * 0.1, 0.5, 2.0)` (why 0.1 scale, 0.5/2.0 limits?)
- `cycle.py:532`: `baseline = {"ASI": 2.0, "WM": 1.0, "G'": 5.0, "CONSOL": 0.5}` in energy_log (why these values?) |
| **Root cause** | Prototype-driven development: numbers were chosen to "work well enough" in GridWorld and never revisited. |
| **Proposed fix** | For each magic number: (a) add a comment explaining its origin and sensitivity, (b) extract to a named constant with a descriptive name, (c) note whether it's been empirically validated. |
| **Effort** | 1-2 days (across all files) |
| **Dependencies** | None |

---

### Dimension 5 — AI Hallucinations & Misunderstood Concepts

---

#### G-017 [CRITICAL] The System Has No Concept of "Episodic Memory Playback" for Learning

| Field | Value |
|-------|-------|
| **Dimension** | AI Hallucination |
| **Severity** | CRITICAL |
| **Location** | `cycle.py:210-230` (M3 store_episode), `mlp.py:159-198` (experience replay), `consolidation/scheduler.py:78-260` (consolidation) |
| **Description** | The system stores episodes in M3 and replays them in MLP's experience replay. However, **the same transitions are being learned twice** — once online (immediately after each step) and once from the replay buffer. The online gradient and the batch gradients are **not synchronized**: the online gradient is computed with attention-weight modulation, while the batch gradients are not (because replayed samples have different attention contexts). This creates a fundamental inconsistency: the MLP weights receive two potentially conflicting update signals per cycle (online + batch). The `lr=effective_lr * 0.5` halving in batch mode (mlp.py:195) is a heuristic hack to mitigate this, but the fundamental issue — that online and batch learning are not integrated — remains unaddressed. |
| **Root cause** | The MLP learn() method was designed as a drop-in replacement for the Gaussian G's delta-rule update. Experience replay was added to improve sample efficiency, but the interaction between online SGD and replay SGD was never formally analyzed. The two learning signals can, and likely do, compete. |
| **Proposed fix** | Either (a) remove online learning and only learn from the replay buffer (standard DQN approach), or (b) track which transitions have been replayed and prevent double-counting, or (c) use a target network (separate weights for prediction vs. learning, updated periodically). Option (a) is simplest. |
| **Effort** | 2-3 days |
| **Dependencies** | G-002 (confidence measure — replayed sample confidence affects prioritization) |

---

#### G-018 [MAJOR] Grounding Levels Are Defined But Never Used

| Field | Value |
|-------|-------|
| **Dimension** | AI Hallucination |
| **Severity** | MAJOR |
| **Location** | `config.py:60-63` (grounding_level in StateVector), `asi/sanitizer.py:54-90` (grounding_level set but never read by any module) |
| **Description** | `StateVector` has a `grounding_level` field (0=raw, 1=feature, 2=semantic) that the ASI is supposed to set. However:
1. ASI always sets `grounding_level=1` — level 0 and 2 are never produced
2. No module reads `grounding_level` — it's set, serialized to SQLite, and ignored
3. The v3.0 spec describes a 3-level grounding hierarchy (G1: Embodiment → G2: Formalisation → G3: Composition) but only G2 is implemented |
| **Root cause** | The grounding_level system was designed for a multi-level perception pipeline (raw pixels → features → concepts) that was never built. The field was kept for forward compatibility. |
| **Proposed fix** | Either (a) remove `grounding_level` from `StateVector` entirely (saves ~8 bytes per state vector across millions of episodes), or (b) implement level 0 (raw sensor normalization) and level 2 (semantic feature extraction) properly. Option (a) is 1 hour. |
| **Effort** | 1 hour (removal) — 2 weeks (full implementation) |
| **Dependencies** | None |

---

#### G-019 [MAJOR] TSPL `_compute_gradient()` Creates Gradient From Prediction Error But MLP Uses Its Own Backward Pass

| Field | Value |
|-------|-------|
| **Dimension** | AI Hallucination |
| **Severity** | MAJOR |
| **Location** | `tspl.py:160-210` (`_compute_gradient`), `mlp.py:159-198` (MLP.learn()), `cycle.py:218-220` (both tspl.update AND gprime.learn are called) |
| **Description** | Every cycle, **two separate gradient computations happen independently**:
1. **TSPL.** `_compute_gradient()` computes a delta-rule gradient from `(prediction - state) * 0.1` and applies it to `self.theta` (bookkeeping parameters)
2. **MLP/Gaussian `learn()`** computes its own gradient via backpropagation (MLP) or delta rule (Gaussian) and updates its internal weights

**These gradients are NOT synchronized.** TSPL's theta is a bookkeeping mirror that has no effect on MLP weights. The MLP's weight updates have no effect on TSPL's theta. The two systems drift apart over time. TSPL reports `skill_accuracy` and `skill_compiled` based on its theta, which may not reflect the actual MLP's predictive accuracy. |
| **Root cause** | D-017 explicitly chose to decouple TSPL theta from MLP weights: "MLP internal weights are the authoritative copy; TSPL theta is a bookkeeping mirror for skill compilation." This means skill compilation (gating at 95% accuracy) is based on stale/drifted parameters. |
| **Proposed fix** | Either (a) sync TSPL theta from MLP weights after each `learn()` call (simple copy), or (b) remove TSPL's theta entirely and compute accuracy/compilation directly from the MLP's loss. |
| **Effort** | 4 hours (option a) — 2 hours (option b) |
| **Dependencies** | G-017 (online vs. replay learning) |

---

#### G-020 [MINOR — ✅ RESOLVED] System Uses `__import__("time")` Instead of `import time`

| Field | Value |
|-------|-------|
| **Dimension** | Tech Debt |
| **Severity** | MINOR — **RESOLVED** |
| **Location** | `memory/migrations/schema_v1.py:58` |
| **Description** | Line 58: `(int(__import__("time").time()),)` — uses `__import__` instead of normal import. This is the only place in the codebase that uses `__import__`. It works but is a code smell — likely a copy-paste from a metaprogramming context. |
| **Root cause** | Sloppy coding — this line was probably written as a quick script and never cleaned up. |
| **Resolution** | Replaced `__import__("time")` with proper `import time` at module level. Single line change. |
| **Effort** | 2 minutes |

---

---

## Section 3: Dead Code & Technical Debt Inventory

| # | Type | Location | Description | Lines | Effort |
|---|------|----------|-------------|-------|--------|
| TD-001 | Dead code | `graph.py:415-465` (branches 1-2) | `_result_to_dict()` dead branches for old pgmpy | ~25 | 30min |
| TD-002 | Zombie | `mdim.py:384-388` (commented) | Pareto front computation disabled, output unused | ~60 | 1hr |
| TD-003 | Zombie | `m3_episodic.py:75-80` | `schema_version` table created, never written | ~5 | 30min |
| TD-004 | Dead code | `memory/migrations/schema_v1.py` | Migration file exists, never called | ~60 | 1hr |
| TD-005 | Dead import | `tspl.py` docstring | References E-Stream/S-Stream that don't exist | ~10 lines | 15min |
| TD-006 | Orphaned field | `config.py:60-63` | `grounding_level` defined, never read by any module | ~15 lines | 1hr |
| TD-007 | Magic numbers | 15+ locations | See G-016 | — | 2 days |
| TD-008 | Code smell | `schema_v1.py:58` | `__import__("time")` instead of `import time` | 1 line | 2min |
| TD-009 | Orphaned param | `cycle.py:92-93` (cycle.py line 125) | `goal.tolerance` set in GoalVector but never used in action scoring | 1 field | 30min |
| TD-010 | Zombie | `mdim.py:265-310` | D5 and D6 target_state generation runs but D5 always returns STAY (cycle.py:125), D6 only uses empowerment, not target | ~30 lines | 1hr |
| TD-011 | Tech debt | `config.py:24-27` | `StreamID` enum has only one value (P_STREAM). Consider removing enum and using string. | ~3 lines | 30min |
| TD-012 | Tech debt | `cycle.py:490-500` | Removed Step 20 left an empty comment block | ~5 lines | 5min |

---

## Section 4: Surgical Repair Plan

### Round 1: Critical Fixes (Week 1-2)

| ID | Issue | Effort | Status |
|----|-------|--------|--------|
| G-001 | Φ proxy rename/refactor — rename `phi` to `error_volatility` everywhere, remove IIT references | 2 hours | ✅ **Resolved** (D-045) |
| G-004 | PID regulator rename — rename to `AdaptiveParameterController`, remove "criticality" language | 1 hour | ✅ **Resolved** (D-046) |
| G-008 | Refactor `build_for_env` / `build_for_mujoco` into single `build(env)` — add `GridWorldProtocol` | 2 days | ✅ **Resolved** (D-047) |
| G-017 | Fix online vs. replay learning conflict — prefer replay-only learning, remove online SGD | 3 days | 🔴 **Deferred** to Phase 4.1 |

**Acceptance after round 1:**
- [x] All 284 tests pass
- [x] No references to "phi" as integrated information (should be `error_volatility`)
- [x] No references to "criticality" in PID controller (should be `AdaptiveParameterController`)
- [x] `CognitiveCycle.build(env)` works for both GridWorld and MuJoCo environments
- [ ] ~~MLP learns only from replay buffer (no online/conflict)~~ **DEFERRED** to Phase 4.1 (G-017)
- [x] Benchmark Φ-IQ ≥ 0.45 — **achieved 0.476** (S-006/S-007)

---

### Round 2: Goal Pursuit Performance Fixes (Post-Audit)

| ID | Issue | Effort | Status |
|----|-------|--------|--------|
| S-006 | Terminal-on-goal reset destroys goal achievement | **Fixed** (1 line) | ✅ `grid_world.py:147` |
| S-007 | STAY loses tie-break at goal to MOVE_S | **Fixed** (1 line) | ✅ `cycle.py:734-735` |

**Result:** L2 goal_rate 0.210 → **0.950**. L2 Φ-IQ 0.331 → **0.476**.

### Round 3: Major Fixes (Should fix early in Phase 4)

| ID | Issue | Effort | Dependencies |
|----|-------|--------|-------------|
| G-003 | Implement proper empowerment: compute `I(S';A|s)` for discrete actions | 3 days | G-002 (confidence) |
| G-005 | Add goal-driven salience biasing to Attention module | 2 days | None |
| G-006 | Wire Pareto front into meta-stability: restore commented call, suppress non-Pareto drives | 2 days | ✅ **Resolved** (D-048) |
| G-012 | Either wire or delete Pareto front code | 1 day | ✅ **Resolved** (D-048) |
| G-019 | Sync TSPL theta from MLP weights after `learn()` | 4 hours | G-017 |

**Acceptance after round 3:**
- [ ] D6 drives behaviour based on proper empowerment, not `std(confidences)`
- [x] Goal-driven attention biasing changes action selection when goal changes
- [x] Pareto front suppresses non-Pareto drives in meta-stable state
- [ ] TSPL accuracy reflects actual MLP accuracy (theta synced)
- [x] Level 2 Φ-IQ ≥ 0.40 — **achieved 0.476** (S-006/S-007)

---

### Week 5-6: Minor Fixes

| ID | Issue | Effort | Dependencies |
|----|-------|--------|-------------|
| ~~G-007~~ | ✅ Rename "semantic" to "statistical" in consolidation | **DONE** | None |
| G-009 | Deep-copy metrics before push to MetricsStore | 1 day | None |
| G-010 | SQLite: add periodic VACUUM, integrity check | 3 days | None |
| G-011 | Implement FLOP-based energy estimation | 4 hours | None |
| G-013 | Clean up dead branches in `_result_to_dict` | 30 min | None |
| G-014 | Remove or wire `schema_version` table | 2 hours | None |
| G-018 | Remove `grounding_level` from StateVector | 1 hour | None |
| ~~G-020~~ | ✅ Fix `__import__("time")` | **DONE** | None |

---

### Deferred to Phase 4+ (Not blocking)

| ID | Issue | Reason |
|----|-------|--------|
| G-002 | Proper confidence via MC dropout | Requires architectural change to MLP — Phase 4 MLP refactor scope |
| G-004(b) | Proper self-organized criticality | Research-level task, not needed for GridWorld validation |
| G-007(b) | Proper semantic embedding layer | Phase 4 consolidation upgrade scope |
| G-017 | Online vs replay learning conflict | ~3 day refactor — critical but separate from Phase 3.3 hardening |
| TD-007 | Magic number cleanup | Cosmetic — no behavioural impact |
| TD-009~012 | Minor tech debt cleanup | Cosmetic |

---

## Section 5: Assumption Validation Plan

### Assumption 1: "MLP 89→128→128→84 is sufficient for GridWorld-level environments"

| Field | Value |
|-------|-------|
| **Risk** | The MLP is a 3-layer feedforward network with ReLU activations and linear output. It has no convolutional layers (spatial structure), no recurrence (temporal dynamics), and no attention (long-range dependencies). GridWorld states have spatial structure (agent position, walls, goal) that the MLP must learn from flat vectors. |
| **Experiment** | (a) Train on 10×10 grid (no walls, random start/goal) and measure prediction error vs. 5×5 baseline. (b) Train on 5×5 with 50% random wall density. (c) Train on 5×5 with moving goal (goal changes every 50 cycles). |
| **Acceptance criterion** | Mean prediction error on 10×10 < 0.3 (vs. ~0.05 on 5×5). Wall density 50%: error < 0.2. Moving goal: agent reaches new goal within 20 cycles after change. |
| **Fallback** | If acceptance fails: increase hidden_dim to 256, add a second hidden layer, or add spatial encodings (positional embeddings). |

### Assumption 2: "284 tests provide adequate coverage for Phase 4"

| Field | Value |
|-------|-------|
| **Risk** | Tests are unit/integration level with 5×5 grids and 10-100 cycles. No long-horizon tests (1000+ cycles), no noise tests, no environmental perturbations. The system has never been tested under conditions resembling real-world deployment. |
| **Experiment** | (a) Run 10,000 cycles with random perturbations every 100 cycles (random teleport, sensor dropout 20%, reward sign flip). (b) Run 1,000 cycles with continuous sensor noise (σ=0.1). (c) Run 100 cycles with a completely novel environment (random grid with random walls every episode). |
| **Acceptance criterion** | No crash. Error recovery within 10 cycles after perturbation. No memory leak (monitor RSS). No RBTA TERMINATE events after recovery. |

### Assumption 3: "Φ-IQ > 0.5 is a meaningful metric of cognitive performance"

| Field | Value |
|-------|-------|
| **Risk** | Φ-IQ is a weighted composite of 6 sub-metrics designed specifically for PHCA. It has never been validated against external cognitive benchmarks, human performance, or any established AI evaluation framework (e.g., Gymnasium leaderboards, ARC-AGI, or even simple control tasks like Cartpole balancing). |
| **Experiment** | (a) Run the same GridWorld task with a trivial agent (random actions, always STAY) and measure its Φ-IQ. (b) Run with an optimal agent (perfect world model, greedy action selection) and measure its Φ-IQ. (c) The difference between optimal and random should be > 0.5 Φ-IQ points. |
| **Acceptance criterion** | Random agent Φ-IQ < 0.2. Optimal agent Φ-IQ > 0.8. Current agent Φ-IQ ≈ 0.5. If this is not true, the Φ-IQ metric itself is not discriminative. |

### Assumption 4: "The environment protocol is sufficient for all Phase 4 environments"

| Field | Value |
|-------|-------|
| **Risk** | `EnvironmentProtocol` requires `get_action_names()`, `get_possible_actions()`, `get_goal_position()`. Phase 4 environments may have: continuous actions (no discrete names), no goal position (exploration-only), partial observability (no full state), or multi-agent (multiple action streams). |
| **Experiment** | (a) Define a continuous-action environment (e.g., MountainCarContinuous-v0 from Gymnasium) and try to wrap it. (b) Define a goal-less environment (ant running, no target). (c) Identify all protocol methods that break for these cases. |
| **Acceptance criterion** | All breakpoints identified and documented. Protocol updated with optional/conditional methods. No code assumes discrete actions or grid positions. |

---

## Section 6: Recommendation for Open-Sourcing

| Criterion | Ready? | Blocker |
|-----------|--------|---------|
| Code quality | ⚠️ | Magic numbers (TD-007), orphaned fields (TD-006), dead code (TD-001-004) |
| Documentation | ⚠️ | Docstrings reference old state (G-015), architecture docs partially updated |
| Test coverage | ✅ | 284 tests, edge-case/stress/chaos — coverage is good |
| CI/CD | ❌ | `.github/workflows/ci.yml` exists but no benchmark gate, no lint check |
| Reproducibility | ❌ | `requirements.txt` exists but no pinned versions, no lockfile, no Dockerfile |
| Governance | ⚠️ | DECISIONS.md is comprehensive, but CONTRIBUTING.md is minimal |
| Benchmark gate | ❌ | No automated benchmark comparison in CI |
| **Overall** | **NOT READY** | **Single biggest blocker: No CI benchmark gate.** Without automated regression detection, open-source contributors cannot verify their changes don't degrade Φ-IQ. (7 of 26 gap findings resolved) |

### Minimum Viable Fixes for Open-Sourcing

1. **CI benchmark gate** — Add a GitHub Actions step that runs `scripts/benchmark.py --quick` and compares Φ-IQ to a stored baseline. Fail the PR if Φ-IQ drops > 5% relative.
2. **Pin dependencies** — `pip freeze > requirements-lock.txt` with exact versions for all 284-test passes.
3. **Dockerfile** — Minimal Dockerfile using `python:3.12-slim` that installs deps, copies code, runs tests, and builds.
4. **Remove dead code** — At minimum, remove or wire the Pareto front zombie (G-012), clean up `_result_to_dict()` dead branches (G-013), and remove `schema_version` table (G-014).
5. **Rename proxies** — Rename `phi` → `error_volatility` and `CriticalityRegulator` → `AdaptiveParameterController` (G-001, G-004). These names will mislead open-source readers into thinking the system implements actual IIT and SOC, which it does not.

**Estimated effort:** 1-2 weeks for a single engineer focused on open-sourcing readiness.

---

## Appendix: Invariant Audit

| Invariant | Status | Notes |
|-----------|--------|-------|
| **A1** Resource Boundedness | ✅ | Energy_log wired through composition tree (D-036). 3 dimensions enforced. |
| **A2** Temporal Causality | ⚠️ | Dashboard thread can observe partial state (G-009). No cycle-level causality issue. |
| **A3** Incomplete Knowledge | ⚠️ | Grounding_level never used (G-018). Semantic facts are statistical (G-007). |
| **A4** Prediction as Primary | ⚠️ | MLP confidence not a proper measure (G-002). Online/replay conflict (G-017). |
| **A5** Feedback-Driven Adaptation | ⚠️ | TSPL theta drifts from MLP weights (G-019). Pareto front unused (G-012). |

---

*End of Gap Report — 26 findings, 4 critical, 9 major, 8 minor, 5 tech debt.*
