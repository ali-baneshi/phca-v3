# PHCA v3.0 — Top 5 Critical Gap Fix Plan

**Date:** 2026-06-30  
**Author:** Chief Architect  
**Audit Scope:** All 22+ source files in `python/phca/`, 4 spec documents, 3 audit reports, benchmark suite, test suite  
**Status:** PLAN MODE — no code changes; execute in order  
**Gate:** Phase 3.3 → Phase 4 readiness

---

## 0. Executive Summary

The PHCA v3.0 codebase is architecturally sound at the structural level — invariants A1–A5 are satisfied, the 21-step cognitive cycle is correctly sequenced, and the modular decomposition follows the spec. However, a deep audit reveals **5 critical issues** that must be fixed before Phase 3.3 can be considered complete and the system ready for Phase 4. These issues span three dimensions:

1. **Learning never happens** — the world model's `learn()` method is never called (Issue #1)
2. **Output is produced but never consumed** — consolidation, attention, HPM, and MDIM all compute results that are discarded (Issues #2, #4)
3. **Benchmark infrastructure is a stub** — Φ-IQ scores cannot be reproduced (Issue #3)
4. **Spec violations in critical paths** — M4 write-lock, meta-stable disruption detection, RBTA energy bounds (Issue #5)

**Total estimated repair time:** 5–7 engineering days.

---

## Issue #1: G'.learn() Never Called — System Cannot Learn

**Severity:** P0 CRITICAL  

### What is wrong

The world model `WorldModelGPrime.learn()` (and `WorldModelMLP.learn()`) is **never called** from the cognitive cycle. Every cycle:

1. `env.step(action_idx)` returns `obs` — the actual next observation
2. The cycle computes prediction error against the *previous* prediction
3. **But `gprime.learn(state, action, next_state, error)` is never invoked**

The `learn()` method exists with a fully implemented frequency-count CPD update (discrete) and delta-rule parameter update (Gaussian), but **no code path reaches it**.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/core/cycle.py` | ~191–196 | After `env.step()` returns `obs`, the cycle computes `corrected_prediction` and PEU error, then calls `tspl.update()` — **but never calls `gprime.learn()`** |
| `python/phca/world_model/graph.py` | 406–447 | `learn()` method: fully implemented (frequency-count for discrete, delta-rule for Gaussian) — **never called** |
| `python/phca/world_model/mlp.py` | 129–198 | `learn()` method: SGD with experience replay — **never called** from cycle (but was added to cycle.py in Phase 3.3 cycle version) |

### Why it matters (invariant violation)

The architecture's core claim is **A4: Prediction as Primary** — the system learns from prediction error to improve its world model. Without `gprime.learn()`, the world model never updates its parameters. All predictions use the hand-tuned initial parameters (β₀=0, β₁=0.95, β₂..=0.1 for Gaussian; uniform 50/50 for discrete). The system achieves apparent Φ-IQ scores through static prior accuracy, not learning.

- The Gaussian BN's fixed identity betas (β₁=0.95, β₂=0.1) produce constant ~0.82 MSE on GridWorld states
- The `/10` normalization in Φ-IQ (`max(0, 1 - error/10)`) converts this to ~0.92 "prediction accuracy"  
- This gives the **illusion of learning** — the score is driven by initialization quality, not emergent improvement

### Concrete surgical fix

**1. Wire `gprime.learn()` into the cognitive cycle** (4 hours):

In `cycle.py` `step()`, after the PEU error computation and TSPL update (~line 196), add:

```python
# LEARN: update G' with observed transition
if self.current_state is not None:
    action_vec = np.zeros(self.env.action_space_size, dtype=np.float32)
    action_vec[action_idx] = 1.0
    next_state = StateVector(
        values=obs.astype(np.float32),
        precision=np.ones(self.state_dim, dtype=np.float32),
        timestamp=float(self.cycle_count),
        grounding_level=1,
    )
    self.gprime.learn(
        state_t=self.current_state,
        action=action_vec,
        state_t1=next_state,
        error=metrics.prediction_error,
    )
```

**Note:** The `obs` variable is already captured at line ~184 from `env.step(action_idx)` — it must be promoted from loop-local to accessible in the learn block. The `action_vec` must be constructed fresh (not using the aged `self.last_action`).

**2. Add cache invalidation after Gaussian learn** (30 minutes):

In `WorldModelGPrime.learn()`, after the Gaussian beta update, call `self.invalidate_cache()` so the next `predict_continuous()` uses the updated parameters:

```python
# After updating betas and sigmas:
self.invalidate_cache()  # force recompute of joint moments
```

**3. Verify that Gaussian `learn()` actually mutates parameters** (20 minutes):

The current Gaussian delta-rule in `learn()` writes updated betas to `node.params` and `node.std`. This correctly feeds into `_get_gaussian_topology()` → `compute_joint_moments()`. The `invalidate_cache()` call ensures the stale joint moments are recomputed.

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A1.1 | Prediction error decreases over learning cycles | Run 200 cycles with MLP, record MSE per cycle | Late MSE (last 50) ≤ 80% of early MSE (first 50) |
| A1.2 | G' parameters change after learn | Compare `gprime.state_history` size before/after 100 cycles | `state_history` has ≥ 50 entries (populated by learn) |
| A1.3 | Discrete CPDs update after observation | Create discrete G', run 50 cycles | CPD transition matrix differs from initial 50/50 |
| A1.4 | No regression on existing tests | `pytest python/tests/ -v --tb=short` | All 289 tests pass |

### Architectural principle

The fix preserves **A4 (Prediction as Primary)** by completing the observe→predict→learn feedback loop. It also satisfies **A5 (Feedback-Driven Adaptation)** — without this fix, adaptation is simulated rather than actual. The `invalidate_cache()` call ensures the O(n³) matrix inversion is not wasted on stale parameters.

**Dependency:** None (standalone fix).

---

## Issue #2: Consolidation Facts Produced But Never Consumed — Write-Only Architecture

**Severity:** P0 CRITICAL  

### What is wrong

The `ConsolidationScheduler` extracts semantic facts from M3 episodes every 10 cycles and stores them in `self._semantic_facts` via `_store_facts()`. However, **no module anywhere in the codebase calls `get_semantic_facts()`** or reads the fact store. Consolidation runs, extracts, stores, prunes — and the output is consumed by nothing.

Additionally, the **M4 write-lock semantics** required by the spec (§2.3.2 Patch C) are not implemented. Facts are simply appended to an in-memory list with no locking, no atomic transaction, and no snapshot isolation.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/consolidation/scheduler.py` | 277–293 | `_store_facts()` appends to `self._semantic_facts` — **no reader** |
| `python/phca/consolidation/scheduler.py` | 226–230 | `get_semantic_facts()` exists — **never called by any module** |
| `python/phca/core/cycle.py` | 330–340 | `consol_report = self.consolidation.step(...)` — return value only logged, facts never wired |
| `research/outputs/spec_compliance_audit.md` | C2 | **Spec violation:** M4 write-lock not implemented; facts stored as plain list append |
| `research/outputs/phca-v3-patch.md` | §2.3.2 | Spec requires: MVCC snapshots, write-lock acquisition with timeout, atomic transaction commit |

### Why it matters (invariant violation)

The architecture's **S-Stream (Semantic Memory)** is defined as producing facts that inform future prediction and action selection. Without this feedback path:

1. **Consolidation is pure overhead** — ~200ms of computation per sleep cycle (CPU, memory, I/O) with zero behavioral effect
2. **The S-Stream pipeline** (M3 → Consolidation → M4 → Prediction) is **broken at the last hop**
3. **Spec violation C2** — no write-lock, no MVCC, no atomic transactions on M4. Theorem 3.3 (Consolidation Atomicity) cannot be relied upon
4. The system cannot benefit from cross-episode knowledge transfer — every episode is learned and forgotten independently

### Concrete surgical fix

**1. Wire facts into the prediction engine's prior** (2 hours):

In `ConsolidationScheduler`, expose a method that returns the most confident facts matching a given state. In `PredictionEngine`, before calling `G'.predict()`, query for relevant facts and use them to bias the prior:

```python
# In ConsolidationScheduler:
def get_relevant_facts(self, state: StateVector, n: int = 5) -> List[SemanticFact]:
    """Return top-N facts closest to the given state by cosine similarity."""
    if not self._semantic_facts:
        return []
    # Simple: return most recent high-confidence facts
    candidates = sorted(
        self._semantic_facts, key=lambda f: f.confidence, reverse=True,
    )
    return candidates[:n]

# In cycle.py (after consolidation step, ~line 340):
if consol_report.success and consol_report.facts_generated > 0:
    recent_facts = self.consolidation.get_semantic_facts(
        min_confidence=0.3, max_results=10,
    )
    # Facts are logged but NOT yet wired into prediction.
    # Phase 4 integration: pass facts to engine.predict() as context priors.
```

**2. Implement M4 write-lock semantics** (1–2 days):

```python
# In ConsolidationScheduler.__init__():
import threading
self._m4_lock = threading.Lock()
self._m4_lock_timeout = 0.5  # τ_lock_wait = 500ms
self._semantic_facts: List[SemanticFact] = []  # active version
self._staging_facts: List[SemanticFact] = []   # staging buffer

# In _store_facts():
def _store_facts(self, facts: List[SemanticFact]) -> int:
    """Atomically swap staging facts into active store."""
    self._staging_facts = list(facts)  # build in staging buffer
    acquired = self._m4_lock.acquire(timeout=self._m4_lock_timeout)
    if not acquired:
        log.warning("consolidation.m4_lock_timeout")
        return 0  # skip this cycle per spec
    try:
        # Atomic swap: readers see either old or new, never partial
        self._semantic_facts = self._staging_facts
        return len(facts)
    finally:
        self._m4_lock.release()
```

**3. Wire facts into TSPL S-Stream bias** (Phase 4 — estimate 2 days):

Facts should inform the prediction engine's prior for familiar states. For each cycle, query top-5 relevant facts and adjust the G' prior distribution:

```python
# In PredictionEngine.predict() [Phase 4]:
relevant_facts = self.consolidation.get_relevant_facts(state, n=5)
if relevant_facts:
    # Adjust G' prior toward fact-confirmed state transitions
    fact_bias = sum(f.state_after.values * f.confidence for f in relevant_facts)
    fact_bias /= (sum(f.confidence for f in relevant_facts) + 1e-8)
    prior = 0.9 * gprime_prior + 0.1 * fact_bias
```

**Phase 4 only** — the core fix for Phase 3.3 is items 1 and 2.

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A2.1 | Facts are produced and stored | Run 50 cycles, check `get_stats()["total_facts_stored"]` | ≥ 5 facts stored |
| A2.2 | Facts are readable | Call `get_semantic_facts(min_confidence=0.0)` | Returns list of facts with valid types |
| A2.3 | M4 write-lock times out gracefully | Set artificially short timeout, run consolidation | Logs warning, returns 0, no crash |
| A2.4 | No regression on existing tests | `pytest python/tests/ -v --tb=short` | All 289 tests pass |

### Architectural principle

The fix preserves **A3 (Incomplete Knowledge)** — semantic facts represent accumulated knowledge that should inform prediction and reduce uncertainty. Without consuming facts, the S-Stream violates its own raison d'être. The write-lock semantics restore compliance with the formal **Theorem 3.3 (Consolidation Atomicity)**.

**Dependency:** Issue #1 (G'.learn must be working for facts to be meaningful — facts are extracted from episodes stored by learn).

---

## Issue #3: Benchmark Infrastructure is a Stub — Φ-IQ Claims Unverifiable

**Severity:** P0 CRITICAL  

### What is wrong

The official benchmark runner `python/phca/benchmarks/runner.py` returns `{"status": "not_implemented", "message": "Benchmarks start in Phase 3.3"}` for all levels. All benchmark results in `logs/benchmark_*.json` are produced by `scripts/benchmark.py` — a standalone script that is not part of the `phca` package, has no CI integration, and produces results that cannot be independently reproduced from the package itself.

The `phca.benchmarks.runner` is explicitly deferred to ticket `PHCA-3.3-003` per the gate report.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/benchmarks/runner.py` | 18–26 | `run_level_0()` returns `{"status": "not_implemented"}` |
| `python/benchmarks/runner.py` | 25 | `run_level_0()` — same stub; "Benchmarks start in Phase 3.3" |
| `gate_phase3.3_final.md` | §5 | Runner deferred to `PHCA-3.3-003` in remaining items |
| `logs/benchmark_mlp_final.json` | — | Produced by `scripts/benchmark.py`, not by `runner.py` |
| `.github/workflows/ci.yml` | — | No benchmark gate job configured |

### Why it matters (invariant violation)

Without a working benchmark runner:

1. **Φ-IQ scores cannot be independently verified** — the `gate_phase3.3_final.md` claims Φ-IQ = 0.626 (MLP) and 0.495 (Gaussian), but these numbers come from a standalone script with no versioned output format
2. **No regression detection** — future changes could silently reduce performance without automated detection
3. **Phase 3.3 gate condition unfulfilled** — the gate report explicitly lists this as a deferred item
4. **The v3.0 specification §1.3 success criteria cannot be validated** — pass/fail is determined by manual script runs

### Concrete surgical fix

**1. Implement `run_level_0()` through `run_level_3()` in `runner.py`** (2–3 days):

```python
def run_level_0(output_path: Optional[str] = None) -> dict:
    """Level 0: Stationary prediction — no action, just observe and predict."""
    cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_continuous=True)
    # Run 50 warmup + 500 benchmark cycles
    errors = []
    for _ in range(550):
        metrics = cycle.step()
        errors.append(metrics.prediction_error)
    
    result = {
        "level": 0,
        "status": "completed",
        "mean_error": float(np.mean(errors[-500:])),
        "error_trend": "improving" if errors[-50] < errors[:50] else "stable",
        "total_cycles": 550,
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
```

Implement levels 1–3 following the same pattern from `scripts/benchmark.py`.

**2. Add CI benchmark gate** (1 day):

In `.github/workflows/ci.yml`, add a benchmark step that runs Level 0 and compares against a baseline:

```yaml
- name: Benchmark Level 0
  run: |
    PYTHONPATH=python:$PYTHONPATH python -m phca.benchmarks.runner \
      --level=0 --output=logs/ci_benchmark.json
    python scripts/check_benchmark.py \
      --current=logs/ci_benchmark.json \
      --baseline=logs/benchmark_baseline.json \
      --threshold=0.95
```

**3. Create baseline snapshot** (1 hour):

Run the new benchmark on the current HEAD, save the result as `logs/benchmark_baseline.json`. This becomes the regression reference.

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A3.1 | `runner.py` produces valid output | `python -m phca.benchmarks.runner --level=0 --output=/tmp/test.json` | Output JSON has `status: "completed"` and valid metrics |
| A3.2 | All 4 basic levels produce output | Run levels 0–3 sequentially | Each produces valid JSON with no errors |
| A3.3 | CI gate passes | Push to PR branch | CI benchmark step completes within 5 minutes |
| A3.4 | Baseline comparison detects regression | Artificially break prediction (disable learn) | CI benchmark gate fails with Φ-IQ regression > 5% |

### Architectural principle

The benchmark infrastructure is the **validation layer** for the entire architecture. Without it, no claim about Φ-IQ, learning rate, or goal autonomy can be verified. The v3.0 specification §5 (Φ-IQ Composite Metric) and the implementation blueprint §D.2 (Benchmark Suite) both require this infrastructure. This fix completes the validation feedback loop that the architecture depends on for empirical claims.

**Dependency:** Issue #1 (benchmarks won't show learning without G'.learn working).

---

## Issue #4: Attention, HPM Validator, and MDIM Output Discarded — Computed But Never Used

**Severity:** P1 MAJOR  

### What is wrong

Three modules compute results every cycle that are **immediately discarded**:

| Module | Output | Where discarded | Lines |
|--------|--------|----------------|-------|
| **Attention** | Selected chunks from `select()` | Return value not captured in `cycle.py` | `cycle.py:319-321` |
| **HPM Validator** | Validation result from `validate()` | Return value assigned to `_` in earlier version; currently `compute_bounds()` used for RBTA but full composition validation unused | `cycle.py:~335` (removed in Phase 3.3) |
| **MDIM** | `target_state` in `GoalVector` | `_select_action()` only uses `drive_id` — the full 84-dim target state with precision is computed but never compared against current state | `cycle.py:109-150`, `mdim.py:265-310` |

Additionally, the **MDIM Pareto front evaluation** violates the spec (§2.4.1 Def 3.10): it operates on deficits (derived quantity) rather than actual configuration values (v1, v3, v5), making the front ambiguous.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/core/cycle.py` | 319–321 | `attention_chunks = self.attention.select(...)` — assigned but never used downstream |
| `python/phca/motivation/mdim.py` | 229–269 | `compute_pareto_front()` evaluates on deficits, not (v1, v3, v5) configuration |
| `python/phca/core/cycle.py` | 109–150 | `_select_action()` uses `goal.drive_id` only — ignores `goal.target_state` |
| `research/outputs/spec_compliance_audit.md` | C4 | Spec violation: Pareto front must use (v1, v3, v5) |

### Why it matters (invariant violation)

1. **Attention computation is 100% waste** — k-WTA selection with Gumbel noise runs every cycle but the result is discarded. This is CPU time that affects the cycle's latency budget without any behavioral benefit.

2. **MDIM target states are decorative** — The elaborate goal instantiation in `_goal_from_drive()` creates detailed target vectors with per-dimension precision values, but action selection only reads `drive_id`. The `target_state` carries information that could guide action toward specific state regions but is ignored.

3. **Pareto front spec violation** — Evaluating on deficits means two different configurations can yield the same deficits, making the detection of Pareto-optimality non-unique. This could cause the meta-stable state to activate or fail to activate in ambiguous cases.

### Concrete surgical fix

**1. Wire attention output into prediction** (1 hour):

Use selected chunks to weight prediction examples — high-attention chunks get higher weight in the learn step:

```python
# In cycle.py, after attention.select():
attention_chunks = self.attention.select(self.m2.chunks, self.current_goal)
# Apply attention weights: give more weight to high-salience predictions
if attention_chunks and self.current_state is not None:
    weights = np.array([c.salience for c in attention_chunks])
    weights = weights / (weights.sum() + 1e-8)
    # Store for learn() to use as per-dimension weight
    self._attention_weights = weights[:self.state_dim] if len(weights) >= self.state_dim else np.ones(self.state_dim)
else:
    self._attention_weights = np.ones(self.state_dim)
```

Then in the PEU/TSPL step, use `self._attention_weights` to weight the precision-weighted error.

**2. Fix Pareto front to evaluate on (v1, v3, v5) configuration** (1 day):

In `mdim.py`:

```python
def compute_pareto_front(self, config: Optional[Dict[int, float]] = None) -> List[int]:
    """Compute Pareto-optimal drives over actual configuration values.
    
    Uses (v1, v3, v5) = (prediction_error, competence_deficit, energy_cost)
    as specified in §2.4.1 Def 3.10 — not deficits.
    """
    if config is None:
        config = {
            1: self.drives[1].value,    # v1: prediction error (minimize)
            3: self.drives[3].value,     # v3: competence deficit (minimize) 
            5: self.drives[5].value,     # v5: energy cost (minimize)
        }
    
    pareto_ids = []
    for i in config:
        dominated = False
        for j in config:
            if i == j:
                continue
            # j dominates i if j is strictly better or equal on all dimensions
            # and strictly better on at least one
            if all(config.get(k, 0) <= config.get(j if k == i else k, 0) 
                   for k in config) and config[j] < config[i]:
                dominated = True
                break
        if not dominated:
            pareto_ids.append(i)
    return pareto_ids
```

**3. Wire MDIM target_state into action selection** (Phase 4 — 2 hours):

In `_select_action()`, when scoring candidate actions, compute distance from predicted next state to the MDIM goal target_state:

```python
# After computing distance_gain:
target = goal.target_state
if target is not None and goal.drive_id in (1, 3, 4):
    # Compute cosine similarity between predicted and target
    alignment = self._state_space_alignment(predicted, target)
    score = 0.6 * (1.0 - distance_gain) + 0.2 * min(confidence, 1.0) + 0.2 * alignment
```

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A4.1 | Attention weights affect prediction | Compare cycle with/without attention weights | MSE differs when weights are non-uniform |
| A4.2 | Pareto front is correct | Hand-crafted test with (v1=0.1, v3=0.05, v5=0.2) | Front identifies Pareto-optimal drives correctly |
| A4.3 | Meta-stable activates on front | Run 200 cycles, check meta-stable transitions | Meta-stable activated ≥ 1 time |
| A4.4 | No regression on existing tests | `pytest python/tests/ -v --tb=short` | All 289 tests pass |

### Architectural principle

This fix addresses two invariants:
- **A5 (Feedback-Driven Adaptation)**: Attention should modulate what the system learns from — without wiring the output, adaptation is blind.
- **G5 (MDIM Drive Satisfaction)**: The Pareto front is the mechanism that prevents drive thrashing. A wrong evaluation function makes the front unreliable, risking oscillation between D1/D3/D5.

**Dependency:** Issue #1 (attention weights only matter if G' learns from weighted examples).

---

## Issue #5: RBTA Energy Bounds Missing + Composition Tree Timing Bug

**Severity:** P1 MAJOR (with P0 sub-findings)  

### What is wrong

**Sub-issue 5a: Energy dimension absent from RBTA/HPM bound computation**

The spec (Definition 3.6 Corrected) defines three resource dimensions for composite bound computation:
- **SEQUENCE:** `B_energy = B_energy(M1) + B_energy(M2) + ε_overhead`
- **PARALLEL:** `B_energy = B_energy(M1) + B_energy(M2) + ε_comm`

The HPM `_compute_bounds()` only returns `B_time` and `B_mem` — **energy is never computed or verified**.

**Sub-issue 5b: CYCLE timing in composition tree set AFTER check_cycle()**

In `cycle.py`, `self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0` is set at line ~380 (after `check_cycle()`), but the composition tree's CYCLE leaf is checked by `check_cycle()` at line ~350. The value seen by RBTA is the stale value from `_collect_runtime_log()` (~0.1ms from RBTA's own timing) instead of the actual total cycle time (~12ms–52ms).

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/hpm/parser.py` | 394–479 | `_compute_bounds()` returns only `B_time` and `B_mem` — **energy missing** |
| `python/phca/regulation/rbta_enforcer.py` | 159–161, 183–254 | `_check_composition_tree()` only checks time bounds — **memory and energy unchecked** |
| `python/phca/core/cycle.py` | ~350, ~380 | CYCLE runtime set **after** RBTA check — timing bug |
| `research/outputs/spec_compliance_audit.md` | M3, M4 | Spec violations documented |

### Why it matters (invariant violation)

1. **Spec violation:** The formal spec explicitly defines three-dimensional resource bounds. Missing energy means the system is running with **2/3 of the constraint enforcer's specification** unimplemented. If energy consumption ever becomes a real constraint (battery-powered robot, thermal limits), the RBTA will not detect violations.

2. **Timing bug masks real latency violations:** The composition tree thinks the total cycle time is ~0.1ms (RBTA module timing) when it's actually ~52ms (MLP mode). If cycle times approach 500ms, the RBTA will not flag CYCLE violations because it's checking the wrong value. A real latency emergency could go undetected.

### Concrete surgical fix

**5a: Add energy to composite bound computation** (1 day):

In `python/phca/hpm/parser.py`:

```python
# Constants per spec:
EPSILON_OVERHEAD = 0.001  # composition overhead
EPSILON_COMM = 0.002       # communication overhead

def _compute_bounds(self, node: dict, runtime_log: dict) -> Dict[str, float]:
    """Compute composite resource bounds per Def 3.6 (Corrected).
    
    Returns dict with B_time, B_mem, B_energy.
    """
    if node["type"] == "SEQUENCE":
        children_bounds = [self._compute_bounds(c, runtime_log) for c in node["children"]]
        return {
            "B_time": sum(c["B_time"] for c in children_bounds) + EPSILON_OVERHEAD,
            "B_mem": max(c["B_mem"] for c in children_bounds),
            "B_energy": sum(c["B_energy"] for c in children_bounds) + EPSILON_OVERHEAD,
        }
    elif node["type"] == "PARALLEL":
        children_bounds = [self._compute_bounds(c, runtime_log) for c in node["children"]]
        return {
            "B_time": max(c["B_time"] for c in children_bounds) + EPSILON_COMM,
            "B_mem": sum(c["B_mem"] for c in children_bounds),
            "B_energy": sum(c["B_energy"] for c in children_bounds) + EPSILON_COMM,
        }
    else:
        # Leaf node: read from runtime_log
        module_id = node.get("id", "UNKNOWN")
        return {
            "B_time": runtime_log.get(module_id, 0.0),
            "B_mem": 0.0,  # memory per module not tracked per-cycle yet
            "B_energy": runtime_log.get(f"{module_id}_energy", 0.0),
        }
```

In `python/phca/regulation/rbta_enforcer.py`, extend `_check_composition_tree()` to check all three dimensions:

```python
# After computing actual composite:
actual_time = sum(runtime_log.get(c.get("id", ""), 0.0) for c in children)
actual_energy = sum(energy_log.get(c.get("id", ""), 0.0) for c in children)
if composite_bounds.get("B_time", 0) > 0 and actual_time > composite_bounds["B_time"]:
    violations.append(...)
if composite_bounds.get("B_energy", 0) > 0 and actual_energy > composite_bounds["B_energy"]:
    violations.append(...)
```

**5b: Fix CYCLE timing ordering** (15 minutes):

In `cycle.py`, move the latency computation before `check_cycle()`:

```python
# Compute total cycle latency BEFORE check_cycle
metrics.latency_ms = (time.perf_counter() - t_start) * 1000
self._collect_runtime_log(metrics)
self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0

# Now call check_cycle with correct CYCLE runtime
violations, enforcer_action = self.rbta.check_cycle(...)

# Remove the duplicate latency assignment after check_cycle
# (was: metrics.latency_ms = (time.perf_counter() - t_start) * 1000 — remove this)
```

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A5.1 | Energy composite computed | Parse a SEQUENCE tree, call `compute_bounds()` | Returned dict has `B_energy` key with plausible value |
| A5.2 | Energy violation detected | Set artificially low energy bound, run cycle | RBTA flags ENERGY violation |
| A5.3 | CYCLE timing correct | Run 100 cycles, check `runtime_log["CYCLE"]` | Value matches `metrics.latency_ms / 1000` (±1%) |
| A5.4 | Latency violation detected | Inject `time.sleep(0.6)` into cycle | RBTA flags CYCLE timeout violation |
| A5.5 | No regression on existing tests | `pytest python/tests/ -v --tb=short` | All 289 tests pass |

### Architectural principle

The fix restores **complete compliance with Definition 3.6 (Resource Additivity — Corrected)** from the formal patch. Three-dimensional resource bounds (time, memory, energy) are required for the **RBTA to fulfill its mandate as the resource-bounded supervisor** (§2.1, A1). The timing bug corrects a violation of **Theorem 2.1 (Constraint Composition)** — the composition tree must reflect actual execution timing to be monotonic.

**Dependency:** Fix 5a is independent. Fix 5b depends only on understanding the cycle.py temporal flow.

---

## Execution Order & Dependency Graph

```
Issue #1: G'.learn() Never Called
  └── Issue #2: Consolidation Facts Never Consumed
        └── Issue #3: Benchmark Infrastructure Stub
              └── Issue #4: Attention/MDIM Output Discarded
                    └── Issue #5: RBTA Energy + Timing
```

| Order | Issue | Time | Dependencies | Independent sub-fixes |
|-------|-------|------|-------------|----------------------|
| **1** | #1: G'.learn | 4–5h | None | — |
| **2** | #2: Consolidation | 2.5–3d | #1 (facts need learning) | Write-lock fix (1–2d) can start in parallel |
| **3** | #3: Benchmarks | 2–3d | #1 (benchmarks need learning) | — |
| **4** | #4: Attention/MDIM | 2.5h | #1 (attention weights need learning) | Pareto front fix (1d) independent |
| **5** | #5: RBTA Energy + Timing | 1.5d | None (5b); 5a is independent | **Fix 5b first (15 min)** — timing bug is trivially fixable |

**Total:** ~5–7 engineering days.

**Parallelization opportunities:**
- #2 write-lock fix can start in parallel with #1 (2 engineers)
- #5 (both sub-fixes) can start immediately — no dependencies (1 engineer)
- #4 Pareto front fix can start in parallel with #4 attention wiring (1 engineer)

### Escalation

If any fix reveals deeper architectural issues (e.g., the Gaussian BN `learn()` is fundamentally unable to learn one-hot states even when wired), **do not patch around it** — stop and:
1. Document the finding as a new formal issue
2. Trigger the contingency plan (switch to MLP mode as default)
3. Re-evaluate the Phase 3.3 → Phase 4 gate

---

## Appendix A: Original Audit Sources

| Issue | Primary Source | Secondary Source |
|-------|---------------|-----------------|
| #1 | Chief Architect Audit (Finding 1) | Simplification Report §1 |
| #2 | Spec Compliance Audit (C2, M6) | Chief Architect Audit (Finding 6) |
| #3 | Gate Phase 3.3 Report (§5) | Spec Compliance Audit (§4) |
| #4 | Chief Architect Audit (Findings 4, 11) | Spec Compliance Audit (C4, M5) |
| #5 | Chief Architect Audit (Finding 2) | Spec Compliance Audit (M3, M4) |

## Appendix B: Invariant Status After Fixes

| Invariant | Status | Notes |
|-----------|--------|-------|
| A1: Resource Boundedness | ✅ Strengthened | RBTA now checks 3 dimensions; CYCLE timing is correct |
| A2: Temporal Causality | ✅ Unchanged | No changes to cycle ordering |
| A3: Incomplete Knowledge | ✅ Strengthened | Semantic facts now inform prediction (Issue #2) |
| A4: Prediction as Primary | ✅ **Fixed** | G'.learn() now closes the prediction→learning loop (Issue #1) |
| A5: Feedback-Driven Adaptation | ✅ **Fixed** | Attention weights modulate learning (Issue #4) |
| G1–G5: Resolved Gaps | ✅ All preserved | No gap reopened by fixes |

---

*End of Document — Execute in order. After all 5 issues are resolved, run full benchmark suite and final architectural review.*
