# PHCA v3.0 — Top 5 Critical Gap Fix Plan (Post-Phase 3.3 Audit)

**Date:** 2026-06-30  
**Author:** Chief Architect  
**Audit Scope:** All source files, 4 spec documents, 3 audit reports, benchmark suite, test suite, gate reports  
**Status:** PLAN MODE — no code changes; execute in order  
**Gate:** Phase 3.3 → Phase 4 readiness (re-assessment)

---

## 0. Executive Summary

Phase 3.3 successfully addressed the original 5 critical issues (G'.learn() wiring, M4 write-lock, benchmark infrastructure, attention output discarding, RBTA timing). The system now passes all 4 pass criteria in MLP mode (Φ-IQ = 0.626, failure rate = 3.4%, latency p95 = 52ms, goal autonomy achieved) with 289 tests passing.

**However, a deep re-audit of the post-Phase 3.3 codebase reveals 5 new critical issues** that must be resolved before Phase 3.3 can be considered truly complete and the system ready for Phase 4:

1. **MLP hidden_dim mismatch** — Documentation claims 128 hidden units (38,868 params), implementation uses 32 hidden units (6,708 params). This 4× capacity reduction directly limits the MLP's ability to learn complex transitions, explaining the Level 2 Φ-IQ = 0.320 bottleneck.

2. **Consolidation facts never consumed by prediction** — The consolidation pipeline (M3 snapshot → extract facts → store M4) produces 210+ semantic facts per run, but `get_semantic_facts()` is only called for logging. No module reads facts to bias the prediction prior.

3. **MLP/Gaussian learn() ignores the error parameter** — The `attn_weighted_error` computed every cycle is passed to `gprime.learn()` but accepted as a no-op: MLP's `learn()` comments "unused — MSE gradient used instead"; Gaussian's `learn()` uses only the state-action pair. Attention-weight modulation of learning is non-functional.

4. **MDIM target_state never used in action selection** — `_goal_from_drive()` creates full StateVector targets with per-dimension precision, but `_select_action()` reads only `goal.drive_id` (integer 1-6). The `_state_space_alignment()` method exists but is never reached for the drives that produce it.

5. **Energy bounds computed but never enforced by RBTA** — HPM `_compute_bounds()` returns `B_energy`, but the composition tree passed to `check_cycle()` only uses `B_time`. 2/3 of the spec's three-dimensional resource bound enforcement (Def 3.6: time, memory, energy) is unimplemented in practice.

**Total estimated repair time:** 4–6 engineering days.

---

## Issue #1: MLP hidden_dim = 32 (Documented as 128) — Capacity Mismatch

**Severity:** P0 CRITICAL  

### What is wrong

The `WorldModelMLP` in `python/phca/world_model/mlp.py` uses `hidden_dim: int = 32` as its default parameter (line 41). However, the architecture documentation (`docs/architecture.md`), release notes (`docs/phase3.3_release_notes.md`), and README all describe the MLP with hidden_dim=128 and 38,868 parameters:

- **README.md**: "MLP mode: The MLP world model (38,868 params) replaces the Gaussian G'..."
- **architecture.md**: References MLP dimensions implicitly via the module map
- **mlp.py**: `self.hidden_dim = 32` with He initialization scaled for 89→32

The actual parameter count with hidden_dim=32:
- Input 89 → W1 (89×32=2,848), b1 (32)
- Hidden → W2 (32×32=1,024), b2 (32)
- Output → W3 (32×84=2,688), b3 (84)
- **Total: ~6,708 parameters** (not 38,868)

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/world_model/mlp.py` | 41 | `hidden_dim: int = 32` default |
| `python/phca/world_model/mlp.py` | 176 | `__repr__` computes param count from `self.hidden_dim` |
| `README.md` | — | "38,868 params" in technical notes |
| `docs/architecture.md` | — | References MLP capacity implicitly |
| `logs/benchmark_mlp_final.json` | — | Level 2 Φ-IQ = 0.320 (symptom) |

### Why it matters (invariant violation)

**A4: Prediction as Primary** requires the world model to have sufficient representational capacity to learn the environment's transition dynamics. With 6,708 parameters (versus the 38,868 the architecture assumes), the MLP has ~17% of the documented capacity:

| Hidden Dim | Parameters | Level 2 Transitions Representable | Level 2 Φ-IQ |
|-----------|-----------|-----------------------------------|--------------|
| 32 (current) | ~6,708 | ~5-7 distinct patterns | 0.320 |
| 64 | ~17,992 | ~15-20 distinct patterns | (est.) 0.40-0.45 |
| 128 (documented) | ~38,868 | ~35-50 distinct patterns | (est.) > 0.50 |

The Level 2 benchmark (5×5 maze with wall barrier) requires the MLP to learn that walls block movement, that the goal is in a fixed position, and that taking path A vs path B leads to different outcomes. With hidden_dim=32, the network has ~7K parameters to model ~84 dimensional binary state transitions — this is likely capacity-constrained for the maze navigation task.

### Concrete surgical fix

**Option A (Recommended): Increase hidden_dim to 128** (1 hour):

```python
# In mlp.py __init__():
# Change default from 32 to 128
hidden_dim: int = 128
```

Then adjust batch training to avoid latency regression:

```python
# Ensure train_steps and batch_size are appropriate for larger network
# Current: train_steps=8, batch_size=64 → 8×64 = 512 forward+backward passes per cycle
# With hidden_dim=128, each pass is ~4x more expensive
# Reduce to: train_steps=4, batch_size=32 → still effective but ~4x fewer passes
# This keeps total compute roughly constant
```

**Option B (Conservative): Increase to 64 with tuned hyperparameters** (2 hours):

```python
hidden_dim: int = 64
# Re-tune lr, train_steps, batch_size for the larger network
lr: float = 0.05  # reduce from 0.1 for larger network
batch_size: int = 32
train_steps: int = 6
```

**Must also update** `build_for_env()` and `build_for_mujoco()` to pass `hidden_dim` through, OR change the default to 128 so the constructor parameter suffices.

**Must update documentation** (30 min):
- README.md: Update parameter count from 38,868 to actual
- docs/architecture.md: Add MLP config section

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A1.1 | MLP has documented capacity | `repr(model)` | Shows ≥ 38,000 parameters |
| A1.2 | No latency regression | Run benchmark Level 0 | p95 latency < 100ms |
| A1.3 | Level 2 Φ-IQ improves | `python scripts/benchmark.py --levels=2 --use-mlp --cycles=500` | Level 2 Φ-IQ > 0.40 |
| A1.4 | All 289 tests pass | `make test-all` | All pass |

### Architectural principle

The fix restores **A4 (Prediction as Primary)** by ensuring the MLP has the representational capacity the architecture design depends on. The documented 38,868-parameter MLP was the architect's intended capacity for GridWorld tasks — the 32-hidden-unit default was a implementation expedient that should have been updated before Phase 3.3 release. This is not a design flaw but a configuration/documentation error.

**Dependency:** None (standalone fix).

---

## Issue #2: Consolidation Facts Produced But Never Consumed — Write-Only Pipeline

**Severity:** P0 CRITICAL  

### What is wrong

The `ConsolidationScheduler` extracts semantic facts from M3 episodes every 10 cycles, stores them in `_committed_facts` via M4 write-lock, and returns statistics. The cycle logs fact types. **But no module reads the facts to bias prediction or inform action selection.**

In `cycle.py` lines 329-340:
```python
consol_report = self.consolidation.step(self.cycle_count)
if consol_report.success and consol_report.episodes_processed > 0:
    recent_facts = self.consolidation.get_semantic_facts(
        min_confidence=0.3, max_results=10,
    )
    fact_types = {}
    for f in recent_facts:
        fact_types[f.fact_type] = fact_types.get(f.fact_type, 0) + 1
    _log(logger, "info", "cycle.consolidation",
         episodes=consol_report.episodes_processed,
         facts=consol_report.facts_generated,
         fact_types=fact_types,
         ...)
```

The facts are retrieved, categorised by type, logged, and **discarded**. The `fact_types` dict is never stored, never passed to the prediction engine, never used to adjust priors.

Meanwhile, the benchmark shows 215 facts stored (Level 3 MLP run) — each representing a learned transition pattern — with zero influence on system behaviour.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/core/cycle.py` | 329–340 | Facts logged but discarded |
| `python/phca/consolidation/scheduler.py` | 226–230 | `get_semantic_facts()` exists, called for logging only |
| `python/phca/prediction/engine.py` | — | `predict()` accepts no fact context |
| `python/phca/consolidation/scheduler.py` | 65–81 | `_extract_facts()` produces transition/novelty/well_known facts but they have no consumer |

### Why it matters (invariant violation)

**A3: Incomplete Knowledge** requires the system to accumulate and use knowledge to reduce uncertainty over time. The consolidation pipeline is the S-Stream's mechanism for this — it extracts stable patterns from episodic experience. Without consuming facts:

1. The entire consolidation pipeline (M3 SQLite reads, serialization, fact extraction, M4 write-lock) is **pure overhead** — ~5ms per consolidation cycle with zero behavioural effect
2. The system exhibits **no long-term memory** — every episode contributes to M3 episodes count (510 on Level 3) but has zero effect on future predictions
3. The S-Stream (semantic memory) is **formally non-functional**: facts accumulate in M4 but no decision-making pathway reads them
4. **Phase 4 cannot start** with a write-only semantic pipeline — the entire Phase 4 plan depends on facts informing prediction (Phase 4 scope: "merge into G' CPDs or use as priors for MLP pre-training")

### Concrete surgical fix

**1. Wire facts into the prediction engine's prior** (2 days):

Add a method to `ConsolidationScheduler` that returns the most relevant facts for a given state:

```python
def get_relevant_facts(self, state: StateVector, n: int = 5) -> List[SemanticFact]:
    """Return top-N facts closest to the given state by cosine similarity."""
    if not self._committed_facts:
        return []
    # Sort by confidence (simple heuristic — cosine similarity would be more precise)
    candidates = sorted(
        self._committed_facts, key=lambda f: f.confidence, reverse=True,
    )
    return candidates[:n]
```

Then in `cycle.py`, pass relevant facts through `mdim_context` so MDIM can bias goals toward fact-confirmed states:

```python
# Before MDIM goal generation:
relevant_facts = self.consolidation.get_relevant_facts(self.current_state, n=5)
mdim_context["consolidation_facts"] = total_facts
mdim_context["relevant_fact_count"] = len(relevant_facts)
mdim_context["fact_confidence_mean"] = float(np.mean([f.confidence for f in relevant_facts])) if relevant_facts else 0.0
```

In `PredictionEngine`, add a `fact_bias` parameter to `predict()`:

```python
# Phase 3.3+: bias G' prior toward fact-confirmed state transitions
def predict(self, state: StateVector, horizon: int = 1,
            action: Optional[np.ndarray] = None,
            fact_bias: Optional[np.ndarray] = None) -> Tuple[StateVector, float]:
    # ... existing prediction logic ...
    if fact_bias is not None and fact_bias.shape == predicted.values.shape:
        # Blend prediction toward fact-biased direction
        predicted.values = 0.9 * predicted.values + 0.1 * fact_bias
```

**2. Gate the fact consumption cost** (30 min):

Add a `max_relevant_facts` constructor parameter to `ConsolidationScheduler` (default 5). The fact similarity search should be capped to avoid latency regression.

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A2.1 | Facts retrievable by relevance | Unit test: store facts, query by state | Returns matching facts ordered by confidence |
| A2.2 | Fact bias changes prediction | Run 50 cycles with/without fact consumption | Prediction output differs when facts present |
| A2.3 | No latency regression | Run benchmark Level 0 | p95 latency < 55ms (within 10% of baseline) |
| A2.4 | All tests pass | `make test-all` | All pass |

### Architectural principle

The fix completes the **S-Stream (Semantic Memory)** feedback loop that the architecture depends on for **A3 (Incomplete Knowledge)**. Facts represent accumulated knowledge that should reduce prediction uncertainty for familiar states. Without this connection, the S-Stream is write-only overhead. The fix does not change the core prediction algorithm — it adds a small bias term that pulls predictions toward fact-confirmed transitions.

**Dependency:** Issue #1 (MLP capacity must be sufficient for facts to be meaningful — low-capacity MLP would produce noisy predictions that facts cannot usefully bias).

---

## Issue #3: MLP/Gaussian learn() Ignore the error Parameter — Attention Weights Decorative

**Severity:** P1 MAJOR  

### What is wrong

The cognitive cycle computes `attn_weighted_error` in `cycle.py` (lines ~304-310) and passes it to `self.gprime.learn()`. However, both world model implementations treat the `error` parameter as a no-op:

**WorldModelMLP.learn()** (mlp.py:129-198):
```python
def learn(
    self,
    state_t: StateVector,
    action: np.ndarray,
    state_t1: StateVector,
    error: float,  # <-- accepted but NOT CONSUMED
) -> None:
    """...
    Args:
        error: Scalar prediction error (unused — MSE gradient used instead).
    """
```

**WorldModelGPrime.learn()** (graph.py:406-447):
```python
def learn(
    self,
    state_t: StateVector,
    action: np.ndarray,
    state_t1: StateVector,
    error: float,
) -> None:
    # error is only used to decide whether to append to state_history
    # not to modulate learning rate, per-dimension weight, or anything else
```

The attention weights computed every cycle (`self._attention_weights` in cycle.py line ~326) are averaged to a scalar and passed as `error` — but since `error` is ignored, the entire attention-weight computation is **cosmetic**.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/world_model/mlp.py` | 129-130, 151 | `error` param documented as "unused" in docstring and implementation |
| `python/phca/world_model/graph.py` | 406, 424 | `error` param used only for `state_history` append decision |
| `python/phca/core/cycle.py` | 304–310 | `attn_weighted_error` computed and passed to `learn()` — no effect |
| `python/phca/core/cycle.py` | 311–326 | `self._attention_weights` computed from chunk saliences — averaged to scalar, loses per-dimension information |

### Why it matters (invariant violation)

**A5: Feedback-Driven Adaptation** requires that prediction error and attention modulate what the system learns. Currently:

1. **Attention weights are 100% computational waste** — the k-WTA selection runs, Gumbel noise is applied, saliences are computed, and the result is averaged to a scalar that is passed to a method that ignores it
2. **No per-dimension learning** — the attention weights are computed at full `state_dim` resolution but then averaged to a scalar, discarding all spatial information about which dimensions are more/less salient
3. **The `attn_weighted_error` variable is misleading** — it creates the impression that attention modulates learning, but it has zero effect on either world model's parameter updates
4. **The Phase 3.3 completion report claims this is fixed** — the completion report says "Attention weights modulate learning" (A5 evidence), but this is incorrect. Weights are computed and passed but never consumed.

### Concrete surgical fix

**1. Wire error into MLP learning rate modulation** (2 hours):

Modify `WorldModelMLP.learn()` to use the error parameter to modulate the learning rate:

```python
def learn(
    self,
    state_t: StateVector,
    action: np.ndarray,
    state_t1: StateVector,
    error: float,
) -> None:
    """Learn from observed transition with error-modulated learning rate.
    
    The error parameter modulates the effective learning rate:
      lr_effective = self.lr * (1.0 + error * 0.1)
    
    Higher error → larger learning steps (correct mistakes quickly).
    Lower error → smaller learning steps (fine-tune).
    """
    # Map error to learning rate modifier
    lr_mod = float(np.clip(1.0 + abs(error) * 0.1, 0.5, 2.0))
    effective_lr = self.lr * lr_mod
    
    # ... existing learn() logic ...
    self._apply_gradient(avg_grad, lr=effective_lr * 0.5)
```

**2. Wire per-dimension attention weights into gradient** (3 hours):

Instead of averaging attention weights, pass them as per-dimension multipliers on the loss gradients:

```python
# In MLP._backward():
# The d_out already has shape (state_dim,)
# Multiply by attention weights to modulate per-dimension learning
if attention_weights is not None:
    d_out = d_out * attention_weights.astype(np.float32)
```

This requires passing `attention_weights` through the learn() interface — or storing them on the cycle and having the MLP read them from its own state. Storing them on the MLP (set before learn() is called, cleared after) is the minimal change:

```python
# In cycle.py, before calling learn():
self.gprime._attention_weights = self._attention_weights  # set before learn

# In MLP.learn():
if hasattr(self, '_attention_weights') and self._attention_weights is not None:
    weights = self._attention_weights
else:
    weights = np.ones(self.state_dim, dtype=np.float32)
# Use weights in backward pass
```

**3. Clean up the Gaussian learn() path** (30 min):

For `WorldModelGPrime.learn()`, the error parameter can modulate the delta-rule learning rate:

```python
# In Gaussian learn(), modify the delta rule:
lr_mod = float(np.clip(1.0 + abs(error) * 0.1, 0.5, 2.0))
lr = 0.05 * lr_mod  # was: lr = 0.05
```

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A3.1 | Error modulates MLP learning rate | Set error=0 vs error=10, compare gradient magnitudes | Higher error → proportionally larger gradients |
| A3.2 | Attention weights affect per-dimension gradients | Set uniform vs non-uniform weights, compare d_out | Non-uniform weights produce non-uniform gradient across dims |
| A3.3 | No regression on prediction accuracy | Run benchmark Level 0 | Mean error within ±10% of baseline |
| A3.4 | All tests pass | `make test-all` | All pass |

### Architectural principle

This fix completes **A5 (Feedback-Driven Adaptation)** by making the attention-weight modulation of learning **actually functional**. The architecture's design intent is clear: attention selects what matters, and the learning system should weight its updates accordingly. Without this fix, A5 is satisfied in documentation but not in runtime behaviour. The per-dimension weight approach (rather than scalar averaging) preserves the spatial information that attention computes, making the mechanism consistent with the theoretical design.

**Dependency:** Issue #1 (larger MLP capacity makes per-dimension weight modulation more meaningful — with 32 hidden units, the gradient is already capacity-constrained).

---

## Issue #4: MDIM target_state Never Used in Action Selection

**Severity:** P1 MAJOR  

### What is wrong

MDIM's `_goal_from_drive()` creates detailed `GoalVector.target_state` objects with per-dimension values and precision arrays. These are full `StateVector` instances designed to represent the agent's desired state:

```python
# D1: Explore uncertain regions where confidence is low
target = StateVector(
    values=np.ones(self.state_dim, dtype=np.float32) * 0.5,
    precision=np.ones(self.state_dim, dtype=np.float32) * 0.3,
)
# D2: Seek criticality
target = StateVector(
    values=np.zeros(self.state_dim, dtype=np.float32),
    precision=np.ones(self.state_dim, dtype=np.float32) * 0.8,
)
# D3: Practice — seek familiar states near current
target = StateVector(
    values=np.zeros(self.state_dim, dtype=np.float32),
    precision=np.ones(self.state_dim, dtype=np.float32) * 0.9,
)
# etc.
```

But `_select_action()` in `cycle.py` (lines 109-158) reads only `goal.drive_id` to determine action scoring strategy:

```python
goal = self.current_goal
goal_id = goal.drive_id if goal else 1
target = goal.target_state if goal else None  # <-- captured but mostly unused

# D5 (Energy Efficiency): prefer STAY
if goal_id == 5:
    return self.env.stay_action

# For action scoring:
if goal_id in (1, 3):
    score = (1.0 - distance_gain) * 0.8 + min(confidence, 1.0) * 0.2
elif goal_id in (2, 4):
    score = (1.0 - min(confidence, 1.0)) * 0.7 + (1.0 - distance_gain) * 0.3
else:
    state_align = self._state_space_alignment(predicted, target)  # D6 only
    score = state_align
```

The `target` variable is captured but **only used in the D6 branch** (`else` clause) via `_state_space_alignment()`. For D1-D5, target is completely ignored.

Additionally, the `_state_space_alignment()` method exists (lines ~164-196) and correctly computes weighted cosine similarity between predicted and target states — but it's only called for D6 (empowerment), which has a generic all-ones target.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/motivation/mdim.py` | 265–310 | `_goal_from_drive()` creates full StateVector targets — decorative |
| `python/phca/core/cycle.py` | 115–118 | `target = goal.target_state` captured but unused for D1-D5 |
| `python/phca/core/cycle.py` | 130–148 | Action scoring uses `goal_id` only, not `target` |
| `python/phca/core/cycle.py` | 164–196 | `_state_space_alignment()` exists — only called for D6 |

### Why it matters (invariant violation)

**G5 (MDIM Drive Satisfaction):** The MDIM module's goal generation creates targets that have **zero effect on action selection** for drives D1-D5. This means:

1. **Goal generation is decorative computation** — ~100 lines of MDIM code produce elaborate target vectors that the action selection ignores
2. **Drive diversity is unrealised** — D1 (error minimisation) and D3 (competence) produce identical action-selection behaviour because they share the same scoring branch (distance_gain + confidence)
3. **D2 (criticality) and D4 (curiosity) produce identical behaviour** — same scoring branch (uncertainty + distance_gain)
4. **The system cannot distinguish between "explore uncertain region D1 goal" and "practice skill D3 goal" at the action level** — they both map to the same equation

### Concrete surgical fix

**1. Wire target_state into D1/D3 action scoring** (1 day):

Modify `_select_action()` to use `_state_space_alignment()` for D1/D3 goals when target_state is available and meaningful:

```python
if goal_id in (1, 3):
    # D1/D3: use distance_gain + confidence + goal alignment
    base_score = (1.0 - distance_gain) * 0.6 + min(confidence, 1.0) * 0.2
    # Add state-space alignment with goal target
    if target is not None:
        alignment = self._state_space_alignment(predicted, target)
        score = base_score + alignment * 0.2
    else:
        score = base_score
elif goal_id in (2, 4):
    # D2/D4: use uncertainty + distance_gain + goal alignment
    base_score = (1.0 - min(confidence, 1.0)) * 0.5 + (1.0 - distance_gain) * 0.2
    if target is not None:
        alignment = self._state_space_alignment(predicted, target)
        score = base_score + alignment * 0.3
    else:
        score = base_score
```

**2. Improve goal target quality for D1-D3** (1 hour):

The current targets are generic (all-0.5, all-0.0). Make them state-dependent:
- **D1 target**: Use current state with low precision (encourage exploring variation from current)
- **D3 target**: Use predicted state with high precision (encourage staying near what's known)
- **D4 target**: Use states with high prediction error (encourage exploring surprising regions)

```python
# D1: Explore uncertain regions — specific to current uncertainty profile
current_uncertain_dims = np.where(state.precision < 0.5)[0]
target = StateVector(
    values=np.ones(self.state_dim, dtype=np.float32) * 0.5,
    precision=np.where(current_uncertain_dims, 0.1, 0.5).astype(np.float32),
)
```

**3. Remove decorative target generation for D5/D6 if unused** (30 min):

If D5 (energy) and D6 (empowerment) targets are never used, either make them meaningful or skip target generation for these drives.

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A4.1 | D1/D3 action selection differs | Same state, different D1 vs D3 targets | Different actions selected |
| A4.2 | D2/D4 action selection differs | Same state, different D2 vs D4 targets | Different actions selected |
| A4.3 | Drive diversity increases | Run 200 cycles, count unique goal_id→action_id pairs | Each drive generates measurably different action distribution |
| A4.4 | No regression on existing tests | `make test-all` | All pass |

### Architectural principle

The fix completes the **MDIM goal → action selection pipeline** that the architecture specifies. Goal vectors carry a drive_id (which drive) AND a target_state (what the goal is). Using only the drive_id reduces MDIM to a single-integer signal, discarding the rich state information that goal generation computes. This fix makes **each drive's target_state influence action selection** in a drive-appropriate way, restoring the architectural intent of **G5 (MDIM Drive Satisfaction)**.

**Dependency:** Issue #1 (MLP must accurately predict state transitions for alignment scoring to be meaningful — with hidden_dim=32, predictions may be too noisy for alignment to help).

---

## Issue #5: Energy Bounds Computed But Not Enforced by RBTA

**Severity:** P1 MAJOR  

### What is wrong

The formal specification (Definition 3.6 Corrected) defines three resource dimensions for composite bound computation: **B_time, B_mem, B_energy**. The HPM `_compute_bounds()` (parser.py:394-479) correctly returns all three:

```python
# SEQUENCE (from parser.py):
return {
    "B_time": sum(c["B_time"] for c in child_bounds) + TAU_COMP,
    "B_mem": max(c["B_mem"] for c in child_bounds) + DELTA_SHARED,
    "B_energy": sum(c.get("B_energy", 1.0) for c in child_bounds) + EPSILON_OVERHEAD,
}
```

However, the cognitive cycle's composition tree in `cycle.py` lines 275-300 only uses `B_time`:

```python
hpm_bounds = self.hpm_validator.compute_bounds(hpm_spec, self.runtime_log)
reg_b_time = hpm_bounds["B_time"] if hpm_bounds else 0.200
# ...composition tree only has bounds for B_time:
composition_tree = {
    ...
    "bounds": {"B_time": reg_b_time * 2 + 0.050},
}
```

The `hpm_bounds["B_energy"]` value is computed by HPM but **never consumed**. The composition tree passes only `B_time` bounds to `RBTA.check_cycle()`. Energy violations (e.g., a module consuming more than its `B_energy` budget) go undetected.

This means 2/3 of the RBTA specification's three-dimensional resource enforcement is effectively unimplemented in the runtime control loop.

### Where

| File | Lines | What |
|------|-------|------|
| `python/phca/core/cycle.py` | 275-278 | `hpm_bounds["B_time"]` extracted, `B_energy` ignored |
| `python/phca/core/cycle.py` | 280-300 | Composition tree only has `"bounds": {"B_time": ...}` — no `B_energy` |
| `python/phca/hpm/parser.py` | 394-479 | `_compute_bounds()` correctly returns `B_energy` — output discarded |
| `python/phca/regulation/rbta_enforcer.py` | — | `_check_composition_tree()` verifies time — energy dimension unchecked |
| `python/phca/config.py` | 140-155 | `ResourceBounds` includes `B_energy` — correctly defined but not enforced |

### Why it matters (invariant violation)

**A1: Resource Boundedness** requires enforcement of all three resource dimensions (time, memory, energy). The current implementation enforces only time:

1. **Spec violation** — Definition 3.6 explicitly defines three-dimensional bounds. Operating with 1/3 enforcement is a clear spec deviation
2. **Energy consumption cannot balloon undetected** — if a module starts consuming 100× its energy budget (e.g., MLP training diverges, causing excessive recomputation), the RBTA will not flag it
3. **Battery-constrained deployment is impossible** — the energy dimension is specifically designed for power-aware deployment (robotics, edge devices). Without enforcement, energy-aware scheduling is blind
4. **The energy dimension exists in two places but the connection is broken** — HPM computes it, config defines it, but the cycle doesn't pass it to RBTA

### Concrete surgical fix

**1. Extract and pass B_energy to composition tree** (30 min):

In `cycle.py`, modify the composition tree construction to include energy bounds:

```python
hpm_bounds = self.hpm_validator.compute_bounds(hpm_spec, self.runtime_log)
reg_b_time = hpm_bounds["B_time"] if hpm_bounds else 0.200
reg_b_energy = hpm_bounds.get("B_energy", 10.0) if hpm_bounds else 10.0

composition_tree = {
    "type": "SEQUENCE", "id": "cognitive_cycle",
    "children": [
        "ASI", "WM", "PE", "PEU", "TSPL-P",
        {
            "type": "PARALLEL", "id": "regulation_block",
            "children": ["MDIM", "CR", "ATTN", "HPM"],
            "bounds": {"B_time": reg_b_time, "B_energy": reg_b_energy},
        },
        "CYCLE",
    ],
    "bounds": {
        "B_time": reg_b_time * 2 + 0.050,
        "B_energy": reg_b_energy * 2 + 0.010,
    },
}
```

**2. Ensure RBTA checks energy in composition tree** (1 hour):

In `python/phca/regulation/rbta_enforcer.py`, modify `_check_composition_tree()` to compute and verify energy composite bounds:

```python
# After computing composite time for a node:
actual_time = ...
actual_energy = sum(
    self.energy_log.get(child.get("id", ""), 0.0) 
    if isinstance(child, dict) else self.energy_log.get(child, 0.0)
    for child in children
)

# Check against node bounds
bounds = node.get("bounds", {})
if bounds.get("B_time", 0) > 0 and actual_time > bounds["B_time"]:
    violations.append(...)
if bounds.get("B_energy", 0) > 0 and actual_energy > bounds["B_energy"]:
    violations.append(...)
```

**3. Verify energy_log accuracy** (1 hour):

The current `energy_log` in `_collect_runtime_log()` estimates energy as `runtime_s * 50.0` — this is a scaling heuristic, not a measurement. For Phase 3.3, add a TODO to instrument actual energy measurement:

```python
# Energy estimates from actual module runtimes (scaled to match original magnitude)
# TODO: Replace with actual energy measurement in Phase 4 (power monitoring hardware)
self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
```

### Acceptance criteria

| # | Criterion | Method | Pass condition |
|---|-----------|--------|----------------|
| A5.1 | Energy bounds in composition tree | Inspect `cycle.py` composition_tree | `bounds` dict contains `B_energy` key |
| A5.2 | Energy violation detected | Set artificially low B_energy bound, run cycle | RBTA flags ENERGY violation |
| A5.3 | HPM energy output consumed | Trace `hpm_bounds["B_energy"]` through cycle | Value flows from HPM → composition tree → RBTA |
| A5.4 | No regression on existing tests | `make test-all` | All pass |

### Architectural principle

The fix restores **full compliance with Definition 3.6 (Resource Additivity — Corrected)** by completing the three-dimensional bound enforcement that the RBTA supervisor requires for **A1 (Resource Boundedness)**. The energy dimension is particularly important for the architecture's claim of being deployable on resource-constrained hardware (robotics, edge devices). Without energy enforcement, the RBTA is effectively a time-bound supervisor that ignores 2/3 of its mandate.

**Dependency:** None (standalone fix — the energy dimension infrastructure already exists in HPM and config; only the wiring from cycle → RBTA is missing).

---

## Execution Order & Dependency Graph

```
Issue #1: MLP hidden_dim (capacity foundation)
  └── Issue #3: learn() error param (needs capacity to matter)
        └── Issue #4: MDIM target_state (needs learning to give meaningful targets)
              └── Issue #2: Consolidation facts (needs facts to improve prediction)
Issue #5: Energy bounds (independent)
```

| Order | Issue | Time | Dependencies | Parallel? |
|-------|-------|------|-------------|-----------|
| **1** | #1: MLP hidden_dim | 3h | None | #5 can run in parallel |
| **2** | #3: learn() error param | 5.5h | #1 (needs capacity) | — |
| **3** | #4: MDIM target_state | 1.5d | #1, #3 (needs learning) | — |
| **4** | #2: Consolidation facts | 2d | #1, #3 (needs learning for meaningful facts) | — |
| **5** | #5: Energy bounds | 2.5h | None | **Can start immediately in parallel with #1** |

**Total:** ~4-6 engineering days (with #5 parallelised).

**Parallelization:** Issue #5 (energy bounds) is fully independent and can be fixed immediately. Assign to a second engineer while the primary engineer works through the dependency chain #1 → #3 → #4 → #2.

---

## Appendix A: Invariant Status After Fixes

| Invariant | Current Status | Post-Fix Status | Notes |
|-----------|---------------|-----------------|-------|
| A1: Resource Boundedness | ⚠️ 2/3 dimensions | ✅ All 3 dimensions | Energy bound wiring (Issue #5) |
| A2: Temporal Causality | ✅ | ✅ Unchanged | No cycle ordering changes |
| A3: Incomplete Knowledge | ⚠️ Write-only | ✅ Facts consumed | Fact consumption (Issue #2) |
| A4: Prediction as Primary | ⚠️ Capacity-constrained | ✅ Full capacity | MLP hidden_dim fix (Issue #1) |
| A5: Feedback-Driven Adaptation | ⚠️ Decorative | ✅ Functional | error param wired (Issue #3) |

## Appendix B: Benchmark Impact Estimates

| Issue | Current Metric | Estimated After Fix | Measurement |
|-------|---------------|-------------------|-------------|
| #1 MLP capacity | L2 Φ-IQ = 0.320 | > 0.40 (est.) | `--levels=2 --use-mlp --cycles=500` |
| #3 error modulation | Constant learning | Faster convergence | `--quick` error trend analysis |
| #4 target state | Drive diversity = 4/5 | 6/6 (fully diverse) | `--levels=3 raw_metrics.active_drives` |
| #2 facts consumed | Facts stored = 215 | Fact-bias reduces L2 error | Compare L2 error with/without fact bias |
| #5 energy bounds | Unenforced | Violations on overload | Inject energy spike, verify RBTA flags |

---

*End of Document — Execute in order with Issue #5 parallelised. After all 5 issues resolved, re-run full benchmark suite (MLP mode, 500 cycles/level) and final architectural review.*
