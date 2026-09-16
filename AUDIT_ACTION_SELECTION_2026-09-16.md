# Action-Selection Ground-Truth Audit — 2026-09-16

## Scope and evidence rule

This is a static audit. It reports coded control flow and documented claims; it does not infer runtime frequencies or causal performance without an experiment. Every finding below points to repository evidence.

## Claims vs Reality

| Claim | Reality | Evidence |
|---|---|---|
| PHCA is prediction-primary | This is environment-scoped. Continuous MPC scores candidates with G′; default discrete GridWorld returns geometric action before the prediction-scored loop. | `python/phca/core/cycle.py:137-138`, `python/phca/core/cycle.py:1939-1982`, `python/phca/core/cycle.py:2173-2183` |
| Default discrete selection is geometric | True in code: `disable_blended_scorer=True` is the default and the selector returns the geometric action when available. | `python/phca/evaluation/interventions.py:47-71`, `python/phca/core/cycle.py:1939-1982` |
| The geometric planner is BFS/Manhattan | Partly true: plain grids of size ≥10 use `bfs_action`; other grid paths use one-step Manhattan neighbor selection over the observed grid. | `python/phca/core/cycle.py:2464-2503`, `python/phca/core/cycle.py:2504-2571` |
| G′ selects default discrete actions | False. G′ prediction is computed before action selection for cycle signals and learning, but the default discrete selector returns before the candidate prediction loop. | `python/phca/core/cycle.py:668-707`, `python/phca/core/cycle.py:1939-1982` |
| G′ supplies uncertainty | True at scalar level: `PredictionEngine.predict()` returns a scalar confidence; MLP confidence is derived from MC-dropout variance. It does not return a per-action score distribution through the public engine contract. | `python/phca/prediction/engine.py:42-86`, `python/phca/world_model/mlp.py:310-366` |
| G′ learns during the cycle | MLP/ensemble/hybrid models learn from observed `(state, action, next_state)` transitions unless learning is disabled, throttled, or feedback is skipped by RBTA. | `python/phca/core/cycle.py:1053-1078`, `python/phca/world_model/mlp.py:387-463` |
| RBTA can affect action selection | True. Preflight `TERMINATE` forces safe behavior and `INTERRUPT` limits candidate evaluation; the prior result is carried into the next cycle. | `python/phca/core/cycle.py:861-870`, `python/phca/core/cycle.py:383-390`, `python/phca/core/cycle.py:2890-2931` |
| Every decision is explainable from logs | Partly true when observability is attached: rationale and candidate scores are captured. The lower-level trace collector records action and aggregate prediction/RBTA fields but not the full rationale. | `python/phca/core/cycle.py:545-590`, `python/phca/monitoring/observability.py:260-263`, `python/phca/evaluation/trace.py:45-76` |
| `action_selection_interpretation()` reports actual selector behavior | False. It reports configuration-level labels and caveats, not per-cycle selector counts, fallback counts, or realized success comparisons. | `python/phca/evaluation/interventions.py:177-213` |

## Entry points and decision flow

### Synchronous path

`CognitiveCycle.step()` → `_run_perception_cycle()` → RBTA preflight → safe action or `_select_action()` → `env.step()` → learning → post-cycle RBTA.

Evidence: `python/phca/core/cycle.py:377-448`, `python/phca/core/cycle.py:618-870`, `python/phca/core/cycle.py:460-490`.

### Asynchronous path

Action thread applies carried RBTA status → perception → safe action or `_select_action()` → environment step → `ActionResult`; learning thread consumes the result and applies learning/RBTA.

Evidence: `python/phca/core/cycle.py:1186-1259`, `python/phca/core/cycle.py:1280-1365`.

### Discrete selector

Emergency entropy stop → no-state STAY → ε-greedy exploration → D5 STAY → geometric suggestion → geometry early return when the blended scorer is disabled or warming up → otherwise candidate prediction/scoring → adaptive confidence/agreement fallback → final action.

Evidence: `python/phca/core/cycle.py:1845-1982`, `python/phca/core/cycle.py:1992-2171`.

### Continuous selector

Emergency entropy stop → ε-greedy action → K bounded random candidates → G′ prediction per candidate → confidence/reference/PGA score → argmax candidate.

Evidence: `python/phca/core/cycle.py:2173-2298`.

## Signal contracts

| Signal/module | Input | Output | Default discrete action influence |
|---|---|---|---|
| Geometric planner | Grid position, goal, grid/observed grid, action names/deltas, optional visited cells | Discrete action and geometric score/components | Direct; final arbiter on default GridWorld path. `python/phca/core/cycle.py:2464-2571` |
| PredictionEngine/G′ | Current `StateVector`, last/current action | Predicted `StateVector`, scalar confidence | Indirect through prediction error, MDIM/regulation, learning, and RBTA; not the default action choice. `python/phca/prediction/engine.py:42-94`, `python/phca/core/cycle.py:668-707` |
| Blended scorer | Candidate action, G′ prediction/confidence, geometry action, goal/drive context | Per-candidate scalar score and selected action | Only when the blended path is enabled and not gated back. `python/phca/core/cycle.py:1998-2171` |
| MDIM | Current state/prediction and memory-derived context | Goal/drive state and planning context | Supplies goal context and planning grid; does not itself return the final action. `python/phca/core/cycle.py:709-845` |
| Attention | Working-memory chunks/state and precision context | Attention weights | Modulates learning gradients; no direct default discrete action arbitration. `python/phca/core/cycle.py:815-841`, `python/phca/core/cycle.py:1053-1071` |
| M1/M2/M3/consolidation | States, transitions, episodes, facts | Memory/context/replay data | M2/M4 context can affect planning state; M3 replay affects learning; no direct default action arbiter. `python/phca/core/cycle.py:650-660`, `python/phca/core/cycle.py:1080-1129` |
| RBTA | Runtime, memory, energy, entropy, sensor failures | Violations and CONTINUE/INTERRUPT/TERMINATE | Can veto/neutralize action or limit candidates. `python/phca/core/cycle.py:861-870`, `python/phca/regulation/rbta_enforcer.py:81-108`, `190-215` |

## Dead Code in Default Path

For default discrete GridWorld cycles with a valid geometric action, the following action-scoring work is not executed by the selector:

- Per-candidate G′ prediction loop and blended score calculation: `python/phca/core/cycle.py:1992-2078`.
- Blended confidence/agreement fallback evaluation: `python/phca/core/cycle.py:2080-2147`.
- Candidate rollouts produced by that loop: `python/phca/core/cycle.py:1989-1990`, `python/phca/core/cycle.py:2149-2155`.

These are not globally dead: they run for explicit blended opt-in and continuous MPC paths. The default path still computes pre-action prediction and post-action learning, so G′ itself is not dead. The distinction is “executed elsewhere or under another mode” versus “influences the default final discrete action.”

## Test coverage

Direct tests verify the default flag, default geometry selector, blended opt-in, learning-invariance of a default goal fraction, and continuous MPC prediction calls. Evidence: `python/phca/core/tests/test_action_selection_defaults.py:18-105`, `python/phca/core/tests/test_action_selection_defaults.py:158-192`.

Existing tests do not provide a dedicated confidence-gated flag, rolling success fallback, or an integration assertion for every rationale field. Most selector tests are single-seed deterministic tests; they are not statistical evidence. Existing causal evaluation defaults to five seeds unless overridden, and the Makefile smoke target uses one seed. Evidence: `scripts/phca_causal_eval.py:843-857`, `Makefile:83-91`.

## UNKNOWNS

- Runtime selector-mode frequencies under each entrypoint require an execution trace.
- Whether scalar confidence is calibrated for action selection requires a confidence-versus-realized-error experiment.
- Whether confidence gating improves behavior requires matched multi-seed evaluation.
- A rolling “G′ success rate versus geometric success rate” is not currently implemented; the precise operational metric must be specified before coding.
- Whether low-level trace logs are enabled in every entrypoint requires running each entrypoint with its logging configuration.

