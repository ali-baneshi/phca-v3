# PHCA Current State — 2026-09-17

> **AUTHORITATIVE SNAPSHOT.** This file describes the repository as implemented on September 17, 2026. Evidence and historical contradictions are recorded in [`AUDIT_ACTION_SELECTION_2026-09-16.md`](../AUDIT_ACTION_SELECTION_2026-09-16.md).

## Action selection

- Default discrete GridWorld selection is geometry-first: `disable_blended_scorer=True` causes `_select_action()` to return the geometric action before the candidate prediction loop.
- The geometric path uses BFS for qualifying large plain grids and one-step Manhattan neighbor selection elsewhere.
- The opt-in blended path evaluates candidate actions through G′ and may fall back to geometry through confidence/agreement gates.
- Continuous MPC samples bounded candidates and scores each through G′.
- RBTA preflight can force safe behavior or limit candidate evaluation before selection.
- When a `TraceCollector` is attached, each science-trace record persists the selector mode,
  decision reason, rationale, and candidate scores; candidate rollouts remain live-only in
  Observatory frames.

Evidence: `python/phca/evaluation/interventions.py:47-75`; `python/phca/core/cycle.py:1939-1982`, `2464-2571`, `2173-2298`, `2890-2931`.

## G′ and learning

- The public prediction contract returns a predicted next `StateVector` and scalar confidence.
- MLP G′ confidence is derived from MC-dropout variance.
- Learning uses observed next-state transitions and is controlled by intervention flags, throttling, and RBTA feedback skipping.

Evidence: `python/phca/prediction/engine.py:42-94`; `python/phca/world_model/mlp.py:310-366`, `387-463`; `python/phca/core/cycle.py:947-1078`.

## Current limitation

The default discrete action is not selected by G′. Therefore default GridWorld goal metrics do not by themselves establish prediction-primary control. An explicit `--confidence-gated` selector now exists for discrete experiments. It is disabled by default, warms up on geometry for 50 cycles, requires confidence ≥0.9 to enter the prediction-scored path, and activates a sticky geometry fallback when the prediction path's rolling realized goal rate is below geometry's after five observations per path. These values are configurable through `InterventionConfig` and the causal/benchmark script flags.

The implementation is experimental. The current 30-seed × 200-cycle 5×5 L2
repair validation completed on September 17, 2026: default goal rate 0.4797,
confidence-gated goal rate 0.5072, and both failed against `greedy_observed`.
The gated run exercised warm-up, prediction, below-threshold geometry, and
rolling fallback. The 10×10 confidence-gated protocol and full multi-level
promotion comparison remain unvalidated. See
[`docs/action_selection_validation_2026-09-17.md`](action_selection_validation_2026-09-17.md)
and the historical [`docs/action_selection_validation_2026-09-16.md`](action_selection_validation_2026-09-16.md).

The causal evaluator now reports confidence intervals and labels evidence as
`smoke_power`, `diagnostic_power`, or `causal_power`; only the 30-seed profile is
marked promotion-ready. The evaluator's existing scenario gate is unchanged.
