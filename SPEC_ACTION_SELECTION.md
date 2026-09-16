# Action-Selection Subsystem Specification

Date: 2026-09-16

This specification is based on the static audit in [`AUDIT_ACTION_SELECTION_2026-09-16.md`](AUDIT_ACTION_SELECTION_2026-09-16.md). It separates the current contract from the proposed opt-in extension.

## Current contract (as-is)

`CognitiveCycle.step()` performs perception/regulation, runs RBTA preflight, then either selects a safe action or calls `_select_action()`. The synchronous and asynchronous paths share this selector. Evidence: `python/phca/core/cycle.py:377-448`, `1186-1259`.

For discrete GridWorld, the default is `disable_blended_scorer=True`. When a geometric action exists, the selector returns the geometric result before candidate G′ scoring. Evidence: `python/phca/evaluation/interventions.py:47-75`, `python/phca/core/cycle.py:1939-1982`.

For continuous environments, MPC samples bounded candidates and scores each candidate using G′ confidence, reference alignment, and predicted-goal alignment. Evidence: `python/phca/core/cycle.py:2173-2298`.

G′ accepts state and action and returns predicted next state plus scalar confidence. MLP confidence is derived from MC-dropout variance; the learned target is the observed next state. Evidence: `python/phca/prediction/engine.py:42-94`, `python/phca/world_model/mlp.py:310-366`, `387-463`.

RBTA preflight can force safe behavior or limit candidate evaluation. Evidence: `python/phca/core/cycle.py:861-870`, `2890-2931`.

## Intended contract per corrected documentation

Prediction-primary behavior is environment-scoped. Continuous MPC is prediction-scored. Default discrete GridWorld remains geometry-primary because the current learned model is not the final arbiter in that path. A future prediction-guided discrete path must be explicit, confidence-gated, and observable. Evidence: `README.md:64-108`, `docs/action_selection.md:3-9`, `docs/limitations.md:71-84`.

## Gap identified before implementation

These were the gaps found by the pre-implementation audit. The implemented opt-in extension below addresses them without changing the default path.

1. There was no separate explicit confidence-gated selector flag.
2. Existing warm-up applied to the blended scorer, but there was no dedicated contract tying warm-up, confidence threshold, and fallback success comparison together.
3. Scalar confidence was available from `predict()`, but no small public convenience method exposed confidence as a standalone selector contract.
4. No rolling comparison of realized outcomes for prediction-scored versus geometric selections existed.
5. Existing rationale recorded selector mode and scores, but not the proposed confidence-gated reason and rolling-success fallback state.

## Minimal-change design proposal — implemented opt-in extension

### Configuration

Add an opt-in `confidence_gated_selector` flag defaulting to `False`. Add configurable values using the repository’s existing selector defaults as the starting point: 50 warm-up cycles, the existing initial calibration threshold of 0.9, and the existing agreement window of 30 cycles. Add a minimum of five observations per path before comparing rolling rates.

The default `disable_blended_scorer=True` behavior remains unchanged.

### Selection behavior

When the explicit flag is false, preserve the current selector exactly.

When true for discrete selection:

1. During the first 50 cycles, return the existing geometric action and record `confidence_gated_warmup`.
2. After warm-up, obtain G′ confidence using the existing prediction output through a public prediction-engine method.
3. If confidence is below threshold, return geometry and record `confidence_gated_geometry`.
4. If confidence meets threshold, run the existing prediction-scored candidate path and record `confidence_gated_prediction`.
5. Maintain realized `goal_reached` outcomes separately for prediction-gated and geometric decisions. Once each path has five observations, if the prediction path’s rolling rate is lower, enter a geometry fallback state and record `confidence_gated_fallback` plus a warning.

The geometric planner and RBTA implementation remain untouched. G′ architecture and training targets remain untouched. G′ continues learning during warm-up because the selector branch occurs before `env.step()` while learning remains in `_run_learning_phase()`.

Implementation status: completed behind `--confidence-gated`; default behavior remains unchanged. The valid 30-seed × 200-cycle 5×5 validation run completed; the planned 10×10 run did not produce a result artifact and remains open.

### Files

- `python/phca/evaluation/interventions.py`: new opt-in configuration and parsing.
- `python/phca/prediction/engine.py`: public confidence convenience method delegating to the existing prediction contract.
- `python/phca/core/cycle.py`: opt-in gating state, rationale, rolling outcome tracking, and branch selection.
- `scripts/phca_causal_eval.py`: explicit CLI flag and configuration wiring.
- `scripts/benchmark.py`: explicit CLI flag if the benchmark entrypoint is used for validation.
- Existing action-selection tests, or a focused new test file if required.

### Acceptance tests

- Default configuration still returns `pure_geometry_ablation` and makes no prediction calls during default discrete selection.
- Opt-in warm-up returns geometry for 50 cycles while G′ learning timing/update state remains active.
- Below-threshold confidence returns geometry.
- Above-threshold confidence reaches the prediction-scored path.
- Rolling prediction success below rolling geometry success activates fallback and emits rationale/log evidence.
- A short integration run records the selector mode and reason every cycle.

## Open experimental questions

- Whether scalar G′ confidence predicts action usefulness.
- Whether the confidence-gated path improves goal rate or distance efficiency.
- Whether five observations per path is adequate for a safety fallback.
- Whether fallback should use `goal_reached`, reward, or distance-to-goal for future revisions.
