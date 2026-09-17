# PHCA Repair Baseline — 2026-09-17

This is the execution baseline for the repair plan. It records static facts and
reproducibility checks before the next change. It does not claim that any
runtime behavior has been experimentally established beyond the referenced
artifacts.

## Repository state

- HEAD at baseline: `1e94291` (`PHCA-Maturation-Hardening-finish-round-69`).
- The action-selection audit, corrected-state documentation, and action-selection
  specification are present: `AUDIT_ACTION_SELECTION_2026-09-16.md`,
  `docs/CURRENT_STATE.md`, and `SPEC_ACTION_SELECTION.md`.
- The default discrete GridWorld selector is documented as pure geometry;
  confidence-gated selection is opt-in. See `README.md:64-114` and
  `docs/CURRENT_STATE.md:1-27`.
- The cycle owns the final discrete/continuous action selection and exposes the
  latest rationale and candidate scores to observability. See
  `python/phca/core/cycle.py:1932-2468` and
  `python/phca/monitoring/observability.py:238-262,345-449`.
- The opt-in confidence-gated branch enforces warmup before legacy exploration,
  continues through the normal learning phase, and records rolling fallback
  state. See `python/phca/core/cycle.py:1991-2039,881-1092,1794-1846`.
- The science trace schema records action and scalar cycle metrics but not the
  selector mode, rationale, or per-action scores. See
  `python/phca/evaluation/trace.py:11-76` and
  `python/phca/core/cycle.py:581-603`.

## Static test inventory

`PYTHONPATH=python pytest --collect-only -q` collected 933 tests on this
baseline. The existing confidence-gated tests cover warmup, threshold routing,
prediction routing, rolling fallback, learning during warmup, and live
observability. See `python/phca/core/tests/test_confidence_gated_selection.py:1-181`.

## Existing evidence and limits

The prior validation artifact contains valid 5×5, 30-seed, 200-cycle default
and gated runs, while the corrected 10×10 run remains unvalidated. See
`docs/action_selection_validation_2026-09-16.md:1-33`.

The causal evaluator reports selector-mode counts and prediction error, but its
gate remains a scenario-metric comparison against controls; prediction error
is explicitly secondary and not gated. See `scripts/phca_causal_eval.py:468-572`.

## Baseline acceptance

This baseline is accepted when the file is present, the test collection count
is recorded, `git diff --check` is clean, and no source code has been changed
by the baseline step. Subsequent phases may add narrowly scoped changes and
must update this document or a linked evidence document when the contract
changes.
