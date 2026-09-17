# PHCA System Assessment — 2026-09-17

## What it is

PHCA is a numerical, resource-bounded cognitive-cycle prototype for embodied
single-agent environments. Its active loop wires sanitized state, prediction,
MDIM drives, attention, RBTA, action selection, post-step learning, episodic
memory, and consolidation. See `python/phca/core/cycle.py:130-186,386-530`.

The strongest verified prediction-primary behavior is continuous MPC: candidate
actions are scored through G′. Default discrete GridWorld action selection is
instead geometry-first, while G′ still predicts and learns for other signals.
See `docs/CURRENT_STATE.md:5-31` and `python/phca/core/cycle.py:1991-2125,2140-2342`.

## What it is not

It is not a general intelligence, language system, vision system, semantic
grounding system, or completed continual-learning architecture. Those limits are
explicit in `docs/limitations.md:1-26`. It also does not currently establish a
uniform prediction-primary claim for default discrete control; the causal evidence
ledger records the geometry/planner confound and unresolved G2-INV-05.

## Good points

- The cycle has a clear central orchestrator and explicit RBTA enforcement points.
- Default discrete behavior is honestly documented as geometry-dominated rather
  than being presented as learned control.
- G′ exposes scalar confidence and MLP uncertainty, and the opt-in selector has
  warmup, threshold, fallback, rationale, and tests.
- Observatory frames and science traces now expose enough selector metadata for
  mode accounting and audit. See `python/phca/evaluation/trace.py:11-125` and
  `python/phca/monitoring/observability.py:238-262`.
- The evaluation harness compares against random and information-matched greedy
  controls and now labels statistical power. See `scripts/phca_causal_eval.py:542-700`.

## Bad points / risks

- Default discrete performance remains planner-dominated and does not validate
  G′-driven action selection; the 10×10 gated validation remains open.
- The causal gate compares means and retains a descriptive PASS/FAIL result even
  when the run is underpowered; power metadata is advisory until a promotion
  policy consumes it.
- RBTA energy, memory, and several entropy signals are estimates rather than
  direct measurements. See `python/phca/core/cycle.py:3135-3215`.
- The codebase contains a large historical decision and experiment surface, so
  old artifacts can be mistaken for current behavior without the documentation
  trust order in `DOCUMENTATION_MAP.md:1-50`.

## Direction of travel

1. Keep pure geometry as the default until a 30-seed, matched-control experiment
   demonstrates a prediction-driven discrete benefit without unacceptable RBTA or
   latency regressions.
2. Use the new trace contract as a hard evidence requirement for every selector
   experiment; reject results with missing rationale coverage.
3. Separate model competence metrics (prediction error/calibration) from planner
   competence metrics (goal rate/distance), especially in continual-learning
   evaluations.
4. Replace estimated resource signals with measured accounting incrementally,
   starting with the action-selection and G′ paths, while protecting the existing
   MuJoCo smoke gates.
5. Expand partial-observation work only after the current viewport and 10×10
   causal gaps are closed with preregistered seeds and controls.
