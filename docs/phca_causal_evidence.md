# PHCA Causal Evidence Gate

This gate checks whether adding the PHCA cognitive cycle improves GridWorld
agent behavior, not whether the runtime can execute without crashing.

## Benchmark Shape

The gate has three scenario levels:

| Level | Scenario | Expected control behavior |
|-------|----------|---------------------------|
| `level1` | Simple goal navigation | Full-information greedy should be the ceiling |
| `level2` | Noisy/delayed observation, partial wall map, dynamic obstacles | Fair `greedy_observed` is the main non-PHCA control |
| `level3` | Long horizon with goal switching, interruption/occlusion windows, partial map | PHCA must show recovery/adaptation, not just navigation |

The evaluator compares:

| Agent | Description | PHCA state? |
|-------|-------------|-------------|
| `phca` | `CognitiveCycle.build(...)` with prediction, learning, action selection, MDIM/APC, M3/consolidation, and RBTA | Yes |
| `random` | Uniform random GridWorld action | No |
| `greedy_observed` | One-step Manhattan-distance controller using the same delayed/partial observed goal and wall map exposed by the scenario wrapper | No |
| `greedy_full_info` | One-step Manhattan-distance controller using true agent/goal/wall state | No |

`greedy_full_info` is reported as a sanity ceiling. It is not gated by default.

## Metrics

Base metrics:

| Metric | Direction | Meaning |
|--------|-----------|---------|
| `goal_rate` | Higher is better | Fraction of cycles on which the goal was reached |
| `first_goal_cycle` | Lower is better | First cycle that reached the goal; missing counts as `cycles + 1` |
| `mean_distance_to_goal` | Lower is better | Mean true Manhattan distance after each action |
| `cumulative_reward` | Higher is better | Sum of GridWorld rewards |

Long-horizon metrics:

| Metric | Direction | Meaning |
|--------|-----------|---------|
| `coverage_rate` | Higher is better | Fraction of grid cells visited |
| `switch_recovery_cycle` | Lower is better | Mean cycles from goal switch to next goal reach |

`rbta_violation_rate`, `prediction_error_mean`, `episode_count`, and
`fact_count` are reported as PHCA health/architecture signals, not counted as
extrinsic wins unless explicitly added to a future gate.

## Gate Rule

PHCA must beat each scenario-gated control on at least 75% of that scenario's
metrics.

**L3 effective bar:** With 6 long-horizon metrics, `ceil(6 × 0.75) = 5` wins
are required. When `first_goal_cycle` ties `greedy_observed` (common at mean
5.6), that metric gives no credit — PHCA must then win **all five** remaining
metrics (`goal_rate`, `mean_distance_to_goal`, `cumulative_reward`,
`coverage_rate`, `switch_recovery_cycle`). A single loss fails the gate.

Nightly CI (D-133) uses `--seeds 5` for level2 and `--level-seeds level3=10`
because L2 aggregate metrics are stable at 5 seeds while L3
`mean_distance_to_goal` needs 10 seeds to clear runner variance.

Default gated controls:

| Level | Gated controls |
|-------|----------------|
| `level1` | `random` |
| `level2` | `random`, `greedy_observed` |
| `level3` | `random`, `greedy_observed` |

This rule is deliberately not tuned to force a pass.

## Current Measurement

Command (Gaussian G', default):

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 5 --output logs/phca_causal_eval.json
```

**Recommended deployment mode** (MLP world model, matches Φ-IQ benchmark):

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 5 --use-mlp --gate \
  --output logs/phca_causal_eval_mlp.json
```

Measured on 2026-07-04 (post P0 gap-closure + L3 coverage fix, D-112):

| Level | Result | Gate detail |
|-------|--------|-------------|
| `level1` | PASS | PHCA beats random on 4/4 metrics; greedy_full_info remains stronger |
| `level2` | PASS | PHCA beats random on 4/4; beats `greedy_observed` on 3/4 |
| `level3` | PASS | PHCA beats random on 5/6; beats `greedy_observed` on 5/6 |

Selected means (Gaussian G', 200 cycles × 5 seeds):

| Level | Agent | Goal rate | First goal | Mean distance | Reward | Coverage | Recovery |
|-------|-------|-----------|------------|---------------|--------|----------|----------|
| `level2` | `phca` | 0.542 | 5.6 | 0.734 | 107.484 | — | — |
| `level2` | `greedy_observed` | 0.420 | 5.6 | 0.935 | 82.840 | — | — |
| `level3` | `phca` | 0.322 | 5.6 | 1.300 | 63.044 | 0.344 | 32.0 |
| `level3` | `greedy_observed` | 0.287 | 5.6 | 1.537 | 55.974 | 0.336 | 55.9 |

Prior measurement (2026-07-04 pre-L3-coverage fix):

| Level | Result | PHCA vs greedy_observed |
|-------|--------|-------------------------|
| `level2` | PASS | 3/4 metrics |
| `level3` | FAIL | 4/6 metrics (`coverage_rate` short) |

## Interpretation

Supported claims:

- PHCA improves measured GridWorld behavior relative to random controls across
  all three levels.
- The benchmark now exposes PHCA architecture signals such as episodic storage,
  consolidation facts, prediction error, RBTA violations, occlusion handling,
  dynamic goals, and recovery after switches.

Unsupported claims:

- PHCA currently beats a full-information greedy controller.

This is a better benchmark shape. Levels 2–3 now pass versus `greedy_observed`
after P0 cycle reorder, task-lock, observed-greedy navigation, and sparse L3
coverage probes (`cycle_count % 50 == 0` or `_goal_switch_cooldown >= 14` on-goal
unvisited steps, D-112/D-133).

