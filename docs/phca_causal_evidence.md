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

Default gated controls:

| Level | Gated controls |
|-------|----------------|
| `level1` | `random` |
| `level2` | `random`, `greedy_observed` |
| `level3` | `random`, `greedy_observed` |

This rule is deliberately not tuned to force a pass.

## Current Measurement

Command:

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 5 --output .tmp/phca_causal_eval_levels_200x5.json
```

Measured on 2026-07-04:

| Level | Result | Gate detail |
|-------|--------|-------------|
| `level1` | PASS | PHCA beats random on 4/4 metrics; greedy_full_info remains stronger |
| `level2` | FAIL | PHCA beats random on 4/4, but beats `greedy_observed` on 0/4 |
| `level3` | FAIL | PHCA beats random on 5/6, but beats `greedy_observed` on 1/6 |

Selected means:

| Level | Agent | Goal rate | First goal | Mean distance | Reward | Coverage | Recovery |
|-------|-------|-----------|------------|---------------|--------|----------|----------|
| `level1` | `phca` | 0.941 | 1.8 | 0.069 | 188.082 | — | — |
| `level1` | `random` | 0.075 | 19.8 | 2.327 | 13.150 | — | — |
| `level1` | `greedy_full_info` | 0.991 | 1.8 | 0.016 | 198.182 | — | — |
| `level2` | `phca` | 0.400 | 7.2 | 1.246 | 78.800 | — | — |
| `level2` | `greedy_observed` | 0.420 | 5.6 | 0.935 | 82.840 | — | — |
| `level3` | `phca` | 0.168 | 7.2 | 2.256 | 31.936 | 0.456 | 71.6 |
| `level3` | `greedy_observed` | 0.287 | 5.6 | 1.537 | 55.974 | 0.336 | 55.9 |
| `level3` | `random` | 0.050 | 19.6 | 3.208 | 8.100 | 0.632 | 128.1 |

## Interpretation

Supported claims:

- PHCA improves measured GridWorld behavior relative to random controls across
  all three levels.
- The benchmark now exposes PHCA architecture signals such as episodic storage,
  consolidation facts, prediction error, RBTA violations, occlusion handling,
  dynamic goals, and recovery after switches.

Unsupported claims:

- PHCA currently beats a fair observed greedy controller in constrained or
  long-horizon GridWorld.
- PHCA currently beats a full-information greedy controller.

This is a better benchmark shape, but it also reveals the next engineering gap:
PHCA's current discrete GridWorld policy is not yet exploiting memory,
consolidation, or prediction strongly enough to beat a simple observed greedy
controller under Level 2/3 constraints.

