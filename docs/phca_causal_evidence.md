# PHCA Causal Evidence Gate

This gate checks whether adding the PHCA cognitive cycle improves GridWorld
agent behavior, not whether the runtime can execute without crashing.

## Honesty (default discrete path)

Under the current default (`InterventionConfig.disable_blended_scorer=True`),
PHCA action selection is **geometry-dominated** (`pure_geometry_ablation`).
Scenario PASS/FAIL therefore measures **planner competence vs baselines**, not
prediction-primary control (D-156/D-161/D-195/D-197).

Gate JSON dual-reports model fit without changing `gate.passed`:

- `prediction_error_mean`
- `secondary_prediction` (`gated: false`) — PHCA-only PE; do not treat scenario
  PASS as proof that G′ chose actions

**30-seed overnight SoT (D-197):** `logs/overnight_20260802_103502/` —
5×5 L2 **FAIL**, 10×10 L2 **PASS**, L3 **FAIL** (both grids) under geometry;
blended opt-in **hurts** L2 at power; H2 hard (learn-off ≡ geometry on scenario
metrics; PE diverges). See
[`investigations/overnight_analysis_2026-08-02.md`](investigations/overnight_analysis_2026-08-02.md).

Shorter diagnostic ablations:
[`investigations/causal_diagnosis_g2inv05_2026-08-02.md`](investigations/causal_diagnosis_g2inv05_2026-08-02.md)
and `logs/diagnosis_causal_g2inv05_*.json`. Continuous MPC remains prediction-primary.

## Benchmark Shape

The core gate has three scenario levels plus three viewport (partial-observability) levels:

| Level | Scenario | Expected control behavior |
|-------|----------|---------------------------|
| `level1` | Simple goal navigation | Full-information greedy should be the ceiling |
| `level2` | Noisy/delayed observation, partial wall map, dynamic obstacles | Fair `greedy_observed` is the main non-PHCA control |
| `level3` | Long horizon with goal switching, interruption/occlusion windows, partial map | PHCA must show recovery/adaptation, not just navigation |
| `viewport1` | GridWorld with `partial_obs_radius=1` (3×3 viewport) | `greedy_observed` sees same restricted goal/walls as PHCA |
| `viewport2` | GridWorld with `partial_obs_radius=2` (5×5 viewport) | Same control structure |
| `viewport3` | GridWorld with `partial_obs_radius=3` (7×7 viewport) | Same control structure |

The evaluator compares:

| Agent | Description | PHCA state? |
|-------|-------------|-------------|
| `phca` | `CognitiveCycle.build(...)` with prediction, learning, action selection, MDIM/APC, M3/consolidation, and RBTA | Yes |
| `random` | Uniform random GridWorld action | No |
| `greedy_observed` | One-step Manhattan-distance controller using the same observed goal and wall map exposed by the scenario wrapper. Under `partial_obs_radius` (viewport levels), `env.get_goal_position()` returns `None` when the goal is outside the viewport, and `env.grid` comes from `env.observed_grid` (only known walls). | No |
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

Default gated controls:

| Level | Gated controls |
|-------|----------------|
| `level1` | `random` |
| `level2` | `random`, `greedy_observed` |
| `level3` | `random`, `greedy_observed` |
| `viewport1` | `random`, `greedy_observed` |
| `viewport2` | `random`, `greedy_observed` |
| `viewport3` | `random`, `greedy_observed` |

This rule is deliberately not tuned to force a pass.

## Current Measurement — overnight SoT (D-197)

Prefer `logs/overnight_20260802_103502/causal_*_l2l3_geometry.json` (geometry default)
and blended counterparts. Summary: 5×5 L2 FAIL, 10×10 L2 PASS, L3 FAIL both grids;
blended hurts L2. Full tables:
[`investigations/overnight_analysis_2026-08-02.md`](investigations/overnight_analysis_2026-08-02.md).

## Historical Measurement (2026-07-12; geometry era; keep for archive)

Re-run after all 4 rounds of fixes (MLP, 200 cycles × 30 seeds, D-151). Treat as
historical if overnight numbers differ; L3 FAIL here already matches overnight pattern:

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 30 --use-mlp --gate \
  --output results/validation/baselines/causal_eval_round4.json
```

### Results

| Level | Result | PHCA vs greedy_observed |
|-------|--------|-------------------------|
| `level1` | **PASS** | PHCA beats random on 4/4; beats `greedy_observed` on ≥75% metrics. Mean goal_rate: phca **0.855** vs greedy 0.793. |
| `level2` | **FAIL** | PHCA beats random on 4/4; beats `greedy_observed` on only **1/3** metrics. Mean goal_rate: phca **0.558** vs greedy **0.598**. Mean distance: phca 1.033 vs greedy 0.884. |
| `level3` | **FAIL** | PHCA beats `greedy_observed` on only **2/6** metrics. Mean goal_rate: phca **0.290** vs greedy **0.291**. Mean distance: phca 2.197 vs greedy 1.981. |

Selected means (MLP, 200 cycles × 30 seeds):

| Level | Agent | Goal rate | First goal | Mean distance | Reward |
|-------|-------|-----------|------------|---------------|--------|
| `level1` | `phca` | 0.855 | 2.7 | 0.299 | 170.61 |
| `level1` | `greedy_observed` | 0.793 | 1.9 | 0.487 | 158.09 |
| `level2` | `phca` | 0.558 | 7.5 | 1.033 | 110.75 |
| `level2` | `greedy_observed` | 0.598 | 4.1 | 0.884 | 118.76 |
| `level3` | `phca` | 0.290 | 24.5 | 2.197 | 56.55 |
| `level3` | `greedy_observed` | 0.291 | 19.5 | 1.981 | 56.68 |

> **Note:** Earlier 5-seed measurements (below) showed PASS at both levels, but this was a statistical false positive caused by underpowered sampling (D-151). The 30-seed run is authoritative.

## Viewport Results (2026-07-13, updated 2026-07-18)

Results after D-160 partial-observability infrastructure + RBTA entropy-floor fix +
frontier-based exploration (MLP, 200 cycles × 30 seeds × 10×10 grid):

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels viewport1,viewport2,viewport3 \
  --cycles 200 --seeds 30 --grid-size 10 --use-mlp
```

| Level | Viewport | Result | PHCA vs greedy_observed |
|-------|----------|--------|-------------------------|
| `viewport1` | 3×3 (`partial_obs_radius=1`) | **PASS** | PHCA goal_rate **0.355** vs greedy_observed **0.133**. PHCA RBTA violation rate: **0.009**. |
| `viewport2` | 5×5 (`partial_obs_radius=2`) | **PASS** | PHCA goal_rate **0.664** vs greedy_observed **0.199**. PHCA RBTA: **0.000**. |
| `viewport3` | 7×7 (`partial_obs_radius=3`) | **PASS** | PHCA goal_rate **0.722** vs greedy_observed **0.391**. PHCA RBTA: **0.000**. |

All three viewport levels confirmed PASS at the project's 30-seed statistical standard
(D-151). RBTA violations are negligible (< 0.01) — the D-160 bound-widening fix
(viewport-proportional RBTA scaling) is sufficient. NEW-14 is resolved.

> **Exploratory 15-seed means (historical; not the 30-seed PASS table above).**
> Do not mix with the 30-seed gate rows. Prefer the PASS table’s goal_rate / RBTA
> (0.355 / 0.664 / 0.722; RBTA ≈ 0.009 / 0 / 0).

| Level | Agent | Goal rate | First goal (med) | Succeeded | RBTA rate |
|-------|-------|-----------|-------------------|-----------|-----------|
| `viewport1` | `phca` | 0.329 | 36 | 7/15 | 0.116 |
| `viewport1` | `greedy_observed` | 0.133 | — | 2/15 | — |
| `viewport1` | `greedy_full_info` | 0.788 | 7 | 13/15 | — |
| `viewport2` | `phca` | 0.351 | 34 | 8/15 | 0.092 |
| `viewport2` | `greedy_observed` | 0.199 | — | 3/15 | — |
| `viewport2` | `greedy_full_info` | 0.788 | 7 | 13/15 | — |
| `viewport3` | `phca` | 0.653 | **17** | 12/15 | 0.101 |
| `viewport3` | `greedy_observed` | 0.391 | — | 6/15 | — |
| `viewport3` | `greedy_full_info` | 0.788 | 7 | 13/15 | — |

`greedy_full_info` uses `env.true_*` accessors (bypasses partial obs) and remains the
unfair ceiling. Under tight viewports (`viewport1`), `greedy_observed` collapses to
13% goal rate because it cannot see far enough to plan a Manhattan route — the core
`observed_grid` delegation correctly imposes the same information constraint on the
baseline.

### RBTA entropy-floor fix

The original RBTA scaling in `CognitiveCycle.build()` applied `_scale = 1.0 + (10 - radius) * 0.1`
to all bounds (B_time, B_mem, B_energy, entropy_floor). Two problems existed:

1. **entropy_floor was multiplied by scale** (raising it, making it stricter), but under partial
   obs G' MC-dropout entropy *drops* because the model becomes confidently wrong with limited
   data. The floor must be **divided** by scale (more lenient). Fixed in eval script.

2. **`gprime_stress_bounds()` overwrote G' bounds** after `build()` completed, undoing the
   scaling. Fixed by applying scaling *after* the stress-bounds update in `run_phca_agent()`.

Before fix: RBTA 57–96% per seed at viewport2 (mean ~60%). After fix: 0–48%, mean 9–12%.

### Frontier-based exploration

When the goal has never been observed, `_compute_distance_gain()` previously returned
0.5 (neutral) for all actions — the agent had no directional guidance. Added `_find_frontier_cell()`
in cycle.py which scans `_observed_cells` (added to GridWorld alongside `_known_walls`) for
the nearest unobserved cell and uses it as a proxy goal. The distance-gain formula then rewards
actions that reduce Manhattan distance to the frontier. This gave:

- viewport2: 5/15 → 8/15 successful seeds, goal_rate 0.269 → 0.351
- viewport3: 8/15 → 12/15 successful seeds, goal_rate 0.489 → 0.653

Frontier exploration is a simple Manhattan-distance heuristic; a coverage-maximization approach
(number of new cells revealed per action) could improve exploration further.

## Interpretation

Supported claims:

- PHCA improves measured GridWorld behavior relative to random controls across all three levels.
- The benchmark exposes PHCA architecture signals (episodic storage, consolidation facts, prediction error, RBTA violations, occlusion handling, dynamic goals, recovery after switches).

Unsupported claims:

- **PHCA does NOT currently beat `greedy_observed` (a simple one-step Manhattan heuristic) at adequate statistical power in L2 (obstacle navigation) or L3 (self-motivated exploration).**
- PHCA does not beat a full-information greedy controller.

### Why this matters (D-151 / D-197)

After removing the task_lock bypass and at adequate seed power, PHCA under **default
geometry** still fails the fair gate on 5×5 L2 and on L3 (both grids); 10×10 L2 PASS
is planner-vs-greedy, not prediction-primary. Overnight H2: learn-off ≡ geometry on
scenario metrics — G′ is off the action path under default. Root causes to keep in mind:

1. **Default action path is geometry** — scenario metrics largely measure planner competence (D-197).
2. **10×10 is already tested** — L2 PASS / L3 FAIL at 30×200; do not re-pilot “run 10×10” as if undone.
3. **Blended opt-in is not a free win** — at 30×200 it hurts L2 vs geometry (D-197); no default flip.

### Recommendation (status)

**Done:** L2/L3 at 10×10 were run at 30 seeds (D-156/D-161). Outcome is **mixed**:
10×10 L2 pure-geometry PASS vs `greedy_observed`; 5×5 L2 and many L3 configs still FAIL;
viewport scenarios PASS. Neither pure geometry nor blended wins universally.
Further work should focus on prediction-quality under opt-in blended scoring and
uncoupling goal_rate from planner confounds — not repeating the "run 10×10" pilot.

## Prior Measurement (5 seeds, historical)

Command (Gaussian G', 5 seeds, pre-round-4):

```bash
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 5 --output logs/phca_causal_eval.json
```

Measured on 2026-07-04 (post P0 gap-closure + L3 coverage fix, D-112) — **underpowered, superseded by 30-seed run above**:

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

