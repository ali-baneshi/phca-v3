# Φ-IQ Metric — Benchmarking & Evaluation

Φ-IQ (Phi-Intelligence Quotient) is a **composite task-performance heuristic** for
**PHCA v3.0** on scripted GridWorld benchmarks. It combines six sub-metrics into a
single score between 0.0 and 1.0.

> **What Φ-IQ is not:** It is not psychometric IQ, general intelligence, or a
> cross-domain capability score. It measures performance on four fixed GridWorld
> scenarios under specific world-model and seed settings.

---

## Definition

```
Φ-IQ = 0.20 × PredictionAccuracy
     + 0.20 × AdaptationSpeed
     + 0.15 × GoalComplexity
     + 0.15 × TransferEfficiency
     + 0.20 × ResourceEfficiency
     - 0.10 × FailureRate
```

| Sub-Metric | Weight | What It Measures |
|---|---|---|
| PredictionAccuracy | 0.20 | Inverse of normalised mean prediction error |
| AdaptationSpeed | 0.20 | Early vs late error/goals improvement |
| GoalComplexity | 0.15 | L2 goal rate; L0/L3 drive diversity; L1 `unique_actions / action_space_size` |
| TransferEfficiency | 0.15 | **`adaptation × prediction` proxy** — not cross-task transfer |
| ResourceEfficiency | 0.20 | `1 − mean_latency_ms / 500` |
| FailureRate | 0.10 | RBTA violations per cycle (subtracted) |

### TransferEfficiency disclaimer

`TransferEfficiency` is computed as `adaptation_speed × prediction_accuracy` in
[`scripts/benchmark.py`](../scripts/benchmark.py). It does **not** measure
retention across tasks or environments. A dedicated cross-task transfer benchmark
is backlog (see [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)).

---

## Benchmark Levels

### Level 0 — Stationary Prediction

Static environment; world model learns identity-like dynamics.

### Level 1 — Reactive Control

Active control; prediction under action. **Goal complexity** =
`unique_actions / env.action_space_size` (GridWorld: 5 actions including STAY).

### Level 2 — Goal Pursuit

5×5 maze with walls; goal reaching rate is the primary signal.

### Level 3 — Self-Motivated Exploration

No external goal. **Adaptation speed** = action diversity /
`env.action_space_size`. **Goal complexity** from MDIM active drives (/5), or
action diversity if MDIM is disabled.

### Emergence trace metrics

Operational emergence (`emergence.py`) uses cycle traces. Key semantics:

- `goal_switched`: intrinsic drive id changed (MDIM)
- `env_goal_relocated`: `GridWorld.relocate_goal()` before this cycle
- `cross_context_reuse`: action-profile similarity around either switch type
- `behavioral_compression`: deterministic zlib baseline (seeded from action bytes)

---

## How to Run Benchmarks

```bash
# Quick smoke (L0, 20 cycles, Gaussian) — matches CI gate
python scripts/benchmark.py --quick

# Canonical benchmark (L0–L3, MLP, 200 cycles, seed 42, 5×5) — full pass criteria
MUJOCO_GL=disabled python scripts/benchmark.py --use-mlp --cycles=200 --grid-size 5

# Scaling / exploratory (10×10 — Φ-IQ may be lower; violation gate often fails)
python scripts/benchmark.py --grid-size 10 --cycles=200 --use-mlp

# Multi-seed report (mean ± std across seeds 42..46)
python scripts/benchmark.py --use-mlp --cycles=200 --seeds=5

# Regression gate (after a canonical run)
python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
```

Scripts bootstrap `python/` automatically (`scripts/_bootstrap.py`). Legacy:
`PYTHONPATH=python python scripts/benchmark.py ...`

MLP mode needs **≥ 200 cycles per level** to stabilise. See
[docs/reproducibility.md](reproducibility.md) for pinned environment details.

---

## Interpreting Results

### Fail column vs pass gate

The report **Fail** column is the Φ-IQ sub-metric `failure_rate = violations / n_cycles`
per level. It is **not** a percentage:

| Fail value | Meaning |
|---|---|
| `0.00` | No RBTA violations |
| `0.10` | 10% of cycles had ≥1 violation (pass gate threshold) |
| `1.00` | ~100% — at least one violation every cycle on average |
| `1.26` | ~126% — more than one violation per cycle on average |

The pass criterion `failure_rate_under_10pct` is **conjunctive**: all four criteria
must pass for Overall PASS. A run can have Φ-IQ > 0.5 but still **Overall FAIL**
if RBTA violations exceed 10% of total cycles.

### Pass Criteria (canonical 5×5 MLP, 200 cycles)

| Pass Criteria | Target | Canonical (5×5 MLP) |
|---|---|---|
| Cycle latency | < 500 ms mean | ~17 ms mean, ~31 ms p95 ✅ |
| Failure rate | < 10% violations | **0 violations** ✅ |
| Goal autonomy (L3) | Drive diversity > 0.1 | Achieved ✅ |
| Overall Φ-IQ | > 0.5 | **0.7323** ✅ |
| Level 2 Φ-IQ | ≥ 0.5 | **0.7924** ✅ |

Source: [STATUS.md](../STATUS.md), canonical `logs/benchmark_report.json`.

### Scaling grids (10×10, 20×20)

Larger grids increase `state_dim`; RBTA bounds are scaled via `grid_rbta_bounds()`
in [`python/phca/world_model/mlp.py`](../python/phca/world_model/mlp.py). Even so,
validation scaling runs (`results/validation/scaling/`) show lower Φ-IQ and the
violation gate may still fail — treat scaling as exploratory, not CI-gated.

| Grid | Typical overall Φ-IQ (MLP, 200 cyc) | `failure_rate_under_10pct` |
|---|---|---|
| 5×5 | ~0.73 | PASS |
| 10×10 | ~0.32 (validation mean) | Often FAIL |
| 20×20 | ~0.15 (validation mean) | FAIL |

### Score bands (heuristic — Φ-IQ only, not Overall PASS)

| Φ-IQ Range | Interpretation |
|---|---|
| 0.00 – 0.30 | Near-random on scripted tasks |
| 0.30 – 0.50 | Learning, not converged |
| 0.50 – 0.70 | Strong Φ-IQ on harder configs; may still FAIL RBTA gate |
| 0.70 – 0.90 | Strong on canonical 5×5 GridWorld suite |
| 0.90 – 1.00 | Theoretical ceiling |

---

## Latest Results

**Date:** 2026-07-05  
**Configuration:** MLP (hidden 128, ~38,868 params), 200 cycles/level, seed 42, 5×5 GridWorld

```
Overall Φ-IQ: 0.7323

Level  Φ-IQ
────────────────
L0     0.7750
L1     0.6748
L2     0.7924
L3     0.6868
```

---

## Technical Notes

- **MLP mode:** 3-layer NumPy MLP, hidden_dim=128, MC-Dropout confidence (D-080).
- **Gaussian mode:** pgmpy Bayesian network; faster per cycle, used by CI quick gate.
- **MuJoCo:** separate single-env latency/error reports — no Φ-IQ composite on physics tasks.
- **Causal gate:** behavioral comparison vs baselines — see [phca_causal_evidence.md](phca_causal_evidence.md).

---

## Related

- [reproducibility.md](reproducibility.md) — reproduce numbers on your machine
- [benchmark_artifacts.md](benchmark_artifacts.md) — which log file is canonical
- [phca_causal_evidence.md](phca_causal_evidence.md) — behavioral gate
- [limitations.md](limitations.md) — scope boundaries
- [action_selection.md](action_selection.md) — discrete vs continuous paths
