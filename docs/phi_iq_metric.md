# Φ-IQ Metric — Benchmarking & Evaluation

Φ-IQ (Phi-Intelligence Quotient) is a **composite task-performance heuristic** for
**PHCA v3.0** on scripted GridWorld benchmarks. It combines five independent
sub-metrics into a single score between 0.0 and 1.0.

> **What Φ-IQ is not:** It is not psychometric IQ, general intelligence, or a
> cross-domain capability score. It measures performance on four fixed GridWorld
> scenarios under specific world-model and seed settings.

---

## Definition

```
Φ-IQ = 0.25 × PredictionAccuracy
     + 0.25 × AdaptationSpeed
     + 0.20 × GoalComplexity
     + 0.20 × ResourceEfficiency
     - 0.10 × FailureRate
```

| Sub-Metric | Weight | What It Measures |
|---|---|---|
| PredictionAccuracy | 0.25 | Inverse of normalised mean prediction error |
| AdaptationSpeed | 0.25 | Early vs late error/goals improvement |
| GoalComplexity | 0.20 | L2 goal rate; L0/L3 drive diversity; L1 `unique_actions / action_space_size` |
| ResourceEfficiency | 0.20 | `1 − mean_latency_ms / 500` |
| FailureRate | 0.10 | RBTA violations per cycle (subtracted) |

### TransferEfficiency note (removed from Φ-IQ 2026-07-11)

`TransferEfficiency` was removed from the Φ-IQ composite per **D-147** (NEW-03):
it is a derived metric (`adaptation_speed × prediction_accuracy`), not an
independent measurement — its 0.15 weight double-counted PA and AS. The weight
was redistributed: PA +0.05, AS +0.05, GC +0.05.

`transfer_efficiency` is still computed and stored on `BenchmarkResult` for
diagnostics, but does not contribute to the Φ-IQ score. Use **Level-4-lite**
for cross-task forgetting measurement:

```bash
PYTHONPATH=python python scripts/benchmark_level4.py \
  --tasks 10 --task-cycles 80 --eval-cycles 20 --seeds 3 --use-mlp \
  --output logs/benchmark_level4.json
```

Gate: `forgetting_rate < 0.05` (max relative drop per task). Implemented in
[`python/phca/evaluation/metrics/forgetting.py`](../python/phca/evaluation/metrics/forgetting.py).

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

### Level 4-lite — Continual learning (forgetting rate)

Sequential GridWorld layouts (`build_task_sequence`) trained one after another.

**Protocol (2026-07-07 maturation):**

1. **Train:** `task_cycles` per task; `on_task_boundary(task_id)` when mitigation enabled.
2. **Baseline:** default `max_rolling` goal rate over window (not last-20 only); tasks with baseline < 0.2 excluded from Δ.
3. **Eval:** `apply_task_layout` → optional `--eval-warmup` (not scored) → `--eval-cycles` scored.
4. **Metric:** `forgetting_rate` = max drop only (negative Δ_perf); gate < 0.05.
5. **Diagnostic:** `--diagnostic` logs train curves, M3 counts, replay totals, B4 counts.
6. **Ablation:** `make bench-level4-ablation` (R0–R6 matrix).

Anti-forgetting: P-Stream `protect_parameters`, M3→G′ replay (`sample_prior_task_episodes`), G′ `replay_boost` on B4. No EWC/GEM (D-020).

Verdict doc: [`docs/l4_root_cause_verdict.md`](l4_root_cause_verdict.md).

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
| Overall Φ-IQ | > 0.5 | **0.7718** ✅ |
| Level 2 Φ-IQ | ≥ 0.5 | **0.7924** ✅ |

Source: [STATUS.md](../STATUS.md), canonical `logs/benchmark_report.json`.

### Scaling grids (10×10, 20×20)

Larger grids increase `state_dim`; RBTA bounds are scaled via `grid_rbta_bounds()`
in [`python/phca/world_model/mlp.py`](../python/phca/world_model/mlp.py). Even so,
validation scaling runs (`results/validation/scaling/`) show lower Φ-IQ and the
violation gate may still fail — treat scaling as exploratory, not CI-gated.
After full module RBTA bound alignment (`grid_rbta_bounds`), 10×10 Gaussian L2
can pass the violation gate (`failure_rate_under_10pct`).

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

**Date:** 2026-07-11  
**Configuration:** MLP (hidden 128, ~38,868 params), 50 cycles/level, seed 42, 5×5 GridWorld (Φ-IQ formula updated D-147)

```
Overall Φ-IQ: 0.7718

Level  Φ-IQ
────────────────
L0     0.7886
L1     0.7837
L2     0.8527
L3     0.6624
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
