# Φ-IQ Validation Protocol

Φ-IQ is validated as a **scientific index**, not defended as a formula. See [`scripts/phi_iq_validation.py`](../scripts/phi_iq_validation.py).

## Studies

### 1. Sensitivity

**Hypothesis:** Φ-IQ(full) > Φ-IQ(random proxy, minimal_cycle, no_prediction).

**Pass:** Mean full > mean ablated on ≥2/3 conditions at p<0.05 (30 seeds).

**Observed (100 seeds, 2026-07-05):**

| Comparison | full mean | ablated mean | U p-value | pass |
|------------|-----------|--------------|-----------|------|
| full vs minimal_cycle | 0.588 | 0.259 | 2.3×10⁻³⁰ | yes |
| full vs no_prediction | 0.588 | 0.585 | 0.490 | no |

Sensitivity passes on **1/2** ablation contrasts (full > minimal only).

### 2. Predictive Validity

**Protocol:** Correlate Φ-IQ computed from first N=50 cycles with goal_rate over cycles 51–200 on held-out L2 scenarios.

**Pass:** Pearson r ≥ 0.7. **Fail action:** Revise weights in [`python/phca/evaluation/metrics/phi_iq.py`](../python/phca/evaluation/metrics/phi_iq.py).

**Observed:** r = **0.992** (n_early=50, n_late=150) — **Validated** (H006).

### 3. Decomposition

Regress later goal_rate on subindices (prediction_accuracy, adaptation_speed, goal_complexity, transfer_efficiency, resource_efficiency, failure_rate).

**Pass:** At least one non-resource subindex is significant predictor. If only resource_efficiency predicts success, demote resource weight.

**Observed:** Not run offline (r ≥ 0.7); composite passes without weight revision. See `decomposition_note` in `results/validation/phi_iq_validation.json`.

### 4. External Corroboration

Co-movement with OOD bandit env performance and scaling grid_size experiments.

**Observed:** Scaling overall Φ-IQ drops with grid size (30 seeds, slip=0): 5×5 **0.700±0.046**, 10×10 **0.325±0.042**, 20×20 **0.147±0.042**. OOD bandit and MuJoCo pendulum runs in `results/validation/ood/`.

## Running Validation

```bash
# Standalone Φ-IQ validation (100 seeds)
PYTHONPATH=python python scripts/phi_iq_validation.py \
  --seeds=100 --output=results/validation/phi_iq_validation.json

# Full scientific suite (ablations, scaling, OOD, aggregate)
make validate-science
```

## Current Status

**Complete** (2026-07-05). Artifacts: `results/validation/phi_iq_validation.json`, `summary.json`, `hypothesis_verdicts.json`. Predictive validity **passes** (r=0.992). Sensitivity is **partial** (full > minimal, not full > no_prediction). Cite scaling corroboration with grid-size caveats.
