# Φ-IQ Validation Protocol

Φ-IQ is validated as a **scientific index**, not defended as a formula. See [`scripts/phi_iq_validation.py`](../scripts/phi_iq_validation.py).

## Studies

### 1. Sensitivity

**Hypothesis:** Φ-IQ(full) > Φ-IQ(random proxy, minimal_cycle, no_prediction).

**Pass:** Mean full > mean ablated on ≥2/3 conditions at p<0.05 (30 seeds).

### 2. Predictive Validity

**Protocol:** Correlate Φ-IQ computed from first N=50 cycles with goal_rate over cycles 51–200 on held-out L2 scenarios.

**Pass:** Pearson r ≥ 0.7. **Fail action:** Revise weights in [`python/phca/evaluation/metrics/phi_iq.py`](../python/phca/evaluation/metrics/phi_iq.py).

### 3. Decomposition

Regress later goal_rate on subindices (prediction_accuracy, adaptation_speed, goal_complexity, transfer_efficiency, resource_efficiency, failure_rate).

**Pass:** At least one non-resource subindex is significant predictor. If only resource_efficiency predicts success, demote resource weight.

### 4. External Corroboration

Co-movement with OOD bandit env performance and scaling grid_size experiments (once run).

## Running Validation

```bash
PYTHONPATH=python python scripts/phi_iq_validation.py --seeds=30 --output=logs/phi_iq_validation.json
```

## Current Status

Run validation locally to populate `logs/phi_iq_validation.json`. Do not cite Φ-IQ as intelligence index until sensitivity + predictive validity pass.
