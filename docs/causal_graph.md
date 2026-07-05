# PHCA Causal Graph

Grounded in [`python/phca/core/cycle.py`](../python/phca/core/cycle.py) runtime order and [`python/phca/evaluation/interventions.py`](../python/phca/evaluation/interventions.py) intervention points.

Machine-readable edge definitions: [`experiments/causal_graph.yaml`](../experiments/causal_graph.yaml).

## Runtime Pipeline

```
RBTA_carry → ASI → M1/M2 → Prediction → MDIM/APC/Attention → RBTA_preflight
  → Action → PEU/TSPL/Learn/M3 → RBTA_post → Consolidation
```

Desync variant runs **regulation before prediction** (see `DESYNC_STAGE_ORDER`).

## Causal Edges

| Edge | Cause | Effect | Intervention | Predicted change if real |
|------|-------|--------|--------------|--------------------------|
| E001 | PredictionEngine | Action diversity, transfer proxy | `enable_prediction=false` | Higher ε-explore; transfer_efficiency ↓ |
| E002 | MDIM | L3 goal complexity | `enable_mdim=false` | self_generated_goals → 0 |
| E003 | M3+Consolidation | cross_context_reuse | `enable_consolidation=false` | reuse < 0.2 under relocation |
| E004 | Attention | adaptation_speed | `enable_attention=false` | slower error reduction |
| E005 | TSPL | adaptation_speed | `enable_tspl=false` | skill compilation stalls |
| E006 | RBTA | exploration/safety tradeoff | tight energy bounds | TERMINATE ↑, diversity ↓ |
| E007 | APC | goal stability | `enable_apc=false` | goal thrashing |
| E008 | Temporal order | synergy | desync stage_order | synergy ↓, interaction_gain ↓ |

## Running Edge Tests

```bash
PYTHONPATH=python python scripts/run_experiment.py experiments/ablations/no_prediction.yaml
PYTHONPATH=python python scripts/run_experiment.py experiments/ablations/no_mdim.yaml
PYTHONPATH=python python scripts/run_experiment.py experiments/ablations/desync.json
```

Each experiment exports the same metric bundle: Φ-IQ subindices, emergence components, synergy, failure logs.

## Falsification Rule

An edge is **supported** only if the intervention changes **behavioral signatures** (diversity, reuse, synergy), not merely composite score. Score-only drops without signature change indicate confounded measurement.
