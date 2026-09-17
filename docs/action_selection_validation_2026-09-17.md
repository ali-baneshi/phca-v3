# Action-Selection Repair Validation — 2026-09-17

This is a current repair-validation record, not a promotion decision. The
default selector was not changed.

## Protocol

Both runs used `scripts/phca_causal_eval.py` with `--levels level2`,
`--grid-size 5`, `--cycles 200`, `--seeds 30`, `--use-mlp`, and the same base
seed. The gated run added `--confidence-gated`. The evaluator marked both
reports `causal_power` because PHCA and both gated controls used 30 seeds.

## Results

| Run | Selector modes | Goal rate | Mean distance | Cumulative reward | Gate |
|---|---|---:|---:|---:|---|
| Default | 99.98% pure geometry; 0.02% emergency | 0.4797 (95% CI 0.3373–0.6070) | 1.0528 | 94.8927 | FAIL |
| Confidence-gated | 25.00% warmup; 10.73% prediction; 19.37% below-threshold geometry; 44.88% fallback; 0.02% emergency | 0.5072 (95% CI 0.3553–0.6374) | 1.0653 | 100.4477 | FAIL |

Both runs lost to `greedy_observed` on goal rate, distance, and reward; first-goal
cycle tied, so neither passed the scenario gate. The gated run exercised all
new selector branches and reduced mean prediction error from 3.7527 to 3.7234,
but this does not establish that confidence-gated control is superior.

## Interpretation

The confidence-gated flag is reachable and its fallback is active at full
sample size. It remains experimental and disabled by default. The 10×10
confidence-gated protocol and a full multi-level promotion comparison remain
open. Raw evidence is stored under `logs/repair_validation/`.
