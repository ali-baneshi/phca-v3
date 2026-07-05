# Scientific Validation Results — Provenance

Multi-seed validation suite output (`make validate-science`). Generated
**2026-07-05** on reference machine (Python 3.11, `MUJOCO_GL=disabled`).

## How to reproduce

```bash
make validate-science
```

Orchestrator: [`scripts/run_validation_suite.py`](../../scripts/run_validation_suite.py)  
Aggregator: [`scripts/aggregate_validation.py`](../../scripts/aggregate_validation.py)

## Run parameters

| Parameter | Value |
|-----------|-------|
| Profile | `full` |
| Base seed | 42 |
| Ablation/scaling/OOD seeds | 30 (42–71) |
| Φ-IQ validation seeds | 100 |
| Horizon | 100,000 cycles (+ resume check) |
| Duration | ~10,893 s (~3.0 h) |

## Key artifacts

| File | Description |
|------|-------------|
| `summary.json` | Cross-experiment Φ-IQ rollup (n=20 experiment means) |
| `hypothesis_verdicts.json` | H001–H006 status and observed values |
| `failure_log.json` | Per-seed RBTA violations and goal-thrash events (n=150 events) |
| `phi_iq_validation.json` | Sensitivity (Mann–Whitney) + predictive validity (r) |
| `manifest_index.json` | Completed step index for `--resume` |

## Summary numbers

- **Aggregate Φ-IQ:** 0.526 ± 0.189 (CI95 0.435–0.603)
- **full_system ablation:** Φ-IQ 0.631 ± 0.029; emergence_composite 0.184 ± 0.004
- **Scaling overall Φ-IQ (slip=0, 30 seeds):** 5×5 0.700±0.046; 10×10 0.325±0.042; 20×20 0.147±0.042
- **Φ-IQ predictive validity:** r = 0.992 (pass ≥ 0.7)

## Hypothesis verdicts

| ID | Status |
|----|--------|
| H001 Memory → cross-context reuse | Refuted |
| H002 Prediction → transfer | Refuted |
| H003 Stage order → synergy | Refuted |
| H004 Single modules → novel behaviour | Partially_supported |
| H005 Full vs minimal cycle | Refuted |
| H006 Φ-IQ predictive validity | Validated |

## Directory layout

```
results/validation/
├── summary.json
├── hypothesis_verdicts.json
├── failure_log.json
├── phi_iq_validation.json
├── manifest_index.json
├── ablations/          # full_system, no_*, desync, minimal_cycle, interaction_test, …
├── scaling/            # grid{5,10,20}_slip{0,1}.json
├── ood/                # bandit, mujoco_pendulum
├── baselines/          # causal_eval.json
├── horizon_100000.json
└── horizon_resume_check.json
```

Checkpoint files (`*.ckpt`) are gitignored and not committed.
