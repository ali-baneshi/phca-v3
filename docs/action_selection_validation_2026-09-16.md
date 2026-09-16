# Action-Selection Validation — 2026-09-16

This is a validation record for the opt-in `--confidence-gated` selector. It is not a claim that the selector should become the default.

## Valid evidence

Command shape:

```text
PYTHONPATH=python python scripts/phca_causal_eval.py --levels level2 --cycles 200 --seeds 30 --use-mlp --output <file>
PYTHONPATH=python python scripts/phca_causal_eval.py --levels level2 --cycles 200 --seeds 30 --use-mlp --confidence-gated --output <file>
```

Both valid runs used the evaluator’s default 5×5 GridWorld and 30 seeds × 200 cycles.

| Run | Action mode | PHCA success rate | Mean prediction error | Selector evidence |
|---|---|---:|---:|---|
| `final_default_5x5_30x200.json` | `pure_geometry_default` | 0.7667 | 3.7567 | 99.98% `pure_geometry_ablation` |
| `final_gated_5x5_30x200.json` | `confidence_gated_opt_in` | 0.7667 | 3.7258 | 25.00% warm-up; 10.73% prediction; 19.37% below-threshold geometry; 44.88% fallback |

The gated run therefore demonstrates that the new branches are reachable and that the rolling fallback activates. It does not establish that confidence-gated selection improves task performance.

## Not evidence

The files named `default_10x10_30x200.json` and `gated_10x10_30x200.json` were produced by an earlier command that selected `level3` without setting `--grid-size 10`; their JSON records `grid_size: 5`. They are excluded from 10×10 conclusions.

The corrected commands were attempted with `--levels level2 --grid-size 10 --cycles 200 --seeds 30`, but no result artifact was produced. The 10×10 benchmark is therefore **OPEN / UNKNOWN**, not PASS or FAIL.

## Test evidence

- Focused selector and regression tests: 49 passed.
- Full non-MuJoCo suite: 888 passed, 1 expected failure.
- Changed implementation/test files pass Ruff.
