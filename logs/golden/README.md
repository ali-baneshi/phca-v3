# Golden benchmark artifacts (Track G / J)

Reference outputs for regression comparison. See [`manifest.json`](manifest.json) for SHA256 checksums.

| Artifact | Source | Tier |
|----------|--------|------|
| `../benchmark_ci_baseline.json` | CI L0 gate | T0 |
| `../benchmark_level4_smoke.json` | `make bench-level4-smoke` | T1 |
| `../l4_ablation.json` | `make bench-level4-ablation` | T3 |
| `../benchmark_recovery.json` | `make bench-recovery` | T3 |
| `../../results/validation/summary.json` | validate-science | T4 |

Bisection: [`docs/maturation_bisection.md`](../../docs/maturation_bisection.md)

**Last maturation-test:** 45 passed (`make maturation-test`)
