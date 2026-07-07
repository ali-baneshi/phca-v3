# Golden benchmark artifacts (Track J)

Reference outputs for regression comparison. Regenerate after intentional behavior changes.

| Artifact | Source command | Purpose |
|----------|----------------|---------|
| `../benchmark_ci_baseline.json` | CI `check_benchmark_gate.py` | T0 Φ-IQ L0 composite |
| `../benchmark_level4.json` | `benchmark_level4.py --tasks 10 ...` | T3 continual forgetting (current: FAIL) |
| `../benchmark_recovery.json` | `benchmark_recovery.py` | T3 injectable recovery |

**Local CI:** use `make setup` (venv) then `make ci-local` — system Python may lack `pytest-timeout` (PEP 668).

**Last full fast suite (2026-07-07):** 95 passed, 3 xfailed (`test_static_contracts` G5 debt).
