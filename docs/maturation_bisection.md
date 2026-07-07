# Maturation Bisection Protocol

When any **T2+** gate flips red, use this protocol before patching.

## 1. Identify gate and tier

| Gate | Command | Tier |
|------|---------|------|
| L0 quick | `make bench-level-0` | T0 |
| Unit maturation | `make maturation-test` | T1 |
| Assumption validation | `PYTHONPATH=python python scripts/assumption_validation.py --ci` | T2 |
| L4 continual | `make bench-level4-ablation` | T3 |
| Recovery injectables | `make bench-recovery` | T3 |
| Full science | `make validate-science` | T4 |

## 2. Locate last green artifact

Check [`logs/golden/manifest.json`](../logs/golden/manifest.json) for archived JSON paths and commit hints.

## 3. Git bisect

```bash
git bisect start
git bisect bad HEAD
git bisect good <last-green-commit>
# use single gate command as test script
git bisect run bash -c 'make maturation-test && make bench-level-0'
git bisect reset
```

## 4. Classify regression

| Class | Symptom | Fix scope |
|-------|---------|-----------|
| protocol | Benchmark script only | `benchmark_level4.py` |
| wiring | Hook exists, not called | `cycle.py` minimal wire |
| behavior | Wiring OK, metric fails | Document limitation or algorithm decision |
| infra | CI deps, paths | `requirements.txt`, `conftest.py`, Makefile |

## 5. Add regression test at appropriate tier

- Protocol → `test_forgetting.py`
- Wiring → `test_static_contracts.py`
- Behavior → diagnostic JSON in `logs/golden/`
- Infra → `ci-local` / workflow step

## 6. Update audit docs

Sync [`maturity_audit_2026-07-07.md`](maturity_audit_2026-07-07.md) last-result column.
