# Baseline — pre-minimization

Captured at commit `f692fac` (tag `pre-minimization`), working tree clean.
All numbers below were produced in this environment on 2026-11-25 and are the
reference for every "smaller/faster" claim later in the report.

## Size

| Metric | Value | How measured |
|---|---|---|
| Production Python files | 97 | `find python/phca -name '*.py' -not -path '*/tests/*'` |
| Production LOC | 30,759 | same set, `cat | wc -l` |
| Test files | 68 | `find python -name 'test_*.py'` |
| Test LOC | 15,059 | same set, `cat | wc -l` |
| Packages (subdirectories) | 16 | `ls -d python/phca/*/` |
| Markdown docs (`docs/`, `research/`) | 131 | `find docs research -name '*.md'` |
| `docs/` on disk | 1.6 MB | `du -sh docs` |
| `research/` on disk | 724 KB | `du -sh research` |
| Git-tracked `logs/` files | 227 | `git ls-files logs | wc -l` |
| Git-tracked `logs/` size | 43 MB | `git ls-files -z logs | du -ch --files0-from=-` |
| `logs/` on disk | 12 GB | mostly untracked runtime output |
| `.git` size | 51 MB | `du -sh .git` |

## Largest production packages (LOC, excluding their tests)

| Package | LOC |
|---|---|
| `monitoring/` | 15,162 |
| `core/` | 3,690 |
| `world_model/` | 2,844 |
| `evaluation/` | 2,157 |
| `memory/` | 1,282 |
| `environments/` | 1,237 |
| `motivation/` | 784 |
| `regulation/` | 690 |
| `resilience/` | 633 |
| `consolidation/` | 516 |
| `learning/` | 335 |
| `asi/` | 318 |
| `attention/` | 272 |
| `prediction/` | 190 |
| `hpm/` | 188 |

`monitoring/` alone is 49% of production code.

## Tests

```
MUJOCO_GL=disabled PYTHONPATH=python python3 -m pytest python/tests/ python/phca/ -q \
  --timeout=60 --benchmark-skip \
  --ignore=python/tests/test_mujoco_env.py \
  --ignore=python/tests/test_cycle_with_mujoco.py \
  --ignore=python/tests/test_continuous_actions.py
```

Result: **894 passed, 1 skipped, 1 xfailed in 58.9 s**. 940 tests collected in 4.2 s.

### Known defect at baseline

`python/phca/core/tests/test_action_selection_defaults.py::test_default_gridworld_selector_is_pure_geometry`
asserts 20/20 `pure_geometry_ablation` selections. It passes in the full-suite
ordering but fails with **18/20** when run alongside `python/tests/test_grid_world.py`
and `python/phca/core/tests` only. The selector mode changes mid-run, so the
failure is cross-test global-state leakage, not a simple bad assertion. This
must be fixed before any golden trace is trusted.

## Wheel

```
python3 -m pip wheel . --no-deps --no-build-isolation --ignore-requires-python -w .gate-dist
```

- Artifact: `phca-0.1.0-py3-none-any.whl`, 488 KB, 162 entries.
- The wheel **contains the test packages** (`phca/asi/tests/__init__.py`, …).
  Tests must not ship in the installed artifact.

## Dependencies

`pyproject.toml` declares 14 hard runtime dependencies: `numpy`, `scipy`,
`networkx`, `opt_einsum`, `tqdm`, `pandas`, `statsmodels`, `scikit-learn`,
`pyro-api`, `orjson`, `click`, `psutil`, `matplotlib`, `PyQt5`.

`PyQt5` and `matplotlib` are hard dependencies although the GUI is conceptually
optional. `matplotlib==3.11.0` does not exist on PyPI as a released version in
this environment's resolver, which is itself a packaging risk.

## Environment quirks (affect how the gate runs)

- Default `python3` is **3.14.7**, outside `requires-python = ">=3.11,<3.13"`.
  It is the only interpreter with `pytest` (9.0.3) and `numpy` (2.4.6), so tests
  run there while the wheel build passes `--ignore-requires-python`.
- `.venv/` is a uv environment on **3.12.12** with `numpy 1.26.4` but no
  `pip`, `setuptools`, `wheel`, or `pytest`.
- `python3.12` on PATH has none of the runtime dependencies.
- Cold `import phca`: **37.4 ms**.

## Non-size baseline

- No `[project.scripts]` entry points exist; all entry points are loose
  `scripts/*.py` files.
- `phca/__init__.py` exports only `__version__` (15 lines), so there is no
  public API contract to preserve.
- `commit.gpgsign=true` is set for this repository; commits use
  `git -c commit.gpgsign=false`.
