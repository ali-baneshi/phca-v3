# Architecture as implemented

Verified against `f692fac` + the warm-up fix (`b5b9e25`). Diagrams and README
descriptions were not used as evidence; every edge below was read from imports
or call sites.

## Layers

```
          scripts/ (40 files)            monitoring/ (15,162 LOC, PyQt5 + JSONL)
                  |                                  |
                  +---------- evaluation/ -----------+   (trace, runner, baselines)
                                   |
                          core/cycle.py  (3,678 LOC)  <-- the only orchestrator
                                   |
   +--------+--------+--------+----+----+--------+--------+--------+--------+
   |        |        |        |         |        |        |        |        |
  asi    memory  prediction  world_model  motivation attention regulation  hpm
   |        |        |        |         |        |        |        |
   +--------+--------+--------+---------+--------+--------+--------+
                                   |
                            config.py, logging.py
```

`core/cycle.py` imports 22 production modules directly. Nothing else orchestrates.
Removing a package therefore always means editing the cycle.

## Module fan-in (production, excluding tests)

| Module | Production modules importing it |
|---|---|
| `phca.config` | 21 |
| `phca.logging` | 13 |
| `phca.core.cycle` | 6 |
| `phca.evaluation.trace` | 6 |
| `phca.core` | 5 |
| `phca.world_model` | 5 |
| `phca.world_model.mlp` | 5 |
| `phca.environments` | 5 |
| `phca.resilience.types` | 4 |
| `phca.environments.grid_world` | 4 |
| `phca.memory` | 3 |
| `phca.evaluation.interventions` | 3 |

## The monitoring seam (the important one)

`core/cycle.py` touches monitoring in exactly four places:

- `TYPE_CHECKING` imports of `MetricsStore` and `ObservabilityStore` (`cycle.py:78-79`).
- Two lazy imports of `ObservabilityFrame` (`cycle.py:580`, `cycle.py:1502`).

Both lazy imports are behind `if self.observability_store is not None` and
`if self.metrics_store is not None`. The stores default to `None`; there is no
production caller that sets them. `scripts/phca_observatory.py` does, from outside
the package.

Consequence: monitoring is a **pluggable observer**, not a dependency. A 6-line
`Protocol` with a `push(frame)` method preserves the seam while deleting 15,162
lines. `environments/mujoco_env.py:389` also imports one helper from
`monitoring.camera_render`; that env is being quarantined.

## Module-level singletons and leakage

- `config.DEFAULT_MODULE_BOUNDS` is a module-level dict. It is copied at cycle
  construction (`cycle.py:180`) and is not mutated anywhere; no leak found.
- `np.random` is global. Tests seed it via an autouse fixture; library code uses
  `np.random.default_rng` in most modules but not all. This is why the same seed
  can produce different traces across orchestration orders.
- `RBTAEnforcer.check_cycle` has three call sites: `step()` (`cycle.py:483`),
  `_finalize_learning_cycle()` (`cycle.py:1412`, async learning path), and
  `_rbta_preflight_check()` (`cycle.py:3098`, before action selection). On the
  default synchronous path it runs **twice per step** (measured). The preflight
  call is what made cold starts enter safe mode.

## Undeclared dependency: pgmpy (and torch)

The default discrete world model, `WorldModelGPrime` (`world_model/graph.py`),
loads `pgmpy` lazily and silently falls back when it is missing. `pgmpy` is listed
in `requirements.txt` but **not** in `pyproject.toml`, so `pip install phca` never
installs it. When present it imports `torch`.

Measured on the default configuration (`build_for_env(size=5, seed=0)`, 60 cycles):

| | pgmpy installed | pgmpy blocked |
|---|---|---|
| First-cycle latency | 1,621 ms (pgmpy + torch import) | 4 ms |
| Actions taken | identical | identical |
| Goals reached | identical | identical |
| Prediction error, cycles 0-7 | 11, 11, 11, 10, 10, 9, 9, 9 | 6, 10, 2, 1, 1, 0, 0, 0 |
| Prediction confidence, cycle 0 | 0.952 | 0.600 |

The actions are the same because the geometric selector ignores the world model.
Every learning metric the project reports is different depending on whether an
undeclared package happens to be installed. **This is a reproducibility defect, not
a style issue.** It is also the real cause of the cold-start latency that the
warm-up fix (`b5b9e25`) addressed.

## The cycle as implemented

`step()` runs: RBTA carry-over → perception (ASI sanitizer, memory write,
prediction, MDIM, attention, PID regulation, partial RBTA) → action selection →
environment step → prediction error → TSPL/learn → full RBTA → resilience
detect/recover → consolidation → metrics/trace. The "12-step" and "21-step"
labels in documentation are naming granularity, not control flow.

Action selection is a single function with four mutually exclusive early returns:
`discrete_safe_mode`, `continuous_safe_mode`, `discrete_explore`,
`energy_stay`, then the geometry path (`pure_geometry_ablation`) or the
prediction-scored path (`prediction_scored`, `confidence_gated_*`).

## Entry points

There are no `[project.scripts]`. Every entry is `python scripts/<name>.py` with
`PYTHONPATH=python`. 40 scripts exist; 9 import `phca.monitoring`.

## Third-party surface in production code

Hard: `numpy` (56 import sites).

Lazy/optional: `structlog` (logging fallback), `orjson` (config serialization
fallback), `psutil` (monitoring), `scipy` (evaluation statistics), `pgmpy`
(graph world model), `matplotlib`/`PIL` (monitoring), `PyQt5` (monitoring),
`gymnasium`/`mujoco` (continuous env).

Everything except `numpy` has a stdlib fallback, is confined to a removable
package, or is behind an optional extra.
