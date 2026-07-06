# MuJoCo Integration

PHCA drives physics simulation environments through MuJoCo via gymnasium. As of
Phase 6/7, **Pendulum-v1** and **Reacher-v5** use true **continuous actions**
with a prediction-driven MPC selector; **Cartpole** remains discrete.

> **Status:** Validated on three envs. Opt-in via `requirements-mujoco.txt`.
> Headless: `export MUJOCO_GL=disabled`.

---

## Supported Environments

| Environment | Gym ID | Action space | State dim | Selector |
|---|---|---|---|---|
| Cartpole | `InvertedPendulum-v5` | Discrete (3: push L / stay / push R) | 4 | Discrete argmax |
| Pendulum | `Pendulum-v1` | **Continuous** torque ∈ [-2, 2], dim 1 | 3 | MPC (K=8 samples) |
| Reacher | `Reacher-v5` | **Continuous** actuator ∈ [-1, 1]², dim 2 | 10 | MPC (K=8 samples) |

100-cycle reference (MLP, D-107): Pendulum ~7 ms mean, error 29.6→0.68; Reacher
~4.4 ms mean, error 105.7→8.4; Cartpole discrete, 0 violations.

MuJoCo RBTA time bounds (D-128, D-131): G' **0.120 s**, ACTION **0.080 s**
(MPC only), ENV **0.250 s** (`env.step()` physics; energy **12.5**) — calibrated
for shared CI runners; local means stay well below these limits.

---

## Installation

```bash
pip install -r requirements-mujoco.txt
export MUJOCO_GL=disabled   # headless CI / servers
```

---

## Usage

```python
from phca.core.cycle import CognitiveCycle

# Continuous Pendulum (prediction-driven MPC)
cycle = CognitiveCycle.build_for_mujoco(
    env_name="Pendulum-v1",
    seed=42,
    use_mlp=True,
)

for i in range(100):
    metrics = cycle.step()
```

MuJoCo builds use lower learning rate (0.05), higher G′ time bound (0.120s),
ACTION bound (0.080s / energy 4.0) for MPC, and ENV bound (0.250s / energy 12.5)
for physics step overhead.

---

## Benchmark

```bash
# Unified benchmark CLI (preferred)
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py \
  --env pendulum --use-mlp --cycles=100

MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py \
  --env cartpole --use-mlp --cycles=100

MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py \
  --env reacher --use-mlp --cycles=100

# Nightly MuJoCo gate
make nightly-mujoco
```

MuJoCo benchmarks report latency, prediction-error trend, and RBTA violations —
**not** GridWorld Φ-IQ (no goal_reached metric on physics tasks).

Gate criteria: finite errors, violations < 10%, error improves or late < 1.0.

---

## Continuous MPC Action Selection (Phase 6/7)

For `ContinuousSpace` environments, `_select_continuous_action()` in
`phca/core/cycle.py`:

1. Sample K=8 candidate actions ~ Uniform(low, high) (A1-capped)
2. Predict next state for each via G′
3. Score: `0.4·confidence + 0.5·goal_ref_alignment + 0.1·PGA`
4. Pick best candidate (ε-greedy exploration)

No reward function, value network, or policy gradient — prediction/goal-driven.
This path is where **A4 is measured** (see `assumption_validation.py`).

Details: [action_selection.md](action_selection.md).

---

## Observation Flow

```
Cycle N:
  _get_observation() → obs_N       (cached from previous step)
  sanitize(obs_N) → M2
  predict(obs_N) → predicted_N+1
  select_action (discrete or MPC)
  env.step(action) → obs_N+1
  PEU: compare obs_N+1 vs predicted_N+1
  learn(obs_N, action, obs_N+1)
```

Standard one-step-ahead prediction learning.

---

## Limitations

| Limitation | Detail |
|---|---|
| Task success not gated | Benchmarks measure error↓, not pole-upright or reach-target success |
| Cartpole balancing | ~10–17 ms cycle may be slow for classic balance task |
| Raw observations | No running normalisation layer (Phase 1 deferral) |
| Limited env set | 3 envs validated; extending requires `EnvironmentProtocol` hooks |
| Camera frames | Observability-only; not fed to G′ |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: gymnasium` | `pip install -r requirements-mujoco.txt` |
| OpenGL errors | `MUJOCO_GL=disabled` or `egl` / `osmesa` |
| State dim mismatch | Use `build_for_mujoco()`, not `build_for_env()` |
| Near-zero predictions | Use `use_mlp=True` for continuous obs |

---

## Related

- [action_selection.md](action_selection.md)
- [architecture.md](architecture.md)
- [reproducibility.md](reproducibility.md)
- [limitations.md](limitations.md)
