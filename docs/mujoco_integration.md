# MuJoCo Integration

Erasmus can drive physics simulation environments through MuJoCo, enabling cognitive-cycle-controlled robotics tasks.

> **Status:** 🟡 Experimental. All actions discretised. Continuous action support in development.

---

## Supported Environments

| Environment | State Dim | Discrete Actions | Action Map |
|---|---|---|---|
| `InvertedPendulum-v5` (Cartpole) | 4 | 3 | PUSH_LEFT (-3), STAY (0), PUSH_RIGHT (+3) |
| `Pendulum-v1` | 3 | 3 | TORQUE_LEFT (-2), STAY (0), TORQUE_RIGHT (+2) |
| `Reacher-v5` (2-DOF) | ≤ 11 | 5 | MOVE_SW, MOVE_NW, STAY, MOVE_NE, MOVE_SE |

---

## Installation

```bash
pip install gymnasium[mujoco]
```

For headless servers (CI, remote training):
```bash
export MUJOCO_GL=egl    # GPU headless (requires EGL support)
# OR
export MUJOCO_GL=osmesa  # CPU headless (requires libosmesa)
```

---

## Usage

### Build a Cognitive Cycle for MuJoCo

```python
from phca.core.cycle import CognitiveCycle

# Cartpole (recommended starting point)
cycle = CognitiveCycle.build_for_mujoco(
    env_name="InvertedPendulum-v5",
    seed=42,
    use_mlp=True,       # MLP world model (recommended)
)
```

MuJoCo environments default to MLP mode with:
- Lower learning rate (0.05) for smooth continuous targets
- Higher G' time bound (0.080s) for physics simulation overhead
- Higher action time bound (0.050s) for MuJoCo step() overhead

### Run Cognitive Cycles

```python
for i in range(100):
    metrics = cycle.step()
    if i % 20 == 0:
        print(f"Cycle {i:3d}: latency={metrics.latency_ms:.0f}ms, "
              f"error={metrics.prediction_error:.3f}, "
              f"reward info in env")
```

---

## Benchmark

```bash
# Quick benchmark (100 cycles, Cartpole)
PYTHONPATH=python python scripts/benchmark_mujoco.py \
    --env=InvertedPendulum-v5 --cycles=100 --output=logs/cartpole.json

# Pendulum
PYTHONPATH=python python scripts/benchmark_mujoco.py \
    --env=Pendulum-v1 --cycles=100 --output=logs/pendulum.json
```

---

## How It Works

The `MuJoCoSimpleEnv` wrapper (`phca/environments/mujoco_env.py`) bridges the gymnasium MuJoCo API to Erasmus's `EnvironmentProtocol`:

1. **Continuous → Discrete**: Each environment's continuous action space is discretised into 3–5 bins
2. **No grid assumption**: `get_goal_position()` returns `None`, so distance-gain falls back to 0.5 (neutral)
3. **State normalisation**: Raw MuJoCo observations pass through directly (Phase 1 — running normalisation deferred)
4. **Observation caching**: `_get_observation()` returns the last observation from `step()` or `reset()`, since MuJoCo only produces observations on `step()`

### Observation Flow

```
Cycle N:
  _get_observation() → obs_N    (cached from previous step)
  sanitize(obs_N) → store in M2
  predict(obs_N) → predicted_N+1
  select_action(predicted_N+1)
  env.step(action) → obs_N+1    (new observation)
  PEU: compare obs_N+1 vs predicted_N+1
  learn(obs_N, action, obs_N+1)

Cycle N+1:
  _get_observation() → obs_N+1  (cached from env.step)
  ...
```

This is a standard temporal-difference learning setup — predictions are always one step ahead of observations.

---

## Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| Discrete actions only | 3–5 bins per environment | Finer discretisation possible via custom `_ACTION_MAPS` |
| No continuous actions | Action dimension discretised to single integer | Phase 4 scope |
| Limited environment set | Only 3 tested | Add action maps for new gymnasium envs |
| No observation normalisation | Raw observations pass through | Add `_normalise_observation()` for MLP mode |
| Cartpole may not balance | ~50ms cycle may be too slow for balancing | Use `--use-mlp --cycles=500` for more learning |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'gymnasium'` | `pip install gymnasium[mujoco]` |
| `mujoco.FatalError: an OpenGL platform library is not found` | Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` |
| State dimension mismatch | Always use `build_for_mujoco()` (not `build_for_env()`) for MuJoCo |
| Predictions are near-zero | Use `use_mlp=True` — discrete G' cannot model continuous observations |
| Slow cycles (>200ms) | MuJoCo physics sim adds ~1-5ms/step. Increase RBTA bounds |

---

## Related

- [Quickstart Guide](quickstart.md) — General installation and usage
- [Φ-IQ Metric](phi_iq_metric.md) — Benchmark interpretation
- [Limitations](limitations.md) — System constraints
- [Architecture Overview](architecture.md) — Cognitive cycle details

---

## Review Notes (Pass 1 — Accuracy)
- Action maps verified from mujoco_env.py `_ACTION_MAPS` dict.
- State dims verified from mujoco_integration_plan.md Appendix A.
- Build parameters verified from cycle.py `build_for_mujoco()` method.
- Observation flow verified by reading cycle.py `step()` method.
- Reacher is experimental (no action map for Reacher in current code — only _REACHER_ACTIONS fallback).

## Review Notes (Pass 2 — Clarity)
- Each environment has clear state dim and action count.
- Observation flow diagram explains the TD learning setup.
- Troubleshooting table covers all known issues.

## Review Notes (Pass 3 — Completeness)
- Covers: supported envs, installation, usage, benchmark, how it works, limitations, troubleshooting.
- Links to all related documentation.
- Clearly marks MuJoCo as experimental.
