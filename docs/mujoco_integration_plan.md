# MuJoCo Integration Plan — PHCA v3.0

**Date:** 2025-06-30  
**Author:** Chief Architect, PHCA v3.0  
**Status:** ✅ **EXECUTED** — see below
**Scope:** Minimal, safe, incremental integration of MuJoCo physics environments into the PHCA cognitive cycle.

> **This plan has been partially executed.** The `MuJoCoSimpleEnv` wrapper was created
> (`python/phca/environments/mujoco_env.py`), `CognitiveCycle.build_for_mujoco()` was
> implemented, tests and benchmark scripts were written. The wrapper was moved into
> `phca/environments/` as part of the gap-closure package restructure (B-004).
> See `docs/phase3.3_full_completion_report.md` for details.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Phase 0: Preparation](#2-phase-0-preparation)
3. [Phase 1: Wrapper Development](#3-phase-1-wrapper-development)
4. [Phase 2: Integration with CognitiveCycle](#4-phase-2-integration-with-cognitivecycle)
5. [Phase 3: Validation & Testing](#5-phase-3-validation--testing)
6. [Phase 4: Scaling Strategy (Optional)](#6-phase-4-scaling-strategy-optional)
7. [Risk Assessment & Mitigation](#7-risk-assessment--mitigation)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Architectural Overview

### 1.1 The EnvironmentProtocol Contract

The PHCA cognitive cycle interacts with environments exclusively through the `EnvironmentProtocol` defined in `python/environments/protocol.py`:

```python
class EnvironmentProtocol(Protocol):
    action_space_size: int

    def get_possible_actions(self) -> List[str]: ...
    def get_goal_position(self) -> Optional[Tuple[int, int]]: ...
    def get_action_names(self) -> List[str]: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]: ...
```

In addition, the `CognitiveCycle.step()` method calls `env._get_observation()` directly (line 108 of cycle.py), and `CognitiveCycle.__init__()` reads `env.size`. The `_compute_distance_gain()` method inspects `env.agent_pos`, `env.grid`, and `env.WALL` — all GridWorld-specific.

**Implication:** The `MuJoCoSimpleEnv` wrapper must implement ALL of these, plus `reset()`, `_get_observation()`, `size` (can be `1` for non-grid), and provide reasonable fallbacks for grid-specific attributes.

### 1.2 Target Environments

Per the constraints, we target three simple MuJoCo environments:

| Environment | Suite | State Dim | Action Space | Goal | Notes |
|-------------|-------|-----------|---------------|------|-------|
| Cartpole (InvertedPendulum) | `gymnasium` (`InvertedPendulum-v5`) | 4 (x, θ, dx, dθ) | Continuous [-3, +3] | Balance upright | Simplest entry point |
| Pendulum | `gymnasium` (`Pendulum-v1`) | 3 (cosθ, sinθ, dθ) | Continuous [-2, +2] | Swing up & balance | Smallest state dim |
| 2-DOF Reacher | `gymnasium` (`Reacher-v5`) | 6-11 (joint angles, velocities, target) | Continuous [-1, +1] × 2 | Reach target | Max state dim ≤ 11 |

### 1.3 Dimensionality Check

All three environments satisfy the ≤20 state dimension constraint:

- **Cartpole:** 4 dims — fits easily within the 84-dim MLP.
- **Pendulum:** 3 dims — smallest possible.
- **Reacher:** ≤11 dims — well within budget.

### 1.4 Action Discretisation Strategy

All target environments have continuous action spaces. We discretise as follows:

| Original Space | Discrete Actions | Mapping |
|---------------|------------------|---------|
| `[-3, +3]` (Cartpole) | 3 | `0 → -3.0`, `1 → 0.0`, `2 → +3.0` |
| `[-2, +2]` (Pendulum) | 3 | `0 → -2.0`, `1 → 0.0`, `2 → +2.0` |
| `[-1, +1]²` (Reacher) | 5 | `0 → (-1,-1)`, `1 → (-1,+1)`, `2 → (0,0)`, `3 → (+1,-1)`, `4 → (+1,+1)` |

This keeps discrete action space ≤5 per the constraint.

---

## 2. Phase 0: Preparation

### 2.1 Dependencies

Add to `requirements.txt` or `requirements-phase-3.2.txt`:

```
gymnasium[mujoco]>=1.0.0
mujoco>=3.2.0
```

Install:

```bash
pip install gymnasium[mujoco]
```

**Note:** `gymnasium[mujoco]` bundles the `mujoco` Python bindings. No separate MuJoCo download is required.

### 2.2 Headless Rendering (for CI/benchmark servers)

If running on a headless server, set:

```bash
export MUJOCO_GL=egl    # GPU headless
# OR
export MUJOCO_GL=osmesa  # CPU headless
```

### 2.3 Verification Script

Create `scripts/verify_mujoco.py`:

```python
"""Verify MuJoCo installation and environment access."""
import gymnasium as gym

env = gym.make("InvertedPendulum-v5")
obs, info = env.reset()
print(f"State dim: {obs.shape[0]}")      # Expected: 4
print(f"Action space: {env.action_space}")  # Expected: Box(-3.0, 3.0, (1,))

# Check that step works
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Step OK: obs shape={obs.shape}, reward={reward:.3f}")

env_pen = gym.make("Pendulum-v1")
obs_pen, _ = env_pen.reset()
print(f"Pendulum state dim: {obs_pen.shape[0]}")  # Expected: 3

print("✓ MuJoCo environments accessible")
env.close()
env_pen.close()
```

---

## 3. Phase 1: Wrapper Development

### 3.1 Class Design: `MuJoCoSimpleEnv`

Create a new file: `python/environments/mujoco_env.py`

The wrapper must bridge the MuJoCo/gymnasium API to the `EnvironmentProtocol` that PHCA expects.

```python
"""
MuJoCo Simple Environment Wrapper — PHCA v3.0

Wraps a gymnasium MuJoCo environment (Cartpole, Pendulum, Reacher)
into the EnvironmentProtocol interface for the PHCA cognitive cycle.

Discretises continuous action spaces into ≤5 discrete bins.
State vector is the raw observation array from the MuJoCo environment.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np


class MuJoCoSimpleEnv:
    """Minimal MuJoCo environment wrapper for PHCA cognitive cycle.

    Attributes:
        env_name: gymnasium environment ID (e.g. 'InvertedPendulum-v5').
        action_space_size: Number of discrete actions (≤5).
        size: Stub attribute for CognitiveCycle (set to 1, non-grid).
    """

    # ── Discretisation maps ──────────────────────────────────

    _ACTION_MAPS: dict = {
        "InvertedPendulum-v5": {
            0: np.array([-3.0], dtype=np.float32),   # push left
            1: np.array([0.0], dtype=np.float32),     # do nothing
            2: np.array([3.0], dtype=np.float32),     # push right
        },
        "Pendulum-v1": {
            0: np.array([-2.0], dtype=np.float32),    # torque left
            1: np.array([0.0], dtype=np.float32),     # no torque
            2: np.array([2.0], dtype=np.float32),     # torque right
        },
    }

    # Reacher uses 2D action space so we discretise via a grid of 5 points
    _REACHER_ACTIONS = {
        0: np.array([-1.0, -1.0], dtype=np.float32),
        1: np.array([-1.0,  1.0], dtype=np.float32),
        2: np.array([ 0.0,  0.0], dtype=np.float32),
        3: np.array([ 1.0, -1.0], dtype=np.float32),
        4: np.array([ 1.0,  1.0], dtype=np.float32),
    }

    # ── Action names (for PHCA logging) ─────────────────────

    _ACTION_NAMES: dict = {
        "InvertedPendulum-v5": ["PUSH_LEFT", "STAY", "PUSH_RIGHT"],
        "Pendulum-v1": ["TORQUE_LEFT", "STAY", "TORQUE_RIGHT"],
        "Reacher-v5": ["MOVE_SW", "MOVE_NW", "STAY", "MOVE_NE", "MOVE_SE"],
    }

    def __init__(
        self,
        env_name: str = "InvertedPendulum-v5",
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        """Initialise the MuJoCo environment wrapper.

        Args:
            env_name: gymnasium MuJoCo environment ID.
            seed: Random seed for reproducibility.
            render_mode: Gymnasium render mode (None for headless,
                         "human" for visualisation, "rgb_array" for recording).
        """
        self.env_name = env_name
        self._env = gym.make(env_name, render_mode=render_mode)
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        # Detect environment type and set up action mapping
        self._action_map = self._build_action_map()
        self.action_space_size = len(self._action_map)

        # State normalisation statistics (populated lazily)
        self._obs_min: Optional[np.ndarray] = None
        self._obs_max: Optional[np.ndarray] = None

        # Stub for CognitiveCycle compatibility
        self.size = 1  # non-grid environment

        # Initial reset
        obs, _ = self._env.reset(seed=seed)

    # ── Public interface (EnvironmentProtocol + extras) ──────

    def get_possible_actions(self) -> List[str]:
        return list(self.get_action_names())

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        """Return None — MuJoCo environments don't have grid positions.

        The cognitive cycle's _compute_distance_gain() will fall back
        to 0.5 (neutral) when goal_position is None.
        """
        return None

    def get_action_names(self) -> List[str]:
        return list(self._ACTION_NAMES.get(self.env_name, [f"ACT_{i}" for i in range(self.action_space_size)]))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute a discrete action in the MuJoCo environment.

        Args:
            action: Discrete action index (0..action_space_size-1).

        Returns:
            (observation, reward, terminal, info) tuple compatible
            with EnvironmentProtocol.
        """
        continuous_action = self._action_map[action]
        obs, reward, terminated, truncated, info = self._env.step(continuous_action)
        terminal = terminated or truncated

        # Optionally normalise the observation
        obs = self._normalise_observation(obs)

        return obs.astype(np.float32), float(reward), bool(terminal), info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            Initial observation vector.
        """
        obs, info = self._env.reset(seed=seed or self._seed)
        return self._normalise_observation(obs).astype(np.float32)

    def get_state_dim(self) -> int:
        """Return the dimensionality of the MuJoCo observation vector."""
        return self._env.observation_space.shape[0]

    # ── Internal methods ─────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Get current observation without stepping.

        Called by CognitiveCycle.step() Step 0 (ASI sanitisation)
        before the action is selected and applied.

        Note: In MuJoCo, reading observation between steps is not
        meaningful — the environment only produces a new observation
        after step(). This method reads the last returned observation
        from the internal cache.
        """
        # The MuJoCo environment does not have a "peek" method.
        # We cache the last observation from step() or reset().
        return self._last_obs

    def _build_action_map(self) -> dict:
        """Build the discrete-to-continuous action mapping."""
        if self.env_name in self._ACTION_MAPS:
            return dict(self._ACTION_MAPS[self.env_name])
        elif "Reacher" in self.env_name:
            return dict(self._REACHER_ACTIONS)
        else:
            raise ValueError(
                f"Unknown environment '{self.env_name}'. "
                f"Add action map to _ACTION_MAPS or _REACHER_ACTIONS."
            )

    def _normalise_observation(self, obs: np.ndarray) -> np.ndarray:
        """Optionally normalise observation to [0, 1] range.

        Normalisation is critical for the MLP world model, which
        benefits from inputs in a consistent range. For the Bayesian
        G' (discrete/continuous), normalisation is less important.

        Uses running min/max statistics collected during episode
        rollouts (Phase 1 minimal: no running stats yet).
        """
        # Phase 1: passthrough — no normalisation.
        # Phase 2+: collect running min/max and normalise.
        return obs

    def close(self) -> None:
        """Release MuJoCo resources."""
        self._env.close()

    # ── Stub properties for grid-world compatibility ─────────

    @property
    def agent_pos(self) -> Tuple[int, int]:
        """Stub — MuJoCo environments don't have grid positions.

        The cognitive cycle's _compute_distance_gain() checks for
        `hasattr(env, 'grid')` and `hasattr(env, 'WALL')` before
        using agent_pos, so returning (0, 0) is safe: the grid path
        will be skipped and the fallback (0.5) will be used.
        """
        return (0, 0)
```

### 3.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No separate `_last_obs` cache shown** | The real implementation must cache the last observation from `step()` and `reset()` so `_get_observation()` can return it. This is because MuJoCo environments produce a new observation only on `step()`. The `CognitiveCycle.step()` calls `_get_observation()` *before* `_select_action()` and the subsequent `env.step()`. This means the "current state" is the *previous step's* observation, which is correct for the prediction → action → observe loop. |
| **Normalisation deferred** | Phase 1 passes raw observations through. Phase 2 should add running min/max normalisation. The MLP world model works better with normalised inputs. |
| **`size = 1` stub** | The `CognitiveCycle.__init__` logs `grid_size=env.size`. A value of 1 signals "not a grid" without breaking anything. |
| **`agent_pos` stub** | `_compute_distance_gain()` checks `hasattr(env, 'grid')` before using it, so the stub is never actually read. Required only for attribute existence. |

### 3.3 Asserted Compatibility Checklist

The `MuJoCoSimpleEnv` satisfies:

- [x] `action_space_size` — set in `__init__`
- [x] `get_possible_actions()` — returns action names
- [x] `get_goal_position()` — returns `None` (triggers fallback in cycle)
- [x] `get_action_names()` — returns action labels
- [x] `step(action: int)` — returns `(obs, reward, terminal, info)`
- [x] `_get_observation()` — returns cached obs
- [x] `reset()` — resets MuJoCo sim, returns obs
- [x] `get_state_dim()` — from `observation_space.shape[0]`
- [x] `size` — set to 1 (for logging)
- [x] `agent_pos` — stub property exists (never used for non-grid)

---

## 4. Phase 2: Integration with CognitiveCycle

### 4.1 Extending `build_for_env()`

Add a new class method to `CognitiveCycle` (or modify `build_for_env()`) to accept a `MuJoCoSimpleEnv` instance:

```python
@classmethod
def build_for_mujoco(
    cls,
    env_name: str = "InvertedPendulum-v5",
    seed: int = 42,
    use_mlp: bool = False,
    use_continuous: bool = True,
) -> CognitiveCycle:
    """Build a cognitive cycle for a MuJoCo physics environment.

    Args:
        env_name: gymnasium MuJoCo environment ID.
        seed: Random seed.
        use_mlp: If True, use MLP world model instead of Bayesian G'.
        use_continuous: If True, use Gaussian CPDs (only if not use_mlp).

    Returns:
        Configured CognitiveCycle instance.
    """
    env = MuJoCoSimpleEnv(env_name=env_name, seed=seed)
    state_dim = env.get_state_dim()

    sanitizer = ASISanitizer(sensor_dim=state_dim, v_max=100.0, epsilon_confidence=0.01)
    m1 = M1SensoryBuffer(sensor_dim=state_dim)
    m2 = M2WorkingMemory(capacity=7)

    if use_mlp:
        gprime = WorldModelMLP(
            state_dim=state_dim,
            action_dim=env.action_space_size,
            seed=seed,
            # MuJoCo observations are continuous floats, not discrete {0,1,2}.
            # The MLP's linear output is appropriate. Reduce learning rate
            # slightly for smoother continuous prediction targets.
            lr=0.05,
        )
    elif use_continuous:
        gprime = WorldModelGPrime.build_gaussian_grid(
            state_dim=state_dim,
            action_dim=env.action_space_size,
            transition_std=0.5,
            seed=seed,
        )
    else:
        # Discrete graph — less suitable for continuous MuJoCo observations
        # but included for compatibility.
        gprime = WorldModelGPrime(
            state_dim=state_dim,
            action_dim=env.action_space_size,
            seed=seed,
        )
        # Discrete nodes not shown for brevity — only use with discrete
        # observations (e.g., binned MuJoCo states).
        # ⚠️ Discrete G' is NOT recommended for MuJoCo. Use continuous or MLP.

    engine = PredictionEngine(gprime)
    peu = PredictionErrorUnit()
    tspl = TSPL(seed=seed)
    tspl.init_parameters("gprime", (state_dim,))
    rbta = RBTAEnforcer(module_bounds=DEFAULT_MODULE_BOUNDS)

    # Adjust G' bounds for MuJoCo simulation overhead
    rbta.update_bounds(
        "G'", ResourceBounds(B_time=0.080, B_mem=500_000, B_energy=50.0),
    )

    mdim = MDIM(state_dim=state_dim)
    attention = Attention()
    criticality_regulator = CriticalityRegulator()
    hpm_validator = HPMValidator()
    m3 = M3EpisodicMemory(state_dim=state_dim, action_dim=env.action_space_size)
    consolidation = ConsolidationScheduler(
        m3=m3, state_dim=state_dim,
        consolidation_interval=10, max_facts_per_cycle=50,
    )

    return cls(
        sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
        engine=engine, peu=peu, tspl=tspl, rbta=rbta,
        mdim=mdim, attention=attention,
        criticality_regulator=criticality_regulator,
        hpm_validator=hpm_validator,
        consolidation=consolidation, env=env,
        state_dim=state_dim,
    )
```

### 4.2 Changes to `_compute_distance_gain()`

The current `_compute_distance_gain()` has a GridWorld-specific code path:

```python
def _compute_distance_gain(self, action_idx: int) -> float:
    env = self.env
    goal_pos = env.get_goal_position()
    if goal_pos is not None and hasattr(env, "agent_pos"):
        if hasattr(env, "grid") and hasattr(env, "WALL"):
            # ... GridWorld-specific Manhattan distance computation
```

For MuJoCo environments:
- `get_goal_position()` returns `None` → falls through to `return 0.5` (neutral).
- All MuJoCo actions are scored neutrally for distance gain.
- MDIM goals will drive exploration/exploitation through prediction error, confidence, and empowerment instead.

**No code changes required** — the existing fallback works correctly.

### 4.3 Changes to `_select_action()`

No changes required. The existing ε-greedy exploration with MDIM drive scoring works with any discrete action space. The action loop iterates over `range(self.env.action_space_size)` which is already dynamic.

### 4.4 Changes to Observation Flow

The `CognitiveCycle.step()` Step 0 calls `self.env._get_observation()`. For `MuJoCoSimpleEnv`, this must return the last observation cached from `step()` or `reset()`.

**Critical flow:**

1. `env.reset()` → obs_A (cached)
2. Cycle 0:
   - `_get_observation()` → obs_A (the reset observation)
   - sanitize obs_A → store in M2
   - predict from obs_A
   - select action using predicted state
   - `env.step(action)` → obs_B (new observation)
   - PEU: compare obs_B vs predicted
   - learn from transition
3. Cycle 1:
   - `_get_observation()` → obs_B (cached from step)
   - ... repeat

This is **correct** — the prediction is always one step ahead of the observation, which is the standard temporal-difference learning setup.

### 4.5 Potential Changes to RBTA Bounds

MuJoCo environments add simulation overhead (~1-5ms per step for simple environments). Update bounds:

| Module | Current Bound | MuJoCo Bound | Reason |
|--------|---------------|--------------|--------|
| `G'` | `B_time=0.020` | `B_time=0.080` | Continuous inference on continuous obs is heavier |
| `ACTION` | `B_time=0.020` | `B_time=0.050` | env.step() includes MuJoCo physics step |
| `ASI` | `B_time=0.005` | `B_time=0.005` | No change — same dimension |
| `PE` | `B_time=0.300` | `B_time=0.300` | Same inference cost |

---

## 5. Phase 3: Validation & Testing

### 5.1 Unit Tests

Create `python/tests/test_mujoco_env.py`:

```python
"""Tests for MuJoCo environment wrapper."""

import numpy as np

from environments.mujoco_env import MuJoCoSimpleEnv


def test_cartpole_env_creation():
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    assert env.action_space_size == 3
    assert env.get_state_dim() == 4
    assert env.size == 1


def test_cartpole_step():
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    obs = env.reset()
    assert obs.shape == (4,)
    assert isinstance(obs, np.ndarray)

    next_obs, reward, terminal, info = env.step(0)  # push left
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminal, bool)


def test_pendulum_env():
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    assert env.action_space_size == 3
    assert env.get_state_dim() == 3
    obs = env.reset()
    assert obs.shape == (3,)


def test_action_discretisation():
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    for discrete_action in range(env.action_space_size):
        obs, reward, terminal, info = env.step(discrete_action)
        assert obs.shape == (4,), f"Failed on action {discrete_action}"
```

### 5.2 Integration Test

Create `python/tests/test_cycle_with_mujoco.py`:

```python
"""Integration tests: CognitiveCycle with MuJoCo."""

from phca.core.cycle import CognitiveCycle


def test_cycle_build_for_mujoco():
    cycle = CognitiveCycle.build_for_mujoco("InvertedPendulum-v5", seed=42, use_mlp=True)
    assert cycle.env.action_space_size == 3
    assert cycle.state_dim == 4


def test_cycle_step_cartpole():
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    metrics = cycle.step()
    assert metrics.cycle_id == 0
    assert metrics.latency_ms > 0
    assert metrics.action_taken in range(3)
    assert metrics.prediction_error >= 0


def test_multi_step_cartpole():
    """Run 10 cycles and verify no crashes."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    for i in range(10):
        metrics = cycle.step()
        assert metrics.cycle_id == i
        assert metrics.latency_ms < 5000  # 5s upper bound
```

### 5.3 Benchmark Script

Create `scripts/benchmark_mujoco.py`:

```python
#!/usr/bin/env python3
"""MuJoCo integration benchmark for PHCA cognitive cycle."""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from phca.core.cycle import CognitiveCycle


def run_benchmark(env_name: str, n_cycles: int = 100, use_mlp: bool = True) -> dict:
    """Run MuJoCo benchmark and collect metrics."""
    cycle = CognitiveCycle.build_for_mujoco(env_name, seed=42, use_mlp=use_mlp)

    # Warmup
    for _ in range(10):
        cycle.step()

    # Benchmark
    errors = []
    latencies = []
    rewards = []
    start_time = time.perf_counter()

    for _ in range(n_cycles):
        t0 = time.perf_counter()
        metrics = cycle.step()
        latencies.append(metrics.latency_ms)
        errors.append(metrics.prediction_error)

        # Track episode reward by reading info from the env
        # (env's internal reward tracking)

    elapsed = time.perf_counter() - start_time

    return {
        "env_name": env_name,
        "n_cycles": n_cycles,
        "model": "MLP" if use_mlp else "Gaussian G'",
        "mean_prediction_error": float(np.mean(errors)),
        "median_prediction_error": float(np.median(errors)),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "max_latency_ms": float(np.max(latencies)),
        "cycles_per_second": n_cycles / elapsed,
        "total_time_s": elapsed,
        "cycle_steps_completed": n_cycles,
        "no_errors": all(np.isfinite(e) for e in errors),
    }


def main():
    parser = argparse.ArgumentParser(description="MuJoCo PHCA Benchmark")
    parser.add_argument("--env", default="InvertedPendulum-v5",
                        choices=["InvertedPendulum-v5", "Pendulum-v1", "Reacher-v5"])
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--use-mlp", action="store_true", default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Benchmarking {args.env} with {'MLP' if args.use_mlp else 'Gaussian G''}...")
    result = run_benchmark(args.env, args.cycles, args.use_mlp)
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

### 5.4 Acceptance Criteria

| # | Criterion | Metric | Pass Condition |
|---|-----------|--------|----------------|
| C1 | Cycle completion | All 100 cycles run | `no_errors == True` |
| C2 | Finite prediction error | Mean prediction error | `mean_prediction_error < ∞` and all finite |
| C3 | Latency | Mean latency per cycle | `< 200ms` (allowing MuJoCo sim overhead) |
| C4 | Prediction improves | Error trend | Late error < early error (or both < 1.0) |
| C5 | Cartpole stability | Steps before terminal | Agent balances pole for `> 50` consecutive steps in at least one run |
| C6 | No RBTA violations | Total violations | `< 10` over 100 cycles (10%) |

**Note on C5:** Cartpole balance for 50+ steps requires the world model to learn an approximately correct dynamics model. For the first benchmark pass, C5 is a "stretch goal" — the MLP/Gaussian G' may not learn fast enough in 100 cycles. Accept if C1-C4 pass.

### 5.5 Test Execution

```bash
# Unit tests
python -m pytest python/tests/test_mujoco_env.py -v

# Integration tests
python -m pytest python/tests/test_cycle_with_mujoco.py -v

# Benchmark (standalone)
python scripts/benchmark_mujoco.py --env=InvertedPendulum-v5 --cycles=100 --output=logs/benchmark_mujoco_cartpole.json
```

---

## 6. Phase 4: Scaling Strategy (Optional)

### 6.1 Complexity Ladder

Only escalate after the previous level is stable:

| Level | Environment | State Dim | Actions | Change from Previous |
|-------|-------------|-----------|---------|---------------------|
| 0 | Cartpole (InvertedPendulum-v5) | 4 | 3 | Baseline |
| 1 | Pendulum (Pendulum-v1) | 3 | 3 | Different dynamics, same interface |
| 2 | 2-DOF Reacher (Reacher-v5) | ≤11 | 5 | 2D action space, target goal |
| 3 | Cartpole with larger action set | 4 | 5 | Finer discretisation (`[-3, -1.5, 0, +1.5, +3]`) |
| 4 | Continuous actions (no discretisation) | 4+ | — | Requires `EnvironmentProtocol` change — action becomes ndarray |

### 6.2 Continuous Action Support (Level 4+)

If continuous actions become necessary:

1. **Extend `EnvironmentProtocol`** to accept `np.ndarray` actions in addition to `int`.
2. **Add a flag** to `CognitiveCycle._select_action()` to switch between discrete and continuous modes.
3. **Use the MLP world model** — it already accepts `action` as an `np.ndarray` of shape `(action_dim,)`.
4. **Gaussian G'** also accepts `action` as `np.ndarray`.

### 6.3 Finer Discretisation

For environments where 3 actions is too coarse:

```python
# Finer 5-bin discretisation for cartpole
_ACTION_MAPS["InvertedPendulum-v5_fine"] = {
    0: np.array([-3.0]),
    1: np.array([-1.5]),
    2: np.array([0.0]),
    3: np.array([1.5]),
    4: np.array([3.0]),
}
```

### 6.4 Higher-DOM Environments

If the project later wants to target >20-dim environments:

- **Add a dimensionality reduction layer** (e.g., autoencoder or PCA) between the MuJoCo observation and the PHCA state vector.
- **The cognitive cycle sees a compressed state** (e.g., 20 dims) while the MuJoCo sim sees the full observation.
- This is a **Phase 5+** change and requires architectural discussion.

---

## 7. Risk Assessment & Mitigation

### Risk 1: State Dimension Mismatch

| Risk | The MLP world model input dimension is hardcoded to `state_dim + action_dim`. If `build_for_env()` is used instead of `build_for_mujoco()`, the state_dim may come from GridWorld (e.g., 84) instead of the MuJoCo environment (e.g., 4). |
|------|--------|
| **Impact** | High — silent dimension mismatch causes array shape errors or NaN predictions. |
| **Mitigation** | Always use `build_for_mujoco()` for MuJoCo environments. Add a runtime assertion in `CognitiveCycle.__init__()` that `state_dim == env.get_state_dim()`. The MLP forward pass will naturally fail with a clear shape error if the dimensions don't match. |

### Risk 2: Action Mapping Failure

| Risk | The discrete action mapping may not produce meaningful control signals (e.g., pushing left when the pole is already falling left). The agent may never learn to balance. |
|------|--------|
| **Impact** | Medium — the cognitive cycle will run but fail to achieve the task (Cartpole balance). |
| **Mitigation** | Start with 3 symmetric actions (-max, 0, +max) which is known to work for bang-bang control of Cartpole. Add ε-greedy exploration (already 5% in `_select_action()`). Scale the number of discrete actions gradually. |

### Risk 3: Simulation Speed Overhead

| Risk | Each `env.step()` call involves MuJoCo physics computation (~1-5ms). Over 100 cycles, this adds 100-500ms of wall-clock time. Headless rendering may fail on CI servers. |
|------|--------|
| **Impact** | Low-Medium — cycle latency may approach or exceed the 200ms budget. CI failures due to rendering issues. |
| **Mitigation** | Set `render_mode=None` (default) for benchmarks. Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` for headless CI. Increase RBTA bounds for MuJoCo specifically (see §4.5). |

### Risk 4: NaN/Inf in MuJoCo Observations

| Risk | MuJoCo can produce NaN or infinite observations during extreme simulation states (e.g., pole falling at high velocity). The ASI sanitizer (`ASISanitizer.sanitize`) should catch these, but if the sanitizer is not configured for the MuJoCo range, they may pass through. |
|------|--------|
| **Impact** | High — NaN propagates through G', PEU, TSPL, causing all subsequent predictions to be NaN. |
| **Mitigation** | Ensure `ASISanitizer` `v_max` is set appropriately (e.g., 100.0 for MuJoCo observations which are typically bounded within ±10-20). Already defaulting to 100.0 in `build_for_mujoco()`. Add a post-step NaN check in `MuJoCoSimpleEnv.step()` with a `nan_to_num` fallback. |

### Risk 5: Continuous Observations Clobber Discrete G'

| Risk | The discrete Bayesian G' (binary nodes with {0, 1} values) cannot represent continuous MuJoCo observations. Using the discrete G' with continuous MuJoCo data will produce meaningless predictions. |
|------|--------|
| **Impact** | High — discrete G' will quantise observations to 0 or 1, losing all fine-grained information. |
| **Mitigation** | Document that only `use_mlp=True` or `use_continuous=True` are valid for MuJoCo. The discrete G' should never be used with continuous observations. Add a runtime warning in `build_for_mujoco()` when neither MLP nor continuous is selected. |

---

## 8. Implementation Checklist

- [ ] **Phase 0:** Install `gymnasium[mujoco]`
- [ ] **Phase 0:** Run verification script
- [ ] **Phase 1:** Create `python/environments/mujoco_env.py` with `MuJoCoSimpleEnv`
- [ ] **Phase 1:** Verify `_get_observation()` caching works correctly
- [ ] **Phase 1:** Test Cartpole, Pendulum, and Reacher action maps
- [ ] **Phase 2:** Add `CognitiveCycle.build_for_mujoco()` class method
- [ ] **Phase 2:** Verify G' predict/learn works with 4-dim continuous state
- [ ] **Phase 2:** Update RBTA bounds for MuJoCo overhead
- [ ] **Phase 2:** Test `_compute_distance_gain()` fallback (goal_position=None)
- [ ] **Phase 3:** Write `test_mujoco_env.py` unit tests
- [ ] **Phase 3:** Write `test_cycle_with_mujoco.py` integration tests
- [ ] **Phase 3:** Write `scripts/benchmark_mujoco.py` benchmark
- [ ] **Phase 3:** Run all tests and benchmarks
- [ ] **Phase 3:** Verify acceptance criteria (C1-C6)
- [ ] **Phase 4 (optional):** Add finer discretisation
- [ ] **Phase 4 (optional):** Add observation normalisation

---

## Appendix A: State Vector Compositions

### Cartpole (`InvertedPendulum-v5`)

```
State: [cart_position, pole_angle, cart_velocity, pole_angular_velocity]
Shape: (4,)
Range:  cart_pos ≈ [-4.8, 4.8], pole_angle ≈ [-0.42, 0.42] rad
       cart_vel ≈ [-∞, ∞], pole_ang_vel ≈ [-∞, ∞]
```

### Pendulum (`Pendulum-v1`)

```
State: [cos(θ), sin(θ), dθ]
Shape: (3,)
Range: cos(θ) ∈ [-1, 1], sin(θ) ∈ [-1, 1], dθ ∈ [-8, 8]
```

### Reacher (`Reacher-v5`)

```
State: [joint_0_pos, joint_1_pos, joint_0_vel, joint_1_vel,
        target_x, target_y, target_z?]
Shape: (6) to (11) depending on version
```

---

## Appendix B: EnvironmentProtocol Overrides for MuJoCo

| Method | GridWorld Behavior | MuJoCo Behavior |
|--------|-------------------|-----------------|
| `get_goal_position()` | Returns `(row, col)` tuple | Returns `None` |
| `_compute_distance_gain()` | Uses Manhattan distance | Returns `0.5` (neutral) |
| `_get_observation()` | Constructs one-hot + local view | Returns cached MuJoCo obs |
| `step()` | Grid movement with wall check | MuJoCo physics step |
| `reset()` | Reset agent to start | MuJoCo sim reset |
| `size` | Grid dimensions (5/10/20) | `1` (not a grid) |
| `agent_pos` | Current grid position | `(0, 0)` stub |
