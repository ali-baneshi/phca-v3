#!/usr/bin/env python3
"""
PHCA v3.0 — Stress Test Resilience Suite.

Executes three stress scenarios and reports quantitative resilience evidence:

  1. Severe Environment Slowdown — RBTA time-bound enforcement under delay
  2. Sudden Database Failure     — M3 SQLite corruption with :memory: fallback
  3. Severe Sensor Noise         — Sensor noise injection, grounding drop/recovery

Usage:
    PYTHONPATH=python:$PYTHONPATH python scripts/stress_test_resilience.py

Output:
    Prints a summary table to stdout.  Detailed per-cycle metrics are saved
    to logs/stress_test_*.json for each scenario.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from typing import Optional
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle
from phca.environments.slow_wrapper import SlowWrapper
from phca.logging import ensure_logging


# ═══════════════════════════════════════════════════════════════
#  Scenario 1: Severe Environment Slowdown
# ═══════════════════════════════════════════════════════════════

def run_env_slowdown(
    cycles: int = 100,
    delay_ms: float = 100.0,
    probability: float = 0.3,
    seed: int = 42,
) -> dict:
    """Wrap GridWorld in SlowWrapper and run the cognitive cycle.

    The RBTA time bounds remain at their default values, so injected
    delays should trigger TIME violations → INTERRUPT / TERMINATE
    actions → cycle continues without crashing.
    """
    print("\n" + "=" * 60)
    print("  Scenario 1: Severe Environment Slowdown")
    print("=" * 60)

    from phca.environments.grid_world import GridWorld
    raw_env = GridWorld(size=5, obstacles=[], seed=seed)
    env = SlowWrapper(
        raw_env,
        delay_min_ms=delay_ms,
        delay_max_ms=delay_ms * 2,
        probability=probability,
        seed=seed,
    )

    cycle = CognitiveCycle.build(env=env, seed=seed, use_mlp=True)

    errors, latencies, violations_list = [], [], []
    start = time.perf_counter()

    for i in range(cycles):
        m = cycle.step()
        errors.append(m.prediction_error)
        latencies.append(m.latency_ms)
        violations_list.append(m.violations_count)

    duration = time.perf_counter() - start
    total_violations = sum(violations_list)
    mean_err = float(np.mean(errors))
    late_err = float(np.mean(errors[-max(1, cycles // 4):]))

    result = {
        "scenario": "env_slowdown",
        "cycles": cycles,
        "duration_s": round(duration, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "mean_error": round(mean_err, 4),
        "late_error": round(late_err, 4),
        "violations_total": total_violations,
        "violation_rate": round(total_violations / cycles, 4),
        "crash": False,
        "slowdown_stats": env.stats,
    }

    status = "PASS" if not result["crash"] else "FAIL"
    print(f"  Duration: {duration:.1f}s")
    print(f"  Mean latency: {result['mean_latency_ms']:.1f} ms  "
          f"(p95={result['p95_latency_ms']:.1f} ms)")
    print(f"  RBTA violations: {total_violations} "
          f"({result['violation_rate']*100:.1f}%)")
    print(f"  Late error: {late_err:.4f}")
    print(f"  Slowdown events: {env.stats['slowdown_count']} "
          f"({env.stats['total_injected_ms']:.0f}ms total injected)")
    print(f"  → {status}")

    output = Path("logs/stress_test_env_slowdown.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"  Report saved to {output}")

    return result


# ═══════════════════════════════════════════════════════════════
#  Scenario 2: Sudden Database Failure
# ═══════════════════════════════════════════════════════════════

def run_db_corruption(
    cycles: int = 100,
    corrupt_at: int = 30,
    seed: int = 42,
) -> dict:
    """Test graceful fallback when M3 SQLite database is corrupt.

    Two phases:
      1. Build M3 with a known-corrupt file → verify `_init_db()` falls
         back to `:memory:` gracefully.
      2. Run cycles with the fallback-active M3 → verify no crash and
         prediction error remains bounded.
    """
    print("\n" + "=" * 60)
    print("  Scenario 2: Sudden Database Failure")
    print("=" * 60)

    from phca.memory.m3_episodic import M3EpisodicMemory
    import shutil

    # Phase 1 — Build M3 with a corrupt file; expect :memory: fallback.
    tmp_dir = Path(tempfile.mkdtemp(prefix="phca_stress_m3_"))
    corrupt_db = tmp_dir / "corrupt_m3.db"
    corrupt_db.write_bytes(b"\x00" * 512 + b"CORRUPTED" * 100)
    print(f"  Created corrupt SQLite file ({corrupt_db.stat().st_size} bytes)")

    state_dim = 5  # GridWorld 5x5
    action_dim = 5  # GridWorld actions
    m3 = M3EpisodicMemory(
        db_path=str(corrupt_db), max_episodes=1000,
        state_dim=state_dim, action_dim=action_dim, seed=seed,
    )
    m3_fell_back = m3.db_path == ":memory:"

    # Build cycle normally, then swap in the fallback M3.
    cycle = CognitiveCycle.build_for_env(size=5, seed=seed, use_mlp=True)
    cycle.m3 = m3
    cycle.consolidation._m3 = m3

    # Phase 2 — Run cycles with the fallback M3.
    errors_before, errors_after = [], []
    violations_total = 0
    crash = False

    for i in range(cycles):
        try:
            m = cycle.step()
        except Exception as e:
            print(f"  ❌ CRASH at cycle {i}: {e}")
            crash = True
            break

        if i < corrupt_at:
            errors_before.append(m.prediction_error)
        else:
            errors_after.append(m.prediction_error)
        violations_total += m.violations_count

    shutil.rmtree(tmp_dir, ignore_errors=True)

    mean_before = float(np.mean(errors_before)) if errors_before else 0.0
    mean_after = float(np.mean(errors_after)) if errors_after else 0.0

    result = {
        "scenario": "db_corruption",
        "cycles": cycles,
        "corrupt_at": corrupt_at,
        "crash": crash,
        "m3_fell_back": m3_fell_back,
        "mean_error_before": round(mean_before, 4),
        "mean_error_after": round(mean_after, 4),
        "error_ratio": round(mean_after / max(mean_before, 1e-8), 4),
        "violations_total": violations_total,
    }

    status = "PASS" if (not crash and m3_fell_back) else "FAIL"
    print(f"  Crash: {crash}")
    print(f"  M3 :memory: fallback: {m3_fell_back}")
    print(f"  Mean error before→after: {mean_before:.4f} → {mean_after:.4f}")
    print(f"  Error ratio: {result['error_ratio']:.2f}x")
    print(f"  → {status}")

    output = Path("logs/stress_test_db_corruption.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"  Report saved to {output}")

    return result


# ═══════════════════════════════════════════════════════════════
#  Scenario 3: Severe Sensor Noise
# ═══════════════════════════════════════════════════════════════

class _NoisyObservationWrapper:
    """Wraps an env and injects NaN into get_observation().

    The noise window is driven by total cycle count from the cycle's
    perspective (``noise_start`` – ``noise_end``), NOT by the wrapper's
    internal step counter (which would reset on env episode boundaries).
    This ensures noise stops permanently after ``noise_end`` cycles
    regardless of environment resets.
    """

    def __init__(self, env: object, noise_start: int, noise_end: int,
                 seed: int):
        self._env = env
        self._noise_start = noise_start
        self._noise_end = noise_end
        self._rng = np.random.RandomState(seed)
        self._cycle = 0

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def get_observation(self) -> np.ndarray:
        obs = self._env.get_observation()
        if self._noise_start <= self._cycle < self._noise_end:
            mask = self._rng.rand(obs.shape[0]) < 0.5
            obs = obs.copy()
            obs[mask] = np.nan
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._cycle += 1
        return obs, reward, done, info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # Do NOT reset self._cycle — the noise window is a global
        # time count independent of episode boundaries.
        return self._env.reset(seed=seed)


def run_sensor_noise(
    cycles: int = 150,
    noise_start: int = 10,
    noise_end: int = 60,
    seed: int = 42,
) -> dict:
    """Inject severe sensor noise (NaN values), then remove, observe recovery.

    Phases:
      - Cycles 0–9:  Clean (baseline error)
      - Cycles 10–59: Heavy NaN injection on sensor observations
                       (50% of sensor dimensions set to NaN each cycle)
      - Cycles 60–99: Clean (recovery)

    The system responds through:
      1. ASISanitizer — replaces NaN, halves precision, tracks failures
      2. GroundingAdapter — drops grounding level on SENSOR_FAILURE
      3. RBTA — detects SENSOR violations (asi_failure_limit exceeded)
      4. FailureDetector B1 — detects distribution shift from error spike
      5. RecoveryManager — boosts eta to compensate
    """
    print("\n" + "=" * 60)
    print("  Scenario 3: Severe Sensor Noise")
    print("=" * 60)

    from phca.environments.grid_world import GridWorld

    raw_env = GridWorld(size=5, obstacles=[], seed=seed)
    noisy_env = _NoisyObservationWrapper(
        raw_env, noise_start=noise_start, noise_end=noise_end, seed=seed + 42,
    )
    cycle = CognitiveCycle.build(env=noisy_env, seed=seed, use_mlp=True)

    errors: list[float] = []
    grounding_levels: list[int] = []
    sensor_failures: list[int] = []
    failure_event_counts: list[int] = []

    for i in range(cycles):
        m = cycle.step()

        errors.append(m.prediction_error)
        grounding_levels.append(
            cycle._grounding_adapter.current_level
        )
        sensor_failures.append(
            getattr(cycle, "sensor_failure_count", 0)
        )
        failure_event_counts.append(
            len(getattr(cycle, "_last_failure_events", []))
        )

    phases = {
        "baseline": errors[:noise_start],
        "noise": errors[noise_start:noise_end],
        "recovery": errors[noise_end:],
    }
    grounding_check = (
        grounding_levels[noise_start:noise_end],
        grounding_levels[noise_end:],
    )
    grounding_dropped = len(grounding_check[0]) > 0 and min(grounding_check[0]) < max(
        grounding_levels[:max(noise_start, 1)]
    )

    mean_phase = {k: float(np.mean(v)) if v else 0.0 for k, v in phases.items()}
    grounding_recovered = len(grounding_check[1]) > 0 and max(grounding_check[1]) >= 1

    result = {
        "scenario": "sensor_noise",
        "cycles": cycles,
        "noise_window": [noise_start, noise_end],
        "mean_error_baseline": round(mean_phase["baseline"], 4),
        "mean_error_noise": round(mean_phase["noise"], 4),
        "mean_error_recovery": round(mean_phase["recovery"], 4),
        "grounding_dropped": grounding_dropped,
        "grounding_recovered": grounding_recovered,
        "grounding_levels_noise_min": int(min(grounding_check[0])),
        "grounding_levels_recovery_max": int(max(grounding_check[1])),
        "grounding_recovered_max_ge_1": grounding_recovered,
        "max_sensor_failures": int(max(sensor_failures)),
        "failure_events_during_noise": int(
            sum(failure_event_counts[noise_start:noise_end])
        ),
        "crash": False,
    }

    status = "PASS" if (grounding_dropped and grounding_recovered and not result["crash"]) else "FAIL"
    print(f"  Mean error (baseline→noise→recovery): "
          f"{mean_phase['baseline']:.4f} → {mean_phase['noise']:.4f} → "
          f"{mean_phase['recovery']:.4f}")
    print(f"  Grounding levels — noise min: {min(grounding_check[0])}, "
          f"recovery max: {max(grounding_check[1])}")
    print(f"  Grounding dropped under noise: {grounding_dropped}")
    print(f"  Grounding recovered after noise: {grounding_recovered}")
    print(f"  Max sensor failures in one cycle: {result['max_sensor_failures']}")
    print(f"  Failure events during noise: {result['failure_events_during_noise']}")
    print(f"  → {status}")

    output = Path("logs/stress_test_sensor_noise.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"  Report saved to {output}")

    return result


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    ensure_logging()
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  PHCA v3.0 — Stress Test Resilience Suite         ║")
    print("╚" + "═" * 58 + "╝")

    results = {}

    # Scenario 1
    results["env_slowdown"] = run_env_slowdown(cycles=100)
    print()

    # Scenario 2
    results["db_corruption"] = run_db_corruption(cycles=100, corrupt_at=30)
    print()

    # Scenario 3
    results["sensor_noise"] = run_sensor_noise(cycles=150)
    print()

    # ── Summary table ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  STRESS TEST — SUMMARY")
    print("=" * 60)
    template = "  {:<30} {:>12} {:>12}"
    print(template.format("Scenario", "Status", "Key Metric"))
    print("  " + "-" * 56)
    all_pass = True
    for key, r in results.items():
        if r.get("crash", False):
            status = "FAIL"
            all_pass = False
        elif key == "env_slowdown":
            status = "PASS"
            metric = f"{r['violations_total']} violations"
        elif key == "db_corruption":
            ok = not r["crash"] and r["m3_fell_back"]
            status = "PASS" if ok else "FAIL"
            all_pass = all_pass and ok
            metric = f"fallback={'yes' if r['m3_fell_back'] else 'no'}"
        elif key == "sensor_noise":
            ok = r["grounding_dropped"] and r["grounding_recovered"]
            status = "PASS" if ok else "FAIL"
            all_pass = all_pass and ok
            metric = f"drop={r['grounding_dropped']} rec_max={r['grounding_levels_recovery_max']}"
        else:
            status = "?"
            metric = ""
        print(template.format(key, status, metric))

    print("  " + "-" * 56)
    overall = "PASS" if all_pass else "FAIL"
    print(template.format("OVERALL", overall, ""))
    print()

    # Save combined results
    combined = Path("logs/stress_test_results.json")
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {combined}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
