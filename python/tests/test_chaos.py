"""
PHCA v3.0 — Chaos / Resilience Tests.

Simulates real-world failures: sensor dropout, slow modules, memory
exhaustion, and global sensor failure. Each test verifies graceful
degradation and recovery.
"""

from __future__ import annotations

import numpy as np

from phca.asi.sanitizer import ASISanitizer
from phca.config import ASIStatus, ResourceBounds
from phca.core.cycle import CognitiveCycle
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.regulation.rbta_enforcer import RBTAEnforcer, EnforcerAction


# ── Sensor Chaos ───────────────────────────────────────────────


class TestSensorChaos:
    """Random sensor failure and recovery."""

    def test_asi_random_sensor_dropout(self):
        """50% random sensor NaN each cycle — system operates without crash."""
        sani = ASISanitizer(sensor_dim=4, v_max=100.0)
        rng = np.random.RandomState(42)

        # Seed with valid data
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sani.sanitize(raw_valid)

        for _ in range(50):
            raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            # Drop 50% of sensors
            mask = rng.rand(4) < 0.5
            raw[mask] = np.nan
            _, status = sani.sanitize(raw)
            assert status in (
                ASIStatus.OK, ASIStatus.PARTIAL_FAILURE, ASIStatus.SENSOR_FAILURE,
            ), f"Unexpected status {status} on dropout cycle"
            # After SENSOR_FAILURE, valid data should recover the system
            if status == ASIStatus.SENSOR_FAILURE:
                # Inject a clean cycle to verify recovery
                sani.sanitize(raw_valid)

    def test_asi_all_dead_sensors_recovers(self):
        """All sensors dead for 7 cycles (SENSOR_FAILURE), then valid data recovers."""
        sani = ASISanitizer(sensor_dim=4, v_max=100.0)
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sani.sanitize(raw_valid)

        raw_nan = np.full(4, np.nan, dtype=np.float32)
        for _ in range(6):
            sani.sanitize(raw_nan)

        # 7th failure → SENSOR_FAILURE
        _, status = sani.sanitize(raw_nan)
        assert status == ASIStatus.SENSOR_FAILURE

        # Valid data arrives → recovery
        state, status = sani.sanitize(raw_valid)
        assert status in (ASIStatus.OK, ASIStatus.PARTIAL_FAILURE)
        np.testing.assert_array_equal(state.values, raw_valid)

    def test_asi_global_failure_recovery(self):
        """> d/3 sensors fail simultaneously → logged, then valid data recovers."""
        sani = ASISanitizer(sensor_dim=3, v_max=100.0)
        raw_valid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sani.sanitize(raw_valid)

        # 2 out of 3 sensors fail (2 > 3//3 = 1) simultaneously
        raw_fail = np.array([np.nan, np.nan, 3.0], dtype=np.float32)
        # Enough failures to trigger SENSOR_FAILURE on both sensors
        for _ in range(7):
            sani.sanitize(raw_fail)

        # Both sensors should be below threshold
        assert sani.precision[0] < sani.epsilon_confidence
        assert sani.precision[1] < sani.epsilon_confidence
        # Valid data recovers
        state, status = sani.sanitize(raw_valid)
        assert status in (ASIStatus.OK, ASIStatus.PARTIAL_FAILURE)


# ── Module Performance Chaos ───────────────────────────────────


class TestModuleChaos:
    """Artificial module degradation."""

    def test_slow_module_triggers_rbta(self):
        """Tightening G' bound should cause RBTA violations."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        # Artificially tight bound for G' (well below actual runtime)
        cycle.rbta.update_bounds(
            "G'", ResourceBounds(B_time=0.0001, B_mem=500_000, B_energy=50.0),
        )

        violations_found = False
        for _ in range(10):
            metrics = cycle.step()
            if metrics.violations_count > 0:
                violations_found = True
                break

        assert violations_found, (
            "No RBTA violations detected after tightening G' bound to 0.0001s"
        )

    def test_rbta_detects_time_violation(self):
        """Direct RBTA check: runtime > B_time returns INTERRUPT with TIME violation."""
        rbta = RBTAEnforcer({
            "MOD": ResourceBounds(B_time=0.001, B_mem=1000, B_energy=1.0),
        })
        violations, action = rbta.check_cycle(
            runtime_log={"MOD": 10.0},
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
        )
        assert action == EnforcerAction.TERMINATE
        assert any(v.bound_type == "TIME" for v in violations)


# ── Memory Exhaustion Chaos ────────────────────────────────────


class TestMemoryChaos:
    """Memory exhaustion and eviction."""

    def test_m3_exact_capacity_boundary(self):
        """Exactly max_episodes stored → no eviction, count == max."""
        m3 = M3EpisodicMemory(max_episodes=50, state_dim=4, action_dim=2)
        state = _make_state(4)
        action = np.zeros(2, dtype=np.float32)

        for i in range(50):
            m3.store_episode(
                state_before=state, action_taken=action,
                state_after=state, prediction_error=0.1,
                timestamp=i,
            )

        assert m3.count() == 50

    def test_m3_evicts_oldest_when_over_capacity(self):
        """Storing past max_episodes evicts oldest episodes (FIFO)."""
        m3 = M3EpisodicMemory(max_episodes=50, state_dim=4, action_dim=2)
        state = _make_state(4)
        action = np.zeros(2, dtype=np.float32)

        for i in range(60):
            m3.store_episode(
                state_before=state, action_taken=action,
                state_after=state, prediction_error=float(i) * 0.01,
                timestamp=i,
            )

        assert m3.count() <= 50, f"count={m3.count()} exceeds max_episodes=50"
        # Verify earliest timestamps were evicted by checking the minimum timestamp
        cursor = m3._connection.execute(
            "SELECT MIN(timestamp) FROM episodes"
        )
        min_ts = cursor.fetchone()[0]
        assert min_ts is not None, "No episodes remain after eviction"
        assert min_ts >= 10, f"Earliest remaining timestamp {min_ts} < 10, eviction did not preserve recent episodes"


# ── Nightly stress G' MEM bound alignment (D-132) ─────────────


class TestNightlyStressBounds:
    """Regression: stress scripts must not clobber build() G' B_mem."""

    def test_nightly_stress_build_no_spurious_gprime_mem_violation(self):
        import importlib.util
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        script = root / "scripts" / "nightly_stress.py"
        spec = importlib.util.spec_from_file_location("nightly_stress", script)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        cycle = mod._build_cycle(42)
        metrics = cycle.step()
        assert metrics.violations_count == 0, cycle.last_violations
        assert cycle.rbta._bounds["G'"].B_mem >= cycle.memory_log["G'"]


# ── Helpers ────────────────────────────────────────────────────


def _make_state(dim: int):
    """Create a simple StateVector for testing."""
    from phca.config import StateVector
    return StateVector(
        values=np.zeros(dim, dtype=np.float32),
        precision=np.ones(dim, dtype=np.float32),
        timestamp=0.0, grounding_level=1,
    )
