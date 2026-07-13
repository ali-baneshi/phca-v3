"""
PHCA v3.0 — Edge Case Tests.

Tests boundary conditions: empty vectors, wrong-sized inputs, full buffers,
empty stores, and RBTA classification boundaries.

Cross-ref: silent_failure_and_fallacy_report.md §4
"""

from __future__ import annotations

import numpy as np
import pytest

from phca.asi.sanitizer import ASISanitizer
from phca.config import ASIStatus, StateVector
from phca.consolidation.scheduler import ConsolidationScheduler
from phca.core.cycle import CognitiveCycle
from phca.memory.m2_working import M2WorkingMemory, Chunk
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.regulation.rbta_enforcer import RBTAEnforcer, EnforcerAction


# ── ASI Edge Cases ─────────────────────────────────────────────


class TestASIEdgeCases:
    """ASI sanitizer boundary conditions."""

    def test_asi_empty_vector(self):
        """sensor_dim=0 should accept an empty float32 array."""
        sani = ASISanitizer(sensor_dim=0, v_max=100.0)
        raw = np.array([], dtype=np.float32)
        state, status = sani.sanitize(raw)
        assert status == ASIStatus.OK
        assert state.values.shape == (0,)

    def test_asi_wrong_size_vector(self):
        """Shape mismatch between raw and sensor_dim should raise AssertionError."""
        sani = ASISanitizer(sensor_dim=4, v_max=100.0)
        raw = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(AssertionError):
            sani.sanitize(raw)

    def test_asi_all_nan_triggers_sensor_failure(self):
        """All sensors NaN for 7 cycles triggers SENSOR_FAILURE."""
        sani = ASISanitizer(sensor_dim=3, v_max=100.0)
        raw_valid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sani.sanitize(raw_valid)

        raw_nan = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        for _ in range(6):
            _, status = sani.sanitize(raw_nan)
            assert status == ASIStatus.PARTIAL_FAILURE
        # 7th failure → SENSOR_FAILURE
        _, status = sani.sanitize(raw_nan)
        assert status == ASIStatus.SENSOR_FAILURE


# ── M2 Edge Cases ──────────────────────────────────────────────


class TestM2EdgeCases:
    """M2 Working Memory boundary conditions."""

    def test_m2_evicts_lowest_salience(self):
        """When M2 is full, the chunk with lowest salience is evicted."""
        m2 = M2WorkingMemory(capacity=7)
        state = _make_state(4)

        # Fill M2 with ascending salience
        for i in range(7):
            sal = 0.1 + i * 0.1  # 0.1, 0.2, ..., 0.7
            m2.write(state, salience=sal)

        assert len(m2.chunks) == 7
        evicted_id = m2.chunks[0].chunk_id  # lowest salience

        # Write one more with high salience
        m2.write(state, salience=0.9)

        assert len(m2.chunks) == 7
        # Evicted chunk should be gone
        remaining_ids = {c.chunk_id for c in m2.chunks}
        assert evicted_id not in remaining_ids

    def test_m2_write_returns_chunk(self):
        """M2.write() should return the created Chunk with assigned chunk_id."""
        m2 = M2WorkingMemory(capacity=7)
        state = _make_state(4)
        chunk = m2.write(state, salience=0.5)
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_id == 0
        assert chunk.salience == 0.5
        assert chunk.age == 1  # age=0 at creation, incremented to 1 during write()


# ── M3 Edge Cases ──────────────────────────────────────────────


class TestM3EdgeCases:
    """M3 Episodic Memory boundary conditions."""

    def test_m3_empty_consolidation(self):
        """Consolidation with no episodes should log warning, not crash."""
        m3 = M3EpisodicMemory(state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(
            m3=m3, state_dim=4,
            consolidation_interval=10, max_facts_per_cycle=50,
        )
        report = cs.step(cycle_count=1, force=True)
        assert report.success
        assert report.episodes_processed == 0
        assert report.facts_generated == 0

    def test_m3_fill_to_capacity_then_evict(self):
        """Storing past max_episodes should FIFO-evict oldest."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        state = _make_state(4)
        action = np.zeros(2, dtype=np.float32)

        for i in range(110):
            m3.store_episode(
                state_before=state, action_taken=action,
                state_after=state, prediction_error=0.1,
                timestamp=i,
            )

        total = m3.count()
        assert total <= 100


# ── RBTA Edge Cases ────────────────────────────────────────────


class TestRBTABoundaries:
    """RBTA enforcer classification boundaries."""

    def test_rbta_no_violations_continue(self, default_resource_bounds):
        """Zero violations → EnforcerAction.CONTINUE."""
        rbta = RBTAEnforcer(default_resource_bounds)
        violations, action = rbta.check_cycle(
            runtime_log={"ASI": 0.001}, memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
        )
        assert action == EnforcerAction.CONTINUE
        assert len(violations) == 0

    def test_rbta_single_violation_interrupt(self, default_resource_bounds):
        """One violation → EnforcerAction.INTERRUPT."""
        rbta = RBTAEnforcer(default_resource_bounds)
        violations, action = rbta.check_cycle(
            runtime_log={"ASI": 0.003}, memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
        )
        assert action == EnforcerAction.INTERRUPT
        assert len(violations) == 1

    def test_rbta_three_violations_terminate(self, default_resource_bounds):
        """Three violations → EnforcerAction.TERMINATE."""
        rbta = RBTAEnforcer(default_resource_bounds)
        violations, action = rbta.check_cycle(
            runtime_log={"ASI": 10.0},
            memory_log={"ASI": 999_999_999},
            energy_log={"ASI": 999_999_999},
            belief_entropies={}, sensor_failure_count=0,
        )
        assert action == EnforcerAction.TERMINATE
        assert len(violations) >= 3

    def test_rbta_missing_module_id_ignored(self, default_resource_bounds):
        """Unknown module in logs should be ignored (no crash)."""
        rbta = RBTAEnforcer(default_resource_bounds)
        violations, action = rbta.check_cycle(
            runtime_log={"BOGUS_MODULE": 999.0}, memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
        )
        # BOGUS_MODULE is not in bounds → no violation
        violations_same, _ = rbta.check_cycle(
            runtime_log={}, memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
        )
        assert len(violations) == len(violations_same)


# ── Cycle Edge Cases ───────────────────────────────────────────


class TestCycleEdgeCases:
    """Cognitive cycle boundary conditions."""

    def test_cycle_wrong_sanitizer_dim(self):
        """Sanitizer with wrong sensor_dim should crash on step."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        # Replace sanitizer with wrong dimension
        cycle.sanitizer = ASISanitizer(sensor_dim=4, v_max=100.0)
        with pytest.raises(Exception):
            cycle.step()

    def test_cycle_zero_cycles(self):
        """run(n_cycles=0) should return safe defaults without error."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=0)
        assert summary["total_cycles"] == 0
        assert summary["total_violations"] == 0
        assert summary["skill_compiled"] is False


# ── Helpers ────────────────────────────────────────────────────


def _make_state(dim: int) -> StateVector:
    """Create a simple StateVector for testing."""
    return StateVector(
        values=np.zeros(dim, dtype=np.float32),
        precision=np.ones(dim, dtype=np.float32),
        timestamp=0.0, grounding_level=1,
    )
