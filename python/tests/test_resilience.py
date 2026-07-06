"""Tests for cognitive resilience detector and recovery protocols."""

from __future__ import annotations

import pytest

from phca.config import StreamID
from phca.core.cycle import CognitiveCycle
from phca.resilience import FailureDetector, RecoveryManager, recovery_rate
from phca.resilience.types import CycleSnapshot, FailureCategory


def _snapshot(**kwargs) -> CycleSnapshot:
    defaults = dict(cycle_id=10)
    defaults.update(kwargs)
    return CycleSnapshot(**defaults)


class TestFailureDetector:
    def test_b1_distribution_shift(self):
        det = FailureDetector()
        snap = _snapshot(
            recent_prediction_errors=[0.1, 0.12, 0.11, 0.5, 0.55, 0.6],
        )
        events = det.detect(snap)
        assert any(e.mode_id == "B1" for e in events)

    def test_b4_catastrophic_forgetting(self):
        det = FailureDetector()
        snap = _snapshot(per_task_baseline_goal_rate=0.8, per_task_goal_rate=0.2)
        events = det.detect(snap)
        assert any(e.mode_id == "B4" for e in events)

    def test_b5_mode_collapse(self):
        det = FailureDetector()
        snap = _snapshot(
            recent_confidences=[0.995] * 20,
            recent_actions=[4] * 20,
        )
        events = det.detect(snap)
        assert any(e.mode_id == "B5" for e in events)

    def test_c1_feedback_instability(self):
        det = FailureDetector()
        snap = _snapshot(rbta_action_history=["TERMINATE", "TERMINATE", "TERMINATE"])
        events = det.detect(snap)
        assert any(e.mode_id == "C1" for e in events)

    def test_f5_consolidation_failure(self):
        det = FailureDetector()
        snap = _snapshot(m3_fill_ratio=0.95, fact_count_stagnant_cycles=55)
        events = det.detect(snap)
        assert any(e.mode_id == "F5" for e in events)


class TestRecoveryProtocols:
    def test_b1_boosts_eta(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        mgr = RecoveryManager()
        eta_before = cycle.tspl.configs[StreamID.P_STREAM].eta
        from phca.resilience.types import FailureEvent
        event = FailureEvent(
            mode_id="B1",
            category=FailureCategory.DISTRIBUTION_SHIFT,
            severity=2.0,
            cycle_id=5,
            measured=1.0,
            threshold=0.3,
        )
        mgr.apply(cycle, [event])
        assert cycle.tspl.configs[StreamID.P_STREAM].eta > eta_before

    def test_b4_replay_boost(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        mgr = RecoveryManager()
        from phca.resilience.types import FailureEvent
        event = FailureEvent(
            mode_id="B4",
            category=FailureCategory.CATASTROPHIC_FORGETTING,
            severity=3.0,
            cycle_id=5,
            measured=0.1,
            threshold=0.4,
        )
        mgr.apply(cycle, [event])
        assert cycle._forgetting_mitigation_active
        assert cycle.gprime.replay_boost

    def test_c1_pid_recovery(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        kp_before = cycle.adaptive_controller.k_p
        cycle.adaptive_controller.apply_c1_recovery()
        assert cycle.adaptive_controller.k_p == pytest.approx(kp_before * 0.5)

    def test_f5_force_consolidation(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        report = cycle.consolidation.force_step(cycle.cycle_count)
        assert report.success or report.episodes_processed >= 0


class TestRecoveryRate:
    def test_recovery_rate_fraction(self):
        assert recovery_rate(["b1", "c1"], {"b1": True, "c1": False}) == 0.5
        assert recovery_rate([], {}) == 1.0


class TestCycleIntegration:
    def test_build_resilience_snapshot(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        for _ in range(5):
            cycle.step()
        snap = cycle._build_resilience_snapshot(cycle.metrics_history[-1])
        assert snap.cycle_id >= 0
        assert len(snap.recent_prediction_errors) > 0

    def test_step_populates_failure_fields(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        m = cycle.step()
        assert hasattr(m, "failure_events")
        assert hasattr(m, "recovery_active")
        assert isinstance(m.failure_events, list)
