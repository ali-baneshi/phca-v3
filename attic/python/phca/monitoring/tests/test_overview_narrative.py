"""Unit tests for overview_narrative helpers."""
from __future__ import annotations

from collections import deque

import numpy as np

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.overview_narrative import (
    _action_status_line,
    _overview_evidence_line,
    _overview_goal_intent_line,
    _overview_moment_flags,
    _overview_outcome_line,
)


def _frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    for k, v in kwargs.items():
        setattr(f, k, v)
    return f


def test_overview_moment_flags_spike():
    f = _frame(cycle_id=1, prediction_error=40.0, env_kind="grid")
    flags = _overview_moment_flags(f, deque([1.0, 2.0]))
    assert flags.get("spike") is True


def test_overview_goal_intent_line_includes_drive():
    f = _frame(
        cycle_id=2,
        active_drive_id=3,
        action_rationale={"goal_id": 3},
    )
    flags = _overview_moment_flags(f, deque())
    line = _overview_goal_intent_line(f, flags)
    assert "D3" in line or "drive" in line.lower()


def test_overview_evidence_and_outcome_non_empty():
    f = _frame(
        cycle_id=5,
        prediction_error=2.0,
        prediction_confidence=0.7,
        env_kind="mujoco_rgb",
        obs_vector=np.zeros(11, dtype=np.float32),
    )
    err_hist = deque([2.0, 2.5, 1.8])
    flags = _overview_moment_flags(f, err_hist)
    evidence = _overview_evidence_line(f, flags, err_hist, deque())
    outcome = _overview_outcome_line(f, flags, err_hist, deque())
    assert len(evidence) > 0
    assert len(outcome) > 0


def test_action_status_line_replay_rollouts():
    f = _frame(cycle_id=0, env_kind="mujoco_rgb")
    line = _action_status_line(
        f,
        [0.2, 0.3],
        1,
        replay=True,
    )
    assert "rollouts" in line.lower()
