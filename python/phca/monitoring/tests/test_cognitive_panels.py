"""Unit tests for cognitive_panels pure helpers."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from phca.monitoring.cognitive_panels import (
    LEARN_MS_MIN,
    action_status_extras,
    apply_decision_shift,
    belief_reference,
    build_moment_series,
    cognitive_moment,
    count_moments,
    execution_dominant_phase,
    flow_action_link_line,
    flow_near_bound_module,
    flow_status_extras,
    overview_spike,
)
from phca.monitoring.observability import ObservabilityFrame


def _frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.env_kind = kwargs.get("env_kind", "mujoco_rgb")
    f.prediction_error = kwargs.get("prediction_error", 0.1)
    f.module_timings = kwargs.get("module_timings", {
        "prediction": 1.0, "action_selection": 2.0, "gprime_learn": 6.0,
    })
    f.obs_vector = kwargs.get("obs_vector", np.zeros(4, dtype=np.float32))
    f.goal_ref = kwargs.get("goal_ref")
    f.sanitized_state = kwargs.get("sanitized_state")
    f.per_dim_peu = kwargs.get("per_dim_peu")
    f.action_rationale = kwargs.get("action_rationale", {})
    f.rbta_bounds = kwargs.get("rbta_bounds", {})
    return f


def test_execution_dominant_phase_learn():
    f = _frame()
    assert execution_dominant_phase(f) == "Learn"


def test_cognitive_moment_learn_burst():
    f = _frame()
    hist = deque([0.1, 0.1], maxlen=10)
    m = cognitive_moment(f, hist)
    assert m["learn_burst"] is True
    assert m["learn_ms"] >= LEARN_MS_MIN
    assert m["dominant_phase"] == "Learn"


def test_cognitive_moment_spike_mujoco():
    f = _frame(env_kind="mujoco_rgb", prediction_error=12.0)
    hist = deque([0.1, 5.0], maxlen=10)
    m = cognitive_moment(f, hist)
    assert m["spike"] is True


def test_overview_spike_parity():
    from phca.monitoring.qt_dashboard import _overview_spike

    hist = deque([0.1, 5.0], maxlen=10)
    assert overview_spike(hist, 12.0, "mujoco_rgb") == _overview_spike(hist, 12.0, "mujoco_rgb")
    hist2 = deque([0.5, 1.0], maxlen=10)
    assert overview_spike(hist2, 3.0, "grid") == _overview_spike(hist2, 3.0, "grid")


def test_flow_action_link_line_helper():
    f = _frame(module_timings={"gprime_learn": 6.0, "action_selection": 2.0})
    f.action_rationale = {"explored": False}
    line = flow_action_link_line(f)
    assert "bottleneck=" in line
    assert "EXPLOIT" in line


def test_belief_reference_priority():
    f = _frame(obs_vector=np.ones(3, dtype=np.float32))
    ref, lbl = belief_reference(f)
    assert lbl == "obs_vector"
    f2 = _frame(sanitized_state=np.ones(5, dtype=np.float32), obs_vector=np.ones(3))
    _, lbl2 = belief_reference(f2)
    assert lbl2 == "sanitized_state"


def test_belief_reference_prefer_goal():
    gref = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    f = _frame(obs_vector=np.zeros(3), goal_ref=gref)
    ref, lbl = belief_reference(f, prefer_goal=True)
    assert lbl == "goal_ref"
    assert np.allclose(ref, gref)


def test_build_moment_series_length():
    frames = [_frame(cycle_id=i, prediction_error=0.1 * i) for i in range(5)]
    series = build_moment_series(frames)
    assert len(series) == 5


def test_build_moment_series_decision_shift():
    frames = []
    for i, bs in enumerate([0.20, 0.55, 0.56, 0.30]):
        f = _frame(cycle_id=i, action_rationale={"best_score": bs})
        frames.append(f)
    series = build_moment_series(frames)
    c = count_moments(series)
    assert c["decision_shift_count"] == 2
    assert series[1]["decision_shift"] is True
    assert series[3]["decision_shift"] is True


def test_apply_decision_shift_threshold():
    assert apply_decision_shift(0.20, 0.55) is True
    assert apply_decision_shift(0.55, 0.56) is False


def test_flow_action_link_human_labels():
    f = _frame(module_timings={"gprime_learn": 6.0, "action_selection": 2.0})
    f.action_rationale = {"explored": False}
    line = flow_action_link_line(f)
    assert "G′lrn" in line
    assert "gprime_learn" not in line


def test_count_moments():
    series = [
        {"spike": True, "learn_burst": False, "decision_shift": False},
        {"spike": False, "learn_burst": True, "decision_shift": True},
    ]
    c = count_moments(series)
    assert c["spike_count"] == 1
    assert c["learn_burst_count"] == 1
    assert c["decision_shift_count"] == 1


def test_flow_status_extras_phase_and_learn():
    f = _frame()
    extras = flow_status_extras(f)
    assert "phase=Learn" in extras
    assert "learn=" in extras


def test_flow_near_bound_module():
    f = _frame(rbta_bounds={"G'": {"time": 1.0}},
               module_timings={"prediction": 0.9, "gprime_learn": 1.0})
    assert flow_near_bound_module(f) == "prediction"


def test_action_status_extras_chosen_and_peu():
    f = _frame(
        per_dim_peu=np.array([0.2, 0.4], dtype=np.float32),
        action_rationale={"continuous": True},
    )
    hist = deque([0.1], maxlen=5)
    moment = cognitive_moment(f, hist)
    extras = action_status_extras(f, [0.2, 0.5], 1, moment=moment)
    assert "chosen=" in extras
    assert "PEU" in extras


def test_action_status_extras_replay_rollouts():
    f = _frame(action_rationale={})
    extras = action_status_extras(f, [0.3], 0, replay=True)
    assert "rollouts=replay" in extras
