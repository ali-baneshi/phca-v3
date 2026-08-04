"""Unit tests for cognitive_panels pure helpers."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from phca.monitoring.cognitive_panels import (
    LEARN_MS_MIN,
    RBTA_TO_FLOW,
    action_status_extras,
    apply_decision_shift,
    belief_reference,
    build_moment_series,
    cognitive_moment,
    count_moments,
    execution_dominant_phase,
    flow_action_link_line,
    flow_near_bound_module,
    classify_action_mechanism,
    data_contract_text,
    format_session_results_lines,
    frame_attention_pairs,
    frame_drive_goal_norms,
    frame_per_dim_peu,
    frame_peu_mean,
    mechanism_histogram,
    moment_tab_badge,
    flow_near_bound_modules,
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
        "prediction": 1.0, "action_selection": 2.0, "gprime_learn": 40.0,
    })
    f.latency_ms = kwargs.get("latency_ms", 80.0)
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


def test_rbta_aliases_cover_recorded_bound_keys():
    assert RBTA_TO_FLOW["TSPL-P"] == "tspl"
    assert RBTA_TO_FLOW["CONSOL"] == "consolidation"
    assert RBTA_TO_FLOW["PE"] == "peu"
    assert RBTA_TO_FLOW["ASI"] == "sanitize"
    f = _frame(
        module_timings={"tspl": 15.0, "consolidation": 60.0, "peu": 210.0},
        rbta_bounds={
            "TSPL-P": {"time": 0.02},
            "CONSOL": {"time": 0.05},
            "PE": {"time": 0.30},
        },
    )
    near = dict(flow_near_bound_modules(f, top_k=3))
    assert "tspl" in near
    assert "consolidation" in near
    assert "peu" in near


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
    f = _frame(rbta_bounds={"G'": {"time": 0.001}},
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
    assert "rollouts=live-only" in extras


def test_frame_per_dim_peu_from_top():
    f = ObservabilityFrame()
    f.state_dim = 5
    f.per_dim_peu_top = [{"idx": 1, "value": 2.0}, {"idx": 4, "value": 3.0}]
    f.per_dim_peu_sum = 5.0
    arr = frame_per_dim_peu(f)
    assert arr is not None
    assert arr.shape == (5,)
    assert float(arr[1]) == 2.0 and float(arr[4]) == 3.0
    assert frame_peu_mean(f) == pytest.approx(1.0)


def test_frame_attention_pairs_lookup_by_chunk_id():
    f = ObservabilityFrame()
    f.attention_indices = [2, 0]
    f.attention_saliences = [0.1, 0.2, 0.9, 0.05]
    pairs = frame_attention_pairs(f)
    assert pairs == [(2, 0.9), (0, 0.1)]
    f.attention_selected_saliences = [0.9, 0.1]
    assert frame_attention_pairs(f) == [(2, 0.9), (0, 0.1)]


def test_frame_drive_goal_norms():
    f = ObservabilityFrame()
    f.drive_goal_norms = [1.0, 0.5, 0.0]
    assert frame_drive_goal_norms(f) == [1.0, 0.5, 0.0]


def test_geometry_chrome_helpers():
    from phca.monitoring.cognitive_panels import (
        frame_is_geometry_control, frame_scores_degenerate,
        geometry_score_label, learn_phase_label,
    )
    f = ObservabilityFrame()
    f.action_rationale = {
        "selector_mode": "geometry",
        "decision_reason": "ablation_pure_geometry",
        "best_score": 1.0,
    }
    f.candidate_scores = [1.0, 1.0, 1.0]
    assert frame_is_geometry_control(f) is True
    assert frame_scores_degenerate(f) is True
    assert "geo-proxy" in geometry_score_label(f)
    assert "WM only" in learn_phase_label(f, learn_burst=True)


def test_classify_action_mechanism_parity():
    assert classify_action_mechanism({"explored": True}) == "explore"
    assert classify_action_mechanism({"continuous": True}) == "continuous"
    assert classify_action_mechanism({"best_score": 0.5}) == "prediction"


def test_mechanism_histogram_window():
    frames = [
        _frame(action_rationale={"explored": i % 2 == 0})
        for i in range(10)
    ]
    counts = mechanism_histogram(frames, window=256)
    assert counts["explore"] == 5
    assert counts["other"] == 5


def test_moment_tab_badge_spike():
    assert moment_tab_badge({"spike": True}) == "SPIKE"
    assert moment_tab_badge({}) == ""


def test_data_contract_all_panels():
    for key in ("overview", "flow", "action", "phase", "retention", "memory", "goals"):
        assert data_contract_text(key, replay=False).startswith("LIVE —")
        assert data_contract_text(key, replay=True).startswith("REPLAY —")
        assert data_contract_text(key, replay=False, review=True).startswith("REVIEW —")
        aborted = data_contract_text(
            key, replay=False, review=True, incomplete=True,
        )
        assert aborted.startswith("ABORTED —")
        assert "INCOMPLETE" in aborted or "ABORTED" in aborted


def test_session_status_aborted_and_incomplete():
    from phca.monitoring.cognitive_panels import session_status_text

    aborted = session_status_text(
        cycle_error="Action dimension mismatch",
        jsonl_count=167,
        total=1000,
    )
    assert "ABORTED" in aborted
    assert "Action dimension mismatch" in aborted

    incomplete = session_status_text(
        incomplete=True,
        jsonl_count=167,
        total=1000,
        verify_status="FAIL",
    )
    assert "INCOMPLETE 167/1000" in incomplete
    assert "verify=FAIL" in incomplete

    safe = session_status_text(rbta_safe_ratio=0.42)
    assert "SAFE-MODE 42%" in safe
    quiet = session_status_text(rbta_safe_ratio=0.05)
    assert "SAFE-MODE" not in quiet


def test_format_early_late():
    from phca.monitoring.cognitive_panels import format_early_late

    assert "↘" in format_early_late("error", 12.0, 6.0)
    assert "↗" in format_early_late("error", 6.0, 12.0)


def test_format_session_results_lines_compare_and_benchmark():
    report = {
        "cycles": 100,
        "explore_ratio": 0.08,
        "goal_reached_count": 2,
        "spike_count": 3,
        "error_early_median": 12.0,
        "error_late_median": 6.0,
        "dist_early_median": 0.5,
        "dist_late_median": 0.2,
        "phase_budget_pct": {"Act": 34.0, "Learn": 28.0},
        "flow_metrics": {"violation_cycle_count": 1},
        "action_metrics": {
            "mechanism_pct": {"prediction": 70.0, "continuous": 20.0, "explore": 10.0},
        },
    }
    compare = {
        "error_late_median": 8.0,
        "explore_ratio": 0.05,
        "action_metrics": {"mechanism_pct": {"prediction": 60.0, "explore": 40.0}},
    }
    benchmark = {
        "overall_phi_iq": 0.732,
        "results": [{"level": 0, "phi_iq": 0.77}, {"level": 1, "phi_iq": 0.65}],
    }
    lines = format_session_results_lines(
        report, compare=compare, benchmark=benchmark, verify_status="PASS")
    text = "\n".join(lines)
    assert "session 100 cycles" in text
    assert "error 12→6" in text.replace(" ", "") or "error 12" in text
    assert "vs prev" in text
    assert "Φ-IQ" in text
    assert "verify: PASS" in text


def test_format_session_results_lines_task_lock_mode():
    report = {
        "cycles": 50,
        "explore_ratio": 0.0,
        "goal_reached_count": 45,
        "spike_count": 0,
        "error_early_median": 4.0,
        "error_late_median": 4.0,
        "flow_metrics": {"violation_cycle_count": 0},
        "action_metrics": {
            "selector_mode_pct": {"task_lock_planner": 100.0},
            "task_lock_planner_dominant": True,
            "mechanism_pct": {"greedy_fallback": 100.0},
        },
        "report_classification": {
            "expected_limitations": ["discrete_task_lock_planner_dominant"],
        },
    }
    text = "\n".join(format_session_results_lines(report))
    assert "selector: task_lock_planner 100%" in text
    assert "task-lock planner dominated" in text
    assert "limits:" in text
