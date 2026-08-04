"""Regression: goal_reached must survive JSONL round-trip for reports."""
from __future__ import annotations

from phca.monitoring.cognitive_panels import classify_action_mechanism
from phca.monitoring.observability import ObservabilityFrame, normalize_observability_json
from phca.monitoring.session_io import frame_from_json, frame_to_json
from phca.monitoring.session_report import _build_agent_report_from_frames


def test_goal_reached_roundtrip_true():
    f = ObservabilityFrame(
        cycle_id=7,
        schema_version=1,
        env_kind="grid",
        goal_reached=True,
        agent_pos=(2, 2),
        goal_pos=(2, 2),
        action_name="STAY",
        action_rationale={
            "selector_mode": "pure_geometry_ablation",
            "decision_reason": "ablation_pure_geometry",
            "mechanism": "greedy_fallback",
            "greedy_fallback": True,
            "chosen_idx": 4,
        },
        candidate_scores=[0.2, 0.2, 0.2, 0.2, 1.0],
        m3_count=12,
        episode_count=1,
    )
    raw = frame_to_json(f)
    assert raw.get("goal_reached") is True
    restored = frame_from_json(raw)
    assert restored.goal_reached is True
    assert restored.m3_count == 12
    assert restored.episode_count == 1


def test_legacy_jsonl_missing_goal_reached_defaults_false():
    obj = normalize_observability_json({"cycle_id": 0, "schema_version": 1})
    assert obj["goal_reached"] is False
    f = frame_from_json(obj)
    assert f.goal_reached is False


def test_geometry_mechanism_not_other():
    r = {
        "selector_mode": "pure_geometry_ablation",
        "decision_reason": "ablation_pure_geometry",
        "mechanism": "other",  # legacy dishonest label
        "k_candidates": 5,
        "chosen_idx": 4,
    }
    assert classify_action_mechanism(r) == "greedy_fallback"


def test_session_report_counts_goal_reached_and_grid_dist():
    frames = []
    for i in range(40):
        on_goal = i >= 5
        frames.append(
            ObservabilityFrame(
                cycle_id=i,
                schema_version=1,
                env_kind="grid",
                goal_reached=on_goal,
                agent_pos=(2, 2) if on_goal else (0, 0),
                goal_pos=(2, 2),
                prediction_error=3.0,
                prediction_confidence=0.99,
                action_name="STAY" if on_goal else "MOVE_E",
                action_rationale={
                    "selector_mode": "pure_geometry_ablation",
                    "decision_reason": "ablation_pure_geometry",
                    "mechanism": "greedy_fallback",
                    "greedy_fallback": True,
                    "chosen_idx": 4 if on_goal else 2,
                    "k_candidates": 5,
                },
                candidate_scores=[0.1, 0.1, 0.5, 0.1, 1.0],
                m3_count=i + 1,
                episode_count=0,
                latency_ms=80.0,
                module_timings={"gprime_learn": 50.0, "action_selection": 1.0},
                rss_bytes=100_000_000 + i * 1000,
            )
        )
    report = _build_agent_report_from_frames(
        {"env": "gridworld", "status": "complete"}, frames,
    )
    assert report["goal_reached_count"] == 35
    assert report["dist_early_median"] is not None
    assert report["dist_late_median"] is not None
    assert report["dist_late_median"] <= report["dist_early_median"]
    mech = report["action_metrics"]["mechanism_pct"]
    assert mech.get("greedy_fallback", 0.0) >= 50.0
    assert mech.get("other", 0.0) < 50.0
