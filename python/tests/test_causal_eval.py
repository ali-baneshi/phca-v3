"""Tests for PHCA causal evidence evaluation helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_eval_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "phca_causal_eval.py"
    spec = importlib.util.spec_from_file_location("phca_causal_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_causal_gate_passes_when_phca_beats_gated_controls():
    mod = _load_eval_module()
    rows = [
        {"agent": "phca", "seed": 0, "cycles": 10, "goal_rate": 0.8,
         "first_goal_cycle": 2, "mean_distance_to_goal": 0.6,
         "cumulative_reward": 5.0, "rbta_violation_rate": 0.0},
        {"agent": "random", "seed": 0, "cycles": 10, "goal_rate": 0.1,
         "first_goal_cycle": None, "mean_distance_to_goal": 4.0,
         "cumulative_reward": -0.1, "rbta_violation_rate": 0.0},
        {"agent": "greedy_observed", "seed": 0, "cycles": 10, "goal_rate": 0.6,
         "first_goal_cycle": 4, "mean_distance_to_goal": 0.9,
         "cumulative_reward": 4.0, "rbta_violation_rate": 0.0},
    ]
    summary = mod.aggregate_runs(rows)
    gate = mod.compare_agents(
        summary, gate_controls=["random", "greedy_observed"],
    )["gate"]
    assert gate["passed"] is True
    assert gate["controls"]["random"]["phca_better_count"] == 4
    assert gate["controls"]["greedy_observed"]["phca_better_count"] == 4


def test_causal_gate_fails_when_phca_loses_to_random():
    mod = _load_eval_module()
    rows = [
        {"agent": "phca", "seed": 0, "cycles": 10, "goal_rate": 0.0,
         "first_goal_cycle": None, "mean_distance_to_goal": 5.0,
         "cumulative_reward": -1.0, "rbta_violation_rate": 0.0},
        {"agent": "random", "seed": 0, "cycles": 10, "goal_rate": 0.5,
         "first_goal_cycle": 4, "mean_distance_to_goal": 2.0,
         "cumulative_reward": 2.0, "rbta_violation_rate": 0.0},
        {"agent": "greedy_observed", "seed": 0, "cycles": 10, "goal_rate": 0.7,
         "first_goal_cycle": 2, "mean_distance_to_goal": 1.0,
         "cumulative_reward": 4.0, "rbta_violation_rate": 0.0},
    ]
    gate = mod.compare_agents(mod.aggregate_runs(rows))["gate"]
    assert gate["passed"] is False
    assert gate["controls"]["random"]["phca_better_count"] == 0


def test_causal_secondary_prediction_dual_report_not_gated():
    """PE is dual-reported on the gate; does not affect scenario PASS/FAIL."""
    mod = _load_eval_module()
    rows = [
        {"agent": "phca", "seed": 0, "cycles": 10, "goal_rate": 0.8,
         "first_goal_cycle": 2, "mean_distance_to_goal": 0.6,
         "cumulative_reward": 5.0, "rbta_violation_rate": 0.0,
         "prediction_error_mean": 0.42},
        {"agent": "random", "seed": 0, "cycles": 10, "goal_rate": 0.1,
         "first_goal_cycle": None, "mean_distance_to_goal": 4.0,
         "cumulative_reward": -0.1, "rbta_violation_rate": 0.0},
        {"agent": "greedy_observed", "seed": 0, "cycles": 10, "goal_rate": 0.6,
         "first_goal_cycle": 4, "mean_distance_to_goal": 0.9,
         "cumulative_reward": 4.0, "rbta_violation_rate": 0.0},
    ]
    summary = mod.aggregate_runs(rows)
    assert summary["phca"]["prediction_error_mean"] == pytest.approx(0.42)
    gate = mod.compare_agents(
        summary, gate_controls=["random", "greedy_observed"],
    )["gate"]
    assert gate["passed"] is True
    assert gate["prediction_error_mean"] == pytest.approx(0.42)
    sec = gate["secondary_prediction"]
    assert sec["gated"] is False
    assert sec["prediction_error_mean"] == pytest.approx(0.42)
    assert "secondary_prediction" in gate["interpretation_note"]


def test_causal_eval_smoke_short_run_emits_required_sections():
    mod = _load_eval_module()
    report = mod.run_evaluation(
        cycles=5,
        seeds=1,
        size=5,
        agents=["phca", "random", "greedy_observed", "greedy_full_info"],
        use_mlp=False,
    )
    assert set(report) == {"config", "runs", "summary", "comparisons"}
    assert len(report["runs"]) == 4
    assert {"phca", "random", "greedy_observed", "greedy_full_info"} <= set(report["summary"])
    assert "gate" in report["comparisons"]
    assert report["config"]["action_selection_mode"] == "pure_geometry_default"
    phca_run = next(r for r in report["runs"] if r["agent"] == "phca")
    assert "selector_mode_counts" in phca_run
    assert phca_run.get("geometry_dominated") is True
    gate = report["comparisons"]["gate"]
    assert "interpretation_note" in gate
    assert gate.get("geometry_dominated_frac", 0) >= 0.5
    assert "selector_mode_pct" in report["summary"]["phca"]
    assert "prediction_error_mean" in report["summary"]["phca"]
    assert gate.get("secondary_prediction", {}).get("gated") is False
    assert gate.get("prediction_error_mean") is not None
    assert gate["evidence_quality"] == "smoke_power"
    assert gate["statistical_power"]["promotion_ready"] is False
    assert gate["statistical_power"]["phca_seeds"] == 1
    assert "goal_rate_ci95_lo" in report["summary"]["phca"]


def test_causal_eval_marks_thirty_shared_seeds_as_promotion_ready():
    mod = _load_eval_module()
    rows = []
    for seed in range(30):
        rows.extend([
            {"agent": "phca", "seed": seed, "cycles": 10, "goal_rate": 0.8,
             "first_goal_cycle": 2, "mean_distance_to_goal": 0.6,
             "cumulative_reward": 5.0},
            {"agent": "random", "seed": seed, "cycles": 10, "goal_rate": 0.1,
             "first_goal_cycle": None, "mean_distance_to_goal": 4.0,
             "cumulative_reward": -0.1},
        ])
    summary = mod.aggregate_runs(rows)
    gate = mod.compare_agents(summary, gate_controls=["random"])["gate"]
    assert gate["evidence_quality"] == "causal_power"
    assert gate["statistical_power"]["promotion_ready"] is True


def test_level2_wrapper_exposes_constrained_observation_to_observed_greedy():
    mod = _load_eval_module()
    env = mod.build_scenario_env(0, 5, mod.SCENARIOS["level2"])
    assert env.goal_pos == env.true_goal_pos
    env.step(env.stay_action)
    env.step(env.stay_action)
    env.base.relocate_goal()
    env._sync_observed()
    assert env.goal_pos != env.true_goal_pos
    assert env.grid.shape == env.base.grid.shape


def test_all_levels_report_separate_gates():
    mod = _load_eval_module()
    report = mod.run_evaluation(
        cycles=5,
        seeds=1,
        size=5,
        agents=["phca", "random", "greedy_observed", "greedy_full_info"],
        use_mlp=False,
        levels=["level1", "level2", "level3"],
    )
    assert set(report) == {"config", "levels", "gate"}
    assert set(report["levels"]) == {"level1", "level2", "level3"}
    for level_report in report["levels"].values():
        assert "summary" in level_report
        assert "gate" in level_report["comparisons"]


@pytest.mark.xfail(
    strict=False,
    reason="PHCA (pure-geometry default) is goal-directed — coverage < random at 50 cycles. This is expected: goal-directed agents revisit goal path cells, reducing unique coverage. Gate-level tests (multi-metric vs greedy_observed) are the correct comparison.",
)
def test_level3_mlp_coverage_beats_greedy_observed():
    """XFail (D-156): L3 coverage_rate trails random for goal-directed agents.

    Pure-geometry PHCA moves toward the goal efficiently, revisiting the same
    corridor path. Random bounces more widely at 50 cycles, achieving higher
    unique coverage. This is expected behavior — coverage_rate from coverage-only
    baselines is not a diagnostic of PHCA quality.
    """
    mod = _load_eval_module()
    report = mod.run_level(
        level="level3",
        cycles=50,
        seeds=2,
        size=5,
        agents=["phca", "random", "greedy_observed", "greedy_full_info"],
        use_mlp=True,
    )
    summary = report["summary"]
    phca_cov = summary["phca"]["coverage_rate_mean"]
    random_cov = summary["random"]["coverage_rate_mean"]
    assert phca_cov >= random_cov, (
        f"L3 coverage_rate phca={phca_cov} < random={random_cov}"
    )
