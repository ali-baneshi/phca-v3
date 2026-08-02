"""Tests for PHCA-3.1-011: Cognitive Cycle Orchestrator."""

from __future__ import annotations


from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phca.config import StateVector, ResourceBounds
from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.evaluation.interventions import InterventionConfig

# Reduce G' inference test load — use small graphs
pytestmark = pytest.mark.timeout(30)


class TestCognitiveCycleBuild:
    """Tests for CognitiveCycle.build_for_env()."""

    def test_build_5x5_defaults(self):
        """Building for 5×5 grid should succeed with all components wired."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        assert cycle.env.size == 5
        assert cycle.state_dim == 5 * 5 * 3 + 9  # 84
        assert cycle.sanitizer is not None
        assert cycle.gprime is not None
        assert cycle.engine is not None
        assert cycle.peu is not None
        assert cycle.tspl is not None
        assert cycle.rbta is not None
        assert cycle.m2 is not None
        assert cycle.mdim is not None
        assert cycle.attention is not None

    def test_build_5x5_cycle_count(self):
        """Build should initialize cycle_count = 0."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        assert cycle.cycle_count == 0

    def test_build_deep_ensemble_prediction_mode(self):
        """prediction_mode=deep_ensemble should build a probabilistic MLP ensemble."""
        cfg = InterventionConfig(prediction_mode="deep_ensemble")
        cycle = CognitiveCycle.build_for_env(
            size=5, seed=42, use_mlp=True, interventions=cfg,
        )
        assert cycle.gprime.uncertainty_snapshot()["kind"] == "mlp_ensemble"
        assert cycle.gprime.position_dim == 25


class TestCognitiveCycleStep:
    """Tests for single cycle execution."""

    def test_single_step_returns_metrics(self):
        """A single step should return CycleMetrics."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        metrics = cycle.step()
        assert isinstance(metrics, CycleMetrics)
        assert metrics.cycle_id == 0
        assert metrics.latency_ms >= 0
        assert metrics.action_taken in range(5)

    def test_single_step_increments_counter(self):
        """Step should increment cycle_count."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.step()
        assert cycle.cycle_count == 1

    def test_single_step_populates_metrics(self):
        """Step metrics should have all fields populated."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        metrics = cycle.step()
        assert metrics.prediction_error >= 0
        assert 0.0 <= metrics.prediction_confidence <= 1.0
        assert metrics.rbta_action in ("CONTINUE", "INTERRUPT", "TERMINATE")
        assert metrics.action_name in ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY")
        assert len(metrics.module_timings) > 0

    def test_two_steps_increment_properly(self):
        """Two steps should result in cycle_count=2."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.step()
        cycle.step()
        assert cycle.cycle_count == 2
        assert len(cycle.metrics_history) == 2


class TestCognitiveCycleRun:
    """Tests for multi-cycle run()."""

    def test_run_10_cycles(self):
        """Run 10 cycles should complete with valid summary."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=10)
        assert summary["total_cycles"] == 10
        assert summary["avg_latency_ms"] > 0
        assert 0 <= summary["total_violations"]
        assert isinstance(summary["skill_compiled"], bool)

    def test_run_stores_all_metrics(self):
        """Run should store one metrics entry per cycle."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=5)
        assert len(summary["early_errors"]) == 5  # 5 cycles, [:10] returns 5
        assert len(summary["late_errors"]) == 5   # 5 cycles, [-10:] returns 5
        assert len(summary["actions_taken"]) == 5

    def test_run_empty_returns_safe_defaults(self):
        """run(n_cycles=0) should return safe defaults."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=0)
        assert summary["total_cycles"] == 0
        assert summary["avg_latency_ms"] == 0.0


class TestCognitiveCycleActionSelection:
    """Tests for action selection (Step 9)."""

    def test_action_in_valid_range(self):
        """_select_action should return 0-4."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.current_state = StateVector(
            values=np.ones(84, dtype=np.float32),
            precision=np.ones(84, dtype=np.float32),
        )
        action = cycle._select_action()
        assert 0 <= action <= 4

    def test_no_state_returns_stay(self):
        """_select_action with no current_state should return STAY (4)."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.current_state = None
        action = cycle._select_action()
        assert action == 4


class TestCognitiveCycleCollectLogs:
    """Tests for runtime log collection."""

    def test_collect_runtime_log_populates_all(self):
        """_collect_runtime_log should populate all logs."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle._collect_runtime_log()
        assert "ASI" in cycle.runtime_log
        assert "WM" in cycle.runtime_log
        assert "G'" in cycle.runtime_log
        assert "CYCLE" in cycle.runtime_log
        assert len(cycle.runtime_log) >= 6

    def test_belief_entropies_decrease_with_cycles(self):
        """Belief entropies should decrease as cycle_count increases."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.gprime._last_mutual_info = 0.8
        cycle._collect_runtime_log()
        e1 = cycle.belief_entropies["G'"]
        cycle.gprime._last_mutual_info = 0.2
        cycle._collect_runtime_log()
        e2 = cycle.belief_entropies["G'"]
        assert e2 <= e1

    def test_epistemic_entropy_uses_mutual_info(self):
        """_epistemic_entropy reflects model's _last_mutual_info directly."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.gprime._last_mutual_info = 0.3
        assert cycle._epistemic_entropy() == pytest.approx(0.3)
        cycle.gprime._last_mutual_info = 0.0
        assert cycle._epistemic_entropy() == pytest.approx(0.0)

    def test_task_lock_low_confidence_uses_blended_scorer(self):
        """Task-lock with low G' confidence uses prediction-scored path (always on)."""
        from phca.prediction.engine import PredictionEngine

        cycle = CognitiveCycle.build_for_env(size=5, seed=42,
            interventions=InterventionConfig(disable_blended_scorer=False))
        cycle._task_lock = True
        cycle.cycle_count = 400
        cycle.current_state = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.ones(cycle.state_dim, dtype=np.float32),
        )
        cycle.last_prediction = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.full(cycle.state_dim, 0.3, dtype=np.float32),
        )
        calls = {"n": 0}
        orig = PredictionEngine.predict

        def _counting_predict(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0
        PredictionEngine.predict = _counting_predict
        try:
            with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
                cycle._select_action()
            assert calls["n"] >= 2
            assert cycle.last_action_rationale.get("selector_mode") == "prediction_scored"
        finally:
            PredictionEngine.predict = orig

    def test_task_lock_high_confidence_still_uses_prediction(self):
        """Task-lock with high G' confidence still uses prediction-scored path (no bypass)."""
        from phca.prediction.engine import PredictionEngine

        cycle = CognitiveCycle.build_for_env(size=5, seed=42,
            interventions=InterventionConfig(disable_blended_scorer=False))
        cycle._task_lock = True
        cycle.cycle_count = 400
        cycle.current_state = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.ones(cycle.state_dim, dtype=np.float32),
        )
        cycle.last_prediction = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.full(cycle.state_dim, 0.8, dtype=np.float32),
        )
        calls = {"n": 0}
        orig = PredictionEngine.predict

        def _counting_predict(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0
        PredictionEngine.predict = _counting_predict
        try:
            with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
                cycle._select_action()
            assert calls["n"] >= 2  # prediction always called, no geometry bypass
            assert cycle.last_action_rationale.get("selector_mode") == "prediction_scored"
        finally:
            PredictionEngine.predict = orig

    def test_prediction_path_marks_selector_mode(self):
        """Low-confidence discrete selection reports prediction_scored path."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42,
            interventions=InterventionConfig(disable_blended_scorer=False))
        cycle._task_lock = True
        cycle.cycle_count = 400
        cycle.current_state = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.ones(cycle.state_dim, dtype=np.float32),
        )
        cycle.last_prediction = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.full(cycle.state_dim, 0.2, dtype=np.float32),
        )
        cycle._select_action()
        assert cycle.last_action_rationale.get("selector_mode") == "prediction_scored"


class TestRBTAEnforcement:
    """P1-01: RBTA INTERRUPT/TERMINATE must alter cycle behavior."""

    def _tighten_all_bounds(self, cycle: CognitiveCycle) -> None:
        tiny = ResourceBounds(B_time=1e-9, B_mem=1, B_energy=1e-9, entropy_floor=0.01)
        for mod_id in list(cycle.rbta._bounds.keys()):
            cycle.rbta.update_bounds(mod_id, tiny)

    def test_terminate_skips_feedback_and_consolidation(self):
        """TERMINATE (3+ violations) → STAY, no learn/consolidation, M3 trace kept."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        self._tighten_all_bounds(cycle)
        metrics = cycle.step()
        assert metrics.rbta_action == "TERMINATE"
        assert metrics.violations_count >= 3
        assert metrics.module_timings.get("gprime_learn", 0.0) == 0.0
        assert metrics.module_timings.get("consolidation", 0.0) == 0.0
        assert metrics.action_taken == cycle.env.stay_action
        assert cycle.consolidation.m3.count() == 1
        [episode] = cycle.consolidation.m3.recent_episodes(1)
        assert episode.planner_mode == "rbta_safe_mode"
        assert episode.trajectory_id == 0
        assert episode.path_length == 1
        assert episode.task_id == cycle._current_task_id

    def test_terminate_continuous_uses_zero_vector(self):
        """CORE-A01: continuous TERMINATE must not pass stay_action int to step."""
        from phca.config import continuous_space

        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.action_space = continuous_space(-1.0, 1.0, 2)
        cycle._is_continuous = True
        stepped = []

        def _step(action):
            arr = np.asarray(action, dtype=np.float32)
            stepped.append(arr.copy())
            assert arr.shape == (2,), f"expected (2,), got {arr.shape}"
            obs = np.zeros(cycle.state_dim, dtype=np.float32)
            return obs, 0.0, False, {}

        cycle.env.step = _step
        self._tighten_all_bounds(cycle)
        metrics = cycle.step()
        assert metrics.rbta_action == "TERMINATE"
        assert len(stepped) == 1
        assert stepped[0].shape == (2,)
        assert np.allclose(stepped[0], 0.0)
        assert np.allclose(cycle.last_action, 0.0)
        r = cycle.last_action_rationale
        assert r.get("decision_reason") == "rbta_safe"
        assert r.get("continuous") is True
        assert r.get("rbta_safe_mode") is True

    def test_interrupt_limits_prediction_candidates(self):
        """INTERRUPT flag → at most one predict rollout when not task-lock."""
        from phca.prediction.engine import PredictionEngine

        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle._task_lock = False
        cycle.env.get_goal_position = lambda: None
        cycle.step()
        cycle._rbta_action_candidate_limit = 1
        calls = {"n": 0}
        orig = PredictionEngine.predict

        def _counting_predict(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        PredictionEngine.predict = _counting_predict
        try:
            before = calls["n"]
            cycle._select_action()
            during = calls["n"] - before
        finally:
            PredictionEngine.predict = orig
        assert during <= 1

    def test_interrupt_skips_consolidation_when_post_check_fires(self):
        """Post-cycle INTERRUPT skips consolidation on same cycle."""
        from phca.config import ResourceBounds

        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.rbta.update_bounds(
            "MDIM",
            ResourceBounds(B_time=1e-9, B_mem=1_000_000, B_energy=10.0),
        )
        cycle.rbta.update_bounds(
            "CR",
            ResourceBounds(B_time=1e-9, B_mem=1_000_000, B_energy=10.0),
        )
        metrics = cycle.step()
        assert metrics.rbta_action in ("INTERRUPT", "TERMINATE")
        if metrics.rbta_action == "INTERRUPT":
            assert metrics.module_timings.get("consolidation", 0.0) == 0.0

    def test_10x10_noisy_moving_obstacle_observation_to_action_records_m3(self):
        """End-to-end smoke test for 10x10 GridWorld with sensor noise and obstacle drift."""
        cycle = CognitiveCycle.build_for_env(
            size=10,
            seed=42,
            use_mlp=True,
            obstacles=[(1, 1), (2, 2), (3, 3), (4, 4)],
            noise_profile="gaussian",
            noise_intensity=0.02,
        )
        assert cycle.gprime.position_dim == 100

        first = cycle.step()
        assert first.action_taken in range(5)

        old_wall = (1, 1)
        if cycle.env.agent_pos != old_wall and cycle.env.goal_pos != old_wall:
            cycle.env.grid[old_wall] = cycle.env.EMPTY
        for candidate in ((0, 1), (1, 0), (5, 5), (6, 4)):
            if candidate != cycle.env.agent_pos and candidate != cycle.env.goal_pos:
                cycle.env.grid[candidate] = cycle.env.WALL
                break

        second = cycle.step()
        assert second.action_taken in range(5)
        assert cycle.consolidation.m3.count() >= 2
        recent = cycle.consolidation.m3.recent_episodes(1)[0]
        assert recent.goal_pos == tuple(cycle.env.goal_pos)
        assert recent.path_length >= 1
        assert recent.action_taken.shape == (cycle.env.action_space_size,)

    def test_her_replay_uses_successful_planner_trajectory(self):
        """Successful fallback/planner trajectories should replay into G'."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        state = StateVector(
            values=np.zeros(cycle.state_dim, dtype=np.float32),
            precision=np.ones(cycle.state_dim, dtype=np.float32),
            timestamp=0.0,
        )
        action = np.zeros(cycle.env.action_space_size, dtype=np.float32)
        action[cycle.env.stay_action] = 1.0
        for step in range(2):
            cycle.consolidation.m3.store_episode(
                state,
                action,
                state,
                0.1,
                timestamp=step,
                trajectory_id=3,
                planner_mode="bfs",
                trajectory_success=(step == 1),
                path_length=step + 1,
            )

        steps = cycle._replay_m3_planner_successes()

        assert steps > 0
        assert cycle._m3_her_total >= steps


class TestActionRationaleEnrichment:
    """Phase 14: action_rationale explain fields on all selection paths."""

    def _state(self, cycle: CognitiveCycle) -> None:
        cycle.current_state = StateVector(
            values=np.ones(cycle.state_dim, dtype=np.float32),
            precision=np.ones(cycle.state_dim, dtype=np.float32),
        )

    def test_discrete_prediction_rationale_fields(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42,
            interventions=InterventionConfig(disable_blended_scorer=False))
        cycle.cycle_count = 400
        self._state(cycle)
        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0
        with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
            cycle._select_action()
        r = cycle.last_action_rationale
        assert r["decision_reason"] == "prediction"
        assert r["mechanism"] == "prediction"
        assert r.get("drive_id") is not None
        assert isinstance(r.get("score_components"), dict)

    def test_discrete_explore_rationale(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        self._state(cycle)
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.0
        mock_rng.randint.return_value = 2
        with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
            cycle._select_action()
        assert cycle.last_action_rationale["decision_reason"] == "explore"

    def test_d5_stay_rationale(self):
        from unittest.mock import Mock

        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        self._state(cycle)
        cycle.current_goal = Mock(drive_id=5, target_state=None)
        cycle._task_lock = False
        cycle.cycle_count = 100
        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0
        with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
            cycle._select_action()
        assert cycle.last_action_rationale["decision_reason"] == "d5_stay"
        assert cycle.last_action_rationale["mechanism"] == "stay"

    def test_rbta_safe_rationale(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        tiny = ResourceBounds(B_time=1e-9, B_mem=1, B_energy=1e-9, entropy_floor=0.01)
        for mod_id in list(cycle.rbta._bounds.keys()):
            cycle.rbta.update_bounds(mod_id, tiny)
        cycle.step()
        r = cycle.last_action_rationale
        assert r.get("decision_reason") == "rbta_safe"
        assert r.get("mechanism") == "rbta_safe"
        assert r.get("rbta_safe_mode") is True

    def test_continuous_mpc_rationale(self):
        from phca.config import continuous_space

        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.action_space = continuous_space(-1.0, 1.0, 2)
        cycle._is_continuous = True
        cycle.cycle_count = 400
        self._state(cycle)
        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0
        mock_rng.uniform.return_value = np.array([0.1, -0.2], dtype=np.float32)
        with patch("phca.core.cycle.np.random.RandomState", return_value=mock_rng):
            cycle._select_continuous_action()
        r = cycle.last_action_rationale
        assert r["decision_reason"] == "continuous_mpc"
        assert r["mechanism"] == "continuous"
        assert isinstance(r.get("score_components"), dict)
        assert r.get("chosen_idx") is not None
