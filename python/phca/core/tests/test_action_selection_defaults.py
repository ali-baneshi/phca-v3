"""Lock default discrete action-selection honesty (investigation H-11).

Default GridWorld must use pure geometry (D-156/D-161). Learning off must not
change goal_rate under that default. Continuous MPC must consult G' during
selection when MuJoCo is available.
"""

from __future__ import annotations

from collections import Counter

import pytest

from phca.core.cycle import CognitiveCycle
from phca.evaluation.interventions import InterventionConfig


def test_default_intervention_disables_blended_scorer() -> None:
    assert InterventionConfig().disable_blended_scorer is True


def test_default_gridworld_selector_is_pure_geometry() -> None:
    cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=False)
    modes: Counter[str] = Counter()
    predict_during_select = {"n": 0}
    selecting = {"on": False}

    orig_predict = cycle.engine.predict

    def wrapped_predict(*args, **kwargs):
        if selecting["on"]:
            predict_during_select["n"] += 1
        return orig_predict(*args, **kwargs)

    cycle.engine.predict = wrapped_predict  # type: ignore[method-assign]
    orig_select = cycle._select_action

    def wrapped_select(*args, **kwargs):
        selecting["on"] = True
        try:
            return orig_select(*args, **kwargs)
        finally:
            selecting["on"] = False

    cycle._select_action = wrapped_select  # type: ignore[method-assign]

    for _ in range(20):
        cycle.step()
        modes[cycle.last_action_rationale.get("selector_mode", "")] += 1

    assert modes["pure_geometry_ablation"] == 20
    assert predict_during_select["n"] == 0


def test_blended_opt_in_uses_prediction_scored_path() -> None:
    cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=False)
    cycle.interventions = InterventionConfig(
        disable_blended_scorer=False,
        before_blended_warmup_cycles=0,
    )
    modes: Counter[str] = Counter()
    predict_during_select = {"n": 0}
    selecting = {"on": False}

    orig_predict = cycle.engine.predict

    def wrapped_predict(*args, **kwargs):
        if selecting["on"]:
            predict_during_select["n"] += 1
        return orig_predict(*args, **kwargs)

    cycle.engine.predict = wrapped_predict  # type: ignore[method-assign]
    orig_select = cycle._select_action

    def wrapped_select(*args, **kwargs):
        selecting["on"] = True
        try:
            return orig_select(*args, **kwargs)
        finally:
            selecting["on"] = False

    cycle._select_action = wrapped_select  # type: ignore[method-assign]

    for _ in range(10):
        cycle.step()
        modes[cycle.last_action_rationale.get("selector_mode", "")] += 1

    assert modes["prediction_scored"] == 10
    assert predict_during_select["n"] >= 10  # at least one predict per select


def test_geometry_default_goal_rate_invariant_to_learning() -> None:
    def goal_frac(*, learn: bool) -> float:
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=False)
        cycle.interventions = InterventionConfig(
            enable_gprime_learn=learn,
            enable_tspl=learn,
        )
        hits = 0
        n = 40
        for _ in range(n):
            hits += int(bool(cycle.step().goal_reached))
        return hits / n

    assert goal_frac(learn=True) == goal_frac(learn=False)


def test_discrete_gprime_lacks_tspl_and_m3_replay_hooks() -> None:
    cycle = CognitiveCycle.build_for_env(size=10, seed=0, use_mlp=False)
    assert type(cycle.gprime).__name__ == "WorldModelGPrime"
    assert not hasattr(cycle.gprime, "set_tspl_bias")
    assert not hasattr(cycle.gprime, "learn_m3_episodes")
    assert cycle.env.get_state_dim() > 10
    assert cycle.tspl_world_model_coupled is False
    assert cycle.m3_gprime_replay_effective is False
    assert cycle.gprime_coverage_ratio < 0.25
    assert cycle.gprime_modeled_dims == 10


def test_mlp_gprime_has_tspl_and_m3_replay_hooks() -> None:
    cycle = CognitiveCycle.build_for_env(size=5, seed=0, use_mlp=True)
    assert hasattr(cycle.gprime, "set_tspl_bias")
    assert hasattr(cycle.gprime, "learn_m3_episodes")
    assert cycle.tspl_world_model_coupled is True
    assert cycle.m3_gprime_replay_effective is True
    assert cycle.gprime_coverage_ratio == pytest.approx(1.0)


def test_rbta_entropy_omits_synthetic_modules() -> None:
    cycle = CognitiveCycle.build_for_env(size=5, seed=0, use_mlp=True)
    for _ in range(3):
        cycle.step()
    assert "G'" in cycle.belief_entropies
    assert "ASI" not in cycle.belief_entropies
    assert "ACTION" not in cycle.belief_entropies


def test_asi_sensor_failure_enters_stale_safe_mode() -> None:
    from phca.config import ASIStatus
    from phca.core.cycle import ASI_STALE_SAFE_THRESHOLD

    cycle = CognitiveCycle.build_for_env(size=5, seed=1, use_mlp=False)
    cycle.step()
    orig = cycle.sanitizer.sanitize

    def fail_san(raw):
        st, _ = orig(raw)
        return st, ASIStatus.SENSOR_FAILURE

    cycle.sanitizer.sanitize = fail_san  # type: ignore[method-assign]
    for _ in range(ASI_STALE_SAFE_THRESHOLD):
        m = cycle.step()
    assert cycle._asi_stale_safe_mode is True
    assert "asi_sensor_failure_stale_state" in m.failure_events
    assert cycle.last_action_rationale.get("decision_reason") == "asi_stale_safe"


def test_continuous_mpc_consults_prediction_during_select() -> None:
    pytest.importorskip("mujoco")
    try:
        cycle = CognitiveCycle.build_for_mujoco(env_name="Pendulum-v1", seed=0)
    except Exception as exc:  # pragma: no cover - env packaging variance
        pytest.skip(f"Pendulum-v1 unavailable: {exc}")

    predict_during_select = {"n": 0}
    selecting = {"on": False}
    orig_predict = cycle.engine.predict

    def wrapped_predict(*args, **kwargs):
        if selecting["on"]:
            predict_during_select["n"] += 1
        return orig_predict(*args, **kwargs)

    cycle.engine.predict = wrapped_predict  # type: ignore[method-assign]
    orig_select = cycle._select_action

    def wrapped_select(*args, **kwargs):
        selecting["on"] = True
        try:
            return orig_select(*args, **kwargs)
        finally:
            selecting["on"] = False

    cycle._select_action = wrapped_select  # type: ignore[method-assign]

    modes: Counter[str] = Counter()
    for _ in range(8):
        cycle.step()
        modes[cycle.last_action_rationale.get("selector_mode", "")] += 1

    assert predict_during_select["n"] >= 8
    assert modes.get("continuous_mpc", 0) + modes.get("continuous_explore", 0) == 8
