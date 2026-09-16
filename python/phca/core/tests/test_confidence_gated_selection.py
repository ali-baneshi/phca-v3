"""Tests for the explicit opt-in confidence-gated discrete selector."""

from __future__ import annotations

import numpy as np

from phca.config import StateVector
from phca.core.cycle import CognitiveCycle
from phca.evaluation.interventions import InterventionConfig
from phca.monitoring.observability import ObservabilityStore
from phca.regulation.rbta_enforcer import EnforcerAction


def _state(cycle: CognitiveCycle) -> StateVector:
    return StateVector(
        values=np.ones(cycle.state_dim, dtype=np.float32),
        precision=np.ones(cycle.state_dim, dtype=np.float32),
    )


def _gated_cycle(**kwargs) -> CognitiveCycle:
    options = {
        "confidence_gated_selector": True,
        "disable_blended_scorer": False,
        "confidence_gated_warmup_cycles": 0,
        "confidence_gated_threshold": 0.9,
        "before_blended_warmup_cycles": 0,
        "agreement_gating": False,
    }
    options.update(kwargs)
    return CognitiveCycle.build_for_env(
        size=5,
        seed=42,
        use_mlp=False,
        interventions=InterventionConfig(**options),
    )


def test_warmup_uses_geometry_only() -> None:
    cycle = _gated_cycle(confidence_gated_warmup_cycles=5)
    cycle.current_state = _state(cycle)
    cycle.cycle_count = 0
    calls = {"confidence": 0, "predict": 0}

    def confidence(*args, **kwargs):
        calls["confidence"] += 1
        return 1.0

    def predict(*args, **kwargs):
        calls["predict"] += 1
        return _state(cycle), 1.0

    cycle.engine.predict_confidence = confidence  # type: ignore[method-assign]
    cycle.engine.predict = predict  # type: ignore[method-assign]
    cycle._task_lock = True

    cycle._select_action()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_warmup"
    assert calls == {"confidence": 0, "predict": 0}


def test_warmup_precedes_legacy_exploration() -> None:
    cycle = _gated_cycle(confidence_gated_warmup_cycles=5)
    cycle.current_state = _state(cycle)
    cycle.cycle_count = 1
    cycle._task_lock = False

    cycle.engine.predict_confidence = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("warm-up must not query G' confidence")
    )  # type: ignore[method-assign]

    cycle._select_action()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_warmup"


def test_low_confidence_uses_geometry() -> None:
    cycle = _gated_cycle()
    cycle.current_state = _state(cycle)
    cycle.cycle_count = 50
    cycle._task_lock = True
    cycle.engine.predict_confidence = lambda *args, **kwargs: 0.2  # type: ignore[method-assign]

    cycle._select_action()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_geometry"
    assert cycle.last_action_rationale["decision_reason"] == "confidence_below_threshold"


def test_high_confidence_uses_prediction_path() -> None:
    cycle = _gated_cycle(confidence_gated_threshold=0.5)
    cycle.current_state = _state(cycle)
    cycle.cycle_count = 50
    cycle._task_lock = True
    calls = {"confidence": 0, "predict": 0}

    def confidence(*args, **kwargs):
        calls["confidence"] += 1
        return 0.95

    def predict(*args, **kwargs):
        calls["predict"] += 1
        return _state(cycle), 0.95

    cycle.engine.predict_confidence = confidence  # type: ignore[method-assign]
    cycle.engine.predict = predict  # type: ignore[method-assign]

    cycle._select_action()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_prediction"
    assert calls["confidence"] == 1
    assert calls["predict"] == cycle.env.action_space_size


def test_rolling_fallback_triggers_and_returns_geometry() -> None:
    cycle = _gated_cycle(confidence_gated_min_samples=2)
    cycle.current_state = _state(cycle)
    cycle.cycle_count = 50
    cycle._confidence_gated_outcomes["prediction"] = [False, False]
    cycle._confidence_gated_outcomes["geometry"] = [True, True]
    cycle._update_confidence_gated_fallback()
    assert cycle._confidence_gated_fallback_active is True

    def unexpected_confidence(*args, **kwargs):
        raise AssertionError("fallback must bypass G' confidence evaluation")

    cycle.engine.predict_confidence = unexpected_confidence  # type: ignore[method-assign]
    cycle._select_action()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_fallback"
    assert cycle.last_action_rationale["decision_reason"] == "confidence_gated_fallback"


def test_gprime_trains_during_confidence_gated_warmup() -> None:
    interventions = InterventionConfig(
        confidence_gated_selector=True,
        disable_blended_scorer=False,
        confidence_gated_warmup_cycles=5,
        before_blended_warmup_cycles=0,
    )
    cycle = CognitiveCycle.build_for_env(
        size=5, seed=42, use_mlp=True, interventions=interventions,
    )
    calls = {"learn": 0}
    original = cycle.gprime.learn

    def learn(*args, **kwargs):
        calls["learn"] += 1
        return original(*args, **kwargs)

    cycle.gprime.learn = learn  # type: ignore[method-assign]
    metrics = cycle.step()

    assert cycle.last_action_rationale["selector_mode"] == "confidence_gated_warmup"
    assert calls["learn"] == 1
    assert metrics.module_timings["gprime_learn"] >= 0.0


def test_integration_observability_records_selector_rationale() -> None:
    store = ObservabilityStore(maxlen=4)
    cycle = CognitiveCycle.build_for_env(
        size=5,
        seed=42,
        use_mlp=False,
        observability_store=store,
        interventions=InterventionConfig(
            confidence_gated_selector=True,
            disable_blended_scorer=False,
            confidence_gated_warmup_cycles=2,
            before_blended_warmup_cycles=0,
        ),
    )
    cycle._rbta_preflight_check = lambda *args, **kwargs: EnforcerAction.CONTINUE  # type: ignore[method-assign]

    cycle.step()
    frame = store.latest()

    assert frame is not None
    assert frame.action_rationale["selector_mode"] == "confidence_gated_warmup"
    assert frame.action_rationale["decision_reason"] == "confidence_gated_warmup"
