"""Tests for adaptive grounding metadata."""

from phca.asi.adapter import GroundingAdapter


def test_grounding_adapter_degrades_after_repeated_sensor_failures():
    adapter = GroundingAdapter()
    assert adapter.update(sensor_failure_count=3, prediction_confidence=0.9) == 0


def test_grounding_adapter_forces_raw_level_after_severe_failure():
    adapter = GroundingAdapter()
    adapter.current_level = 2
    assert adapter.update(sensor_failure_count=5, prediction_confidence=0.9) == 0


def test_grounding_adapter_recovers_one_level_after_healthy_window():
    adapter = GroundingAdapter()
    adapter.current_level = 0
    for _ in range(adapter.HEALTHY_THRESHOLD - 1):
        assert adapter.update(0, 0.9) == 0
    assert adapter.update(0, 0.9) == 1


def test_grounding_adapter_does_not_recover_on_low_confidence():
    adapter = GroundingAdapter()
    adapter.current_level = 0
    for _ in range(adapter.HEALTHY_THRESHOLD + 2):
        assert adapter.update(0, 0.5) == 0
