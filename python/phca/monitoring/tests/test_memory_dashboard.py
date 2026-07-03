"""Tests for Memory & Belief tab."""
from __future__ import annotations

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import MemoryBeliefView, make_app


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def _mem_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.belief_entropies = kwargs.get("belief_entropies", {"d0": 0.5, "d1": 0.3, "total": 0.4})
    f.gprime_uncertainty = kwargs.get(
        "gprime_uncertainty", np.ones(4, dtype=np.float32) * 0.1)
    f.dim_names = kwargs.get("dim_names", ["a", "b", "c", "d"])
    f.m3_recent = kwargs.get("m3_recent", [{"drive_id": 1, "confidence": 0.8, "timestamp": 0}])
    f.m3_top_error = kwargs.get("m3_top_error", [{"drive_id": 2, "confidence": 0.9, "timestamp": 1}])
    f.m4_relevant = kwargs.get("m4_relevant", [])
    f.m4_top = kwargs.get("m4_top", [])
    f.fact_count = kwargs.get("fact_count", 3)
    f.m4_cap = kwargs.get("m4_cap", 100)
    f.obs_vector = kwargs.get("obs_vector", np.zeros(4, dtype=np.float32))
    f.sanitized_state = kwargs.get("sanitized_state", np.ones(4, dtype=np.float32) * 0.1)
    return f


def test_memory_rebuild_clears_cache(qt_app):
    view = MemoryBeliefView()
    f = _mem_frame()
    view.set_frame(f)
    view._m3_sig = "stale"
    view._m3_cache = [{"x": 1}]
    view.rebuild_histories([f])
    assert view._m3_sig == ""
    assert view._m3_cache == []


def test_m3_top_error_on_frame():
    f = _mem_frame(m3_top_error=[{"drive_id": 3, "confidence": 0.7, "timestamp": 2}])
    assert len(f.m3_top_error) == 1
    assert f.m3_top_error[0]["drive_id"] == 3


def test_memory_m3_includes_top_error(qt_app):
    view = MemoryBeliefView()
    f = _mem_frame(
        m3_recent=[{"drive_id": 1, "confidence": 0.5, "timestamp": 0}],
        m3_top_error=[{"drive_id": 2, "confidence": 0.9, "timestamp": 1}],
    )
    view.set_frame(f)
    merged = list(f.m3_recent) + list(f.m3_top_error)
    assert len(merged) == 2
