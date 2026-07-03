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


def test_m3_top_error_serialized_but_bulk_memory_lists_dropped():
    f = _mem_frame(
        m3_recent=[{"drive_id": 1, "confidence": 0.5, "timestamp": 0}],
        m3_top_error=[{"drive_id": 3, "confidence": 0.7, "timestamp": 2}],
        m4_relevant=[{"fact_type": "x", "confidence": 0.4, "timestamp": 1}],
        m4_top=[{"fact_type": "y", "confidence": 0.6, "timestamp": 2}],
    )
    payload = f.to_json()
    assert payload["m3_top_error"][0]["drive_id"] == 3
    assert "m3_recent" not in payload
    assert "m4_relevant" not in payload
    assert "m4_top" not in payload


def test_memory_m3_includes_top_error(qt_app):
    view = MemoryBeliefView()
    f = _mem_frame(
        m3_recent=[{"drive_id": 1, "confidence": 0.5, "timestamp": 0}],
        m3_top_error=[{"drive_id": 2, "confidence": 0.9, "timestamp": 1}],
    )
    view.set_frame(f)
    merged = list(f.m3_recent) + list(f.m3_top_error)
    assert len(merged) == 2


def test_memory_replay_empty_banner(qt_app):
    view = MemoryBeliefView()
    f = _mem_frame(m3_recent=[], m3_top_error=[], m4_relevant=[], m4_top=[])
    view.set_frame(f, replay=True)
    assert view._replay is True
    assert view._memory_live_empty(f)


def test_memory_replay_jsonl_uncertainty_list_broadcasts_entropy(qt_app):
    from phca.monitoring.render import frame_from_json

    f = frame_from_json({
        "cycle_id": 1,
        "belief_entropies": {"G'": 0.5},
        "gprime_uncertainty": [0.1, 0.2, 0.3],
        "dim_names": ["a", "b", "c"],
    })
    view = MemoryBeliefView()
    ent = view._per_dim_entropy(f.belief_entropies, f.dim_names, f)
    assert isinstance(f.gprime_uncertainty, np.ndarray)
    assert ent == [pytest.approx(0.5)] * 3


def test_m4_retention_score_not_support(qt_app):
    from phca.monitoring.qt_dashboard import _retention_score

    f = _mem_frame(cycle_id=100)
    fac = {"timestamp": 90, "confidence": 0.5, "frequency": 42, "support": 42}
    score = _retention_score(
        max(0, int(f.cycle_id) - int(fac.get("timestamp", f.cycle_id))),
        max(float(fac.get("confidence", 0.1) or 0.1) * 80.0, 5.0))
    assert score <= 1.0
    assert score < 1.0
