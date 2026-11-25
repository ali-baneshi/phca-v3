"""Quarantined from python/tests/test_maturation.py with phca.monitoring.

Covered GUI frame serialization (ObservabilityFrame.to_json), not cycle behavior.
"""

import json

from phca.core.cycle import CognitiveCycle
from phca.monitoring.observability import ObservabilityFrame


class TestTrackObservatoryExport:
    def test_observability_frame_resilience_json_roundtrip(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        cycle.on_task_boundary(2)
        m = cycle.step()
        m.failure_events = ["B4"]
        m.recovery_active = True
        m.task_id = 2
        cycle.metrics_history[-1] = m
        frame = ObservabilityFrame.from_cycle(cycle)
        raw = frame.to_json()
        assert raw["task_id"] == 2
        assert raw["failure_events"] == ["B4"]
        assert raw["recovery_active"] is True
        restored = json.loads(json.dumps(raw))
        assert restored["task_id"] == 2


# Quarantined from python/tests/test_static_contracts.py::TestObservabilityContracts.
def test_observability_frame_exports_resilience_fields():
    import ast
    from pathlib import Path

    import pytest

    src = (Path(__file__).resolve().parents[1] / "phca" / "monitoring" / "observability.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ObservabilityFrame":
            field_names = {
                t.target.id
                for t in node.body
                if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
            }
            missing = {"failure_events", "recovery_active", "task_id"} - field_names
            assert not missing, f"ObservabilityFrame missing fields: {missing}"
            return
    pytest.fail("ObservabilityFrame class not found")
