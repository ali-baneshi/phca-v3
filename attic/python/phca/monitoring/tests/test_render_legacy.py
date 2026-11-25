"""Smoke tests for legacy matplotlib dashboard path (--from-jsonl)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.render import build_dashboard, update_dashboard


def test_build_and_update_dashboard_grid_smoke():
    fig = plt.figure(figsize=(10, 6))
    handle = build_dashboard(fig, is_grid=True)
    frame = ObservabilityFrame(
        cycle_id=0,
        env_kind="grid",
        agent_pos=(1, 1),
        goal_pos=(4, 4),
        prediction_error=0.5,
        drive_levels=[0.5] * 6,
        action_rationale={"explored": False, "eps": 0.1, "goal_id": 1, "best_score": 0.5},
    )
    update_dashboard(handle, frame)
    assert handle is not None
    assert len(fig.axes) >= 1
    plt.close(fig)
