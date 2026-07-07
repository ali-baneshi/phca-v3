"""Static architecture contract tests (Track B).

Encodes the contract map from docs/static_audit_2026-07-07.md.
Known G5 gaps use xfail so CI stays green while debt remains visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PHCA = _REPO / "python" / "phca"
_SCRIPTS = _REPO / "scripts"


def _python_sources_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]


def _grep_callers(symbol: str, *, exclude_def: Path) -> list[str]:
    """Return paths that invoke ``symbol`` (not mere string mentions)."""
    pattern = f".{symbol}("
    hits: list[str] = []
    search_roots = [_REPO / "python" / "phca", _SCRIPTS]
    for root in search_roots:
        for path in _python_sources_under(root):
            if path == exclude_def:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern in text:
                hits.append(str(path.relative_to(_REPO)))
    return hits


class TestContinualLearningContracts:
    def test_benchmark_level4_calls_on_task_boundary(self):
        src = (_SCRIPTS / "benchmark_level4.py").read_text(encoding="utf-8")
        assert "on_task_boundary" in src
        assert "cycle.on_task_boundary(task.task_id)" in src

    def test_cycle_stores_task_id_in_m3(self):
        src = (_PHCA / "core" / "cycle.py").read_text(encoding="utf-8")
        assert "task_id=self._current_task_id" in src

    @pytest.mark.xfail(
        reason="G5-01: M3 sample_episodes has no callers outside m3_episodic.py",
        strict=True,
    )
    def test_m3_sample_episodes_wired_to_learning_path(self):
        def_file = _PHCA / "memory" / "m3_episodic.py"
        callers = _grep_callers("sample_episodes", exclude_def=def_file)
        assert callers, (
            "sample_episodes must be called from cycle/gprime learning path; "
            f"found no callers outside {def_file.name}"
        )


class TestObservabilityContracts:
    def test_cycle_metrics_has_resilience_fields(self):
        src = (_PHCA / "core" / "cycle.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CycleMetrics":
                field_names = {
                    t.target.id
                    for t in node.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                }
                assert "failure_events" in field_names
                assert "recovery_active" in field_names
                assert "task_id" in field_names
                return
        pytest.fail("CycleMetrics class not found")

    @pytest.mark.xfail(
        reason="G5-02..04: ObservabilityFrame does not export resilience/task fields",
        strict=True,
    )
    def test_observability_frame_exports_resilience_fields(self):
        src = (_PHCA / "monitoring" / "observability.py").read_text(encoding="utf-8")
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


class TestResilienceContracts:
    def test_resilience_runs_after_rbta_in_cycle(self):
        src = (_PHCA / "core" / "cycle.py").read_text(encoding="utf-8")
        rbta_pos = src.index("self.rbta.check_cycle")
        res_pos = src.index("self._resilience_detector.detect")
        assert res_pos > rbta_pos, "Resilience detect must run after RBTA check_cycle"

    def test_b4_recovery_calls_on_forgetting_detected(self):
        src = (_PHCA / "resilience" / "recovery.py").read_text(encoding="utf-8")
        assert "on_forgetting_detected" in src


class TestReplayArchitecture:
    def test_gprime_internal_replay_buffer_documented_capacity(self):
        src = (_PHCA / "world_model" / "mlp.py").read_text(encoding="utf-8")
        assert "replay_capacity" in src
        assert "_replay_buffer" in src

    @pytest.mark.xfail(
        reason="G5-01: continual learning relies on G' FIFO only; M3 not wired",
        strict=True,
    )
    def test_m3_and_gprime_both_used_in_learn(self):
        cycle_src = (_PHCA / "core" / "cycle.py").read_text(encoding="utf-8")
        assert ".sample_episodes(" in cycle_src, (
            "cycle.py should call m3.sample_episodes for continual replay"
        )
