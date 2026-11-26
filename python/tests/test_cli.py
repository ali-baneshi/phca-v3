"""Behavioral tests for the `phca` command line surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PYTHONPATH = str(_REPO / "python")


def _phca(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _PYTHONPATH
    env["MUJOCO_GL"] = "disabled"
    env["PHCA_LOG_LEVEL"] = "ERROR"
    return subprocess.run(
        [sys.executable, "-m", "phca", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _result(tmp_path: Path, name: str = "r.json") -> dict:
    path = tmp_path / name
    proc = _phca("run", "--cycles", "30", "--seed", "0", "--quiet", "--out", str(path))
    assert proc.returncode == 0, proc.stderr
    return json.loads(path.read_text())


class TestRun:
    def test_run_writes_result_with_documented_schema(self, tmp_path: Path) -> None:
        result = _result(tmp_path)
        assert result["schema"] == "phca.result.v1"
        assert result["command"] == "run"
        assert result["config"]["env"] == "grid"
        assert result["config"]["cycles"] == 30
        summary = result["summary"]
        assert summary["cycles"] == 30
        assert sum(summary["action_counts"].values()) == 30
        assert sum(summary["selector_mode_counts"].values()) == 30
        assert summary["first_goal_cycle"] is not None

    def test_run_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        first = _result(tmp_path, "a.json")
        second = _result(tmp_path, "b.json")
        assert first["summary"] == second["summary"]
        assert first["config"] == second["config"]

    def test_run_prints_a_human_summary_unless_quiet(self) -> None:
        loud = _phca("run", "--cycles", "8", "--seed", "0")
        assert loud.returncode == 0, loud.stderr
        assert "cycles" in loud.stdout
        assert "first goal" in loud.stdout.lower()

        quiet = _phca("run", "--cycles", "8", "--seed", "0", "--quiet")
        assert quiet.returncode == 0, quiet.stderr
        assert quiet.stdout.strip() == ""

    def test_relocating_goals_records_one_latency_per_goal(self, tmp_path: Path) -> None:
        path = tmp_path / "relocate.json"
        proc = _phca(
            "run", "--cycles", "120", "--seed", "1", "--quiet",
            "--relocate-goal-every", "30", "--out", str(path),
        )
        assert proc.returncode == 0, proc.stderr
        summary = json.loads(path.read_text())["summary"]
        assert summary["goal_events"] >= 1
        assert len(summary["goal_latencies"]) == summary["goal_events"]

    def test_learned_selector_is_selectable_and_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "learned.json"
        proc = _phca(
            "run", "--cycles", "15", "--seed", "0", "--quiet",
            "--selector", "learned", "--out", str(path),
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(path.read_text())
        assert result["config"]["selector"] == "learned"
        assert "prediction_scored" in result["summary"]["selector_mode_counts"]


class TestArgumentValidation:
    @pytest.mark.parametrize(
        "args",
        [
            ["run", "--cycles", "0"],
            ["run", "--env", "lunar-lander"],
            ["run", "--selector", "oracle"],
            ["run", "--size", "7"],
        ],
    )
    def test_invalid_arguments_exit_2(self, args: list[str]) -> None:
        proc = _phca(*args)
        assert proc.returncode == 2, proc.stderr
        assert proc.stderr.strip()

    def test_missing_mujoco_exits_3_with_install_hint(self, tmp_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _PYTHONPATH
        env["MUJOCO_GL"] = "disabled"
        blocker = tmp_path / "block" / "mujoco"
        blocker.mkdir(parents=True)
        (blocker / "__init__.py").write_text('raise ImportError("blocked")\n')
        env["PYTHONPATH"] = f"{blocker.parent}{os.pathsep}{_PYTHONPATH}"
        proc = subprocess.run(
            [sys.executable, "-m", "phca", "run", "--env", "pendulum", "--cycles", "2"],
            capture_output=True, text=True, env=env, check=False,
        )
        assert proc.returncode == 3, proc.stderr
        assert "phca[mujoco]" in proc.stderr


class TestBench:
    def test_bench_compares_both_selectors_per_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "bench.json"
        proc = _phca(
            "bench", "--sizes", "5", "--seeds", "0", "--cycles", "80",
            "--relocate-goal-every", "20", "--quiet", "--out", str(path),
        )
        assert proc.returncode == 0, proc.stderr
        rows = json.loads(path.read_text())["rows"]
        assert {row["selector"] for row in rows} == {"geometry", "learned"}
        for row in rows:
            assert row["size"] == 5
            assert row["goal_events"] >= 1
            assert row["mean_goal_latency"] is not None

    def test_bench_prints_a_table_unless_quiet(self) -> None:
        proc = _phca("bench", "--sizes", "5", "--seeds", "0", "--cycles", "30",
                     "--relocate-goal-every", "25")
        assert proc.returncode == 0, proc.stderr
        assert "geometry" in proc.stdout
        assert "learned" in proc.stdout


class TestReplay:
    def test_replay_summarizes_a_result_file(self, tmp_path: Path) -> None:
        result_path = tmp_path / "r.json"
        _result(tmp_path)
        proc = _phca("replay", "--session", str(result_path))
        assert proc.returncode == 0, proc.stderr
        assert "selector" in proc.stdout.lower()

    def test_missing_session_file_exits_2(self, tmp_path: Path) -> None:
        proc = _phca("replay", "--session", str(tmp_path / "nope.json"))
        assert proc.returncode == 2
        assert "not found" in proc.stderr.lower()

    def test_malformed_session_file_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        proc = _phca("replay", "--session", str(bad))
        assert proc.returncode == 2
        assert "json" in proc.stderr.lower()


def test_cli_import_loads_no_gui_or_graph_stack() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _PYTHONPATH
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, phca.cli\n"
         "bad = [m for m in ('PyQt5', 'matplotlib', 'pgmpy', 'torch') if m in sys.modules]\n"
         "assert not bad, bad\n"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode == 0, proc.stderr


class TestReplacedHarnessGuarantees:
    """Coverage that used to live in script-level tests (benchmark.py,
    reproduce.py, test_benchmark_harness.py), restated against the CLI."""

    def test_trace_records_selector_evidence(self, tmp_path: Path) -> None:
        from phca.core.cycle import CognitiveCycle
        from phca.evaluation.trace import TraceCollector, summarize_action_selection

        trace = TraceCollector(capacity=64)
        cycle = CognitiveCycle.build_for_env(size=5, seed=0, trace_collector=trace)
        for _ in range(6):
            cycle.step()
        records = trace.snapshot()
        assert len(records) == 6
        summary = summarize_action_selection(records)
        assert summary["n_records"] == 6
        assert summary["rationale_coverage"] == 1.0
        assert records[0].selector_mode

    def test_result_file_round_trips_through_replay(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        _result(tmp_path)
        payload = json.loads(path.read_text())
        assert payload["summary"]["goal_latencies"]
        assert all(lat >= 1 for lat in payload["summary"]["goal_latencies"])
        proc = _phca("replay", "--session", str(path))
        assert proc.returncode == 0, proc.stderr
        assert payload["schema"] in proc.stdout

    def test_wrong_schema_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "old.json"
        bad.write_text(json.dumps({"schema": "phca.result.v0", "command": "run"}))
        proc = _phca("replay", "--session", str(bad))
        assert proc.returncode == 2
        assert "schema" in proc.stderr.lower()

    def test_bench_selector_choice_is_actually_different(self, tmp_path: Path) -> None:
        path = tmp_path / "bench.json"
        proc = _phca(
            "bench", "--sizes", "5", "--seeds", "0", "--cycles", "60",
            "--relocate-goal-every", "20", "--quiet", "--out", str(path),
        )
        assert proc.returncode == 0, proc.stderr
        rows = {r["selector"]: r for r in json.loads(path.read_text())["rows"]}
        assert "pure_geometry_ablation" in rows["geometry"]["selector_mode_counts"]
        assert "prediction_scored" in rows["learned"]["selector_mode_counts"]
