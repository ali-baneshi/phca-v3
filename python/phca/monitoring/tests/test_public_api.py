"""Public API stability tests for phca.monitoring (Phase 15)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import phca.monitoring as monitoring
from phca.monitoring import (
    __all__,
    export_session_csv,
    frame_from_json,
    frame_to_json,
    normalize_observability_json,
)
from phca.monitoring.export import SESSION_SCALAR_COLUMNS

_REPO = Path(__file__).resolve().parents[4]
_PYTHONPATH = str(_REPO / "python")
_FIXTURE_JSONL = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _PYTHONPATH
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_import_smoke():
    assert monitoring.__all__


def test_all_exports_exist():
    for name in __all__:
        obj = getattr(monitoring, name)
        assert obj is not None, name


def test_import_does_not_load_pyqt5():
    proc = _run_isolated(
        "import sys\n"
        "import phca.monitoring\n"
        "assert 'PyQt5' not in sys.modules, sorted(sys.modules)\n"
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_frame_round_trip():
    line = _FIXTURE_JSONL.read_text().splitlines()[0]
    obj = json.loads(line)
    f = frame_from_json(obj)
    out = frame_to_json(f)
    norm = normalize_observability_json(out)
    assert int(norm["cycle_id"]) == int(obj["cycle_id"])
    assert float(norm["prediction_error"]) == float(obj["prediction_error"])
    assert norm.get("env_kind") == obj.get("env_kind")


def test_export_session_csv_round_trip(tmp_path):
    d = tmp_path / "session"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"env": "Reacher-v5", "cycles": 12}))
    (d / "timeseries.jsonl").write_text(_FIXTURE_JSONL.read_text())
    out = export_session_csv(d)
    assert out == d / "scalars.csv"
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12
    assert list(rows[0].keys()) == SESSION_SCALAR_COLUMNS
    assert float(rows[0]["prediction_error"]) == 12.0
    assert rows[5]["spike"] in ("True", "False")
