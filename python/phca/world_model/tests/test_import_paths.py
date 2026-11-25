"""Import-path tests: MLP/Observatory paths must not eagerly load pgmpy."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_PYTHONPATH = str(_REPO / "python")


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


def test_cycle_import_does_not_load_pgmpy():
    proc = _run_isolated(
        "import sys\n"
        "from phca.core.cycle import CognitiveCycle\n"
        "assert 'pgmpy' not in sys.modules, list(sys.modules.keys())\n"
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_core_import_does_not_load_gui_stack():
    proc = _run_isolated(
        "import sys\n"
        "import phca\n"
        "from phca.core.cycle import CognitiveCycle\n"
        "loaded = [m for m in ('PyQt5', 'matplotlib', 'PIL') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
