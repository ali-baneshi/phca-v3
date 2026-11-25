"""Import-path and dependency-surface tests for the core package."""
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


def test_declared_runtime_dependencies_are_numpy_only():
    """The core runs on numpy alone; everything else is an optional extra."""
    import re

    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M)
    assert block is not None
    names = {
        re.split(r"[<>=!~\[ ]", item.strip().strip('"'))[0].lower()
        for item in block.group(1).split(",")
        if item.strip()
    }
    assert names == {"numpy"}, names
