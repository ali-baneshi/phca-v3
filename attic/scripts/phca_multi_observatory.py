#!/usr/bin/env python3
"""Thin launcher for aligned 2-agent Observatory runs (Phase 17)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OBS = Path(__file__).resolve().parent / "phca_observatory.py"


def main() -> int:
    agents = os.environ.get("PHCA_OBSERVATORY_AGENTS", "2")
    cmd = [
        sys.executable,
        str(_OBS),
        f"--agents={agents}",
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_ROOT / "python"))
    return int(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
