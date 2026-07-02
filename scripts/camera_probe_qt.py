#!/usr/bin/env python3
"""Compare MuJoCo render_rgb with and without an active Qt window.

Usage:
    PYTHONPATH=python python scripts/camera_probe_qt.py --env reacher
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
from PyQt5 import QtWidgets

from phca.core.cycle import CognitiveCycle
from phca.monitoring.camera_render import is_glitchy_rgb_frame


def _stats(arr: np.ndarray) -> dict:
    g_frac = float(
        ((arr[:, :, 1] > 120) & (arr[:, :, 1] > arr[:, :, 0] + 30)
         & (arr[:, :, 1] > arr[:, :, 2] + 30)).mean()
    )
    return {
        "shape": tuple(arr.shape),
        "std": float(arr.std()),
        "mean_rgb": arr.mean(axis=(0, 1)).tolist(),
        "green_frac": g_frac,
        "glitchy": is_glitchy_rgb_frame(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo camera probe inside Qt")
    parser.add_argument("--env", default="reacher", choices=["pendulum", "reacher", "cartpole"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    env_name = {
        "pendulum": "Pendulum-v1",
        "reacher": "Reacher-v5",
        "cartpole": "InvertedPendulum-v5",
    }[args.env]

    print(f"Building {env_name} …")
    cycle = CognitiveCycle.build_for_mujoco(
        env_name, seed=args.seed, use_mlp=True, enable_camera=True)
    for _ in range(args.steps):
        cycle.step()

    # Without Qt window
    arr0 = cycle.env.render_rgb()
    print("\n=== without Qt window ===")
    if arr0 is None:
        print("  None")
    else:
        for k, v in _stats(np.asarray(arr0, dtype=np.uint8)).items():
            print(f"  {k}: {v}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lbl = QtWidgets.QLabel("camera probe")
    lbl.resize(320, 240)
    lbl.show()
    app.processEvents()

    arr1 = cycle.env.render_rgb()
    print("\n=== with Qt window visible ===")
    if arr1 is None:
        print("  None")
    else:
        for k, v in _stats(np.asarray(arr1, dtype=np.uint8)).items():
            print(f"  {k}: {v}")

    try:
        cycle.env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
