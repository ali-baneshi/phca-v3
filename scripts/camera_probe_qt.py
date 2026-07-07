#!/usr/bin/env python3
"""Compare MuJoCo render_rgb with and without an active Qt window.

Usage:
    PYTHONPATH=python python scripts/camera_probe_qt.py --env reacher
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

import numpy as np
from PyQt5 import QtWidgets

from phca.core.cycle import CognitiveCycle
from phca.monitoring.camera_render import camera_frame_stats, is_glitchy_rgb_frame


def _stats(arr: np.ndarray) -> dict:
    stats = camera_frame_stats(arr)
    stats["glitchy"] = is_glitchy_rgb_frame(arr)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo camera probe inside Qt")
    parser.add_argument("--env", default="reacher", choices=["pendulum", "reacher", "cartpole"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    env_name = {
        "pendulum": "Pendulum-v1",
        "reacher": "Reacher-v5",
        "cartpole": "InvertedPendulum-v5",
    }[args.env]

    print(f"Building {env_name} …")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '(auto)')}")
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

    print("\n=== probe sequence ===")
    for idx in range(max(int(args.samples), 1)):
        arr = cycle.env.render_rgb()
        if arr is None:
            print(f"  sample[{idx}] none")
            continue
        st = _stats(np.asarray(arr, dtype=np.uint8))
        print(
            f"  sample[{idx}] std={st['std']:.2f} green_frac={st['green_frac']:.3f} "
            f"glitchy={st['glitchy']} shape={st['shape']}"
        )

    try:
        cycle.env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
