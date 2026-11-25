#!/usr/bin/env python3
"""Compare MuJoCo camera capture on main thread vs background thread.

Usage:
    PYTHONPATH=python python scripts/camera_probe.py --env reacher
    MUJOCO_GL=egl PYTHONPATH=python python scripts/camera_probe.py --env reacher
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.monitoring.camera_render import is_glitchy_rgb_frame


def _frame_stats(arr: np.ndarray) -> dict:
    g_frac = float(
        ((arr[:, :, 1] > 120) & (arr[:, :, 1] > arr[:, :, 0] + 30)
         & (arr[:, :, 1] > arr[:, :, 2] + 30)).mean()
    )
    return {
        "shape": tuple(arr.shape),
        "mean_rgb": arr.mean(axis=(0, 1)).tolist(),
        "std": float(arr.std()),
        "green_frac": g_frac,
        "glitchy": is_glitchy_rgb_frame(arr),
    }


def _save_png(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception as exc:
        print(f"  (could not save {path}: {exc})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo camera thread probe")
    parser.add_argument("--env", default="reacher",
                        choices=["pendulum", "reacher", "cartpole"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    env_name = {
        "pendulum": "Pendulum-v1",
        "reacher": "Reacher-v5",
        "cartpole": "InvertedPendulum-v5",
    }[args.env]

    gl = __import__("os").environ.get("MUJOCO_GL", "(default)")
    print(f"MuJoCo GL: {gl}")
    print(f"Building {env_name} with enable_camera=True …")

    cycle = CognitiveCycle.build_for_mujoco(
        env_name, seed=args.seed, use_mlp=True, enable_camera=True)
    env = cycle.env
    for _ in range(args.steps):
        cycle.step()

    # Main thread capture
    main_frame = env.render_rgb()
    print("\n=== main thread ===")
    if main_frame is None:
        print("  render_rgb() returned None")
    else:
        arr = np.asarray(main_frame, dtype=np.uint8)
        stats = _frame_stats(arr)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        _save_png(Path("/tmp/phca_camera_main.png"), arr)
        print("  saved: /tmp/phca_camera_main.png")

    # Background thread capture
    holder: dict = {}

    def _bg():
        try:
            holder["frame"] = env.render_rgb()
        except Exception as exc:
            holder["error"] = str(exc)

    t0 = time.monotonic()
    th = threading.Thread(target=_bg, daemon=True)
    th.start()
    th.join(timeout=10.0)
    dt = time.monotonic() - t0

    print(f"\n=== background thread ({dt:.2f}s) ===")
    if holder.get("error"):
        print(f"  error: {holder['error']}")
    elif holder.get("frame") is None:
        print("  render_rgb() returned None")
    else:
        arr = np.asarray(holder["frame"], dtype=np.uint8)
        stats = _frame_stats(arr)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        _save_png(Path("/tmp/phca_camera_thread.png"), arr)
        print("  saved: /tmp/phca_camera_thread.png")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
