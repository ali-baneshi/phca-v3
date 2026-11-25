#!/usr/bin/env python3
"""PHCA v3.0 — Cognitive Dashboard (Observability v2, live visualiser).

Live, graphical, non-invasive view of a cognitive cycle run. The dashboard
surfaces what the architecture is actually doing: the active goal drive, MDIM
drive values vs targets, the G' predicted next-cell distribution (confidence-
guarded), prediction-error/confidence trends, attention focus with chunk ids,
RBTA interrupt reasons, action-selection rationale (explore vs exploit, score,
K), and M3/M4 retention-cap engagement.

Threading: the cognitive cycle (incl. M3 SQLite) runs in a daemon thread — the
ONLY writer to the lock-guarded ObservabilityStore. The main thread runs the
matplotlib render loop + recording. Zero overhead when no store is attached.

Recording (Observability v2): no PNG firehose. One JSONL line per cycle (full-
fidelity, ~2KB) for analytics/replay-time reconstruction, plus a SINGLE encoded
video file (mp4 via ffmpeg, gif via Pillow fallback) captured from the live
canvas at --record-fps. Zero PNGs.

Usage (needs a display + matplotlib):
    PYTHONPATH=python python scripts/phca_visualise.py --cycles=1500 --mlp
    PYTHONPATH=python python scripts/phca_visualise.py --env reacher --cycles=300 --mlp
    PYTHONPATH=python python scripts/phca_visualise.py --cycles=200 --no-record   # live only
    MPLBACKEND=Agg PYTHONPATH=python python scripts/phca_visualise.py --cycles=50 --mlp  # headless smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from collections import deque
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.monitoring.observability import (
    ObservabilityStore, SessionRecorder, VideoRecorder,
)
from phca.monitoring.render import build_dashboard, update_dashboard


def _build_cycle(args, store: ObservabilityStore) -> CognitiveCycle:
    if args.env == "gridworld":
        rng = np.random.RandomState(args.seed + 2)
        obstacles = []
        gap_row = rng.randint(0, args.grid_size)
        for r in range(args.grid_size):
            if r != gap_row and args.grid_size >= 5:
                obstacles.append((r, max(1, args.grid_size // 2)))
        return CognitiveCycle.build_for_env(
            size=args.grid_size, seed=args.seed + 2, use_mlp=args.mlp,
            use_continuous=True, obstacles=obstacles,
            observability_store=store,
        )
    return CognitiveCycle.build_for_mujoco(
        args.env, seed=args.seed, use_mlp=args.mlp,
        observability_store=store,
    )


def main() -> None:
    ensure_logging()
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt
    import time

    parser = argparse.ArgumentParser(description="PHCA Cognitive Dashboard (live)")
    parser.add_argument("--env", default="gridworld",
                        choices=["gridworld", "pendulum", "reacher", "cartpole"],
                        help="gridworld (default) | pendulum | reacher | cartpole")
    parser.add_argument("--cycles", type=int, default=1500)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp", action="store_true", help="use MLP world model")
    parser.add_argument("--fps", type=float, default=12.0,
                        help="live screen render rate (frames/sec)")
    parser.add_argument("--record-fps", type=float, default=6.0,
                        help="video capture rate (<= --fps to keep GUI smooth)")
    parser.add_argument("--record-format", default="auto", choices=["auto", "mp4", "gif"],
                        help="video format (auto: mp4 if ffmpeg else gif)")
    parser.add_argument("--no-record", action="store_true", help="live view only, no session/video")
    parser.add_argument("--record-dir", default="logs/sessions")
    args = parser.parse_args()

    if args.env == "cartpole":
        args.env = "InvertedPendulum-v5"
    elif args.env == "pendulum":
        args.env = "Pendulum-v1"
    elif args.env == "reacher":
        args.env = "Reacher-v5"

    is_grid = (args.env == "gridworld")
    store = ObservabilityStore(maxlen=max(1000, args.cycles))
    recorder = SessionRecorder(root=args.record_dir, fps=args.record_fps,
                               record=not args.no_record)
    video = VideoRecorder(fps=args.record_fps, dpi=90, format=args.record_format)
    session_dir = recorder.start({"env": args.env, "seed": args.seed,
                                  "cycles": args.cycles, "grid_size": args.grid_size,
                                  "mlp": args.mlp, "record_fps": args.record_fps,
                                  "format": args.record_format})
    if session_dir:
        print(f"Recording session -> {session_dir}")

    # The cognitive cycle (incl. M3 SQLite) MUST be built and run in the SAME
    # thread (SQLite objects are thread-affine). The store is the only shared
    # object (lock-guarded). The main thread runs the matplotlib render loop.
    stop_flag = threading.Event()
    cycle_holder: dict = {}

    def _run_cycle():
        try:
            cycle = _build_cycle(args, store)
            cycle_holder["cycle"] = cycle
            for _ in range(args.cycles):
                if stop_flag.is_set():
                    break
                cycle.step()
        except Exception as e:
            cycle_holder["error"] = str(e)
            print(f"[cycle thread] error: {e}", file=sys.stderr)
        finally:
            # Close M3's SQLite connection IN THIS THREAD before exit. If left
            # open, its __del__ runs at interpreter shutdown in the main thread
            # and raises sqlite3.ProgrammingError, which + Qt teardown aborts
            # the process (SIGABRT). close() sets _conn=None so __del__ no-ops.
            try:
                m3 = getattr(getattr(cycle_holder.get("cycle", None),
                                     "consolidation", None), "m3", None)
                if m3 is not None:
                    m3.close()
            except Exception:
                pass
            try:
                if "cycle" in cycle_holder:
                    cycle_holder["cycle"].env.close()
            except Exception:
                pass
            stop_flag.set()

    ct = threading.Thread(target=_run_cycle, daemon=True)
    ct.start()

    fig = plt.figure(figsize=(14, 7))
    handle = build_dashboard(fig, is_grid=is_grid)
    video_path, video_fmt = (None, None)
    if not args.no_record and session_dir is not None:
        video_path, video_fmt = video.start(session_dir, fig)
        if video_path:
            print(f"Video -> {video_path} ({video_fmt})")
        else:
            print(f"Video unavailable ({video_fmt}); JSONL-only recording.")

    interval = 1.0 / max(args.fps, 0.1)
    record_interval = 1.0 / max(args.record_fps, 0.1)
    last_recorded_cycle = -1   # last cycle_id written to JSONL
    last_rendered_cycle = -1   # last cycle_id rendered to the screen
    n_video = 0
    last_grab_time = -record_interval  # so the first new frame is captured
    plt.show(block=False)
    try:
        while True:
            # Drain all new frames since last tick: JSONL every cycle (cheap),
            # render only the latest, grab video at record-fps.
            new = [f for f in store.latest_n(256) if f.cycle_id > last_recorded_cycle]
            if new:
                for f in new:
                    recorder.record(f)
                last_recorded_cycle = new[-1].cycle_id
                latest = new[-1]
                if latest.cycle_id > last_rendered_cycle:
                    update_dashboard(handle, latest)
                    last_rendered_cycle = latest.cycle_id
                # Video capture: reuse the canvas just drawn by plt.pause below.
                # We grab after the first update of a record tick.
            done = stop_flag.is_set() and len(store) >= args.cycles
            err = cycle_holder.get("error")
            if err:
                # Surface the cycle-thread error into the dashboard and exit.
                handle.artists["action_txt"].set_text(
                    f"CYCLE THREAD ERROR:\n  {err}\n\nRun aborted.")
                handle.artists["action_txt"].set_color("#d62728")
                fig.canvas.draw_idle()
                break
            if done:
                break
            try:
                plt.pause(interval)
            except KeyboardInterrupt:
                stop_flag.set()
                break
            if not plt.fignum_exists(fig.number):
                stop_flag.set()
                break
            # Grab a video frame at record-fps, after the screen has rendered.
            if video.enabled and video_path:
                now = time.monotonic()
                if now - last_grab_time >= record_interval and last_rendered_cycle >= 0:
                    video.grab(fig)
                    n_video = video.frames
                    last_grab_time = now
    finally:
        stop_flag.set()
        ct.join(timeout=5.0)
        # Ensure the terminal frame is recorded (JSONL) + captured (video).
        snap = store.latest_n(64)
        if snap and snap[-1].cycle_id > last_recorded_cycle:
            recorder.record(snap[-1])
            last_recorded_cycle = snap[-1].cycle_id
            if video.enabled and video_path:
                try:
                    update_dashboard(handle, snap[-1])
                    fig.canvas.draw()
                    video.grab(fig)
                    n_video = video.frames
                except Exception:
                    pass
        recorder.flush()
        recorder.close()
        video.close()
        try:
            plt.close(fig)
        except Exception:
            pass
        err = cycle_holder.get("error")
        msg = (f"Done. {len(store)} cycles run; {recorder.count} JSONL lines; "
               f"{n_video} video frames.")
        if video_path:
            try:
                msg += f" video={video_path.name}({video_path.stat().st_size} B)"
            except Exception:
                pass
        if session_dir:
            msg += f" session={session_dir}"
        if err:
            msg += f" [ERROR: {err}]"
        print(msg)


if __name__ == "__main__":
    main()
