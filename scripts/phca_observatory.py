#!/usr/bin/env python3
"""PHCA v3.0 — Cognitive Observatory (Observability v3, PyQt5 launcher).

Live, multi-tab Qt dashboard that reveals the PHCA cognition as it happens:
overview, animated cognitive-flow pipeline (per-module ms + RBTA violations),
action-selection candidates, phase-space/trajectory, and a retention/resources
console. Reuses the v2 data layer + JSONL recorder unchanged; only the
rendering is a real Qt GUI. Zero overhead when no store is attached.

Threading: the cognitive cycle (incl. M3 SQLite) runs in a daemon thread — the
ONLY writer to the lock-guarded ObservabilityStore. The main thread runs the
QApplication + a QTimer (~--poll-ms) that polls the store and mutates the
persistent Qt widgets (no widget rebuild per tick). M3 is closed in-thread
(same pattern as v2) to avoid the SIGABRT on interpreter shutdown.

Recording: one JSONL line per cycle (full-fidelity) via SessionRecorder; an
optional single mp4 is captured from the widget via ``QPixmap.grab()`` piped to
``ffmpeg`` when ``--record-video`` is given. Zero PNGs.

Usage (needs a display + PyQt5):
    PYTHONPATH=python python scripts/phca_observatory.py --cycles=1500 --mlp
    PYTHONPATH=python python scripts/phca_observatory.py --env reacher --cycles=300 --mlp
    PYTHONPATH=python python scripts/phca_observatory.py --cycles=200 --no-record
    QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp  # headless smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.monitoring.observability import ObservabilityStore, SessionRecorder
from phca.monitoring.qt_dashboard import ObservatoryWindow, make_app
from PyQt5 import QtCore


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
        render_mode="rgb_array",
    )


class _VideoPipe:
    """Optional: grab the QMainWindow widget to an mp4 via an ffmpeg pipe."""

    def __init__(self, path: Path, fps: float, w: int, h: int):
        self.path = path
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
               str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.frames = 0

    def grab(self, widget) -> None:
        if self.proc.stdin is None:
            return
        try:
            from PyQt5.QtGui import QImage
            pm = widget.grab()
            img = pm.toImage().scaled(widget.width(), widget.height())
            # Force RGB888 (3 bytes/pixel, R-G-B order) so ffmpeg rgb24 matches.
            fmt_rgb888 = getattr(QImage, "Format_RGB888", None)
            if fmt_rgb888 is not None:
                img = img.convertToFormat(fmt_rgb888)
            ptr = img.bits(); ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 3)
            self.proc.stdin.write(arr.tobytes())
            self.frames += 1
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA Cognitive Observatory (live, PyQt5)")
    parser.add_argument("--env", default="gridworld",
                        choices=["gridworld", "pendulum", "reacher", "cartpole"],
                        help="gridworld (default) | pendulum | reacher | cartpole")
    parser.add_argument("--cycles", type=int, default=1500)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp", action="store_true", help="use MLP world model")
    parser.add_argument("--poll-ms", type=int, default=30, help="Qt poll interval (ms)")
    parser.add_argument("--render-fps", type=float, default=30.0,
                        help="max dashboard render rate (decoupled from poll/cycle; "
                             "JSONL still records every cycle). 0 = render every poll.")
    parser.add_argument("--no-record", action="store_true", help="live view only, no JSONL/video")
    parser.add_argument("--record-video", action="store_true",
                        help="capture an mp4 from the widget (requires ffmpeg)")
    parser.add_argument("--record-fps", type=float, default=8.0, help="video capture rate")
    parser.add_argument("--record-dir", default="logs/sessions")
    parser.add_argument("--video-cycle-ms", type=int, default=1500,
                        help="when recording video, auto-cycle tabs every N ms so the "
                             "mp4 captures all 7 tabs (0 = off, stay on clicked tab)")
    args = parser.parse_args()

    if args.env == "cartpole":
        args.env = "InvertedPendulum-v5"
    elif args.env == "pendulum":
        args.env = "Pendulum-v1"
    elif args.env == "reacher":
        args.env = "Reacher-v5"

    store = ObservabilityStore(maxlen=max(1000, args.cycles))
    recorder = SessionRecorder(root=args.record_dir, fps=args.record_fps,
                               record=not args.no_record)
    session_dir = recorder.start({"env": args.env, "seed": args.seed,
                                  "cycles": args.cycles, "grid_size": args.grid_size,
                                  "mlp": args.mlp, "record_fps": args.record_fps,
                                  "renderer": "qt"})
    if session_dir:
        print(f"Recording session -> {session_dir}")

    # Cognitive cycle (incl. M3 SQLite) MUST be built+run in ONE thread.
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
            # Close M3's SQLite connection IN THIS THREAD (else __del__ at
            # interpreter shutdown runs in the main thread -> SIGABRT).
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

    app = make_app()
    win = ObservatoryWindow()
    win.show()
    ctrl = win.controller

    # Optional video pipe.
    video: "_VideoPipe | None" = None
    if args.record_video and session_dir is not None:
        # Pipe size set after the window is shown (real geometry).
        app.processEvents()
        video = _VideoPipe(session_dir / "session.mp4", args.record_fps,
                           win.width(), win.height())
        print(f"Video -> {video.path}")

    last_recorded_cycle = -1
    last_rendered_cycle = -1
    record_interval = 1.0 / max(args.record_fps, 0.1)
    last_grab = -record_interval
    target = args.cycles
    # Render throttle: decouple dashboard repaint from poll/cycle rate so the
    # visible tab is repainted at most --render-fps times/sec (stable, no
    # flicker). JSONL still records every cycle (above). 0 => render every poll.
    render_interval = (1.0 / args.render_fps) if args.render_fps > 0 else 0.0
    last_render = -render_interval
    # Auto-cycle tabs during video recording so one mp4 captures all 7 tabs.
    tabs = win._tabs
    n_tabs = tabs.count()
    cycle_tab_ms = args.video_cycle_ms if (args.record_video and args.video_cycle_ms > 0) else 0
    last_tab_switch = 0.0
    user_tab = {"i": tabs.currentIndex()}  # remember user's tab when not auto-cycling

    def _tick():
        nonlocal last_recorded_cycle, last_rendered_cycle, last_grab, last_tab_switch
        nonlocal last_render
        try:
            now = time.monotonic()
            # Drain new frames: JSONL every cycle (always), render only the latest
            # and only when the render throttle allows.
            new = [f for f in store.latest_n(256) if f.cycle_id > last_recorded_cycle]
            if new:
                for f in new:
                    recorder.record(f)
                last_recorded_cycle = new[-1].cycle_id
                latest = new[-1]
                if (latest.cycle_id > last_rendered_cycle
                        and (render_interval <= 0 or now - last_render >= render_interval)):
                    ctrl.update(latest, new, cycle_holder.get("error"))
                    last_rendered_cycle = latest.cycle_id
                    last_render = now
            err = cycle_holder.get("error")
            if err:
                ctrl.update(store.latest(), None, err)
                _finish(); return
            done = stop_flag.is_set() and len(store) >= target
            if done:
                _finish(); return
            if video is not None:
                now = time.monotonic()
                # Auto-cycle tabs so the mp4 records every tab.
                if cycle_tab_ms > 0 and now - last_tab_switch >= cycle_tab_ms / 1000.0:
                    cur = tabs.currentIndex()
                    tabs.setCurrentIndex((cur + 1) % n_tabs)
                    last_tab_switch = now
                if now - last_grab >= record_interval and last_rendered_cycle >= 0:
                    video.grab(win)
                    last_grab = now
        except Exception as e:
            import traceback
            print(f"[tick] exception: {e}", file=sys.stderr)
            traceback.print_exc()
            cycle_holder["error"] = f"tick: {e}"
            ctrl.update(store.latest(), None, f"tick: {e}")

    def _finish():
        nonlocal last_recorded_cycle
        # Stop the timer, flush recording, close the video pipe.
        try:
            timer.stop()
        except Exception:
            pass
        snap = store.latest_n(64)
        if snap and snap[-1].cycle_id > last_recorded_cycle:
            recorder.record(snap[-1])
            last_recorded_cycle = snap[-1].cycle_id
            try:
                ctrl.update(snap[-1], snap, cycle_holder.get("error"))
                if video is not None:
                    video.grab(win)
            except Exception:
                pass
        recorder.flush(); recorder.close()
        if video is not None:
            video.close()
        msg = (f"Done. {len(store)} cycles; {recorder.count} JSONL lines"
               + (f"; {video.frames} video frames" if video else "") + ".")
        print(msg)
        win.close()

    timer = QtCore.QTimer(win)
    timer.timeout.connect(_tick)
    timer.start(max(args.poll_ms, 5))

    # Clean exit when the window is closed by the user.
    def _on_close(_ev):
        stop_flag.set()
        try:
            timer.stop()
        except Exception:
            pass
        # In case _finish() hasn't run (user closed mid-run): flush recorder.
        try:
            recorder.flush(); recorder.close()
            if video is not None:
                video.close()
        except Exception:
            pass
    win.closeEvent = _on_close  # type: ignore[assignment]

    try:
        rc = app.exec_()
    except KeyboardInterrupt:
        stop_flag.set()
        _finish()
        rc = 0
    stop_flag.set()
    ct.join(timeout=5.0)
    sys.exit(int(getattr(rc, "real", rc)) if rc else 0)


if __name__ == "__main__":
    main()
