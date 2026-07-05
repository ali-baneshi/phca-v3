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
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_pkg_root = Path(__file__).resolve().parent.parent / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.monitoring.observability import (
    OBSERVABILITY_SCHEMA_VERSION,
    ObservabilityStore,
    SessionRecorder,
)
from phca.monitoring.playback import CyclePacer, PlaybackClock
from phca.monitoring.cognitive_panels import format_session_results_lines
from phca.monitoring.qt_dashboard import ObservatoryWindow, make_app, _TransportBar
from PyQt5 import QtCore


def _load_optional_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _session_summary(
    session_dir: Path | None,
    n_lines: int,
    expected: int,
    *,
    recorder_error: str | None = None,
) -> None:
    """One-line post-run recording health (matches meta.recorded_cycles when present)."""
    if session_dir is None:
        return
    if recorder_error:
        print(
            f"Session ERROR: JSONL write failure: {recorder_error}",
            file=sys.stderr,
        )
    recorded = None
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        try:
            recorded = json.loads(meta_path.read_text()).get("recorded_cycles")
        except Exception:
            pass
    recorded_ok = recorded is None or n_lines == int(recorded)
    if recorder_error:
        print(
            f"Session WARN: {n_lines} JSONL lines; requested {expected}, "
            f"recorded={recorded}",
            file=sys.stderr,
        )
    elif n_lines == expected and recorded_ok:
        print(f"Session OK: {n_lines} JSONL lines (recorded_cycles match).")
    else:
        print(
            f"Session WARN: {n_lines} JSONL lines; requested {expected}, "
            f"recorded={recorded}",
            file=sys.stderr,
        )


def _print_post_run_summary(
    session_dir: Path | None,
    *,
    n_lines: int,
    expected: int,
    verify_status: str,
    report_path: Path | None,
    warnings: list[str],
    exit_code: int,
) -> None:
    """Structured terminal summary after a recorded session."""
    print("=== Session summary ===")
    if session_dir is not None:
        print(f"  dir:      {session_dir}")
    print(f"  cycles:   {n_lines} / {expected}")
    print(f"  verify:   {verify_status}")
    if report_path is not None:
        print(f"  report:   {report_path}")
    if warnings:
        print(f"  warnings: {'; '.join(warnings)}")
    print(f"=== Exit {exit_code} ===")


def _post_run_pipeline(
    session_dir: Path | None,
    *,
    n_lines: int,
    expected: int,
    verify: bool,
    warnings: list[str],
    recorder_error: str | None = None,
) -> tuple[int, str, Path | None]:
    """Verify session, write session_report.json, return (exit_code, status, report_path)."""
    verify_status = "SKIP"
    verify_rc = 0
    report_path: Path | None = None
    if session_dir is None:
        return 0, verify_status, report_path
    warn_list = list(warnings)
    if recorder_error:
        err_msg = f"JSONL write error: {recorder_error}"
        if err_msg not in warn_list:
            warn_list.append(err_msg)
        print(f"[record] {err_msg}", file=sys.stderr)
    if recorder_error and verify:
        verify_rc = 1
        verify_status = "FAIL (recorder error)"
    elif verify:
        verify_rc = _run_session_verify(session_dir)
        verify_status = (
            f"PASS (contiguous cycle_id, schema v{OBSERVABILITY_SCHEMA_VERSION})"
            if verify_rc == 0
            else "FAIL"
        )
        if verify_rc != 0:
            print("[verify] session check FAILED", file=sys.stderr)
    try:
        from phca.monitoring.session_report import write_session_report
        write_session_report(session_dir)
        report_path = session_dir / "session_report.json"
    except Exception as exc:
        print(f"[report] session_report.json failed: {exc}", file=sys.stderr)
        if verify_rc == 0:
            verify_rc = 1
            verify_status = "FAIL (report write)"
    exit_code = verify_rc if (verify or recorder_error) else 0
    _print_post_run_summary(
        session_dir,
        n_lines=n_lines,
        expected=expected,
        verify_status=verify_status,
        report_path=report_path,
        warnings=warn_list,
        exit_code=exit_code,
    )
    return exit_code, verify_status, report_path


def _run_session_verify(session_dir: Path, *, allow_incomplete: bool = False) -> int:
    """Run phca_replay.py --check on the recorded session (inline verify)."""
    replay = Path(__file__).resolve().parent / "phca_replay.py"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_pkg_root))
    cmd = [sys.executable, str(replay), "--check", str(session_dir)]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


def _camera_self_test(env: Any) -> bool:
    """One-shot MuJoCo RGB capture; False on EGL green slab or missing GL."""
    from phca.monitoring.camera_render import is_glitchy_rgb_frame

    render_rgb = getattr(env, "render_rgb", None)
    if not callable(render_rgb):
        return False
    try:
        frame = render_rgb()
    except Exception:
        return False
    return frame is not None and not is_glitchy_rgb_frame(frame)


def _camera_capture_period(camera_hz: float) -> float:
    """Seconds between live camera captures."""
    hz = max(float(camera_hz), 0.1)
    return 1.0 / hz


def _can_capture_live_camera(now_t: float, *, next_capture_t: float,
                             cooldown_until_t: float) -> bool:
    """Gate live capture by cadence + temporary cooldown window."""
    if now_t < cooldown_until_t:
        return False
    return now_t >= next_capture_t


def _read_latest_camera_packet(cycle_holder: dict) -> Any:
    """Return latest captured frame packet without extra copying."""
    lock = cycle_holder.get("camera_lock")
    if lock is None:
        return None
    with lock:
        frame = cycle_holder.get("camera_frame")
        if frame is None:
            return None
        return {
            "frame": frame,
            "cycle_id": cycle_holder.get("camera_cycle_id"),
        }


def _build_cycle(
    args,
    store: ObservabilityStore,
    *,
    seed_offset: int = 0,
    enable_camera: bool = False,
) -> CognitiveCycle:
    base_seed = args.seed + 2 + int(seed_offset)
    if args.env == "gridworld":
        rng = np.random.RandomState(base_seed)
        obstacles = []
        gap_row = rng.randint(0, args.grid_size)
        for r in range(args.grid_size):
            if r != gap_row and args.grid_size >= 5:
                obstacles.append((r, max(1, args.grid_size // 2)))
        return CognitiveCycle.build_for_env(
            size=args.grid_size, seed=base_seed, use_mlp=args.mlp,
            use_continuous=True, obstacles=obstacles,
            observability_store=store,
        )
    return CognitiveCycle.build_for_mujoco(
        args.env, seed=base_seed, use_mlp=args.mlp,
        observability_store=store,
        enable_camera=enable_camera,
    )


def _parse_agent_labels(raw: str, n: int) -> list[str]:
    if raw and str(raw).strip():
        parts = [p.strip() for p in str(raw).split(",")]
        while len(parts) < n:
            parts.append(f"agent_{len(parts)}")
        return parts[:n]
    return [f"agent_{i}" for i in range(n)]


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
            w, h = img.width(), img.height()
            bpl = img.bytesPerLine()
            ptr = img.constBits(); ptr.setsize(img.byteCount())
            raw = np.frombuffer(ptr, dtype=np.uint8)
            # v6: respect the per-scanline stride (Format_RGB888 may be padded
            # to 4 bytes on some platforms — a naive (h,w,3) reshape drops every
            # grab silently). Copy the 3-byte RGB window of each scanline.
            if bpl == w * 3:
                arr = raw.reshape(h, w, 3)
            else:
                arr = np.zeros((h, w, 3), dtype=np.uint8)
                arr[:] = raw.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
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
    parser.add_argument("--heartbeat-hz", type=float, default=10.0,
                        help="dashboard heartbeat rate (Hz) — the only rate at which "
                             "canvases are fed new data. Low = calm/no-flicker. JSONL "
                             "still records every cycle.")
    parser.add_argument("--render-fps", type=float, default=30.0,
                        help="hard repaint cap (kept from v5; heartbeat is the effective "
                             "rate). 0 = no cap.")
    parser.add_argument("--render-hz", type=float, default=6.0,
                        help="v8 calm-render pacer rate (Hz) — the only rate at which "
                             "the visible tab's dirty canvases repaint. 6 = calm/no-flicker.")
    parser.add_argument("--no-record", action="store_true", help="live view only, no JSONL/video")
    parser.add_argument("--record-video", action="store_true",
                        help="capture an mp4 from the widget (requires ffmpeg)")
    parser.add_argument("--record-fps", type=float, default=8.0, help="video capture rate")
    parser.add_argument("--record-dir", default="logs/sessions")
    parser.add_argument("--video-cycle-ms", type=int, default=1500,
                        help="when recording video, auto-cycle tabs every N ms so the "
                             "mp4 captures all 7 tabs (0 = off, stay on clicked tab)")
    parser.add_argument("--camera-debug", action="store_true",
                        help="log one-line camera stats on stderr each capture (main thread)")
    parser.add_argument("--camera", default=None, choices=["auto", "live", "schematic"],
                        help="camera: auto (GL+2D fallback), live (MuJoCo GL), "
                             "schematic (2D arm). Reacher default: schematic")
    parser.add_argument("--profile", default="custom",
                        choices=["custom", "near_real", "readable"],
                        help="runtime profile: custom | near_real | readable")
    parser.add_argument("--camera-hz", type=float, default=5.0,
                        help="live camera capture rate on cycle thread (Hz)")
    parser.add_argument("--camera-fail-cooldown-ms", type=int, default=250,
                        help="cooldown after burst camera capture failures (ms)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip post-run phca_replay.py --check (report still written)")
    parser.add_argument("--close-at-end", action="store_true",
                        help="close the window when the run completes (default: stay open for review)")
    parser.add_argument("--compare-report", default=None,
                        help="optional prior session_report.json for delta comparison in Overview")
    parser.add_argument("--benchmark-report", default="logs/benchmark_report.json",
                        help="optional offline Φ-IQ benchmark JSON for Overview context (skip if missing)")
    parser.add_argument("--no-benchmark-display", action="store_true",
                        help="do not show Φ-IQ benchmark lines in Overview results")
    parser.add_argument(
        "--agents",
        type=int,
        default=int(os.environ.get("PHCA_OBSERVATORY_AGENTS", "1")),
        help="number of local agents (aligned lockstep); default 1",
    )
    parser.add_argument(
        "--agent-labels",
        default=os.environ.get("PHCA_OBSERVATORY_AGENT_LABELS", ""),
        help="comma-separated labels for each agent (optional)",
    )
    args = parser.parse_args()
    args.agents = max(1, int(args.agents))
    args.agent_labels_list = _parse_agent_labels(args.agent_labels, args.agents)

    if args.env == "cartpole":
        args.env = "InvertedPendulum-v5"
    elif args.env == "pendulum":
        args.env = "Pendulum-v1"
    elif args.env == "reacher":
        args.env = "Reacher-v5"

    if args.camera is None:
        args.camera = "schematic" if args.env == "Reacher-v5" else "auto"

    if args.profile == "near_real":
        args.camera_hz = 5.0
        args.heartbeat_hz = 6.0
        args.render_hz = 4.0
    elif args.profile == "readable":
        args.camera_hz = 4.0
        args.heartbeat_hz = 4.0
        args.render_hz = 3.0

    store_maxlen = max(1000, args.cycles * args.agents)
    store = ObservabilityStore(maxlen=store_maxlen)
    total_jsonl_lines = args.cycles * args.agents
    recorder = SessionRecorder(root=args.record_dir, fps=args.record_fps,
                               record=not args.no_record)
    start_meta = {
        "env": args.env, "seed": args.seed,
        "cycles": total_jsonl_lines if args.agents > 1 else args.cycles,
        "cycles_per_agent": args.cycles,
        "grid_size": args.grid_size,
        "mlp": args.mlp, "record_fps": args.record_fps,
        "renderer": "qt",
        "profile": args.profile,
        "camera": args.camera,
        "heartbeat_hz": args.heartbeat_hz,
        "render_hz": args.render_hz,
        "camera_hz": args.camera_hz,
    }
    if args.agents > 1:
        start_meta["agent_count"] = args.agents
        start_meta["agents"] = [
            {"agent_id": i, "label": args.agent_labels_list[i], "recorded_cycles": 0}
            for i in range(args.agents)
        ]
        start_meta["timeline_mode"] = "aligned"
        start_meta["recording_layout"] = "single_jsonl"
    session_dir = recorder.start(start_meta)
    if session_dir:
        print(f"Recording session -> {session_dir}")

    gl_backend = os.environ.get("MUJOCO_GL", "(default)")
    if args.env != "gridworld":
        print(f"MuJoCo GL: {gl_backend} (camera via mujoco.Renderer, not gym viewer)")
        print(f"Runtime profile: {args.profile}")
        if args.env == "Reacher-v5" and args.camera == "schematic":
            print("Reacher default: 2D schematic (no green screen)")
            print("Real MuJoCo camera: add --camera live")
        elif args.camera == "schematic":
            print("Camera mode: schematic (2D arm, no GPU)")
        elif args.camera == "live":
            print("Camera mode: live (MuJoCo GL on cycle thread; 2D fallback if fail)")
        else:
            print("Camera mode: auto (live GL, fallback to 2D schematic if GL fails)")
        print("Debug: add --camera-debug  |  probe: scripts/camera_probe_qt.py")

    # Cognitive cycle (incl. M3 SQLite) MUST be built+run in ONE thread.
    stop_flag = threading.Event()
    cycle_holder: dict = {}

    def _on_sigterm(signum, frame) -> None:
        stop_flag.set()

    signal.signal(signal.SIGTERM, _on_sigterm)
    want_live_camera = args.camera in ("live", "auto") and args.env != "gridworld"
    cycle_ready = threading.Event()
    pacer = CyclePacer()  # v6: optional cycle throttle/pause (no-op at full speed)

    def _camera_provider_from_holder() -> Any:
        """Return latest RGB + cycle id captured on the cycle thread."""
        return _read_latest_camera_packet(cycle_holder)

    def _run_cycle():
        try:
            if args.agents <= 1:
                cycle = _build_cycle(args, store, enable_camera=want_live_camera)
                cycles = [cycle]
            else:
                cycles = [
                    _build_cycle(
                        args, store,
                        seed_offset=aid * 7,
                        enable_camera=want_live_camera and aid == 0,
                    )
                    for aid in range(args.agents)
                ]
            cycle_holder["cycles"] = cycles
            cycle_holder["cycle"] = cycles[0]
            next_capture_t = time.monotonic()
            cooldown_until_t = 0.0
            fail_streak = 0
            cap_period = _camera_capture_period(args.camera_hz)
            fail_cooldown = max(float(args.camera_fail_cooldown_ms), 0.0) / 1000.0
            primary = cycles[0]
            if want_live_camera:
                ok = _camera_self_test(primary.env)
                cycle_holder["camera_ok"] = ok
                if ok:
                    cycle_holder["camera_lock"] = threading.Lock()
                    cycle_holder["camera_frame"] = None
            else:
                cycle_holder["camera_ok"] = False
            cycle_ready.set()
            labels = args.agent_labels_list
            for step in range(args.cycles):
                if stop_flag.is_set():
                    break
                for aid, cycle in enumerate(cycles):
                    if stop_flag.is_set():
                        break
                    cycle.observability_agent_id = aid
                    cycle.observability_agent_label = labels[aid]
                    cycle.observability_timeline_step = step
                    cycle_id_for_step = cycle.cycle_count
                    pacer.wait()
                    cycle.step()
                    if want_live_camera and aid == 0 and cycle_holder.get("camera_ok"):
                        now_t = time.monotonic()
                        if not _can_capture_live_camera(
                            now_t,
                            next_capture_t=next_capture_t,
                            cooldown_until_t=cooldown_until_t,
                        ):
                            continue
                        try:
                            frame = cycle.env.render_rgb()
                        except Exception:
                            frame = None
                        if frame is not None:
                            fail_streak = 0
                            next_capture_t = now_t + cap_period
                            with cycle_holder["camera_lock"]:
                                cycle_holder["camera_frame"] = np.asarray(
                                    frame, dtype=np.uint8).copy()
                                cycle_holder["camera_cycle_id"] = cycle_id_for_step
                        else:
                            fail_streak += 1
                            next_capture_t = now_t + cap_period
                            if fail_streak >= 3:
                                cooldown_until_t = now_t + fail_cooldown
                                fail_streak = 0
        except Exception as e:
            cycle_holder["error"] = str(e)
            print(f"[cycle thread] error: {e}", file=sys.stderr)
        finally:
            for cycle in cycle_holder.get("cycles", [cycle_holder.get("cycle")]):
                if cycle is None:
                    continue
                try:
                    m3 = getattr(getattr(cycle, "consolidation", None), "m3", None)
                    if m3 is not None:
                        m3.close()
                except Exception:
                    pass
                try:
                    cycle.env.close()
                except Exception:
                    pass
            stop_flag.set()

    ct = threading.Thread(target=_run_cycle, daemon=True)
    ct.start()
    if want_live_camera:
        # MuJoCo/GLFW must init before QApplication (Qt also uses GLFW on Linux).
        if not cycle_ready.wait(timeout=60.0):
            print("[camera] cycle thread did not become ready in time",
                  file=sys.stderr)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.debug=false")
    app = make_app()
    startup_warnings: list[str] = []
    if app.platformName().lower() == "wayland":
        startup_warnings.append(
            "libdecor-gtk (Wayland cosmetic — see SETUP.md; try QT_QPA_PLATFORM=xcb)"
        )
        print(
            "Wayland note: cosmetic 'libdecor-gtk.so' plugin noise is harmless; "
            "use QT_QPA_PLATFORM=xcb to silence (see SETUP.md)."
        )
    win = ObservatoryWindow()
    win.set_session_context({
        "env": args.env,
        "camera": args.camera,
        "target_cycles": args.cycles,
        "session_dir": str(session_dir) if session_dir else "",
        "recording": not args.no_record,
        "agent_count": args.agents,
        "agent_labels": args.agent_labels_list,
    })
    if args.agents > 1:
        win.init_multi_agent(args.agents, args.agent_labels_list)
    ctrl = win.controller
    camera_wired = False

    if args.camera == "schematic":
        win.set_camera_provider(None, mode="schematic")
        camera_wired = True
    else:
        win.overview.bind_camera_tabs(win._tabs, overview_tab_index=0)

    def _wire_camera_if_ready() -> None:
        nonlocal camera_wired
        if camera_wired:
            return
        cycle = cycle_holder.get("cycle")
        if cycle is None:
            return

        mode = args.camera
        provider = None
        if mode in ("live", "auto"):
            if "camera_ok" not in cycle_holder:
                return
            if not cycle_holder.get("camera_ok"):
                print(
                    "[camera] capture failed on this system; using 2D schematic",
                    file=sys.stderr,
                )
                mode = "schematic"
            else:
                provider = _camera_provider_from_holder
        elif mode != "schematic":
            if "camera_ok" not in cycle_holder:
                return
            if cycle_holder.get("camera_ok"):
                provider = _camera_provider_from_holder

        if mode == "schematic":
            win.set_camera_provider(None, debug=args.camera_debug, mode="schematic")
        else:
            win.set_camera_provider(provider, debug=args.camera_debug, mode=mode)
        camera_wired = True
        win.overview._sync_camera_from_provider()
        win.overview.mark_dirty()
        win.overview.update()
        app.processEvents()

    def _wait_cycle_and_wire() -> None:
        if camera_wired:
            return
        if "cycle" not in cycle_holder:
            QtCore.QTimer.singleShot(30, _wait_cycle_and_wire)
            return
        if args.camera in ("live", "auto") and "camera_ok" not in cycle_holder:
            QtCore.QTimer.singleShot(30, _wait_cycle_and_wire)
            return
        _wire_camera_if_ready()

    # v6 transport: PlaybackClock drives the dashboard at the heartbeat rate;
    # _TransportBar gives pause / speed / step / scrub / cycle-throttle.
    clock = PlaybackClock(heartbeat_hz=args.heartbeat_hz, mode="live", maxlen=store_maxlen)
    clock.on_update = lambda f, rolling, err: ctrl.update(f, rolling, err)
    transport = _TransportBar(clock, pacer=pacer)
    win.install_transport(transport)
    # heartbeat timer — the only thing that feeds canvases new data.
    hb_timer = QtCore.QTimer(win)
    hb_timer.setInterval(int(1000.0 / max(args.heartbeat_hz, 0.5)))
    hb_timer.timeout.connect(clock.tick)
    # keep the transport scrubber in sync with the live cursor
    def _sync_transport():
        transport.set_range(clock.n)
        transport._sync_slider()
        latest = store.latest()
        if latest is not None:
            win.update_session_strip(
                latest,
                jsonl_count=recorder.count,
                clock_cursor=clock.cursor_int,
            )
    hb_sync = QtCore.QTimer(win)
    hb_sync.setInterval(250)
    hb_sync.timeout.connect(_sync_transport)

    win.show()
    _wait_cycle_and_wire()
    # v8: start the calm-render pacer (repaints the visible tab's dirty canvases
    # at --render-hz; paused/no-new-data → 0 repaints).
    win.start_render(args.render_hz)

    # Optional video pipe.
    video: "_VideoPipe | None" = None
    if args.record_video and session_dir is not None:
        # Pipe size set after the window is shown (real geometry).
        app.processEvents()
        video = _VideoPipe(session_dir / "session.mp4", args.record_fps,
                           win.width(), win.height())
        print(f"Video -> {video.path}")

    last_recorded_by_agent = {aid: -1 for aid in range(args.agents)}
    expected_jsonl = total_jsonl_lines
    record_interval = 1.0 / max(args.record_fps, 0.1)
    last_grab = -record_interval
    target = args.cycles
    # v6: the heartbeat clock drives rendering; the poll timer only drains +
    # records JSONL + pushes frames into the clock buffer (so JSONL stays
    # 1 line/cycle and the scrubber can reach any past cycle).
    # Auto-cycle tabs during video recording so one mp4 captures all 7 tabs.
    tabs = win._tabs
    n_tabs = tabs.count()
    cycle_tab_ms = args.video_cycle_ms if (args.record_video and args.video_cycle_ms > 0) else 0
    last_tab_switch = 0.0
    user_tab = {"i": tabs.currentIndex()}  # remember user's tab when not auto-cycling

    def _tick():
        nonlocal last_recorded_by_agent, last_grab, last_tab_switch
        try:
            _wire_camera_if_ready()
            new_frames = []
            for aid in range(args.agents):
                new = store.frames_after(last_recorded_by_agent[aid], agent_id=aid)
                if new:
                    new_frames.extend(new)
                    last_recorded_by_agent[aid] = new[-1].cycle_id
            if new_frames:
                if args.agents > 1:
                    from phca.monitoring.multi_agent import interleave_frames_for_record
                    new_frames = interleave_frames_for_record(new_frames)
                for f in new_frames:
                    recorder.record(f)
                if args.agents > 1:
                    win.append_observability_frames(new_frames)
                    projected = win.project_frames_for_agent(win._all_frames)
                    clk = transport.clock
                    clk.reload_frames(
                        projected,
                        preserve_transport=True,
                        follow_live=not (clk.paused or clk.scrubbing),
                    )
                    transport.set_range(len(projected))
                    transport._sync_slider()
                else:
                    for f in new_frames:
                        clock.push(f)
            err = cycle_holder.get("error")
            if err:
                clock.error = err
                ctrl.update(store.latest(), None, err)
                _on_run_complete(); return
            from phca.monitoring.multi_agent import frames_for_agent
            snap = store.snapshot()
            if args.agents <= 1:
                done = stop_flag.is_set() and len(snap) >= target
            else:
                done = stop_flag.is_set() and all(
                    len(frames_for_agent(snap, aid)) >= target
                    for aid in range(args.agents)
                )
            if done:
                _on_run_complete(); return
            if video is not None:
                now = time.monotonic()
                # Auto-cycle tabs so the mp4 records every tab.
                if cycle_tab_ms > 0 and now - last_tab_switch >= cycle_tab_ms / 1000.0:
                    cur = tabs.currentIndex()
                    tabs.setCurrentIndex((cur + 1) % n_tabs)
                    last_tab_switch = now
                if now - last_grab >= record_interval and recorder.count > 0:
                    video.grab(win)
                    last_grab = now
        except Exception as e:
            import traceback
            print(f"[tick] exception: {e}", file=sys.stderr)
            traceback.print_exc()
            cycle_holder["error"] = f"tick: {e}"
            ctrl.update(store.latest(), None, f"tick: {e}")

    def _stop_production_timers() -> None:
        """Stop cycle poll + heartbeat + camera; keep render pacer for review scrub."""
        cam_timer = getattr(win.overview, "_capture_timer", None)
        if cam_timer is not None:
            try:
                cam_timer.stop()
            except Exception:
                pass
        for t in (timer, hb_timer, hb_sync):
            try:
                t.stop()
            except Exception:
                pass
        try:
            pacer.stop()
        except Exception:
            pass

    def _stop_all_timers() -> None:
        _stop_production_timers()
        try:
            win.render_pacer.stop()
        except Exception:
            pass

    post_run_exit = 0
    run_completed = False

    def _finalize_early_session() -> None:
        if args.no_record or session_dir is None:
            return
        n_lines = int(recorder.count)
        try:
            from phca.monitoring.session_recovery import finalize_session
            finalize_session(session_dir, reason="user_close", write_report=True)
        except Exception as exc:
            print(f"[recover] early finalize failed: {exc}", file=sys.stderr)
            return
        if n_lines > 0 and args.verify:
            verify_rc = _run_session_verify(session_dir, allow_incomplete=True)
            status = "PASS (allow-incomplete)" if verify_rc == 0 else "FAIL"
            print(f"[verify] early close check: {status}", file=sys.stderr)

    def _on_run_complete() -> None:
        nonlocal last_recorded_by_agent, post_run_exit, run_completed
        if run_completed:
            return
        run_completed = True
        _stop_production_timers()
        new_frames = []
        for aid in range(args.agents):
            snap_new = store.frames_after(last_recorded_by_agent[aid], agent_id=aid)
            if snap_new:
                new_frames.extend(snap_new)
                last_recorded_by_agent[aid] = snap_new[-1].cycle_id
        if new_frames:
            for f in new_frames:
                recorder.record(f)
            if args.agents > 1:
                win.append_observability_frames(new_frames)
                projected = win.project_frames_for_agent(win._all_frames)
                clk = transport.clock
                clk.reload_frames(
                    projected,
                    preserve_transport=True,
                    follow_live=not (clk.paused or clk.scrubbing),
                )
                transport.set_range(len(projected))
                transport._sync_slider()
            else:
                for f in new_frames:
                    clock.push(f)
        if args.agents > 1:
            from phca.monitoring.multi_agent import multi_agent_meta_patch
            patch = multi_agent_meta_patch(store.snapshot(), timeline_mode="aligned")
            if patch:
                recorder._patch_meta(patch)
        recorder.flush()
        recorder.close()
        if video is not None:
            video.close()
        msg = (f"Done. {len(store)} cycles; {recorder.count} JSONL lines"
               + (f"; {video.frames} video frames" if video else "") + ".")
        print(msg)
        win.enter_review_mode(resync_only=True)
        rolling = win.project_frames_for_agent(store.snapshot())
        if rolling:
            ctrl.rebuild_all_histories(rolling)
        ctrl._repaint_visible_tab(force_sync=True)
        app.processEvents()
        win.refresh_session_strip(jsonl_count=recorder.count)

        def _deferred_post_run() -> None:
            nonlocal post_run_exit
            verify_status = ""
            if not args.no_record:
                _session_summary(
                    session_dir,
                    recorder.count,
                    expected_jsonl,
                    recorder_error=recorder.error,
                )
                if session_dir is not None:
                    post_run_exit, verify_status, _ = _post_run_pipeline(
                        session_dir,
                        n_lines=recorder.count,
                        expected=expected_jsonl,
                        verify=not args.no_verify,
                        warnings=startup_warnings,
                        recorder_error=recorder.error,
                    )
                    report = _load_optional_json(
                        str(session_dir / "session_report.json"))
                    compare = _load_optional_json(args.compare_report)
                    benchmark = None
                    if not args.no_benchmark_display:
                        benchmark = _load_optional_json(args.benchmark_report)
                    if report:
                        lines = format_session_results_lines(
                            report,
                            compare=compare,
                            benchmark=benchmark,
                            verify_status=verify_status,
                        )
                        win.set_session_results(lines)
            if verify_status:
                win.set_verify_status(verify_status)
            print("Review mode: window stays open — scrub tabs and close when done.",
                  file=sys.stderr)
            if args.close_at_end:
                win.close()

        QtCore.QTimer.singleShot(0, _deferred_post_run)

    timer = QtCore.QTimer(win)
    timer.timeout.connect(_tick)
    timer.start(max(args.poll_ms, 5))
    hb_timer.start()
    hb_sync.start()

    # Clean exit when the window is closed by the user.
    def _on_close(ev):
        stop_flag.set()
        _stop_all_timers()
        if not run_completed:
            try:
                recorder.flush()
                if recorder.count < expected_jsonl:
                    recorder.abort("user_close")
                else:
                    recorder.close()
                if video is not None:
                    video.close()
                _finalize_early_session()
            except Exception:
                pass
        ev.accept()
        app.quit()

    win.closeEvent = _on_close  # type: ignore[assignment]

    try:
        rc = app.exec_()
    except KeyboardInterrupt:
        stop_flag.set()
        _on_run_complete()
        rc = 0
    if not run_completed:
        stop_flag.set()
        ct.join(timeout=5.0)
        if ct.is_alive():
            print("[cycle thread] did not exit within 5s", file=sys.stderr)
    ui_rc = int(getattr(rc, "real", rc)) if rc else 0
    sys.exit(ui_rc or post_run_exit)


if __name__ == "__main__":
    main()
