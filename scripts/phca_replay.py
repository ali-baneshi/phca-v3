#!/usr/bin/env python3
"""PHCA v3.0 — Session Replay Tool (Observability v2).

Replays a recorded observability session produced by phca_visualise.py.

A session dir contains:
  - meta.json          run metadata
  - timeseries.jsonl   one full-fidelity frame per cycle (analytics + replay)
  - session.mp4|.gif   the captured live dashboard video (when recorded)

Modes:
    <session_dir>              Play the recorded video (mp4/gif) in a window.
    <session_dir> --from-jsonl Reconstruct the dashboard from the JSONL and play
                               it back at any fps (full-fidelity, no video needed).
    <session_dir> --check      Consistency gate (used by the acceptance suite):
                               verify meta parses, JSONL lines all parse, and the
                               video file (if present) is non-zero. Exit 0/1.

Usage:
    python scripts/phca_replay.py logs/sessions/20260701_170450/
    python scripts/phca_replay.py logs/sessions/20260701_170450/ --from-jsonl --fps 12
    python scripts/phca_replay.py --check logs/sessions/20260701_170450/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_session(session_dir: str):
    d = Path(session_dir)
    meta_p = d / "meta.json"
    jsonl_p = d / "timeseries.jsonl"
    if not meta_p.exists() or not jsonl_p.exists():
        print(f"ERROR: not a valid session dir: {d}", file=sys.stderr)
        return None, None, None
    meta = json.loads(meta_p.read_text())
    lines = [ln for ln in jsonl_p.read_text().splitlines() if ln.strip()]
    video = None
    for ext in (".mp4", ".gif"):
        p = d / f"session{ext}"
        if p.exists():
            video = p
            break
    return meta, lines, video


def _check(session_dir: str) -> int:
    meta, lines, video = _load_session(session_dir)
    if meta is None:
        return 1
    ok = True
    print(f"Session: {session_dir}")
    print(f"  meta       : env={meta.get('env')} cycles={meta.get('cycles')} "
          f"record_fps={meta.get('record_fps')} mlp={meta.get('mlp')}")
    print(f"  jsonl      : {len(lines)} lines")
    parse_fail = 0
    for ln in lines:
        try:
            json.loads(ln)
        except Exception:
            parse_fail += 1
    if parse_fail:
        print(f"  [FAIL] {parse_fail} unparseable JSONL lines")
        ok = False
    else:
        print(f"  [PASS] all JSONL lines parse")
    if video is not None:
        sz = video.stat().st_size
        if sz > 0:
            print(f"  [PASS] video {video.name} ({sz} B)")
        else:
            print(f"  [FAIL] video {video.name} is empty")
            ok = False
    else:
        print(f"  [WARN] no video file (live-only or --no-record run)")
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _play_video(video: Path, fps: float) -> int:
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt
    if video.suffix == ".gif":
        try:
            from PIL import Image
        except Exception as e:
            print(f"Need Pillow to replay gif: {e}", file=sys.stderr)
            return 1
        frames = []
        with Image.open(video) as im:
            try:
                while True:
                    frames.append(im.convert("RGB"))
                    im.seek(im.tell() + 1)
            except EOFError:
                pass
        if not frames:
            print("Empty gif.", file=sys.stderr)
            return 1
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111); ax.axis("off")
        im_ax = ax.imshow(frames[0])
        fig.suptitle(f"PHCA replay — {video.parent.name} ({len(frames)} frames)")
        interval = int(1000 / max(fps, 0.1))
        import matplotlib.animation as mplan
        def _u(i):
            im_ax.set_data(frames[i % len(frames)])
            return [im_ax]
        anim = mplan.FuncAnimation(fig, _u, frames=len(frames), interval=interval,
                                   blit=False, cache_frame_data=False, repeat=False)
        print(f"Replaying {len(frames)} gif frames. Close window to exit.")
        try:
            plt.show()
        finally:
            pass
        return 0
    # mp4: try imageio, else fall back to --from-jsonl
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(video))
        frames = [f for f in reader]
        reader.close()
    except Exception as e:
        print(f"Cannot decode {video} ({e}). Use --from-jsonl to reconstruct.",
              file=sys.stderr)
        return 1
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111); ax.axis("off")
    im_ax = ax.imshow(frames[0])
    fig.suptitle(f"PHCA replay — {video.parent.name} ({len(frames)} frames)")
    interval = int(1000 / max(fps, 0.1))
    import matplotlib.animation as mplan
    def _u(i):
        im_ax.set_data(frames[i % len(frames)])
        return [im_ax]
    anim = mplan.FuncAnimation(fig, _u, frames=len(frames), interval=interval,
                               blit=False, cache_frame_data=False, repeat=False)
    print(f"Replaying {len(frames)} video frames. Close window to exit.")
    try:
        plt.show()
    finally:
        pass
    return 0


def _play_jsonl(session_dir: str, fps: float) -> int:
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt
    _pkg = Path(__file__).resolve().parent.parent / "python"
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))
    from phca.monitoring.render import build_dashboard, update_dashboard, frame_from_json
    meta, lines, video = _load_session(session_dir)
    if meta is None:
        return 1
    if not lines:
        print("No JSONL frames to replay.", file=sys.stderr)
        return 1
    is_grid = (meta.get("env") == "gridworld")
    frames = [frame_from_json(json.loads(ln)) for ln in lines]
    fig = plt.figure(figsize=(14, 7))
    handle = build_dashboard(fig, is_grid=is_grid)
    # Prime the trend with the full history so the rolling window is populated.
    for f in frames:
        update_dashboard(handle, f)
    fig.canvas.draw()
    print(f"Replaying {len(frames)} frames from JSONL at {fps} fps. Close to exit.")
    interval = int(1000 / max(fps, 0.1))
    import matplotlib.animation as mplan

    def _u(i):
        update_dashboard(handle, frames[i % len(frames)])
        return []

    anim = mplan.FuncAnimation(fig, _u, frames=len(frames), interval=interval,
                               blit=False, cache_frame_data=False, repeat=False)
    try:
        plt.show()
    finally:
        pass
    return 0


def _play_qt(session_dir: str, fps: float, close_at_end: bool = False) -> int:
    """Replay a session in the PyQt5 Cognitive Observatory dashboard with the
    unified v6 transport (pause / speed / step / scrub)."""
    _pkg = Path(__file__).resolve().parent.parent / "python"
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))
    from phca.monitoring.qt_dashboard import ObservatoryWindow, make_app, _TransportBar
    from phca.monitoring.playback import PlaybackClock
    from phca.monitoring.render import frame_from_json
    from PyQt5 import QtCore
    meta, lines, video = _load_session(session_dir)
    if meta is None:
        return 1
    if not lines:
        print("No JSONL frames to replay.", file=sys.stderr)
        return 1
    frames = [frame_from_json(json.loads(ln)) for ln in lines]
    n = len(frames)

    app = make_app()
    win = ObservatoryWindow(title=f"PHCA Observatory — replay {Path(session_dir).name}")
    ctrl = win.controller
    # v6: unified transport. heartbeat = fps; speed dial multiplies it.
    clock = PlaybackClock(heartbeat_hz=max(fps, 0.5), mode="replay")
    clock.set_frames(frames)
    ended = {"v": False}

    def _on_update(f, rolling, err):
        ctrl.update(f, rolling, err)

    def _on_end():
        ended["v"] = True
        if close_at_end:
            win.close()

    clock.on_update = _on_update
    clock.on_end = _on_end
    transport = _TransportBar(clock, pacer=None)
    transport.set_range(n)
    transport.play_btn.setText("▶ Play")  # start paused so user can scrub/step
    transport.play_btn.setChecked(True)
    win.install_transport(transport)
    win.show()
    win.start_render(6.0)

    hb = QtCore.QTimer(win)
    hb.setInterval(int(1000.0 / max(fps, 0.5)))
    hb.timeout.connect(clock.tick)

    def _sync():
        transport._sync_slider()

    sync = QtCore.QTimer(win)
    sync.setInterval(200)
    sync.timeout.connect(_sync)

    # start at first frame
    clock.seek(0)
    hb.start(); sync.start()
    print(f"Replaying {n} frames in Qt dashboard (transport: pause / speed / step / scrub). Close to exit.")
    try:
        rc = app.exec_()
    except KeyboardInterrupt:
        hb.stop(); rc = 0
    return int(rc) if rc else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA session replay tool")
    parser.add_argument("session", help="session dir (logs/sessions/<ts>/)")
    parser.add_argument("--check", action="store_true",
                        help="consistency check only (exit 0/1), no playback")
    parser.add_argument("--from-jsonl", action="store_true",
                        help="reconstruct the matplotlib dashboard from JSONL")
    parser.add_argument("--qt", action="store_true",
                        help="reconstruct the PyQt5 Observatory dashboard from JSONL")
    parser.add_argument("--qt-close-at-end", action="store_true",
                        help="with --qt, close the window when playback reaches the end "
                             "(default: pause so the scrubber stays usable)")
    parser.add_argument("--fps", type=float, default=10.0, help="playback fps")
    args = parser.parse_args()
    if args.check:
        sys.exit(_check(args.session))
    if args.qt:
        sys.exit(_play_qt(args.session, args.fps, close_at_end=args.qt_close_at_end))
    if args.from_jsonl:
        sys.exit(_play_jsonl(args.session, args.fps))
    meta, lines, video = _load_session(args.session)
    if meta is None:
        sys.exit(1)
    if video is not None:
        sys.exit(_play_video(video, args.fps))
    # No video -> reconstruct from JSONL.
    print("No video file; reconstructing from JSONL.")
    sys.exit(_play_jsonl(args.session, args.fps))


if __name__ == "__main__":
    main()
