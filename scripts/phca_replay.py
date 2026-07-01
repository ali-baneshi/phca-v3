#!/usr/bin/env python3
"""PHCA v3.0 — Session Replay Tool (Phase 7 extension).

Replays a recorded observability session (produced by phca_visualise.py) from
its frames/*.png + timeseries.jsonl + meta.json.

Modes:
    --check <session_dir>   Assert frame count == JSONL line count + meta
                            parseable. Exit 0 if consistent, 1 otherwise.
                            (Used by the observability acceptance gate.)
    (default)               Play back the frames in a matplotlib window at
                            the recorded FPS, and print a JSONL summary.

Usage:
    python scripts/phca_replay.py logs/sessions/20260701_153000/
    python scripts/phca_replay.py --check logs/sessions/20260701_153000/
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
    frames_p = d / "frames"
    if not meta_p.exists() or not jsonl_p.exists():
        print(f"ERROR: not a valid session dir: {d}", file=sys.stderr)
        return None, None, None
    meta = json.loads(meta_p.read_text())
    frames = sorted(frames_p.glob("*.png")) if frames_p.exists() else []
    lines = [ln for ln in jsonl_p.read_text().splitlines() if ln.strip()]
    return meta, frames, lines


def _check(session_dir: str) -> int:
    meta, frames, lines = _load_session(session_dir)
    if meta is None:
        return 1
    ok = True
    print(f"Session: {session_dir}")
    print(f"  meta       : env={meta.get('env')} cycles={meta.get('cycles')} "
          f"fps={meta.get('fps')} mlp={meta.get('mlp')}")
    print(f"  frames     : {len(frames)} PNGs")
    print(f"  jsonl      : {len(lines)} lines")
    if len(frames) != len(lines):
        print(f"  [FAIL] frame count ({len(frames)}) != JSONL line count ({len(lines)})")
        ok = False
    else:
        print(f"  [PASS] frame count == JSONL line count")
    # Spot-check a few JSONL entries parse
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
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _playback(session_dir: str) -> int:
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    meta, frames, lines = _load_session(session_dir)
    if meta is None:
        return 1
    if not frames:
        print("No frames to replay (session recorded with --no-record?).", file=sys.stderr)
        return 1
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111); ax.axis("off")
    img_ax = ax.imshow(mpimg.imread(frames[0]))
    fig.suptitle(f"PHCA replay — {Path(session_dir).name}  ({len(frames)} frames)")
    interval_ms = int(1000 / max(float(meta.get("fps", 5.0)), 0.1))

    def _update(i):
        img_ax.set_data(mpimg.imread(frames[i % len(frames)]))
        return [img_ax]

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=interval_ms,
                         blit=False, cache_frame_data=False, repeat=False)
    print(f"Replaying {len(frames)} frames at {meta.get('fps')} fps. Close window to exit.")
    try:
        plt.show()
    finally:
        pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA session replay tool")
    parser.add_argument("session", help="session dir (logs/sessions/<ts>/)")
    parser.add_argument("--check", action="store_true",
                        help="consistency check only (exit 0/1), no playback")
    args = parser.parse_args()
    if args.check:
        sys.exit(_check(args.session))
    sys.exit(_playback(args.session))


if __name__ == "__main__":
    main()
