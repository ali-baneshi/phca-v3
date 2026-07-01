#!/usr/bin/env python3
"""PHCA v3.0 — Visual Observability Layer (Phase 7 extension).

Live, graphical, non-invasive two-panel view of a cognitive cycle run:

  Left  (world): GridWorld — grid, walls, goal, agent, path trail, and a
                  semi-transparent heatmap of the agent's PREDICTED next-cell
                  distribution (G' prediction, A4 made visible).
                  MuJoCo (pendulum/reacher) — observation-vector bar chart +
                  goal-reference alignment (no grid; honest fallback).
  Right (mind):  D1-D6 drive levels (bar); prediction-error + confidence
                  (rolling trend lines); attention focus (bar); RBTA status
                  (coloured); last action; cycle latency; violations; M3
                  episode_count + M4 fact_count + RSS (retention caps visible).

Non-blocking: the cycle runs in a daemon thread; matplotlib FuncAnimation
runs on the main thread and polls a lock-guarded ObservabilityStore every
~200ms. Zero-overhead when not used (the store is opt-in on the cycle).

Auto-records every rendered frame to logs/sessions/<timestamp>/ as
timeseries.jsonl + frames/<id>.png + meta.json (replay via phca_replay.py).

Usage (needs a display + matplotlib):
    PYTHONPATH=python python scripts/phca_visualise.py --cycles=300 --mlp --grid-size=5
    PYTHONPATH=python python scripts/phca_visualise.py --env pendulum --cycles=200 --mlp
    PYTHONPATH=python python scripts/phca_visualise.py --cycles=100 --no-record      # live only
    MPLBACKEND=Agg PYTHONPATH=python python scripts/phca_visualise.py --cycles=50 --mlp  # headless smoke
"""
from __future__ import annotations

import argparse
import os
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
from phca.monitoring.observability import (
    ObservabilityStore, ObservabilityFrame, SessionRecorder,
)

DRIVE_NAMES = {1: "D1 PredErr", 2: "D2 Crit", 3: "D3 Compet",
               4: "D4 Curious", 5: "D5 Energy", 6: "D6 Empower"}


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


def _render_world(ax, f: ObservabilityFrame, path_trail: list):
    ax.clear()
    ax.set_title("External world — agent + predicted next state")
    if f.grid is not None and f.agent_pos is not None:
        size = f.grid.shape[0]
        # Base grid: 0 empty, 1 wall, 2 goal, 3 hazard (GridWorld codes)
        base = np.zeros((size, size), dtype=np.float32)
        g = np.asarray(f.grid)
        base[g == 1] = -1.0   # walls
        goal = f.goal_pos
        if goal is not None:
            base[goal[0], goal[1]] = 0.6
        # Prediction heatmap (predicted next-cell distribution) — first size^2 dims
        heat = None
        if f.predicted_state is not None and f.predicted_state.shape[0] >= size * size:
            heat = f.predicted_state[:size * size].reshape(size, size).astype(np.float32)
            heat = heat - heat.min()
            if heat.max() > 1e-9:
                heat = heat / heat.max()
        ax.imshow(base, cmap="RdGy", vmin=-1, vmax=1, origin="upper")
        if heat is not None:
            ax.imshow(heat, cmap="Blues", alpha=0.45, origin="upper")
        # Path trail
        if path_trail:
            rows = [p[0] for p in path_trail]
            cols = [p[1] for p in path_trail]
            ax.plot(cols, rows, "-", color="orange", linewidth=1.5, alpha=0.6)
        # Agent
        ax.plot(f.agent_pos[1], f.agent_pos[0], "o", color="royalblue",
                markersize=14, markeredgecolor="white")
        if goal is not None:
            ax.plot(goal[1], goal[0], "*", color="lime", markersize=16)
        ax.set_xticks(range(size)); ax.set_yticks(range(size))
        ax.grid(True, color="grey", linewidth=0.3, alpha=0.5)
    elif f.obs_vector is not None:
        # MuJoCo fallback: observation vector bar chart
        ax.bar(range(len(f.obs_vector)), f.obs_vector, color="steelblue")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("obs dim")
        ax.set_title(f"Observation vector (dim={len(f.obs_vector)}) — no grid for MuJoCo")
    else:
        ax.text(0.5, 0.5, "waiting for first cycle...", ha="center", va="center")


def _render_mind(axs, f: ObservabilityFrame, history: list):
    a_drive, a_trend, a_att, a_status = axs
    # D1-D6 drives
    a_drive.clear()
    labels = [DRIVE_NAMES.get(i, f"D{i}") for i in range(1, len(f.drive_levels) + 1)]
    a_drive.bar(labels, f.drive_levels, color="teal")
    a_drive.set_ylim(bottom=0)
    a_drive.set_title("MDIM drives (D1-D6)")
    a_drive.tick_params(axis="x", labelrotation=30, labelsize=8)
    # Error + confidence trend
    a_trend.clear()
    if history:
        errs = [h.prediction_error for h in history]
        confs = [h.prediction_confidence for h in history]
        xs = list(range(len(history)))
        a_trend.plot(xs, errs, "-", color="crimson", label="pred error")
        a_trend.plot(xs, confs, "-", color="darkgreen", label="confidence")
        a_trend.set_ylim(0, max(1.0, max(errs) * 1.05) if errs else 1.0)
    a_trend.set_title("Prediction error / confidence trend")
    a_trend.legend(loc="upper right", fontsize=7)
    # Attention focus
    a_att.clear()
    if f.attention_saliences:
        a_att.bar(range(len(f.attention_saliences)), f.attention_saliences,
                  color="purple")
        a_att.set_title(f"Attention focus ({len(f.attention_saliences)} chunks)")
        a_att.set_ylim(bottom=0)
    else:
        a_att.text(0.5, 0.5, "no attention yet", ha="center", va="center")
    # Status text panel
    a_status.clear(); a_status.axis("off")
    rbta_color = {"CONTINUE": "green", "INTERRUPT": "darkorange"}.get(f.rbta_action, "red")
    rss_mb = f.rss_bytes / 1e6
    lines = [
        f"cycle        : {f.cycle_id}",
        f"latency      : {f.latency_ms:.1f} ms",
        f"action       : {f.action_name}",
        f"RBTA         : {f.rbta_action}",
        f"violations   : {f.violations_count}",
        f"goal_reached : {f.goal_reached}",
        f"pred error   : {f.prediction_error:.4f}",
        f"confidence   : {f.prediction_confidence:.3f}",
        f"M3 episodes  : {f.episode_count}",
        f"M4 facts     : {f.fact_count}",
        f"RSS          : {rss_mb:.1f} MB",
    ]
    for i, ln in enumerate(lines):
        col = rbta_color if ln.startswith("RBTA") else "black"
        a_status.text(0.02, 0.95 - i * 0.085, ln, transform=a_status.transAxes,
                      fontsize=9, color=col, family="monospace")


def main() -> None:
    ensure_logging()
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="PHCA Visual Observability Layer")
    parser.add_argument("--env", default="gridworld", choices=["gridworld", "pendulum", "reacher", "cartpole"],
                        help="gridworld (default) | pendulum | reacher | cartpole")
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp", action="store_true", help="use MLP world model")
    parser.add_argument("--fps", type=float, default=5.0, help="render + record fps")
    parser.add_argument("--no-record", action="store_true", help="live view only, no session recording")
    parser.add_argument("--record-dir", default="logs/sessions")
    args = parser.parse_args()

    if args.env == "cartpole":
        args.env = "InvertedPendulum-v5"
    elif args.env == "pendulum":
        args.env = "Pendulum-v1"
    elif args.env == "reacher":
        args.env = "Reacher-v5"

    store = ObservabilityStore(maxlen=max(1000, args.cycles))
    recorder = SessionRecorder(root=args.record_dir, fps=args.fps, record=not args.no_record)
    session_dir = recorder.start({"env": args.env, "seed": args.seed,
                                  "cycles": args.cycles, "grid_size": args.grid_size,
                                  "mlp": args.mlp})
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
            print(f"[cycle thread] error: {e}", file=sys.stderr)
        finally:
            try:
                if "cycle" in cycle_holder:
                    cycle_holder["cycle"].env.close()
            except Exception:
                pass
            stop_flag.set()

    ct = threading.Thread(target=_run_cycle, daemon=True)
    ct.start()

    path_trail: list = []
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("PHCA v3.0 — Visual Observability Layer  (live, non-invasive)")
    ax_world = fig.add_subplot(1, 2, 1)
    ax_drive = fig.add_subplot(2, 4, 3)
    ax_trend = fig.add_subplot(2, 4, 4)
    ax_att = fig.add_subplot(2, 4, 7)
    ax_status = fig.add_subplot(2, 4, 8)
    mind_axs = (ax_drive, ax_trend, ax_att, ax_status)

    interval = 1.0 / max(args.fps, 0.1)
    last_recorded = -1
    plt.show(block=False)
    try:
        while True:
            snap = store.snapshot()
            new = [f for f in snap if f.cycle_id > last_recorded]
            if new:
                last_recorded = new[-1].cycle_id
                for f in new:
                    if f.agent_pos is not None and (
                        not path_trail or path_trail[-1] != f.agent_pos
                    ):
                        path_trail.append(f.agent_pos)
                        if len(path_trail) > 300:
                            del path_trail[: len(path_trail) - 300]
                    recorder.record(f, fig)  # JSONL + PNG per frame
                latest = new[-1]
                _render_world(ax_world, latest, path_trail)
                _render_mind(mind_axs, latest, snap)
            done = stop_flag.is_set() and (last_recorded >= args.cycles - 1 or len(store) >= args.cycles)
            if done:
                break
            try:
                plt.pause(interval)
            except KeyboardInterrupt:
                stop_flag.set()
                break
            # If the window was closed by the user (real backend), stop.
            if not plt.fignum_exists(fig.number):
                stop_flag.set()
                break
    finally:
        # Final drain: record any frames produced after the last render tick.
        stop_flag.set()
        ct.join(timeout=2.0)
        snap = store.snapshot()
        extra = [f for f in snap if f.cycle_id > last_recorded]
        for f in extra:
            recorder.record(f, fig)
            last_recorded = f.cycle_id
        recorder.close()
        print(f"Done. {len(store)} frames captured; "
              f"{last_recorded + 1} recorded to disk."
              + (f" Session: {session_dir}" if session_dir else ""))


if __name__ == "__main__":
    main()
