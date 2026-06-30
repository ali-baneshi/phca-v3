#!/usr/bin/env python3
"""PHCA v3.0 — Live Terminal Monitor.

Displays real-time cognitive cycle metrics in a curses-based terminal
dashboard while the cycle is running.

Usage:
    python scripts/phca-monitor.py --cycles=200
    python scripts/phca-monitor.py --cycles=100 --mlp
    python scripts/phca-monitor.py --cycles=50 --grid-size=10

Keys:
    q  Quit (returns terminal to normal)
    r  Reset display counters
"""

from __future__ import annotations

import argparse
import curses
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.logging import ensure_logging
from phca.monitoring.metrics_store import MetricsStore


# ── Dashboard Thread ─────────────────────────────────────────


def _dashboard_thread(store: MetricsStore, stdscr) -> None:
    """Dashboard render loop (runs every 500ms).

    Reads from the shared MetricsStore and renders a live display
    using curses. Uses non-blocking getch() for keyboard input.

    Args:
        store: Shared metrics store written by the cognitive cycle.
        stdscr: Curses screen object.
    """
    curses.curs_set(0)
    stdscr.nodelay(1)
    running = True
    cycle_offset = 0  # for 'r' key reset

    while running:
        key = stdscr.getch()
        if key == ord('q'):
            running = False
            break
        elif key == ord('r'):
            cycle_offset = store.latest().cycle_id if store.latest() else 0

        latest = store.latest()
        history = store.snapshot()

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        if latest is not None and history:
            latencies = [m.latency_ms for m in history]
            errors = [m.prediction_error for m in history]
            actions = [m.action_name for m in history if m.action_name]
            violations = sum(m.violations_count for m in history)
            n_cycles = len(history)

            _draw_header(stdscr, latest, n_cycles, h, w)
            _draw_latency(stdscr, latencies, h, w)
            _draw_prediction(stdscr, latest, errors, h, w)
            _draw_goal(stdscr, latest, h, w)
            _draw_actions(stdscr, actions, h, w)
            _draw_rbta(stdscr, latest, violations, h, w)
            _draw_drives(stdscr, latest, h, w)
            _draw_skill(stdscr, latest, h, w)
            _draw_footer(stdscr, h, w)
        else:
            stdscr.addstr(0, 0, "PHCA v3.0 — Live Monitor  [waiting for first cycle...]")

        stdscr.refresh()
        time.sleep(0.5)


# ── Render Functions ─────────────────────────────────────────


def _draw_header(stdscr, m: CycleMetrics, n_total: int, h: int, w: int) -> None:
    """Draw the top status bar with system status."""
    if m.rbta_action == "CONTINUE":
        status = "OK"
        color = curses.A_REVERSE
    elif m.rbta_action == "INTERRUPT":
        status = "WARN"
        color = curses.A_REVERSE | curses.A_BOLD
    else:
        status = "CRIT"
        color = curses.A_REVERSE | curses.A_BLINK

    text = f" PHCA v3.0 — Live Monitor    Cycle: {m.cycle_id:04d}  [{status}]  Total: {n_total}"
    stdscr.addstr(0, 0, text.ljust(w - 1)[:w - 1], color)


def _draw_latency(stdscr, latencies: List[float], h: int, w: int) -> None:
    """Draw latency statistics row."""
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95)) if len(latencies) > 1 else 0.0
    max_lat = float(max(latencies))
    text = f" LATENCY  Mean: {mean_lat:.1f}ms  p95: {p95_lat:.1f}ms  Max: {max_lat:.1f}ms"
    stdscr.addstr(2, 0, text[:w - 1])


def _draw_prediction(stdscr, m: CycleMetrics, errors: List[float], h: int, w: int) -> None:
    """Draw prediction metrics row."""
    mean_err = float(np.mean(errors)) if errors else 0.0
    text = (
        f" PREDICT  Error: {m.prediction_error:.4f}  "
        f"Mean: {mean_err:.4f}  "
        f"Conf: {m.prediction_confidence:.3f}"
    )
    stdscr.addstr(3, 0, text[:w - 1])


def _draw_goal(stdscr, m: CycleMetrics, h: int, w: int) -> None:
    """Draw current goal information (drive_id from CycleMetrics)."""
    goal_text = f" GOAL     D{m.drive_id}"
    if m.goal_reached:
        goal_text += "  [GOAL REACHED!]"
    stdscr.addstr(4, 0, goal_text[:w - 1])


def _draw_actions(stdscr, actions: List[str], h: int, w: int) -> None:
    """Draw action distribution."""
    if not actions:
        return
    counts: dict = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    items = sorted(counts.items(), key=lambda x: -x[1])
    parts = [f"{name}({count})" for name, count in items]
    text = " ACTIONS  " + " ".join(parts)
    stdscr.addstr(5, 0, text[:w - 1])


def _draw_rbta(stdscr, m: CycleMetrics, total_violations: int, h: int, w: int) -> None:
    """Draw RBTA enforcement status."""
    if m.rbta_action == "CONTINUE":
        color = 0
    elif m.rbta_action == "INTERRUPT":
        color = curses.A_BOLD
    else:
        color = curses.A_BOLD | curses.A_BLINK

    text = (
        f" RBTA     {m.rbta_action}"
        f"  {m.violations_count} violations (total: {total_violations})"
    )
    stdscr.addstr(6, 0, text[:w - 1], color)


def _draw_drives(stdscr, m: CycleMetrics, h: int, w: int) -> None:
    """Draw MDIM drive (current active drive from goal)."""
    drive_names = {1: "Explore", 2: "Critical", 3: "Practice",
                   4: "Curious", 5: "Energy", 6: "Empower"}
    name = drive_names.get(m.drive_id, "Unknown")
    text = f" DRIVE    D{m.drive_id} ({name})"
    stdscr.addstr(7, 0, text[:w - 1])


def _draw_skill(stdscr, m: CycleMetrics, h: int, w: int) -> None:
    """Draw skill compilation status and memory info."""
    compiled = "YES" if m.skill_compiled else "no"
    text = f" SKILL    Acc: {m.skill_accuracy:.3f}  Compiled: {compiled}  Facts: {m.fact_count}  Episodes: {m.episode_count}"
    stdscr.addstr(8, 0, text[:w - 1])


def _draw_footer(stdscr, h: int, w: int) -> None:
    """Draw bottom status bar with keybindings."""
    text = " q=quit  r=reset"
    stdscr.addstr(h - 1, 0, text.ljust(w - 1)[:w - 1], curses.A_REVERSE)


# ── Main ─────────────────────────────────────────────────────


def run_monitor(
    size: int = 5,
    n_cycles: int = 500,
    use_mlp: bool = False,
    seed: int = 42,
    obstacles: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Run the cognitive cycle with live terminal monitoring.

    Builds a CognitiveCycle with a MetricsStore, starts a curses
    dashboard in a daemon thread, then runs cycles in the main
    thread. The dashboard reads metrics every 500ms without
    blocking the cycle.

    Args:
        size: GridWorld size.
        n_cycles: Number of cognitive cycles to run.
        use_mlp: Use MLP world model instead of Gaussian G'.
        seed: Random seed.
        obstacles: Wall positions for GridWorld.
    """
    # Ensure file logging is configured
    ensure_logging()

    store = MetricsStore(maxlen=1000)
    cycle = CognitiveCycle.build_for_env(
        size=size, seed=seed, use_mlp=use_mlp,
        obstacles=obstacles, metrics_store=store,
    )

    def _curses_main(stdscr) -> None:
        """Curses wrapper: start dashboard thread, run cycles."""
        dash_thread = threading.Thread(
            target=_dashboard_thread, args=(store, stdscr),
            daemon=True,
        )
        dash_thread.start()

        try:
            for _ in range(n_cycles):
                cycle.step()
                if not dash_thread.is_alive():
                    break
        except KeyboardInterrupt:
            pass

    curses.wrapper(_curses_main)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA Live Terminal Monitor")
    parser.add_argument("--cycles", type=int, default=200,
                        help="Number of cognitive cycles to run (default: 200)")
    parser.add_argument("--mlp", action="store_true",
                        help="Use MLP world model")
    parser.add_argument("--grid-size", type=int, default=5,
                        help="GridWorld size (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_monitor(
        size=args.grid_size,
        n_cycles=args.cycles,
        use_mlp=args.mlp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
