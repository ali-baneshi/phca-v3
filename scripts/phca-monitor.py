#!/usr/bin/env python3
"""PHCA v3.0 — Live Terminal Monitor.

Displays real-time cognitive cycle metrics in a curses-based terminal
dashboard while the cycle is running. Falls back to printing metrics
to stdout every 2 seconds if not in a TTY (e.g., CI, piped output).

Includes ASCII sparkline trend plots showing latency and prediction
error over the last N cycles.

Usage:
    python scripts/phca-monitor.py --cycles=200          # curses mode (TTY)
    python scripts/phca-monitor.py --cycles=50 --text    # force text mode
    python scripts/phca-monitor.py --cycles=100 --mlp
    python scripts/phca-monitor.py --spark-width=30      # wider sparklines

Keys (curses mode):
    q  Quit (returns terminal to normal)
    r  Reset display counters
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

_pkg_root = Path(__file__).resolve().parent.parent / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.logging import ensure_logging
from phca.monitoring.metrics_store import MetricsStore


# ── Sparkline Rendering ──────────────────────────────────────

# Unicode block chars for sparklines: 9 levels (blank to full)
SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

def _sparkline(values: List[float], width: int = 20) -> str:
    """Render an ASCII sparkline from a list of values.

    Maps values to Unicode block characters (U+2581-U+2588) to
    create a compact trend visualization. Downsamples by averaging
    blocks if there are more values than width.

    Args:
        values: List of numeric values to plot.
        width: Number of characters in the sparkline (default 20).

    Returns:
        String of sparkline characters representing the trend.
    """
    if not values:
        return " " * width

    # Downsample if more values than width
    if len(values) > width:
        block = len(values) // width
        downsampled = [
            float(np.mean(values[i * block:(i + 1) * block]))
            for i in range(width)
        ]
    else:
        downsampled = list(values)

    vmin = float(np.min(downsampled))
    vmax = float(np.max(downsampled))
    span = vmax - vmin

    n_levels = len(SPARK_CHARS)
    out: List[str] = []
    for v in downsampled:
        if span < 1e-12:
            idx = n_levels // 2  # midpoint if all values are the same
        else:
            norm = (v - vmin) / span
            idx = min(int(norm * n_levels), n_levels - 1)
        out.append(SPARK_CHARS[idx])
    return "".join(out)


# ── Text Fallback Mode ───────────────────────────────────────

def _run_text_mode(store: MetricsStore, n_cycles: int,
                   cycle: CognitiveCycle, spark_width: int = 20) -> None:
    """Run cycles and print metrics summary to stdout.

    Prints a header once, a one-line summary every 2 seconds with a
    latency sparkline, and a final summary at the end.

    Args:
        store: Shared metrics store.
        n_cycles: Number of cycles to run.
        cycle: The cognitive cycle instance.
        spark_width: Width of sparkline plots.
    """
    print("PHCA v3.0 — Monitor (text mode)")
    print(f"{'Cycle':>6} {'Lat(ms)':>8} {'Error':>8} {'Conf':>6} "
          f"{'RBTA':>10} {'Action':>8} {'Drive':>5}  Sparkline(20)")
    print("-" * 80)

    last_print = time.perf_counter()
    for i in range(n_cycles):
        cycle.step()
        now = time.perf_counter()
        if now - last_print >= 2.0:
            m = store.latest()
            if m is not None:
                history = store.snapshot()
                lat_vals = [x.latency_ms for x in history]
                spark = _sparkline(lat_vals, width=spark_width)
                print(f"{m.cycle_id:>6} {m.latency_ms:>8.1f} "
                      f"{m.prediction_error:>8.4f} "
                      f"{m.prediction_confidence:>6.3f} "
                      f"{m.rbta_action:>10} {m.action_name:>8} D{m.drive_id}  {spark}")
            last_print = now

    # Final summary
    history = store.snapshot()
    if history:
        latencies = [m.latency_ms for m in history]
        errors = [m.prediction_error for m in history]
        print("-" * 80)
        print(f"Completed {n_cycles} cycles. "
              f"Mean latency: {np.mean(latencies):.1f}ms, "
              f"Mean error: {np.mean(errors):.4f}")
        violations = sum(m.violations_count for m in history)
        print(f"Total violations: {violations}")
        # Show final sparklines
        if len(latencies) > 1:
            print(f"Latency trend:  {_sparkline(latencies, width=30)}")
        if len(errors) > 1:
            print(f"Error trend:    {_sparkline(errors, width=30)}")


# ── Curses Dashboard ─────────────────────────────────────────


def _dashboard_thread(store: MetricsStore, stdscr,
                      spark_width: int = 20) -> None:
    """Dashboard render loop (runs every 500ms).

    Reads from the shared MetricsStore and renders a live display
    using curses. Uses non-blocking getch() for keyboard input.

    Args:
        store: Shared metrics store written by the cognitive cycle.
        stdscr: Curses screen object.
        spark_width: Width of sparkline plots.
    """
    import curses
    curses.curs_set(0)
    stdscr.nodelay(1)
    running = True

    while running:
        key = stdscr.getch()
        if key == ord('q'):
            running = False
            break

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

            # Adjust sparkline width to fit terminal
            sw = min(spark_width, max(10, w - 14))

            _draw_header(stdscr, latest, n_cycles, h, w)
            _draw_latency(stdscr, latencies, h, w)
            _draw_sparkline(stdscr, "LAT TREND", latencies, 9, sw, w)
            _draw_prediction(stdscr, latest, errors, h, w)
            _draw_sparkline(stdscr, "ERR TREND", errors, 10, sw, w)
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


def _draw_header(stdscr, m: CycleMetrics, n_total: int, h: int, w: int) -> None:
    """Draw the top status bar with system status."""
    import curses
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


def _draw_sparkline(stdscr, label: str, values: List[float],
                    row: int, width: int, max_width: int = 80) -> None:
    """Draw a sparkline trend at the given row.

    Args:
        stdscr: Curses screen object.
        label: Short label (e.g. "LAT TREND").
        values: Data values to plot.
        row: Terminal row to draw at.
        width: Width of the sparkline in characters.
        max_width: Terminal width in characters (for truncation).
    """
    spark = _sparkline(values, width=width)
    if len(values) >= 2:
        pct = ""
        first = values[0]
        last = values[-1]
        if abs(first) > 1e-12:
            change = (last - first) / abs(first) * 100
            pct = f"  ({change:+.1f}%)"
    else:
        pct = ""
    text = f" {label:<10} {spark}{pct}"
    stdscr.addstr(row, 0, text[:max_width - 1])


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
    import curses
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
    import curses
    text = " q=quit  r=reset"
    stdscr.addstr(h - 1, 0, text.ljust(w - 1)[:w - 1], curses.A_REVERSE)


# ── Main ─────────────────────────────────────────────────────


def run_monitor(
    size: int = 5,
    n_cycles: int = 500,
    use_mlp: bool = False,
    seed: int = 42,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    force_text: bool = False,
    spark_width: int = 20,
) -> None:
    """Run the cognitive cycle with live terminal monitoring.

    Auto-detects TTY: uses curses if available, falls back to text
    mode otherwise. Use force_text=True to always use text mode.

    Args:
        size: GridWorld size.
        n_cycles: Number of cognitive cycles to run.
        use_mlp: Use MLP world model instead of Gaussian G'.
        seed: Random seed.
        obstacles: Wall positions for GridWorld.
        force_text: Force text mode even if TTY is available.
        spark_width: Width of sparkline plots.
    """
    ensure_logging()

    store = MetricsStore(maxlen=1000)
    cycle = CognitiveCycle.build_for_env(
        size=size, seed=seed, use_mlp=use_mlp,
        obstacles=obstacles, metrics_store=store,
    )

    # Text mode: no curses, just print summaries
    if force_text or not sys.stdout.isatty():
        _run_text_mode(store, n_cycles, cycle, spark_width=spark_width)
        return

    # Curses mode: requires a real TTY
    import curses

    def _curses_main(stdscr) -> None:
        """Curses wrapper: start dashboard thread, run cycles."""
        dash_thread = threading.Thread(
            target=_dashboard_thread, args=(store, stdscr, spark_width),
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
    parser.add_argument("--text", action="store_true",
                        help="Force text mode (no curses)")
    parser.add_argument("--spark-width", type=int, default=20,
                        help="Width of sparkline plots (default: 20)")
    args = parser.parse_args()

    run_monitor(
        size=args.grid_size,
        n_cycles=args.cycles,
        use_mlp=args.mlp,
        seed=args.seed,
        force_text=args.text,
        spark_width=args.spark_width,
    )


if __name__ == "__main__":
    main()
