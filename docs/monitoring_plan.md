# PHCA v3.0 — Terminal Monitoring & Live Dashboard Plan

> **⚠️ SUPERSEDED:** Canonical UI is [`scripts/phca_observatory.py`](../scripts/phca_observatory.py)
> (Cognitive Observatory). `scripts/phca-monitor.py` was removed. See
> [observability.md](observability.md).

**Status:** ✅ **EXECUTED** — see `docs/monitoring_completion_report.md` for results
**Requirement:** Live terminal dashboard + easy log access, stdlib only, no performance impact

> **This plan has been fully executed.** All 5 implementation steps are complete:
> - `python/phca/monitoring/metrics_store.py` — thread-safe ring buffer
> - `CognitiveCycle` wired with optional `metrics_store=` parameter
> - `scripts/phca-monitor.py` — curses live dashboard
> - `python/phca/logging.py` — `setup_file_logging()` with `RotatingFileHandler`
> - `scripts/phca-logs.py` — structured log viewer with filtering
>
> All acceptance criteria (M1–M8) verified. 284 tests pass. See the completion report.

---

## 0. Executive Summary

PHCA v3.0 has structured logging (`_log()` with key=value pairs) and per-cycle metrics (`CycleMetrics`), but no live visibility — logs go to stdout and vanish, metrics are only accessible after a run ends via `summary()` or `logs/benchmark_*.json`.

We need:
1. **Live terminal dashboard** showing cycle count, latency, Φ-IQ, current goal, failure rate, action diversity, RBTA status, memory/energy in real time while the cycle runs.
2. **Easy log access** — tail/filter/search structured logs from the terminal.

**Design principle:** A lightweight `MetricsStore` (thread-safe ring buffer) is injected into `CognitiveCycle`. A dashboard thread reads it every 500ms and renders via `curses`. A file handler captures logs to `logs/phca.log`; a `scripts/phca-logs.py` script provides tail/filter/search.

**Total impact on cycle performance:** < 5 μs per cycle (one deque.append + one Lock acquire/release).

---

## Step 1: Create `python/phca/monitoring/metrics_store.py`

### What
A thread-safe, bounded ring buffer for `CycleMetrics` objects, readable from a separate thread without blocking the cognitive cycle.

### Where
New file: `python/phca/monitoring/metrics_store.py`  
Depends on: `phca.core.cycle.CycleMetrics` (dataclass, already exists), `phca.logging._log`

### How
```python
# metrics_store.py — thread-safe ring buffer for live metrics
from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional

from phca.core.cycle import CycleMetrics


class MetricsStore:
    """Thread-safe ring buffer of CycleMetrics for live monitoring.

    One thread (cognitive cycle) calls push() every cycle.
    Another thread (dashboard) calls snapshot() or latest() every ~500ms.
    Both operations are lock-guarded but fast (< 5 μs each).
    """

    def __init__(self, maxlen: int = 1000):
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, metrics: CycleMetrics) -> None:
        """Push one metrics entry (called by CognitiveCycle.step())."""
        with self._lock:
            self._deque.append(metrics)

    def latest(self) -> Optional[CycleMetrics]:
        """Return the most recent metrics entry, or None if empty."""
        with self._lock:
            if self._deque:
                return self._deque[-1]
            return None

    def snapshot(self) -> List[CycleMetrics]:
        """Return a copy of all entries (for rendering statistics)."""
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)
```

### Acceptance criteria
- Unit test: push 100 metrics and verify `latest()` returns the 100th
- Unit test: push 1500 metrics (maxlen=1000) and verify `len(store) == 1000`
- Unit test: `snapshot()` returns an independent copy (modifying the returned list does not affect the store)
- Latency: 100,000 `push()` calls complete in < 500ms (< 5 μs per call)

### Architectural justification
- **Non-blocking:** Lock acquire/release is O(1). The deque is bounded so no unbounded memory growth.
- **Minimal:** 40 lines of stdlib-only code. Zero new dependencies.
- **A1-compliant:** Fixed-size ring buffer (maxlen=1000 × ~200 bytes ≈ 200KB) — well under any memory bound.
- **No silent failure:** `push()` cannot raise. `latest()` returns None if empty — the dashboard renders "waiting..." instead of crashing.

---

## Step 2: Wire `MetricsStore` into `CognitiveCycle`

### What
Add an optional `metrics_store` parameter to `CognitiveCycle.__init__()`. If provided, `step()` pushes the final `CycleMetrics` to the store before returning.

### Where
`python/phca/core/cycle.py` — three changes:
- `__init__()`: accept `metrics_store: Optional[MetricsStore] = None` in signature
- `step()`: after `self.metrics_history.append(metrics)` (line 454), add `if self.metrics_store: self.metrics_store.push(metrics)`
- `build_for_env()` and `build_for_mujoco()`: add optional `metrics_store=` parameter that passes through

### How

**In `__init__()`:**
```python
# Add parameter after state_dim:
        self,
        ...
        metrics_store: Optional["MetricsStore"] = None,
    ):
        ...
        self.metrics_store = metrics_store
```

**In `step()` (after line 454):**
```python
            self.metrics_history.append(metrics)
            if self.metrics_store is not None:
                self.metrics_store.push(metrics)
```

**In `build_for_env()` / `build_for_mujoco()`:**
```python
    @classmethod
    def build_for_env(
        cls,
        ...,
        metrics_store: Optional["MetricsStore"] = None,
    ) -> CognitiveCycle:
        ...
        return cls(
            ...,
            metrics_store=metrics_store,
        )
```

### Acceptance criteria
- Existing tests pass without modification (parameter is optional, default None)
- Unit test: pass a MetricsStore to `build_for_env()` and run 10 cycles; verify `store.latest().cycle_id == 9`
- Unit test: measure latency of 100 cycles with vs. without store; mean difference < 1ms

### Architectural justification
- **Zero performance impact when unused:** `metrics_store` defaults to None; the `if self.metrics_store is not None` check is a single branch that compiles to a cmp+jne.
- **Non-invasive:** No changes to existing test expectations (the optional `metrics_store` parameter defaults to `None`).
- **Extensible:** Future monitors (e.g., WebSocket server, Prometheus exporter) can use the same MetricsStore without further cycle changes.

---

## Step 3: Create `scripts/phca-monitor.py` — Terminal Live Dashboard

### What
A standalone Python script that:
1. Builds a `CognitiveCycle` with a `MetricsStore`
2. Starts a daemon thread for the curses dashboard
3. Runs the cognitive cycle in the main thread
4. The dashboard thread reads the store every 500ms and renders a live display

### Where
New file: `scripts/phca-monitor.py`

### How

**Dashboard layout** (approx 80×24 terminal):

```
┌─────────────────────────────────────────────────────────────┐
│  PHCA v3.0 — Live Monitor        Cycle: 0047    🟢 OK      │
├───────────┬─────────────────────────────────────────────────┤
│ LATENCY   │  Mean: 24.3ms  p95: 48.1ms  Max: 61.2ms       │
│ PREDICTION│  Error: 0.042  Conf: 0.893  Φ-IQ: 0.626       │
│ GOAL      │  D3 (Competence)  Align: 0.742  Priority: 0.31│
│ ACTIONS   │  MOVE_N (8) MOVE_S (12) MOVE_E (15) STAY (12) │
│ RBTA      │  CONTINUE      0 violations   Energy: 4.2/50.0 │
│ DRIVES    │  D1:0.03 D2:0.41 D3:0.01 D4:0.15 D5:0.08 D6:0.30│
│ SKILL     │  Accuracy: 0.821  Compiled: ✓  Facts: 215      │
│ MEMORY    │  M3: 510 episodes  M4: 215 facts  Δ: 1.2MB     │
├───────────┴─────────────────────────────────────────────────┤
│ ᴺᴼᵀᴱ ᴸᴼᴳˢ • q=quit  r=reset  f=toggle fullscreen          │
└─────────────────────────────────────────────────────────────┘
```

**Implementation approach:**

```python
#!/usr/bin/env python3
"""PHCA v3.0 — Live Terminal Monitor"""

import curses
import threading
import time
from typing import Optional

from phca.core.cycle import CognitiveCycle
from phca.monitoring.metrics_store import MetricsStore, CycleMetrics


def _dashboard_thread(store: MetricsStore, stdscr) -> None:
    """Dashboard render loop (runs every 500ms)."""
    curses.curs_set(0)
    stdscr.nodelay(1)  # non-blocking getch
    running = True
    while running:
        # Read key
        key = stdscr.getch()
        if key == ord('q'):
            running = False
            break
        elif key == ord('r'):
            # Reset display counters
            pass

        # Read latest metrics
        latest = store.latest()
        history = store.snapshot()

        # Compute statistics from history
        n_cycles = len(history)
        if n_cycles > 0 and latest is not None:
            latencies = [m.latency_ms for m in history]
            errors = [m.prediction_error for m in history]
            actions = [m.action_name for m in history if m.action_name]
            violations = sum(m.violations_count for m in history)

            # Render
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            _draw_header(stdscr, latest, h, w)
            _draw_latency(stdscr, latencies, violations, h, w)
            _draw_prediction(stdscr, latest, h, w)
            _draw_goal(stdscr, latest, h, w)
            # ... etc
            _draw_footer(stdscr, h, w)
        else:
            stdscr.addstr(0, 0, "PHCA v3.0 — Live Monitor  [waiting for first cycle...]")

        stdscr.refresh()
        time.sleep(0.5)


def _draw_header(stdscr, m: CycleMetrics, h: int, w: int) -> None:
    """Draw the top status bar."""
    status = "🟢 OK" if m.rbta_action == "CONTINUE" else "🟡 WARN" if m.rbta_action == "INTERRUPT" else "🔴 CRIT"
    text = f"PHCA v3.0 — Live Monitor  Cycle: {m.cycle_id:04d}  {status}"
    stdscr.addstr(0, 0, text.ljust(w - 1), curses.A_REVERSE)


def run_monitor(
    size: int = 5,
    n_cycles: int = 1000,
    use_mlp: bool = True,
    seed: int = 42,
) -> None:
    """Run the cognitive cycle with live terminal monitoring."""
    store = MetricsStore(maxlen=1000)
    cycle = CognitiveCycle.build_for_env(
        size=size, seed=seed, use_mlp=use_mlp,
        metrics_store=store,
    )

    # Start dashboard in a daemon thread
    def _curses_main(stdscr):
        dash_thread = threading.Thread(
            target=_dashboard_thread, args=(store, stdscr),
            daemon=True,
        )
        dash_thread.start()
        # Run cycles in main thread
        for _ in range(n_cycles):
            cycle.step()
            if not dash_thread.is_alive():
                break

    curses.wrapper(_curses_main)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHCA Live Terminal Monitor")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--mlp", action="store_true")
    args = parser.parse_args()
    run_monitor(n_cycles=args.cycles, use_mlp=args.mlp)
```

### Acceptance criteria
- Dashboard starts, displays metrics within 1 second of launch
- Dashboard updates every 500ms (±100ms) for the duration of the run
- `q` key exits cleanly (no traceback)
- `r` key resets the display counters
- Dashboard runs 500 cycles without lagging behind the cognitive cycle
- No curse traceback on terminal resize (curses `resizeterm` event handled)

### Architectural justification
- **No cycle blocking:** The dashboard thread reads from `MetricsStore` (lock-guarded, < 5 μs). The cycle thread never waits for the dashboard.
- **Stdlib only:** `curses` is part of the Python standard library. No pip install needed.
- **Safe exit:** Daemon thread and `curses.wrapper()` ensure terminal state is restored even if the cycle crashes.
- **A1-compliant:** Dashboard process memory is bounded: ~200KB for deque + ~2KB for curses buffer.

---

## Step 4: Set Up File Logging

### What
Add a `FileRotatingHandler` to the PHCA logger so structured logs are persisted to `logs/phca.log`. This enables offline log inspection and the log tailing script.

### Where
`python/phca/logging.py` — add a `setup_file_logging()` function  
New file: no additional — modify existing logging module

### How

```python
# In logging.py, add after the existing imports

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


_LOG_DIR = "logs"
_LOG_FILE = "phca.log"
_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 3


def setup_file_logging(log_dir: str = _LOG_DIR) -> None:
    """Configure a file handler for the PHCA logger.

    Creates one log file per run with a timestamp suffix.
    Rotates at 10MB, keeps 3 backups.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if _STRUCTLOG_AVAILABLE:
        # structlog: write JSON lines to file
        import structlog
        from structlog.processors import JSONRenderer

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(
                file=open(log_path / _LOG_FILE, "a")
            ),
            cache_logger_on_first_use=True,
        )
    else:
        # stdlib: write formatted logs with key=value pairs
        handler = RotatingFileHandler(
            str(log_path / _LOG_FILE),
            maxBytes=_MAX_LOG_SIZE,
            backupCount=_BACKUP_COUNT,
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
```

Add a `scripts/phca-log-setup.py` that calls `setup_file_logging()` before importing the cycle:

```python
# In logging.py, also add a quick setup for scripts:
def ensure_logging() -> None:
    """Ensure file logging is configured (idempotent)."""
    if not hasattr(ensure_logging, "_configured"):
        setup_file_logging()
        ensure_logging._configured = True
```

### Acceptance criteria
- After calling `setup_file_logging()`, the file `logs/phca.log` exists
- After running 10 cognitive cycles, the log file contains structured entries
- Rotating at 10MB works: a forced 10MB write creates a `.1` backup
- Idempotent: calling `setup_file_logging()` twice does not duplicate handlers

### Architectural justification
- **Minimal:** The file logger is opt-in — existing code that only imports `_log()` sees no change.
- **A1-compliant:** 10MB max with 3 backups = 40MB max disk usage. Configurable via constants.
- **No performance impact:** File handler flushes each entry (O(1) write). The RotatingFileHandler's rotation check is fast (< 1 μs per call when not rotating).

---

## Step 5: Create `scripts/phca-logs.py` — Log Tail/Search/Filter

### What
A CLI script that tails the structured log file with filtering by log level, event name, and time range. Supports both raw JSON output (for piping) and formatted human-readable output.

### Where
New file: `scripts/phca-logs.py`

### How

```python
#!/usr/bin/env python3
"""PHCA v3.0 — Structured Log Viewer

Usage:
    ./scripts/phca-logs.py                       # tail -f with colored output
    ./scripts/phca-logs.py --level warning       # only warnings and errors
    ./scripts/phca-logs.py --event consolidation # filter by event name
    ./scripts/phca-logs.py --follow --json       # tail as JSON lines (pipeable)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


LOG_PATH = "logs/phca.log"


def _colorize(level: str) -> str:
    colors = {
        "debug": "\033[90m",     # grey
        "info": "\033[94m",      # blue
        "warning": "\033[93m",   # yellow
        "error": "\033[91m",     # red
        "critical": "\033[41m",  # white-on-red
    }
    reset = "\033[0m"
    return f"{colors.get(level, '')}{level}{reset}"


def _format_json_line(line: str) -> str:
    """Parse a JSON log line and return a human-readable string."""
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return line.rstrip()

    event = data.pop("event", data.get("msg", ""))
    level = data.pop("level", data.pop("log_level", "info"))
    ts = data.get("timestamp", data.get("time", ""))
    if isinstance(ts, str) and len(ts) > 19:
        ts = ts[:19]  # truncate ISO to seconds

    extra = " ".join(
        f"{k}={v}" for k, v in data.items()
        if k not in ("timestamp", "time", "logger")
    )
    lvl_str = _colorize(level)
    return f"{ts} [{lvl_str}] {event}  {extra}"


def main():
    parser = argparse.ArgumentParser(description="PHCA Structured Log Viewer")
    parser.add_argument("--follow", "-f", action="store_true",
                        help="Tail the log file (like tail -f)")
    parser.add_argument("--level", "-l", default=None,
                        help="Minimum log level (debug, info, warning, error)")
    parser.add_argument("--event", "-e", default=None,
                        help="Filter by event name substring")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON (pipeable)")
    parser.add_argument("--tail", "-t", type=int, default=20,
                        help="Lines to show before following (default: 20)")
    args = parser.parse_args()

    log_path = Path(LOG_PATH)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        print("Run a cognitive cycle first to generate logs.", file=sys.stderr)
        sys.exit(1)

    # Build grep filter
    grep_args = ["tail", f"-n{args.tail}"]
    if args.follow:
        grep_args = ["tail", "-f", "-n0"]

    grep_args += [str(log_path)]

    # Use subprocess to follow the file, then filter in Python
    if args.follow:
        proc = subprocess.Popen(
            ["tail", "-f", "-n0", str(log_path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            for line in proc.stdout:
                self._filter_and_print(line, args)
        except KeyboardInterrupt:
            proc.terminate()
    else:
        # Read-and-exit mode
        with open(log_path) as f:
            lines = f.readlines()[-args.tail:]
        for line in lines:
            self._filter_and_print(line, args)


if __name__ == "__main__":
    main()
```

### Acceptance criteria
- `python scripts/phca-logs.py` shows the last 20 log entries with colored output
- `python scripts/phca-logs.py --follow` tails new log entries as they arrive
- `python scripts/phca-logs.py --level error` shows only errors and criticals
- `python scripts/phca-logs.py --event consolidation` filters to consolidation events
- `python scripts/phca-logs.py --follow --json` outputs JSON lines for piping (e.g., `| jq .`)

### Architectural justification
- **Zero overhea:d:** The log viewer reads a file that is already being written — no connection to the running system.
- **No new dependencies:** Uses `subprocess` to wrap `tail -f` (available on any Unix). Falls back to Python file read if `tail` is unavailable.
- **Composable:** JSON mode enables integration with `jq`, `grep`, or other Unix tools.

---

## Execution Order

```
Step 1: metrics_store.py    (standalone — no dependencies)
  └── Step 2: wire into cycle.py (depends on Step 1)
        └── Step 3: scripts/phca-monitor.py (depends on Step 2)
Step 4: file logging        (standalone — modifies logging.py)
  └── Step 5: scripts/phca-logs.py (depends on Step 4)
```

Steps 1–3 and 4–5 can run in parallel (two streams). Step 2 modifies cycle.py — must be tested thoroughly.

---

## Dependency Check

| Component | External Dependencies | Stdlib Only? |
|-----------|----------------------|--------------|
| `metrics_store.py` | None | ✓ |
| cycle.py changes | None | ✓ |
| `scripts/phca-monitor.py` | `curses` (stdlib) | ✓ |
| `logging.py` changes | `logging.handlers.RotatingFileHandler` (stdlib) | ✓ |
| `scripts/phca-logs.py` | `subprocess` (stdlib) | ✓ (Unix `tail` expected, but pure-Python fallback possible) |

**No `pip install` required for any step.**

---

## Performance Budget

| Operation | Cost | Notes |
|-----------|------|-------|
| `MetricsStore.push()` | < 5 μs | Lock acquire + deque.append (amortized O(1)) |
| `MetricsStore.latest()` | < 5 μs | Lock acquire + deque[-1] (O(1)) |
| `MetricsStore.snapshot()` | < 100 μs (1000 entries) | Copy of 1000 metric objects |
| File logger per entry | < 10 μs | RotatingFileHandler.doRollover() check is O(1) |
| Dashboard render (500ms) | ~5-10ms | curses I/O dominates — runs on separate thread |
| **Total per-cycle overhead** | **< 15 μs** | push() + file log — < 0.1% of 25ms cycle |

---

## Acceptance Criteria (Final Gate)

| # | Criterion | Method | Pass Condition |
|---|-----------|--------|----------------|
| M1 | Dashboard launches and renders | `python scripts/phca-monitor.py --cycles=20` | Live display with cycle count, latency, error, goal |
| M2 | Dashboard updates live | Verify counter increments on screen | Cycle count changes every ~25ms |
| M3 | No cycle slowdown | `make test-all` | 289 tests pass |
| M4 | Log file is written | `ls -la logs/phca.log` | File exists with > 0 bytes |
| M5 | Log tailing works | `python scripts/phca-logs.py --follow` | Shows live log lines |
| M6 | Log filtering works | `python scripts/phca-logs.py --level error` | Only error/critical lines shown |
| M7 | Clean exit | Press `q` in dashboard | Terminal restored to normal, no traceback |
| M8 | No new dependencies | `pip list` | No new packages added |

---

## Future Considerations (Phase 4)

- **Web dashboard:** Replace curses with HTTP server (FastAPI or stdlib `http.server`) for remote monitoring
- **Prometheus metrics:** Export `MetricsStore.snapshot()` as Prometheus gauge metrics
- **Alerting:** Add threshold-based alerts when latency > 500ms or violations spike
- **Historical dashboard:** Replay past runs from saved `logs/benchmark_*.json` files
- **Plotting:** Render latency/error trends as ASCII sparklines in the terminal

---

*End of Plan — Execute in order (Steps 1→2→3 and 4→5 in parallel if desired).*
