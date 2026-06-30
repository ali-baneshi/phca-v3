#!/usr/bin/env python3
"""PHCA v3.0 — Structured Log Viewer.

Tails, filters, and searches the structured PHCA log file from the terminal.

Usage:
    python scripts/phca-logs.py                       # show last 20 entries
    python scripts/phca-logs.py --follow               # tail -f mode
    python scripts/phca-logs.py --level warning        # only warnings+
    python scripts/phca-logs.py --event consolidation  # filter by event
    python scripts/phca-logs.py --follow --json        # raw JSON (pipeable)
    python scripts/phca-logs.py --tail=50              # show last 50 lines
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


LOG_PATH = "logs/phca.log"

LEVEL_COLORS = {
    "debug": "\033[90m",     # grey
    "info": "\033[94m",      # blue
    "warning": "\033[93m",   # yellow
    "error": "\033[91m",     # red
    "critical": "\033[41m\033[97m",  # white-on-red
}
RESET = "\033[0m"

LEVEL_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}


def _colorize_level(level: str) -> str:
    """Return the colorized level string, or plain if level unknown."""
    color = LEVEL_COLORS.get(level, "")
    return f"{color}{level}{RESET}"


def _passes_level_filter(level: str, min_level: str | None) -> bool:
    """Check if level passes the minimum level filter."""
    if min_level is None:
        return True
    min_val = LEVEL_ORDER.get(min_level, 0)
    actual_val = LEVEL_ORDER.get(level, 1)
    return actual_val >= min_val


def _format_line(line: str, output_json: bool) -> str | None:
    """Parse a log line and return formatted string, or None if invalid.

    Handles both JSON lines (structlog) and plain text (stdlib fallback).
    """
    line = line.rstrip()
    if not line:
        return None

    # Try JSON parsing first (structlog mode)
    if line.startswith("{"):
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON — treat as plain text
            return line

        if output_json:
            return line  # pass through raw

        event = data.get("event", data.get("msg", ""))
        level = data.get("level", data.get("log_level", "info"))
        ts = data.get("timestamp", data.get("time", ""))
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]

        extra = " ".join(
            f"{k}={v}" for k, v in data.items()
            if k not in ("event", "msg", "level", "log_level",
                         "timestamp", "time", "logger")
        )
        lvl_str = _colorize_level(level)
        if extra:
            return f"{ts} [{lvl_str}] {event}  {extra}"
        return f"{ts} [{lvl_str}] {event}"

    # Plain text (stdlib fallback)
    return line


def _filter_and_print(line: str, args: argparse.Namespace) -> None:
    """Filter a log line by level and event, then print if it passes."""
    # For JSON lines, extract level for filtering
    level = "info"
    if line.startswith("{"):
        try:
            data = json.loads(line)
            level = data.get("level", data.get("log_level", "info"))
        except (json.JSONDecodeError, ValueError):
            pass

    # Level filter
    if not _passes_level_filter(level, args.level):
        return

    # Event filter (substring match against the formatted line)
    formatted = _format_line(line, args.json)
    if formatted is None:
        return
    if args.event and args.event.lower() not in formatted.lower():
        return

    print(formatted, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA Structured Log Viewer")
    parser.add_argument("--follow", "-f", action="store_true",
                        help="Tail the log file (like tail -f)")
    parser.add_argument("--level", "-l", default=None,
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Minimum log level to show")
    parser.add_argument("--event", "-e", default=None,
                        help="Filter by event name substring")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON lines (pipeable)")
    parser.add_argument("--tail", "-t", type=int, default=20,
                        help="Number of recent lines to show (default: 20)")
    args = parser.parse_args()

    log_path = Path(LOG_PATH)
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        print("Run a cognitive cycle or benchmark first to generate logs.",
              file=sys.stderr)
        sys.exit(1)

    if args.follow:
        # Tail mode: use subprocess tail -f, pipe through Python filter
        try:
            proc = subprocess.Popen(
                ["tail", "-f", "-n", "0", str(log_path)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            print("Error: 'tail' command not found (required for --follow mode)",
                  file=sys.stderr)
            sys.exit(1)

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                _filter_and_print(line, args)
        except KeyboardInterrupt:
            proc.terminate()
    else:
        # Read-and-exit mode
        try:
            with open(log_path) as f:
                lines = f.readlines()
        except OSError as e:
            print(f"Error reading log file: {e}", file=sys.stderr)
            sys.exit(1)

        for line in lines[-args.tail:]:
            _filter_and_print(line, args)


if __name__ == "__main__":
    main()
