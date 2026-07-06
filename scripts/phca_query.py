#!/usr/bin/env python3
"""PHCA v3.0 — Observatory cognitive-moment query CLI (Phase 18).

Query recorded sessions for spikes, violations, explore cycles, drive changes,
near-bound modules, and more. Uses the shared ``session_query`` engine.

Usage:
    PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --spike
    PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --violation --json
    PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --near-bound prediction
    PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --explore --export /tmp/out.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import _bootstrap  # noqa: F401

from phca.monitoring.multi_agent import is_multi_agent_session
from phca.monitoring.session_io import load_session_frames
from phca.monitoring.session_query import (
    MomentQuery,
    export_matches_jsonl,
    format_match_line,
    matches_to_json,
    query_frames,
    query_session_dir,
)


def _load_session_meta(session_dir: str) -> Tuple[Optional[dict], Optional[Path]]:
    d = Path(session_dir)
    meta_p = d / "meta.json"
    jsonl_p = d / "timeseries.jsonl"
    if not meta_p.exists() or not jsonl_p.exists():
        print(f"ERROR: not a valid session dir: {d}", file=sys.stderr)
        return None, None
    meta = json.loads(meta_p.read_text())
    return meta, d


def _parse_cycle_range(raw: str) -> Tuple[int, int]:
    if ":" not in raw:
        raise ValueError("cycle-range must be A:B")
    a, b = raw.split(":", 1)
    return int(a), int(b)


def _build_query(args: argparse.Namespace) -> MomentQuery:
    cycle_min = cycle_max = None
    if args.cycle_range:
        cycle_min, cycle_max = _parse_cycle_range(args.cycle_range)
    return MomentQuery(
        spike=bool(args.spike),
        violation=bool(args.violation),
        explore=bool(args.explore),
        learn_burst=bool(args.learn_burst),
        decision_shift=bool(args.decision_shift),
        drive_id=args.drive,
        near_bound_module=args.near_bound,
        cycle_min=cycle_min,
        cycle_max=cycle_max,
        agent_id=args.agent_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query Observatory sessions for cognitive moments",
    )
    parser.add_argument("session_dir", help="path to logs/sessions/<ts>/")
    parser.add_argument("--spike", action="store_true", help="prediction-error spike")
    parser.add_argument("--violation", action="store_true", help="RBTA violation")
    parser.add_argument("--explore", action="store_true", help="explore action")
    parser.add_argument("--learn-burst", action="store_true", help="G′ learn burst")
    parser.add_argument("--decision-shift", action="store_true", help="decision score shift")
    parser.add_argument("--drive", type=int, default=None, metavar="N", help="active drive/goal id")
    parser.add_argument("--cycle-range", default=None, metavar="A:B",
                        help="inclusive cycle_id range")
    parser.add_argument("--near-bound", default=None, metavar="MODULE",
                        help="module near RBTA bound (e.g. prediction)")
    parser.add_argument("--agent-id", type=int, default=None, metavar="N",
                        help="filter to one agent (multi-agent sessions)")
    parser.add_argument("--json", action="store_true", help="structured JSON output")
    parser.add_argument("--count-only", action="store_true", help="print match count only")
    parser.add_argument("--export", nargs="?", const="", default=None, metavar="PATH",
                        help="export matching frames to JSONL (default: session/query_matches.jsonl)")
    parser.add_argument("--allow-empty", action="store_true",
                        help="exit 0 when no matches (default: exit 1)")
    args = parser.parse_args()

    meta, sess = _load_session_meta(args.session_dir)
    if meta is None or sess is None:
        return 1

    try:
        query = _build_query(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        matches = query_session_dir(sess, query)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not matches and not args.allow_empty:
        print("No matches.", file=sys.stderr)
        return 1

    multi = is_multi_agent_session(meta) or query.agent_id is not None

    if args.count_only:
        print(len(matches))
    elif args.json:
        print(json.dumps(matches_to_json(matches, query=query, meta=meta), indent=2))
    else:
        for m in matches:
            print(format_match_line(m, multi_agent=multi))

    if args.export is not None:
        frames = load_session_frames(sess)
        if query.agent_id is not None:
            from phca.monitoring.multi_agent import frames_for_agent
            frames = frames_for_agent(frames, int(query.agent_id))
        out = Path(args.export) if args.export else sess / "query_matches.jsonl"
        export_matches_jsonl(matches, frames, out)
        if not args.count_only and not args.json:
            print(f"Exported {len(matches)} frames -> {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
