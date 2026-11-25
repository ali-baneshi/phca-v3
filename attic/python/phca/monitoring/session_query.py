"""Qt-free cognitive-moment query engine for Observatory sessions (Phase 18)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from phca.monitoring.cognitive_panels import (
    FLOW_ALL_MODULES,
    build_moment_series,
    flow_near_bound_module,
    goal_id_from_frame,
)
from phca.monitoring.multi_agent import DEFAULT_AGENT_ID, frames_for_agent
from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.session_io import frame_to_json, load_session_frames

PathLike = Union[str, Path]

_MOMENT_FLAG_KEYS = (
    "spike",
    "violation",
    "explored",
    "learn_burst",
    "decision_shift",
)


@dataclass
class MomentQuery:
    """Filter specification for cognitive-moment queries (AND semantics)."""

    spike: bool = False
    violation: bool = False
    explore: bool = False
    learn_burst: bool = False
    decision_shift: bool = False
    drive_id: Optional[int] = None
    near_bound_module: Optional[str] = None
    cycle_min: Optional[int] = None
    cycle_max: Optional[int] = None
    agent_id: Optional[int] = None

    def has_moment_filters(self) -> bool:
        return any((
            self.spike,
            self.violation,
            self.explore,
            self.learn_burst,
            self.decision_shift,
            self.drive_id is not None,
            self.near_bound_module is not None,
        ))

    def validate(self) -> None:
        mod = self.near_bound_module
        if mod is not None and mod not in FLOW_ALL_MODULES:
            allowed = ", ".join(FLOW_ALL_MODULES)
            raise ValueError(f"unknown near-bound module {mod!r} (allowed: {allowed})")


@dataclass
class MomentMatch:
    """One frame index that satisfies a MomentQuery."""

    index: int
    cycle_id: int
    agent_id: int = DEFAULT_AGENT_ID
    flags: Dict[str, Any] = field(default_factory=dict)


def _match_flags_snapshot(moment: Dict[str, Any], f: ObservabilityFrame) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for key in _MOMENT_FLAG_KEYS:
        if key in moment:
            snap[key] = moment[key]
    nb = moment.get("near_bound")
    if nb:
        snap["near_bound"] = nb
    dc = moment.get("drive_change")
    if dc:
        snap["drive_change"] = dc
    gid = goal_id_from_frame(f)
    if gid is not None:
        snap["drive_id"] = gid
    return snap


def _frame_in_cycle_range(f: ObservabilityFrame, q: MomentQuery) -> bool:
    cid = int(getattr(f, "cycle_id", 0) or 0)
    if q.cycle_min is not None and cid < int(q.cycle_min):
        return False
    if q.cycle_max is not None and cid > int(q.cycle_max):
        return False
    return True


def _moment_matches(f: ObservabilityFrame, moment: Dict[str, Any], q: MomentQuery) -> bool:
    if not _frame_in_cycle_range(f, q):
        return False
    if q.spike and not moment.get("spike"):
        return False
    if q.violation and not moment.get("violation"):
        return False
    if q.explore and not moment.get("explored"):
        return False
    if q.learn_burst and not moment.get("learn_burst"):
        return False
    if q.decision_shift and not moment.get("decision_shift"):
        return False
    if q.drive_id is not None:
        gid = goal_id_from_frame(f)
        if gid is None or int(gid) != int(q.drive_id):
            return False
    if q.near_bound_module is not None:
        if flow_near_bound_module(f) != q.near_bound_module:
            return False
    return True


def query_frames(
    frames: Sequence[ObservabilityFrame],
    q: MomentQuery,
) -> List[MomentMatch]:
    """Return all frame indices matching ``q`` (read-only, O(n))."""
    q.validate()
    if not frames:
        return []
    series = build_moment_series(list(frames))
    matches: List[MomentMatch] = []
    for idx, (f, moment) in enumerate(zip(frames, series)):
        if q.has_moment_filters():
            if not _moment_matches(f, moment, q):
                continue
        elif not _frame_in_cycle_range(f, q):
            continue
        matches.append(MomentMatch(
            index=idx,
            cycle_id=int(getattr(f, "cycle_id", idx) or idx),
            agent_id=int(getattr(f, "agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID),
            flags=_match_flags_snapshot(moment, f),
        ))
    return matches


def query_session_dir(
    session_dir: PathLike,
    q: MomentQuery,
) -> List[MomentMatch]:
    """Load a session directory and query its frames."""
    frames = load_session_frames(session_dir)
    if q.agent_id is not None:
        frames = frames_for_agent(frames, int(q.agent_id))
    return query_frames(frames, q)


def navigate_match(
    matches: Sequence[MomentMatch],
    cursor: int,
    direction: int,
) -> Optional[int]:
    """Return the frame index of the prev (-1) or next (+1) match from ``cursor``."""
    if not matches or direction not in (-1, 1):
        return None
    indices = [m.index for m in matches]
    cur = int(cursor)
    if direction > 0:
        for i in indices:
            if i > cur:
                return i
        return indices[0]
    for i in reversed(indices):
        if i < cur:
            return i
    return indices[-1]


def current_match_position(
    matches: Sequence[MomentMatch],
    cursor: int,
) -> int:
    """1-based position of the nearest match at or before ``cursor`` (0 if none)."""
    if not matches:
        return 0
    cur = int(cursor)
    pos = 0
    for j, m in enumerate(matches, start=1):
        if m.index <= cur:
            pos = j
        else:
            break
    return pos


def matches_to_json(
    matches: Sequence[MomentMatch],
    *,
    query: MomentQuery,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured JSON payload for CLI / API consumers."""
    return {
        "count": len(matches),
        "filters": {k: v for k, v in asdict(query).items() if v not in (False, None, "")},
        "meta": dict(meta or {}),
        "matches": [
            {
                "index": m.index,
                "cycle_id": m.cycle_id,
                "agent_id": m.agent_id,
                "flags": dict(m.flags),
            }
            for m in matches
        ],
    }


def export_matches_jsonl(
    matches: Sequence[MomentMatch],
    frames: Sequence[ObservabilityFrame],
    out_path: PathLike,
) -> Path:
    """Write matching frames to JSONL without mutating inputs."""
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for m in matches:
        if 0 <= m.index < len(frames):
            lines.append(json.dumps(frame_to_json(frames[m.index])))
    dest.write_text("\n".join(lines) + ("\n" if lines else ""))
    return dest


def format_match_line(m: MomentMatch, *, multi_agent: bool = False) -> str:
    """Plain-text line for CLI default output."""
    if multi_agent:
        return f"{m.agent_id}:{m.cycle_id}"
    return str(m.cycle_id)
