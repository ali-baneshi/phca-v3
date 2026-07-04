"""Qt-free helpers for multi-agent Observatory sessions (Phase 17)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from phca.monitoring.observability import ObservabilityFrame

DEFAULT_AGENT_ID = 0


def session_agent_ids(frames: Sequence[ObservabilityFrame]) -> List[int]:
    """Sorted unique agent ids present in a frame sequence."""
    ids = {int(getattr(f, "agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID) for f in frames}
    return sorted(ids)


def frames_for_agent(
    frames: Sequence[ObservabilityFrame],
    agent_id: int,
) -> List[ObservabilityFrame]:
    """Filter frames to one agent, preserving JSONL order."""
    aid = int(agent_id)
    return [
        f for f in frames
        if int(getattr(f, "agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID) == aid
    ]


def agent_meta_from_frames(
    frames: Sequence[ObservabilityFrame],
) -> List[Dict[str, Any]]:
    """Build ``meta.agents[]`` summary entries from recorded frames."""
    counts: Dict[int, int] = defaultdict(int)
    labels: Dict[int, str] = {}
    for f in frames:
        aid = int(getattr(f, "agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID)
        counts[aid] += 1
        lbl = str(getattr(f, "agent_label", "") or "")
        if lbl and aid not in labels:
            labels[aid] = lbl
    return [
        {
            "agent_id": aid,
            "label": labels.get(aid, ""),
            "recorded_cycles": counts[aid],
        }
        for aid in sorted(counts)
    ]


def is_multi_agent_session(
    meta: Optional[Dict[str, Any]] = None,
    frames: Optional[Sequence[ObservabilityFrame]] = None,
    parsed: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """True when the session has more than one distinct agent."""
    if meta is not None:
        ac = meta.get("agent_count")
        if isinstance(ac, (int, float)) and int(ac) > 1:
            return True
        agents = meta.get("agents")
        if isinstance(agents, list) and len(agents) > 1:
            return True
    if frames is not None:
        return len(session_agent_ids(frames)) > 1
    if parsed is not None:
        ids = {
            int(obj.get("agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID)
            for obj in parsed
        }
        return len(ids) > 1
    return False


def validate_agent_cycle_contiguity(
    parsed: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Per-agent ``cycle_id`` must be contiguous 0..n-1."""
    by_agent: Dict[int, List[int]] = defaultdict(list)
    for obj in parsed:
        aid = int(obj.get("agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID)
        cid = int(obj.get("cycle_id", -1))
        by_agent[aid].append(cid)
    for aid in sorted(by_agent):
        cids = by_agent[aid]
        for i, cid in enumerate(cids):
            if cid != i:
                return False, (
                    f"agent_id={aid}: cycle_id={cid} expected {i} "
                    f"(per-agent contiguous 0..{len(cids) - 1})"
                )
    return True, None


def validate_aligned_timeline(
    parsed: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """When timeline_step is set, each step must have one line per agent."""
    steps: Dict[int, Dict[int, int]] = defaultdict(dict)
    for obj in parsed:
        ts = obj.get("timeline_step", -1)
        if ts is None or int(ts) < 0:
            return True, None
        aid = int(obj.get("agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID)
        steps[int(ts)][aid] = int(obj.get("cycle_id", -1))
    if not steps:
        return True, None
    agent_ids = {
        int(obj.get("agent_id", DEFAULT_AGENT_ID) or DEFAULT_AGENT_ID)
        for obj in parsed
    }
    for ts in sorted(steps):
        present = set(steps[ts])
        if present != agent_ids:
            missing = agent_ids - present
            return False, f"timeline_step={ts}: missing agent_id(s) {sorted(missing)}"
    return True, None


def multi_agent_meta_patch(
    frames: Sequence[ObservabilityFrame],
    *,
    timeline_mode: str = "aligned",
) -> Dict[str, Any]:
    """Meta fields to patch on SessionRecorder close for multi-agent runs."""
    agents = agent_meta_from_frames(frames)
    if len(agents) <= 1:
        return {}
    return {
        "agent_count": len(agents),
        "agents": agents,
        "timeline_mode": timeline_mode,
        "recording_layout": "single_jsonl",
    }
