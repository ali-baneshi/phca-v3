"""PHCA v3.0 — Monitoring & Live Dashboard Support.

Stable public API for programmatic observability access. Import from this
package root; submodule paths remain supported for internal scripts.

See ``docs/observability_api.md`` for the stability contract.
"""
from __future__ import annotations

from phca.monitoring.cognitive_panels import (
    append_cognitive_moment,
    build_moment_series,
    classify_action_mechanism,
    cognitive_moment,
    count_moments,
    goal_id_from_frame,
)
from phca.monitoring.multi_agent import (
    DEFAULT_AGENT_ID,
    agent_meta_from_frames,
    frames_for_agent,
    is_multi_agent_session,
    multi_agent_meta_patch,
    session_agent_ids,
    validate_agent_cycle_contiguity,
    validate_aligned_timeline,
)
from phca.monitoring.export import export_session_csv
from phca.monitoring.observability import (
    OBSERVABILITY_SCHEMA_VERSION,
    SUPPORTED_OBSERVABILITY_SCHEMA_VERSIONS,
    ObservabilityFrame,
    ObservabilityStore,
    normalize_observability_json,
    observability_schema_version,
)
from phca.monitoring.playback import CyclePacer, PlaybackClock, throttle_period
from phca.monitoring.session_io import (
    frame_from_json,
    frame_to_json,
    load_jsonl_lines,
    load_jsonl_path,
    load_session_frames,
)
from phca.monitoring.session_query import (
    MomentMatch,
    MomentQuery,
    current_match_position,
    export_matches_jsonl,
    format_match_line,
    matches_to_json,
    navigate_match,
    query_frames,
    query_session_dir,
)
from phca.monitoring.session_report import (
    build_session_report,
    compare_all_agents,
    compare_session_reports,
    load_session_report,
    write_session_report,
)

__all__ = [
    # Schema contract
    "OBSERVABILITY_SCHEMA_VERSION",
    "SUPPORTED_OBSERVABILITY_SCHEMA_VERSIONS",
    "normalize_observability_json",
    "observability_schema_version",
    # Core types
    "ObservabilityFrame",
    "ObservabilityStore",
    # Frame I/O
    "frame_from_json",
    "frame_to_json",
    "load_jsonl_lines",
    "load_jsonl_path",
    "load_session_frames",
    # Session report
    "build_session_report",
    "compare_session_reports",
    "compare_all_agents",
    "write_session_report",
    "load_session_report",
    # Multi-agent (Phase 17)
    "DEFAULT_AGENT_ID",
    "agent_meta_from_frames",
    "frames_for_agent",
    "is_multi_agent_session",
    "multi_agent_meta_patch",
    "session_agent_ids",
    "validate_agent_cycle_contiguity",
    "validate_aligned_timeline",
    # Export
    "export_session_csv",
    # Session query (Phase 18)
    "MomentQuery",
    "MomentMatch",
    "query_frames",
    "query_session_dir",
    "navigate_match",
    "current_match_position",
    "matches_to_json",
    "export_matches_jsonl",
    "format_match_line",
    # Cognitive moments
    "cognitive_moment",
    "append_cognitive_moment",
    "build_moment_series",
    "count_moments",
    "goal_id_from_frame",
    "classify_action_mechanism",
    # Playback basics
    "PlaybackClock",
    "CyclePacer",
    "throttle_period",
]
