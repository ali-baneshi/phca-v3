"""PHCA v3.0 — Cognitive Observatory (Observability v3.1, PyQt5).

Panel views (imported from separate modules): Overview, Flow, Action,
Phase, Retention, Memory, Goals. Shared base + drawing helpers are in
``qt_base.py``.

Tabs (7 — indices 0–6):
  0. Overview              — grid/MuJoCo world, drives, error/confidence, session-results (review)
  1. Cognitive Flow      — ASI→M2→G′→PE→PEU→TSPL→Action→RBTA pipeline; timings, near-bound, violations
  2. Action Selection    — candidate scores, explore/exploit, rollouts (live-only)
  3. Phase Space & Trajectory — trajectory, radar, per-dim traces
  4. Retention & Resources — RSS/M3/M4, mechanism rollup, RBTA bounds table
  5. Memory & Belief     — M1/M2/M3 snapshots, m3_top_error
  6. Goals & Motivation  — drives D1–D6, goal stack, pareto

All polling-side: read-only on the frame; the cycle thread never touches Qt.
"""

# Direct imports for view classes (ruff-visible).
from __future__ import annotations
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# The rest from qt_base (shared constants, drawing helpers, monitoring imports).
from phca.monitoring.qt_base import *  # noqa: F401, F403

# Explicit re-exports from qt_base needed by view classes + tests.
from phca.monitoring.qt_base import (
    _BaseCanvas, _ChartCanvas, _RolloutCloudCache,
    PANEL_BG, PANEL_BG_ALT, PANEL_BORDER, CHIP_FILL_ALPHA,
    GRID_COL, TEXT_COL, DIM_COL, _CAPTION_COL, _FOOTER_COL, ACCENT,
    DRIVE_NAMES, DRIVE_SHORT, DRIVE_COLORS,
    PIPELINE, SIDE_MODULES, GRID_ACTIONS, _FLOW_ALL_MODULES,
    EXECUTION_PHASE_STEPS,
    _MECH_BAR_COLORS,
    _F_TITLE, _F_AXIS, _F_LABEL, _F_LABEL_B, _F_CAPTION, _F_DIMSEL,
    _OVERVIEW_HEADER_H, _OVERVIEW_PHASE_H, _OVERVIEW_NARRATIVE_H,
    _OVERVIEW_EVENT_LOG_H, _OVERVIEW_RIBBON_H, _OVERVIEW_MARGIN,
    _OVERVIEW_HZ, _OVERVIEW_LEARN_MS_MIN, _OVERVIEW_EVENT_HOLD,
    _FLOW_REPLAY_Y0, _CONTRACT_BANNER_Y0, _FLOW_HDR_CAPTION_Y, _FLOW_HDR_CAPTION_H,
    _ACTION_TITLE_H, _ACTION_STATUS_H, _ACTION_CHIPS_H,
    _ACTION_DECISION_H, _ACTION_CTX_H, _ACTION_EXPLAIN_H,
    _ACTION_MECH_H, _ACTION_SPARK_W, _ACTION_SPARK_H,
    _ACTION_SPARK_GAP, _ACTION_FOOTER_H, _ACTION_EPSILON_H,
    _OVERVIEW_SEMANTIC_MAP,
    _REACHER_L1, _REACHER_L2, _REACHER_TRAIL_MAX,
    _TAU_POS, _TAU_NEG,
    _apply_decision_shift,
    _qss, _phase_layout, _flow_layout, _action_layout, _traj_canvas_layout,
    _flow_bound_for, _heatmap_cell_alpha, _heatmap_column_percentile,
    _heatmap_cell_color, _phase_grid_caption, _grid_err_summary_line,
    _draw_belief_rollout_cloud, _draw_rollout_score_legend,
    _draw_mechanism_stacked_bar, _draw_action_decision_card,
    _draw_score_proxy_cloud, _flow_update_active_idx,
    _action_chosen_idx, _action_candidate_label,
    _draw_moment_ticks, _draw_moment_chips, _elide_line, _dim_label,
    _n_drives, _drive_color, _drive_short, _drive_name,
    _limb_line_start, _vec_pca2, _dim_arrow_label,
    _cost_color, _to_qcolor, _arrow, _radial_sankey_link, _map_pt, _ellipse_pixel_axes,
    _retention_score, _draw_decay_sparkline, _draw_measured_sparkline,
    _window_session_incomplete, _draw_data_contract_banner,
    _draw_cached_pixmap, _draw_qimage, _draw_sparkline, _draw_tau_bar,
    _overview_grid_chip, _overview_chip,
    _overview_plain_story, _overview_metrics_line, _overview_new_events,
    _execution_phase_segments, _overview_phase_segments,
    _execution_dominant_phase, _overview_dominant_phase,
    _draw_execution_phase_strip, _draw_overview_phase_strip,
    _draw_overview_event_log, _draw_overview_narrative,
    _draw_mini_drive_ring, _draw_vitals_ribbon,
    _draw_overview_tau_bar, _draw_overview_pca_fallback,
    _draw_reacher_schematic, _render_reacher_schematic_pixmap,
    _apply_overview_camera_overlay,
    _draw_agent_limbs, _draw_agent_glyph, _draw_obs_vector_bars,
    _draw_overview_grid, _draw_overview_body, _draw_overview_header,
    _draw_phase_grid_base, _draw_session_results_panel,
    _Smoother, freeze_sig,
    _normalize_rgb_frame,
    _prediction_heatmap,
    _AUTOSCALE_FROZEN,
    set_autoscale_frozen,
    _TransportBar,
    # --- re-exported from overview_narrative through qt_base ---
    _overview_moment_flags,
    _overview_goal_id,
    _overview_goal_intent_line,
    _overview_evidence_line,
    _overview_outcome_line,
    _overview_spike,
    _phase_frame_is_grid,
    _phase_ms,
    _phase_status_line,
    _flow_bottleneck_key,
    _flow_status_line,
    _action_score_margin,
    _action_status_line,
    _reacher_kinematics_from_obs,
    TREND_WINDOW,
)


# ----- Overview tab (extracted to qt_overview.py) ---------------------------
from phca.monitoring.qt_overview import (  # noqa: E402, F401
    CameraCaptureTimer,
    OverviewAgentView,
    AgentPortraitView,
    WorldCanvas,
    DrivesCanvas,
    TrendCanvas,
    AttentionCanvas,
    StatusPanel,
)

# ----- Cognitive Flow tab (extracted to qt_flow.py) -----------------------
from phca.monitoring.qt_flow import CognitiveFlowView  # noqa: E402, F401


# ----- Action Selection tab (extracted to qt_action.py) -------------------
from phca.monitoring.qt_action import CandidateScoreView  # noqa: E402, F401


# ----- Phase Space tab (extracted to qt_phase.py) --------------------------
from phca.monitoring.qt_phase import (  # noqa: E402, F401
    TrajectoryView,
    DriveRadarView,
    _DimSelector,
    _PhasePortraitView,
)


# ----- Retention tab (extracted to qt_retention.py) ------------------------
from phca.monitoring.qt_retention import (  # noqa: E402, F401
    RetentionView,
    ViolationTable,
    RBTABoundsView,
)


# ----- NEW Memory & Belief tab ------------------------------------------------


# ----- Memory / Goals panels (extracted) ------------------------------------
from phca.monitoring.qt_memory import MemoryBeliefView  # noqa: E402, F401
from phca.monitoring.qt_goals import GoalsMotivationView  # noqa: E402, F401

# ----- Controller + main window (extracted to qt_app.py) ---------------------
from phca.monitoring.qt_app import (  # noqa: E402, F401
    DashboardController,
    RenderPacer,
    _PhaseSpaceTab,
    ObservatoryWindow,
    make_app,
)

