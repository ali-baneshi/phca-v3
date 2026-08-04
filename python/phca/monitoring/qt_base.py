"""Shared base canvas, constants, and drawing helpers for Cognitive Observatory panels."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .camera_render import (
    camera_frame_stats,
    fit_pixmap_to_box,
    is_glitchy_pixmap,
    is_glitchy_rgb_frame,
    rgb_frame_to_pixmap,
)
from .observability import ObservabilityFrame, _normalize_rgb_frame
from .playback import _Smoother, freeze_sig
from .render import _prediction_heatmap
from .cognitive_panels import (
    MOMENT_COLORS,
    PIPELINE_LABEL,
    RBTA_TO_FLOW,
    append_cognitive_moment,
    apply_decision_shift,
    belief_reference,
    build_moment_series,
    data_contract_text,
    flow_action_link_line,
    flow_near_bound_modules,
    classify_action_mechanism,
    mechanism_histogram,
    mechanism_pct,
    pipeline_time_budget_ms,
    rbta_bound_for_module,
    rbta_time_bound_ms,
)
from phca.monitoring.action_explain import build_explain_chain
from phca.monitoring.belief_projection import (
    BeliefProjection,
    ScaleState,
    _AUTOSCALE_FROZEN,
    set_autoscale_frozen,  # noqa: F401 — re-export for tests
)
from phca.monitoring.qt_transport import TransportBar as _TransportBar  # noqa: F401
from phca.monitoring.overview_narrative import (
    TREND_WINDOW,
    _OVERVIEW_PHASE_STEPS,
    _action_score_margin,
    _action_status_line,
    _flow_bottleneck_key,  # noqa: F401 — re-export for tests
    _flow_status_line,
    _overview_evidence_line,
    _overview_goal_id,
    _overview_goal_intent_line,
    _overview_moment_flags,
    _overview_outcome_line,
    _overview_spike,  # noqa: F401 — re-export for tests
    _phase_frame_is_grid,  # noqa: F401 — re-export for tests
    _phase_ms,
    _phase_status_line,
    _reacher_kinematics_from_obs,
)

from PyQt5 import QtWidgets, QtCore, QtGui

_apply_decision_shift = apply_decision_shift

# ----- shared constants ------------------------------------------------------

DRIVE_NAMES = {
    1: "D1 PredErr", 2: "D2 Critical", 3: "D3 Compet",
    4: "D4 Curious", 5: "D5 Energy", 6: "D6 Empower",
}
DRIVE_SHORT = {1: "D1", 2: "D2", 3: "D3", 4: "D4", 5: "D5", 6: "D6"}
DRIVE_COLORS = {
    1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd",
    5: "#1f77b4", 6: "#17becf",
}
PIPELINE = ["sanitize", "memory_write", "prediction", "peu", "tspl",
            "action_selection", "rbta"]
SIDE_MODULES = ["gprime_learn", "mdim", "attn", "hpm", "cr", "consolidation"]
# PIPELINE_LABEL lives in cognitive_panels; re-exported here for legacy imports.
# Verified via env.get_action_names(): ['MOVE_N','MOVE_S','MOVE_E','MOVE_W','STAY']
GRID_ACTIONS = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY"]

_FLOW_ALL_MODULES = PIPELINE + SIDE_MODULES

# Real execution order (matches cycle.py and Overview phase strip).
EXECUTION_PHASE_STEPS = _OVERVIEW_PHASE_STEPS

# Cached reacher schematic pixmaps (keyed by observation hash).
_REACHER_CACHE: Dict[str, QtGui.QPixmap] = {}


def _flow_bound_for(mod: str, bounds: dict) -> Optional[float]:
    """RBTA time bound in seconds (legacy alias)."""
    return rbta_bound_for_module(mod, bounds)


def _heatmap_cell_alpha(ms: float, rmax: float) -> int:
    """Heatmap cell opacity; floor keeps fast Reacher cycles visible."""
    scale = rmax if rmax > 1e-6 else 25.0
    if ms <= 0:
        return 0
    return max(6, int(np.clip(ms / scale, 0, 1) * 230))


def _heatmap_column_percentile(ms: float, col_vals: List[float]) -> int:
    """Alpha from rank within a module's history column (non-zero cells only)."""
    if ms <= 0:
        return 0
    nz = [float(v) for v in col_vals if float(v) > 0]
    if not nz:
        return 20
    rank = sum(1 for v in nz if v <= ms) / len(nz)
    return max(20, int(rank * 230))


def _heatmap_cell_color(ms: float, col_vals: List[float]) -> QtGui.QColor:
    """Cost-semantic heatmap cell; alpha = column percentile for contrast."""
    a = _heatmap_column_percentile(ms, col_vals)
    if a <= 0:
        return QtGui.QColor(0, 0, 0, 0)
    col = _to_qcolor(_cost_color(ms))
    col.setAlpha(a)
    return col


def _heatmap_cell_color_precomputed(ms: float, p_rank: float) -> QtGui.QColor:
    """Cost-semantic heatmap cell with precomputed percentile rank.

    Skips per-cell sort - caller pre-computes ranks for the entire column.
    """
    a = max(20, int(p_rank * 230)) if ms > 0 else 0
    if a <= 0:
        return QtGui.QColor(0, 0, 0, 0)
    col = _to_qcolor(_cost_color(ms))
    col.setAlpha(a)
    return col


_FLOW_REPLAY_Y0 = 16
_CONTRACT_BANNER_Y0 = 16
_FLOW_HDR_CAPTION_Y = 74
_FLOW_HDR_CAPTION_H = 14

_ACTION_TITLE_H = 18
_ACTION_STATUS_H = 14
_ACTION_CHIPS_H = 16
_ACTION_DECISION_H = 28
_ACTION_CTX_H = 28
_ACTION_EXPLAIN_H = 36
_ACTION_MECH_H = 16
_ACTION_SPARK_W = 62
_ACTION_SPARK_H = 28
_ACTION_SPARK_GAP = 4
_ACTION_FOOTER_H = 28
_ACTION_EPSILON_H = 32


def _flow_layout(w: int, h: int, n_mods: int, *, replay: bool = False) -> Dict[str, Any]:
    """Reserve header, graph, and heatmap bands so they do not overlap."""
    y0 = _CONTRACT_BANNER_Y0
    header_h = _FLOW_HDR_CAPTION_Y + _FLOW_HDR_CAPTION_H + 2 + y0
    margin = 6
    label_w = 36 if w < 520 else 44
    heatmap_h = max(90, min(int(h * 0.18), h - header_h - margin - 120))
    graph_h = max(120, h - header_h - heatmap_h - margin)
    graph_top = header_h
    heatmap_top = graph_top + graph_h + margin
    row_h = max(7.0, heatmap_h / max(n_mods, 1))
    return {
        "header_h": header_h,
        "y0": y0,
        "caption_bottom": header_h,
        "heatmap_h": int(heatmap_h),
        "graph_h": int(graph_h),
        "graph_top": graph_top,
        "heatmap_top": int(heatmap_top),
        "row_h": row_h,
        "label_w": label_w,
        "margin": margin,
        "compact": w < 520,
        "node_scale": 0.85 if w < 520 else 1.0,
    }


def _action_layout(w: int, h: int, *, replay: bool = False) -> Dict[str, Any]:
    """Explicit row-based geometry for Action Selection (no band overlap)."""
    y0 = _CONTRACT_BANNER_Y0
    title_y = y0 + 2
    status_y = title_y + _ACTION_TITLE_H
    chips_y = status_y + _ACTION_STATUS_H
    decision_y = chips_y + _ACTION_CHIPS_H
    ctx_y = decision_y + _ACTION_DECISION_H
    explain_y = ctx_y + _ACTION_CTX_H
    mech_y = explain_y + _ACTION_EXPLAIN_H
    body_top = mech_y + _ACTION_MECH_H + 4
    margin = 8
    spark_gutter = 3 * _ACTION_SPARK_W + 2 * _ACTION_SPARK_GAP + 12
    body_h = max(100, h - body_top - margin)
    left_w = max(160, w // 2 - 10)
    right_x = left_w + 16
    right_w = max(120, w - right_x - margin)
    footer_h = _ACTION_FOOTER_H
    epsilon_h = _ACTION_EPSILON_H
    cloud_h = max(60, body_h - epsilon_h - footer_h)
    tau_slot_h = 44
    tau_slot_y = body_top + body_h - tau_slot_h
    list_h = body_h - tau_slot_h - 6
    spark_x0 = w - spark_gutter
    spark_y0 = title_y
    return {
        "y0": y0,
        "title_y": title_y,
        "status_y": status_y,
        "chips_y": chips_y,
        "decision_y": decision_y,
        "ctx_y": ctx_y,
        "explain_y": explain_y,
        "mech_y": mech_y,
        "body_top": body_top,
        "body_h": body_h,
        "left_x": margin,
        "left_y": body_top,
        "left_w": left_w,
        "left_h": list_h,
        "tau_slot_y": tau_slot_y,
        "tau_slot_h": tau_slot_h,
        "right_x": right_x,
        "right_w": right_w,
        "cloud_top": body_top,
        "cloud_h": cloud_h,
        "cloud_bot": body_top + cloud_h,
        "footer_y": body_top + cloud_h,
        "footer_h": footer_h,
        "epsilon_y": body_top + cloud_h + footer_h,
        "epsilon_h": epsilon_h,
        "spark_x0": spark_x0,
        "spark_y0": spark_y0,
        "spark_w": _ACTION_SPARK_W,
        "spark_h": _ACTION_SPARK_H,
        "status_w": max(80, spark_x0 - margin - 4),
    }


def _phase_layout(w: int, h: int, env_kind: str, is_grid: bool,
                  *, tab_w: Optional[int] = None) -> Dict[str, Any]:
    """Explicit geometry for Phase Space tab regions."""
    header_h = 40 if not is_grid else 36
    pager_h = 0 if is_grid else 28
    perdim_h = 0 if is_grid else max(120, int(h * 0.38))
    traj_h = max(140, h - perdim_h - pager_h - 8)
    if not is_grid and h < 500:
        traj_h = max(60, traj_h)
        perdim_h = max(100, min(perdim_h, max(h - traj_h - pager_h - 8, 100)))
    show_perdim = not is_grid
    tw = tab_w if tab_w is not None else w
    show_radar = (not is_grid) and tw >= 900
    radar_w = 180 if show_radar else 0
    return {
        "header_h": header_h,
        "traj_h": traj_h,
        "perdim_h": perdim_h,
        "pager_h": pager_h,
        "show_perdim": show_perdim,
        "show_radar": show_radar,
        "radar_w": radar_w,
        "chart_top": header_h + 4,
        "chart_bot": header_h + traj_h - 8,
    }


def _traj_canvas_layout(w: int, h: int, y0: int, *, is_grid: bool) -> Dict[str, Any]:
    """Widget-local chart geometry for TrajectoryView (not the full Phase Space tab)."""
    spark_w = 96
    footer_h = 18 if is_grid else 36
    title_y = y0 + (12 if is_grid else 28)
    caption_y = y0 + (38 if is_grid else 42)
    chart_top = caption_y + 6
    max_bot = h - footer_h
    avail = max(0, max_bot - chart_top - 4)
    min_h = 24 if is_grid else 24
    chart_h = min(avail, max(min_h, avail)) if avail >= min_h else avail
    chart_bot = min(max_bot, chart_top + chart_h, h - 4)
    legend_y = h - 8
    axis_y = legend_y - 14
    return {
        "chart_top": chart_top,
        "chart_bot": chart_bot,
        "footer_h": footer_h,
        "footer_top": chart_bot + 4,
        "header_top": title_y,
        "header_bot": chart_top - 4,
        "title_y": title_y,
        "caption_y": caption_y,
        "legend_y": legend_y,
        "axis_y": axis_y,
        "spark_w": spark_w,
    }


def _phase_grid_caption(*, low_conf: bool = False) -> str:
    cap = ("orange = |pred cell dist − actual| · gold ghost = argmax pred · "
           "(no confidence layer)")
    if low_conf:
        cap += " · low confidence — err heat decaying"
    return cap


def _grid_err_summary_line(errmap: Optional[np.ndarray], trail: Deque[Any]) -> str:
    tl = len(trail)
    if errmap is None:
        return f"trail={tl} · per-dim portrait N/A for grid"
    idx = np.unravel_index(int(np.argmax(errmap)), errmap.shape)
    val = float(errmap[idx])
    return f"worst cell {idx}={val:.2f} · trail={tl} · per-dim portrait N/A for grid"


class _RolloutCloudCache:
    """Mutable rollout-draw cache shared by Action + Phase Space views."""

    __slots__ = ("sig", "drawn")

    def __init__(self) -> None:
        self.sig: tuple = ()
        self.drawn: List[Tuple] = []

    def clear(self) -> None:
        self.sig = ()
        self.drawn = []


def _draw_belief_rollout_cloud(
    p: QtGui.QPainter,
    f: ObservabilityFrame,
    proj: "BeliefProjection",
    px0: int, py0: int, px1: int, py1: int,
    cache: _RolloutCloudCache,
    *,
    replay: bool = False,
    is_continuous: bool = False,
    draw_anchor_label: bool = True,
) -> bool:
    """Score-colored rollout cloud in the shared PCA plane. Returns True if drawn.

    Callers must set a QPainter clip rect on the chart body before calling
    (Action Selection rollout cloud and Phase Space trajectory both do).
    """
    rollouts = list(getattr(f, "candidate_rollouts", []) or [])
    if not rollouts or proj is None or not proj.history:
        return False
    pareto = set(int(x) for x in (getattr(f, "pareto_front", []) or []))
    scores = [float(r.get("score", 0.0)) for r in rollouts]
    smin, smax = (min(scores), max(scores)) if scores else (0.0, 1.0)
    srange = (smax - smin) or 1.0
    bounds = proj.bounds()
    sig = freeze_sig((tuple(scores), int(f.cycle_id), bounds))
    cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
    cur = proj.project(cur_v) if cur_v is not None else None
    cx = cy = None
    if cur is not None:
        cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
        p.setBrush(ACCENT)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        p.drawEllipse(cx - 3, cy - 3, 6, 6)
        if draw_anchor_label:
            p.setPen(DIM_COL)
            p.setFont(_F_AXIS)
            p.drawText(cx + 6, cy + 3, "now")
    elif replay:
        p.setPen(DIM_COL)
        p.setFont(_F_AXIS)
        p.drawText(px0 + 4, py0 + 14, "state anchor unavailable (replay)")
    if sig != cache.sig:
        cache.sig = sig
        drawn: List[Tuple] = []
        for i, r in enumerate(rollouts):
            pt = proj.project(r.get("predicted"))
            if pt is None:
                continue
            rx, ry = _map_pt(pt, bounds, px0, py0, px1, py1)
            s = float(r.get("score", 0.0))
            t = (s - smin) / srange
            base = QtGui.QColor(int(231 - 180 * t), int(60 + 140 * t), int(60 + 60 * t))
            drawn.append((rx, ry, i, base, bool(r.get("chosen")), i in pareto))
        cache.drawn = drawn
    for rx, ry, i, base, chosen, is_pareto in cache.drawn:
        if chosen:
            continue
        col = QtGui.QColor(base.red(), base.green(), base.blue(), 165)
        if is_continuous and cx is not None and cy is not None:
            p.setPen(QtGui.QPen(col, 1, QtCore.Qt.DashLine))
            p.drawLine(cx, cy, rx, ry)
        p.setBrush(col)
        p.setPen(QtGui.QPen(col.darker(140), 1))
        p.drawEllipse(rx - 4, ry - 4, 8, 8)
        if is_pareto:
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
            p.setBrush(QtGui.QColor(0, 0, 0, 0))
            p.drawEllipse(rx - 7, ry - 7, 14, 14)
    for rx, ry, i, base, chosen, is_pareto in cache.drawn:
        if not chosen:
            continue
        col = QtGui.QColor(base.red(), base.green(), base.blue(), 230)
        if is_continuous and cx is not None and cy is not None:
            p.setPen(QtGui.QPen(col, 2))
            p.drawLine(cx, cy, rx, ry)
        p.setBrush(col)
        p.setPen(QtGui.QPen(ACCENT, 2))
        p.drawEllipse(rx - 6, ry - 6, 12, 12)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        p.setBrush(QtGui.QColor(0, 0, 0, 0))
        p.drawEllipse(rx - 8, ry - 8, 16, 16)
        if is_pareto:
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
            p.drawEllipse(rx - 10, ry - 10, 20, 20)
    return True


def _draw_rollout_score_legend(
    p: QtGui.QPainter, x: int, y: int, w: int, h: int = 10,
) -> None:
    """3-stop mini gradient bar clarifying rollout dot score coloring."""
    if w < 24 or h < 4:
        return
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    p.drawText(x, y - 2, "low")
    p.drawText(x + w - 22, y - 2, "high score")
    for i in range(w):
        t = i / max(w - 1, 1)
        col = QtGui.QColor(int(231 - 180 * t), int(60 + 140 * t), int(60 + 60 * t))
        p.setPen(col)
        p.drawLine(x + i, y, x + i, y + h - 1)


_MECH_BAR_COLORS: Dict[str, QtGui.QColor] = {
    "explore": QtGui.QColor(52, 152, 219),
    "prediction": QtGui.QColor(46, 204, 113),
    "continuous": QtGui.QColor(155, 89, 182),
    "stay": QtGui.QColor(127, 140, 141),
    "greedy_fallback": QtGui.QColor(230, 126, 34),
    "other": QtGui.QColor(100, 100, 110),
}


def _draw_mechanism_stacked_bar(
    p: QtGui.QPainter,
    frames: List[ObservabilityFrame],
    x: int, y: int, w: int, h: int,
) -> None:
    """Horizontal stacked bar of action mechanism mix (last N frames)."""
    if w < 40 or h < 6 or not frames:
        return
    counts = mechanism_histogram(frames, window=256)
    pct = mechanism_pct(counts)
    if not pct:
        return
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    p.drawText(x, y - 2, f"mechanism mix (last {len(frames)})")
    bar_y = y + 2
    bar_h = max(h - 4, 6)
    cx = x
    for key, val in sorted(pct.items(), key=lambda kv: kv[1], reverse=True):
        if val <= 0:
            continue
        seg_w = int(w * val / 100.0)
        if seg_w < 1:
            continue
        col = _MECH_BAR_COLORS.get(key, _MECH_BAR_COLORS["other"])
        p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
        p.fillRect(cx, bar_y, seg_w, bar_h, col)
        if val >= 8 and seg_w >= 18:
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(cx + 2, bar_y + bar_h - 2, f"{key[:6]} {val}%")
        cx += seg_w


def _draw_action_decision_card(
    p: QtGui.QPainter,
    f: ObservabilityFrame,
    scores: List[float],
    chosen: int,
    moment: Optional[Dict[str, Any]],
    pareto: set,
    x: int, y: int, w: int, h: int,
) -> None:
    """Two-column decision summary inset."""
    r = f.action_rationale or {}
    mech = classify_action_mechanism(r)
    gid = r.get("goal_id") or getattr(f, "active_drive_id", None)
    lvls = list(getattr(f, "drive_levels", []) or [])
    lvl_s = ""
    if gid and lvls and 0 < int(gid) <= len(lvls):
        lvl_s = f"{lvls[int(gid) - 1]:.2f}"
    emp = float(getattr(f, "empowerment", 0.0) or 0.0)
    peu = (moment or {}).get("peu_mean")
    if peu is None and getattr(f, "per_dim_peu", None) is not None:
        arr = np.asarray(f.per_dim_peu, dtype=np.float32).reshape(-1)
        if arr.size:
            peu = float(np.mean(arr))
    eps = r.get("eps")
    eps_s = f"{float(eps):.3f}" if isinstance(eps, (int, float)) else "—"
    k = r.get("k_candidates")
    k_s = str(int(k)) if isinstance(k, (int, float)) else "—"
    bs = r.get("best_score")
    bs_s = f"{float(bs):.3f}" if isinstance(bs, (int, float)) else "—"
    margin = _action_score_margin(scores)
    margin_s = f"{margin:+.3f}" if margin is not None else "—"
    cr_t = float(getattr(f, "cr_temperature", 0.0) or 0.0)
    n_par = len(pareto)
    n_sc = len(scores)
    p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
    p.drawRoundedRect(x, y, w, h, 4, 4)
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    mid = x + w // 2
    left = f"mech={mech}  goal={gid}  lvl={lvl_s or '—'}"
    if emp > 0:
        left += f"  emp={emp:.2f}"
    if peu is not None:
        left += f"  PEŪ={peu:.2f}"
    right = (f"ε={eps_s}  k={k_s}  score={bs_s}  Δ2nd={margin_s}"
             f"  Pareto {n_par}/{n_sc}  cr_T={cr_t:.2f}")
    p.drawText(x + 6, y + 12, _elide_line(p, left, mid - x - 8))
    p.drawText(mid + 4, y + 12, _elide_line(p, right, x + w - mid - 8))
    p.drawText(x + 6, y + h - 6, _elide_line(p, flow_action_link_line(f), w - 12))


def _draw_score_proxy_cloud(
    p: QtGui.QPainter,
    f: ObservabilityFrame,
    proj: "BeliefProjection",
    px0: int, py0: int, px1: int, py1: int,
    scores: List[float],
    chosen_idx: int,
) -> bool:
    """Score-ranked radial proxy when candidate_rollouts are unavailable."""
    import math
    if not scores or proj is None or not proj.history:
        return False
    bounds = proj.bounds()
    cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
    cur = proj.project(cur_v) if cur_v is not None else None
    if cur is None:
        return False
    cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
    smin, smax = min(scores), max(scores)
    srange = (smax - smin) or 1.0
    R = min(px1 - px0, py1 - py0) * 0.38
    n = len(scores)
    p.setBrush(ACCENT)
    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
    p.drawEllipse(cx - 3, cy - 3, 6, 6)
    for i, s in enumerate(scores):
        ang = -math.pi / 2 + 2 * math.pi * i / max(n, 1)
        t = (float(s) - smin) / srange
        rad = (0.2 + 0.8 * t) * R
        rx = int(cx + rad * math.cos(ang))
        ry = int(cy + rad * math.sin(ang))
        tt = t
        col = QtGui.QColor(int(231 - 180 * tt), int(60 + 140 * tt), int(60 + 60 * tt))
        is_ch = (i == chosen_idx)
        p.setBrush(col)
        p.setPen(QtGui.QPen(ACCENT if is_ch else col.darker(140), 2 if is_ch else 1))
        p.drawEllipse(rx - (6 if is_ch else 4), ry - (6 if is_ch else 4),
                      12 if is_ch else 8, 12 if is_ch else 8)
    return True


def _flow_update_active_idx(
    prev_heat: Dict[str, float],
    cur_heat: Dict[str, float],
    delta_smooth: Dict[str, _Smoother],
    active_idx: int,
    active_hold: int,
) -> Tuple[int, int]:
    """Pure active-node hysteresis (mirrors CognitiveFlowView.set_frame)."""
    for m in PIPELINE:
        d = float(cur_heat.get(m, 0.0)) - float(prev_heat.get(m, 0.0))
        if m in delta_smooth:
            delta_smooth[m].value(d)
    if active_hold > 0:
        return active_idx, active_hold - 1
    sm = {m: delta_smooth[m]._v for m in PIPELINE}
    new = int(max(range(len(PIPELINE)), key=lambda i: sm[PIPELINE[i]]))
    cur_v = sm[PIPELINE[active_idx]]
    if new != active_idx and sm[PIPELINE[new]] > cur_v * 1.3 + 0.05:
        return new, 3
    return active_idx, 0


def _action_chosen_idx(f: ObservabilityFrame, scores: List[float]) -> int:
    """Authoritative chosen candidate index from telemetry, else argmax."""
    r = f.action_rationale or {}
    ci = r.get("chosen_idx")
    if isinstance(ci, (int, float)):
        return int(ci)
    return int(np.argmax(scores)) if scores else -1


def _action_candidate_label(idx: int, names: List[str], is_continuous: bool) -> str:
    if idx < len(names) and names[idx]:
        nm = str(names[idx])
        if is_continuous and nm.startswith("MOVE_"):
            return f"τ cand {idx}"
        return nm
    return f"τ cand {idx}" if is_continuous else f"a{idx}"


def _draw_moment_ticks(
    p: QtGui.QPainter,
    x0: int,
    y_top: int,
    x1: int,
    y_bot: int,
    moments: List[Dict[str, Any]],
) -> None:
    """Spike / learn / decision-shift ticks along a horizontal history axis."""
    n = len(moments)
    if n < 2:
        return
    span = x1 - x0
    for i, m in enumerate(moments):
        cx = int(x0 + i * span / max(n - 1, 1))
        ty = y_bot - 2
        if m.get("spike"):
            p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["spike"]), 2))
            p.drawLine(cx, ty - 10, cx, ty)
        if m.get("learn_burst"):
            p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["learn_burst"]), 2))
            p.drawLine(cx, ty - 18, cx, ty - 12)
        if m.get("decision_shift"):
            p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["decision_shift"]), 2))
            p.drawLine(cx, ty - 26, cx, ty - 20)
        if m.get("violation"):
            p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["spike"]), 1))
            p.drawLine(cx, ty - 32, cx, ty - 28)


def _draw_moment_chips(
    p: QtGui.QPainter,
    x: int,
    y: int,
    moment: Optional[Dict[str, Any]],
) -> int:
    """Draw compact moment chips; return x after last chip."""
    if not moment:
        return x
    chips: List[Tuple[str, str]] = []
    if moment.get("spike"):
        chips.append(("spike", MOMENT_COLORS["spike"]))
    if moment.get("learn_burst"):
        chips.append(("learn", MOMENT_COLORS["learn_burst"]))
    if moment.get("near_bound"):
        nb = moment["near_bound"]
        nb_lbl = PIPELINE_LABEL.get(nb, str(nb))[:8]
        chips.append((nb_lbl, MOMENT_COLORS["near_bound"]))
    if moment.get("drive_change"):
        chips.append(("driveΔ", MOMENT_COLORS["drive_change"]))
    if moment.get("decision_shift"):
        chips.append(("shift", MOMENT_COLORS["decision_shift"]))
    cx = x
    p.setFont(_F_AXIS)
    for lbl, col_s in chips:
        col = _to_qcolor(col_s)
        tw = 8 + len(lbl) * 6
        p.setPen(QtGui.QPen(col, 1))
        p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), CHIP_FILL_ALPHA))
        p.drawRoundedRect(cx, y, tw, 14, 3, 3)
        p.setPen(col)
        p.drawText(cx + 4, y + 11, lbl)
        cx += tw + 4
    return cx


PANEL_BG = QtGui.QColor(18, 18, 24)
PANEL_BG_ALT = QtGui.QColor(28, 28, 38)
PANEL_BORDER = QtGui.QColor(42, 42, 54)
CHIP_FILL_ALPHA = 120
GRID_COL = QtGui.QColor(60, 60, 70)
TEXT_COL = QtGui.QColor(210, 210, 220)
DIM_COL = QtGui.QColor(150, 150, 160)
_CAPTION_COL = QtGui.QColor(175, 175, 188)
_FOOTER_COL = QtGui.QColor(190, 190, 200)
ACCENT = QtGui.QColor(241, 196, 15)

# v5: cached QFont instances (no per-frame QFont construction → stable metrics).
_F_TITLE = QtGui.QFont("Sans", 10, QtGui.QFont.Bold)
_F_AXIS = QtGui.QFont("Sans", 7)
_F_LABEL = QtGui.QFont("Sans", 8)
_F_LABEL_B = QtGui.QFont("Sans", 8, QtGui.QFont.Bold)
_F_CAPTION = QtGui.QFont("Sans", 8)
_F_DIMSEL = QtGui.QFont("Sans", 8, QtGui.QFont.Bold)


def _elide_line(p: QtGui.QPainter, text: str, max_w: int) -> str:
    if max_w <= 0:
        return text
    return p.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, max_w)


def _dim_label(i: int, f: Optional[ObservabilityFrame]) -> str:
    """Named dimension label from frame.dim_names, else d{i} fallback."""
    if f is not None and f.dim_names and i < len(f.dim_names):
        nm = f.dim_names[i]
        return nm if nm else f"d{i}"
    return f"d{i}"


def _n_drives(f: Optional[ObservabilityFrame] = None,
              levels: Optional[List[float]] = None) -> int:
    """Drive count from frame or an explicit levels list (dim-agnostic, not fixed 6)."""
    if levels is not None and len(levels) > 0:
        return len(levels)
    if f is not None:
        for attr in ("drive_levels", "drive_deficits", "drive_targets"):
            arr = list(getattr(f, attr, []) or [])
            if arr:
                return len(arr)
    return 6


_DRIVE_EXTRA: Dict[int, str] = {}


def _drive_color(did: int) -> QtGui.QColor:
    """1-based drive id → colour; generates a stable HSV swatch for did > 6."""
    if did in DRIVE_COLORS:
        c = DRIVE_COLORS[did]
        return QtGui.QColor(c) if isinstance(c, str) else QtGui.QColor(*c)
    if did not in _DRIVE_EXTRA:
        hue = int((did * 137.508) % 360)
        _DRIVE_EXTRA[did] = QtGui.QColor.fromHsv(hue, 180, 220).name()
    return QtGui.QColor(_DRIVE_EXTRA[did])


def _drive_short(did: int) -> str:
    return DRIVE_SHORT.get(did, f"D{did}")


def _drive_name(did: int) -> str:
    return DRIVE_NAMES.get(did, f"Drive {did}")


def _limb_line_start(cx: int, cy: int, cr: int, ang: float
                     ) -> Tuple[int, int, int, int]:
    """Limb segment from core edge outward (never through the conf disc)."""
    import math
    pad = 2
    r0 = cr + pad
    sx = int(cx + r0 * math.cos(ang))
    sy = int(cy + r0 * math.sin(ang))
    return sx, sy, cx, cy


def _vec_pca2(v: np.ndarray) -> Tuple[float, float, List[int]]:
    """Project a vector to 2-D for arrow/glyph drawing.

    d≤2 → raw components; d>2 → the two dims with largest |value| (labelled
    via dim_names in the caller).  Returns (vx, vy, source_dim_indices).
    """
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return 0.0, 0.0, []
    if v.size == 1:
        return float(np.clip(v[0], -1, 1)), 0.0, [0]
    if v.size == 2:
        return float(np.clip(v[0], -1, 1)), float(np.clip(v[1], -1, 1)), [0, 1]
    idx = [int(x) for x in np.argsort(np.abs(v))[-2:][::-1]]
    return float(np.clip(v[idx[0]], -1, 1)), float(np.clip(v[idx[1]], -1, 1)), idx


def _dim_arrow_label(indices: List[int], dim_names: Optional[List[str]] = None) -> str:
    names = dim_names or []
    parts = [names[i] if i < len(names) and names[i] else f"d{i}" for i in indices]
    return "/".join(parts)


def _qss() -> str:
    """Dark stylesheet — kills the default 'prototype' Qt look."""
    return """
    QMainWindow, QWidget { background: #121218; color: #d2d2dc; }
    QTabWidget::pane { border: 1px solid #2a2a36; top: -1px; }
    QTabBar::tab { background: #1c1c26; color: #b0b0bc;
        padding: 8px 18px; margin-right: 2px;
        border: 1px solid #2a2a36; border-bottom: none;
        border-top-left-radius: 6px; border-top-right-radius: 6px; }
    QTabBar::tab:selected { background: #2b2b3a; color: #f1c40f; font-weight: bold; }
    QTabBar::tab:hover { background: #242432; }
    QTableWidget { background: #16161e; alternate-background-color: #1b1b24;
        gridline-color: #2a2a36; selection-background-color: #3a3a4e; color: #d2d2dc; }
    QHeaderView::section { background: #1c1c26; color: #b0b0bc;
        padding: 4px; border: 1px solid #2a2a36; font-weight: bold; }
    QSlider::groove:horizontal { height: 6px; background: #2a2a36; border-radius: 3px; }
    QSlider::handle:horizontal { background: #f1c40f; width: 14px;
        margin: -5px 0; border-radius: 7px; }
    QSlider::sub-page:horizontal { background: #b8860b; border-radius: 3px; }
    QToolButton { background: #1c1c26; color: #d2d2dc; padding: 6px 12px;
        border: 1px solid #2a2a36; border-radius: 5px; }
    QToolButton:hover { background: #2b2b3a; }
    QLabel { color: #b0b0bc; }
    """


def _cost_color(ms: float) -> str:
    if ms < 5.0:
        return "#2ca02c"
    if ms < 20.0:
        return "#ff7f0e"
    return "#d62728"


def _to_qcolor(hex_or_rgb: Any) -> "QtGui.QColor":
    if isinstance(hex_or_rgb, str):
        return QtGui.QColor(hex_or_rgb)
    r, g, b = hex_or_rgb
    return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))


def _arrow(p: QtGui.QPainter, x0: int, y0: int, x1: int, y1: int, col: QtGui.QColor,
           size: int = 8) -> None:
    """Draw a line with an arrowhead at (x1,y1)."""
    p.setPen(QtGui.QPen(col, 2))
    p.drawLine(x0, y0, x1, y1)
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    a1 = ang + 2.6; a2 = ang - 2.6
    p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
    poly = QtGui.QPolygonF([QtCore.QPointF(x1, y1),
                            QtCore.QPointF(x1 - size * math.cos(a1), y1 - size * math.sin(a1)),
                            QtCore.QPointF(x1 - size * math.cos(a2), y1 - size * math.sin(a2))])
    p.drawPolygon(poly)


def _radial_sankey_link(p: QtGui.QPainter, x0: int, y0: int, x1: int, y1: int,
                        cx: int, cy: int, col: QtGui.QColor, width: float = 2.0) -> None:
    """Curved radial Sankey link — pen width encodes activation mass share."""
    path = QtGui.QPainterPath(QtCore.QPointF(x0, y0))
    midx = (x0 + x1) * 0.35 + cx * 0.65
    midy = (y0 + y1) * 0.35 + cy * 0.65
    path.quadTo(midx, midy, x1, y1)
    pen = QtGui.QPen(col, max(1, int(width)))
    pen.setCapStyle(QtCore.Qt.RoundCap)
    p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
    p.drawPath(path)


def _retention_score(t: float, S: float) -> float:
    """Ebbinghaus retention R = e^{-t/S}."""
    import math
    return math.exp(-max(t, 0.0) / max(float(S), 1e-6))


def _draw_decay_sparkline(p: QtGui.QPainter, x: int, y: int, w: int, h: int,
                          S: float, col: QtGui.QColor, n: int = 24) -> None:
    """Sparkline of R=e^{-t/S} for t=0..n-1."""
    import math
    S = max(float(S), 0.5)
    p.setPen(QtGui.QPen(col, 1))
    for i in range(1, n):
        r0 = math.exp(-(i - 1) / S)
        r1 = math.exp(-i / S)
        x0 = x + (i - 1) * w / max(n - 1, 1)
        x1 = x + i * w / max(n - 1, 1)
        y0 = y + h - r0 * h
        y1 = y + h - r1 * h
        p.drawLine(int(x0), int(y0), int(x1), int(y1))


def _draw_measured_sparkline(p: QtGui.QPainter, x: int, y: int, w: int, h: int,
                             hist: Deque[float], col: QtGui.QColor, *,
                             fallback_S: float = 8.0) -> None:
    """Plot measured history; fall back to decay curve when too few samples."""
    if len(hist) < 2:
        _draw_decay_sparkline(p, x, y, w, h, fallback_S, col)
        return
    vals = list(hist)
    lo, hi = float(min(vals)), float(max(vals))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(vals)
    p.setPen(QtGui.QPen(col, 1))
    for i in range(1, n):
        x0 = x + (i - 1) * w / max(n - 1, 1)
        x1 = x + i * w / max(n - 1, 1)
        y0 = y + h - (vals[i - 1] - lo) / (hi - lo) * h
        y1 = y + h - (vals[i] - lo) / (hi - lo) * h
        p.drawLine(int(x0), int(y0), int(x1), int(y1))


def _map_pt(pt: Tuple[float, float], bounds: Tuple[float, float, float, float],
            px0: int, py0: int, px1: int, py1: int) -> Tuple[int, int]:
    xlo, xhi, ylo, yhi = bounds
    xr = (xhi - xlo) or 1.0; yr = (yhi - ylo) or 1.0
    x = px0 + (pt[0] - xlo) / xr * (px1 - px0)
    y = py1 - (pt[1] - ylo) / yr * (py1 - py0)
    x = int(max(px0, min(px1, x)))
    y = int(max(py0, min(py1, y)))
    return (x, y)


def _ellipse_pixel_axes(cur_pt: Tuple[float, float], ax_p: float, ay_p: float,
                        bounds: Tuple[float, float, float, float],
                        px0: int, py0: int, px1: int, py1: int) -> Tuple[int, int, float, float]:
    """Map belief-space σ axes to pixel radii for uncertainty ellipse."""
    cx, cy = _map_pt(cur_pt, bounds, px0, py0, px1, py1)
    px_ax, _ = _map_pt((cur_pt[0] + ax_p, cur_pt[1]), bounds, px0, py0, px1, py1)
    _, py_ay = _map_pt((cur_pt[0], cur_pt[1] + ay_p), bounds, px0, py0, px1, py1)
    return cx, cy, max(1.0, abs(float(px_ax - cx))), max(1.0, abs(float(py_ay - cy)))


def _draw_qimage(p: QtGui.QPainter, frame: np.ndarray, x: int, y: int,
                 max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    """Fit an RGB uint8 array into a box; return the placed rect."""
    if max_w < 2 or max_h < 2:
        return (x, y, 0, 0)
    if is_glitchy_rgb_frame(frame):
        return (x, y, 0, 0)
    return _draw_cached_pixmap(p, rgb_frame_to_pixmap(frame), x, y, max_w, max_h)


def _draw_cached_pixmap(p: QtGui.QPainter, pixmap: Optional[QtGui.QPixmap],
                        x: int, y: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    """Fit a cached pixmap into a box; return the placed rect."""
    if pixmap is None or pixmap.isNull():
        return (x, y, 0, 0)
    scaled, sw, sh = fit_pixmap_to_box(pixmap, max_w, max_h)
    if scaled.isNull() or sw < 1 or sh < 1:
        return (x, y, 0, 0)
    dx = x + (max_w - sw) // 2
    dy = y + (max_h - sh) // 2
    p.drawPixmap(dx, dy, scaled)
    return (dx, dy, sw, sh)


# ----- v8 Overview 1.5 shared draw helpers -----------------------------------

def _draw_sparkline(p: QtGui.QPainter, vals: List[float], x: int, y: int, w: int, h: int,
                    col: QtGui.QColor, *, log: bool = False) -> None:
    """Mini sparkline with per-series autoscale (no misleading 0..1 clamp)."""
    if len(vals) < 2:
        return
    import math
    data = [math.log10(max(float(v), 1e-6)) for v in vals] if log else [float(v) for v in vals]
    lo, hi = min(data), max(data)
    if hi - lo < 1e-9:
        hi = lo + 1
    n = len(data)
    p.setPen(QtGui.QPen(col, 1))
    for i in range(1, n):
        x0 = x + (i - 1) * w / max(n - 1, 1)
        x1 = x + i * w / max(n - 1, 1)
        y0 = y + h - (data[i - 1] - lo) / (hi - lo) * h
        y1 = y + h - (data[i] - lo) / (hi - lo) * h
        p.drawLine(int(x0), int(y0), int(x1), int(y1))


_TAU_POS = QtGui.QColor(26, 188, 156)   # teal — positive continuous action
_TAU_NEG = QtGui.QColor(243, 156, 18)   # amber — negative continuous action


def _draw_tau_bar(p: QtGui.QPainter, act, dim_names: List[str],
                  x: int, y: int, w: int, h: int) -> None:
    """Signed per-dim continuous action bar (Reacher 2-D friendly)."""
    if act is None:
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + 10, "τ — (no continuous action)")
        return
    v = np.asarray(act, dtype=np.float32).reshape(-1)
    n = min(int(v.size), 8)
    if n == 0:
        return
    mid = y + h // 2 + 2
    p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, mid, x + w, mid)
    bw = w / n
    for i in range(n):
        val = float(np.clip(v[i], -1, 1))
        bx = int(x + i * bw)
        bh = int(abs(val) * max(h // 2 - 6, 4))
        col = _TAU_POS if val >= 0 else _TAU_NEG
        p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
        if val >= 0:
            p.fillRect(bx + 1, mid - bh, max(int(bw) - 2, 1), bh, col)
        else:
            p.fillRect(bx + 1, mid, max(int(bw) - 2, 1), bh, col)
        lbl = dim_names[i] if i < len(dim_names) else f"τ{i}"
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(bx + 1, y + h - 2, f"{lbl[:5]}={val:+.2f}")


def _draw_agent_limbs(p: QtGui.QPainter, f: ObservabilityFrame,
                      cx: int, cy: int, cr: int, R: int) -> None:
    import math
    ca = getattr(f, "continuous_action", None)
    names = list(getattr(f, "action_names", []) or [])
    dim_names = list(getattr(f, "dim_names", []) or [])
    if ca is not None:
        vec = np.asarray(ca, dtype=np.float32).reshape(-1)
        if vec.size >= 1:
            # Reacher (2-torque) is better read as two independent channels.
            # Converting (tau0,tau1) to a single angle causes fast 180° flips.
            kind = (getattr(f, "env_kind", "") or "").lower()
            if vec.size == 2 and kind == "mujoco_rgb":
                box_w = max(40, int(R * 0.72))
                box_h = max(18, int(R * 0.26))
                bx = int(cx - box_w // 2)
                by = int(cy + cr + 8)
                p.setPen(QtGui.QPen(GRID_COL, 1))
                p.setBrush(QtGui.QColor(20, 20, 28, 200))
                p.drawRoundedRect(bx, by, box_w, box_h, 4, 4)
                mid = by + box_h // 2
                p.setPen(QtGui.QPen(GRID_COL, 1))
                p.drawLine(bx + 4, mid, bx + box_w - 4, mid)
                half = (box_w - 8) // 2
                for i in range(2):
                    val = float(np.clip(vec[i], -1.0, 1.0))
                    ch_x = bx + 4 + i * half
                    ch_w = max(10, half - 2)
                    bar_h = int(abs(val) * max(box_h // 2 - 3, 2))
                    col = _TAU_POS if val >= 0 else _TAU_NEG
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(col)
                    if val >= 0:
                        p.fillRect(ch_x, mid - bar_h, ch_w, bar_h, col)
                    else:
                        p.fillRect(ch_x, mid, ch_w, bar_h, col)
                    lbl = dim_names[i] if i < len(dim_names) and dim_names[i] else f"τ{i}"
                    p.setPen(DIM_COL); p.setFont(_F_AXIS)
                    p.drawText(ch_x, by + box_h - 1, f"{lbl[:6]}={val:+.2f}")
                return
            vx, vy, dims = _vec_pca2(vec)
            mag = float(min(1.0, math.hypot(vx, vy)))
            ang = math.atan2(vy, vx)
            reach = cr + int(R * 0.5 * mag)
            sx, sy, _cx, _cy = _limb_line_start(cx, cy, cr, ang)
            ex = int(cx + reach * math.cos(ang))
            ey = int(cy + reach * math.sin(ang))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
            p.drawLine(sx, sy, ex, ey)
            p.setBrush(QtGui.QColor(155, 89, 182)); p.setPen(QtCore.Qt.white)
            p.drawEllipse(ex - 4, ey - 4, 8, 8)
            p.setPen(QtGui.QColor(155, 89, 182)); p.setFont(_F_AXIS)
            if vec.size == 2:
                n0 = dim_names[0] if dim_names else "τ₀"
                n1 = dim_names[1] if len(dim_names) > 1 else "τ₁"
                p.drawText(cx - R, cy + cr + 16,
                           f"τ {n0}={float(vec[0]):+.2f}  {n1}={float(vec[1]):+.2f}")
            else:
                lbl = _dim_arrow_label(dims, dim_names) if dims else "τ"
                p.drawText(cx - R, cy + cr + 16, f"τ({lbl}) ‖·‖={mag:.2f}")
            return
        mag = float(min(1.0, abs(float(vec[0]))))
        p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
        p.drawArc(cx - cr - 6, cy - cr - 6, (cr + 6) * 2, (cr + 6) * 2,
                  90 * 16, int(-mag * 360 * 16))
        return
    scores = list(getattr(f, "candidate_scores", []) or [0])
    chosen = int(np.argmax(scores)) if scores else 0
    a = -math.pi / 2
    sx, sy, _cx, _cy = _limb_line_start(cx, cy, cr, a)
    ex = int(cx + (cr + R * 0.4) * math.cos(a))
    ey = int(cy + (cr + R * 0.4) * math.sin(a))
    p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
    p.drawLine(sx, sy, ex, ey)
    lbl = names[chosen] if chosen < len(names) else f"a{chosen}"
    p.setPen(ACCENT); p.setFont(_F_LABEL_B)
    p.drawText(cx - R, cy + cr + 16, f"act: {lbl}")


def _draw_agent_glyph(p: QtGui.QPainter, f: ObservabilityFrame,
                      cx: int, cy: int, R: int,
                      glyph_hist: Deque[Tuple[List[float], float, int]],
                      t0: float, emp_value: float,
                      *, active_pulse: float = 0.5,
                      show_head_label: bool = False) -> None:
    """Mind glyph: drive halo + limbs + core + head (shared by portrait + overview)."""
    import math
    levels = list(getattr(f, "drive_levels", []) or [])
    deficits = list(getattr(f, "drive_deficits", []) or [])
    n = _n_drives(f, levels)
    while len(levels) < n:
        levels.append(0.0)
    while len(deficits) < n:
        deficits.append(0.0)
    active = int(_overview_goal_id(f) or 0)
    conf = float(getattr(f, "prediction_confidence", 0.0) or 0.0)
    ms = getattr(f, "meta_stable", None) or {}
    stable = bool(ms.get("is_meta_stable", ms.get("stable", False)))
    for gi, (gh_lvls, _gh_conf, _gh_act) in enumerate(list(glyph_hist)[:-1]):
        fade = int(25 + 35 * (gi + 1))
        for i in range(min(n, len(gh_lvls))):
            lvl = max(0.0, min(1.0, float(gh_lvls[i])))
            a0 = -math.pi / 2 + i * 2 * math.pi / n
            a1 = a0 + 2 * math.pi / n - 0.08
            r_out = int(R * (0.50 + 0.35 * lvl))
            col = _drive_color(i + 1); col.setAlpha(fade)
            p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
            path = QtGui.QPainterPath()
            path.moveTo(cx + R * 0.42 * math.cos(a0), cy + R * 0.42 * math.sin(a0))
            path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2,
                       math.degrees(a0), math.degrees(a1 - a0))
            path.lineTo(cx + R * 0.42 * math.cos(a1), cy + R * 0.42 * math.sin(a1))
            p.drawPath(path)
    if _AUTOSCALE_FROZEN:
        breath = 0.5
    else:
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * (time.monotonic() - t0) * 0.4)
    br = int(R * 1.20)
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, int(60 + 30 * breath)), 2))
    p.drawEllipse(cx - br, cy - br, br * 2, br * 2)
    unc = getattr(f, "gprime_uncertainty", None)
    if unc is not None:
        u = np.asarray(unc, dtype=np.float32).reshape(-1)
        if u.size:
            u_mean = float(np.clip(np.mean(u), 0, 1))
            ur = int(R * (1.28 + min(0.4, u_mean)))
            p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, int(50 + 100 * u_mean)),
                                1, QtCore.Qt.DashLine))
            p.drawEllipse(cx - ur, cy - ur, ur * 2, ur * 2)
    elif conf < 0.5:
        # MLP fallback: use 1.0 - confidence as approximate uncertainty
        u_approx = 1.0 - conf
        ur = int(R * (1.28 + min(0.4, u_approx)))
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, int(50 + 100 * u_approx)),
                            1, QtCore.Qt.DashLine))
        p.drawEllipse(cx - ur, cy - ur, ur * 2, ur * 2)
    for i in range(n):
        did = i + 1
        lvl = float(levels[i])
        dfc = float(deficits[i])
        lvl = max(0.0, min(1.0, lvl))
        a0 = -math.pi / 2 + i * 2 * math.pi / n
        a1 = a0 + 2 * math.pi / n - 0.18 * dfc
        r_out = int(R * (0.55 + 0.45 * lvl))
        col = _drive_color(did); col.setAlpha(int(80 + 160 * lvl))
        pen_w = 1
        if did == active:
            pen_w = int(2 + active_pulse)
            col.setAlpha(min(255, int(120 + 135 * lvl + 40 * active_pulse)))
        p.setBrush(col); p.setPen(QtGui.QPen(col.darker(140), pen_w))
        path = QtGui.QPainterPath()
        path.moveTo(cx + R * 0.45 * math.cos(a0), cy + R * 0.45 * math.sin(a0))
        path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2,
                   math.degrees(a0), math.degrees(a1 - a0))
        path.lineTo(cx + R * 0.45 * math.cos(a1), cy + R * 0.45 * math.sin(a1))
        p.drawPath(path)
    cr = int(R * 0.4)
    _draw_agent_limbs(p, f, cx, cy, cr, R)
    cval = max(0.0, min(1.0, conf))
    if cval > 0.66:
        core = QtGui.QColor(46, 204, 113)
    elif cval > 0.33:
        core = QtGui.QColor(241, 196, 15)
    else:
        core = QtGui.QColor(231, 76, 60)
    if 1 <= active <= n:
        dom = _drive_color(active)
        core = QtGui.QColor(
            int(0.6 * core.red() + 0.4 * dom.red()),
            int(0.6 * core.green() + 0.4 * dom.green()),
            int(0.6 * core.blue() + 0.4 * dom.blue()))
    b = int(120 + 110 * max(0.0, min(1.0, emp_value)))
    core.setAlpha(min(255, b))
    p.setBrush(core); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 2))
    p.drawEllipse(cx - cr, cy - cr, cr * 2, cr * 2)
    if stable:
        p.setBrush(QtGui.QColor(241, 196, 15)); p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        p.drawEllipse(cx + cr - 12, cy - cr + 4, 8, 8)
    if 1 <= active <= n:
        i = active - 1
        a = -math.pi / 2 + i * 2 * math.pi / n + math.pi / n
        hx = cx + (R * 1.02) * math.cos(a); hy = cy + (R * 1.02) * math.sin(a)
        p.setBrush(_drive_color(active))
        p.setPen(QtGui.QPen(QtCore.Qt.white, 2))
        p.drawEllipse(int(hx) - 5, int(hy) - 5, 10, 10)
        if show_head_label:
            p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
            lbl = _drive_short(active)
            p.drawText(int(hx) + 8, int(hy) + 4, lbl)
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    order = sorted(range(n), key=lambda i: -deficits[i])[:3]
    dy = cy + br + 12
    for i in order:
        if deficits[i] > 0.02:
            p.setPen(_drive_color(i + 1))
            p.drawText(cx - br, dy, f"Δ {_drive_short(i + 1)}={deficits[i]:.2f}")
            dy += 11
    p.setPen(QtGui.QColor(20, 20, 24)); p.setFont(_F_LABEL_B)
    p.drawText(cx - cr, cy - cr, cr * 2, cr * 2, 0x84, f"conf\n{conf:.2f}")


def _draw_obs_vector_bars(p: QtGui.QPainter, f: ObservabilityFrame,
                          rect: QtCore.QRect) -> bool:
    """Bar chart of obs/sanitized vector when camera is unavailable or glitched."""
    v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
    if v is None:
        return False
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    n = min(int(vec.size), 24)
    if n == 0:
        return False
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    p.setPen(TEXT_COL); p.setFont(_F_LABEL)
    p.drawText(x + 4, y + 14, "obs vector (camera fallback)")
    gy = y + 20
    gh = max(h - 28, 8)
    lo = float(vec[:n].min())
    hi = float(vec[:n].max())
    span = hi - lo
    if span < 1e-9:
        span = max(abs(lo), abs(hi), 1.0)
        lo, hi = -span, span
    mid = gy + gh // 2
    p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x + 4, mid, x + w - 4, mid)
    bw = (w - 8) / n
    for i in range(n):
        val = float(vec[i])
        frac = (val - lo) / span
        bx = int(x + 4 + i * bw)
        bar_h = int(abs(frac - 0.5) * (gh - 8))
        col = _TAU_POS if val >= 0 else _TAU_NEG
        p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
        if val >= 0:
            p.fillRect(bx + 1, mid - bar_h, max(int(bw) - 2, 1), bar_h, col)
        else:
            p.fillRect(bx + 1, mid, max(int(bw) - 2, 1), bar_h, col)
    return True


def _draw_overview_pca_fallback(p: QtGui.QPainter, f: ObservabilityFrame,
                                rect: QtCore.QRect,
                                proj: "BeliefProjection") -> None:
    """PCA projection fallback inside the camera clip rect."""
    px0, py0, px1, py1 = rect.x() + 4, rect.y() + 4, rect.right() - 4, rect.bottom() - 4
    p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
    bounds = proj.bounds()
    hist = proj.history
    if len(hist) >= 2:
        n = len(hist)
        for i in range(1, n):
            a = int(40 + 200 * i / n)
            p0 = _map_pt(hist[i - 1], bounds, px0, py0, px1, py1)
            p1 = _map_pt(hist[i], bounds, px0, py0, px1, py1)
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
            p.drawLine(p0[0], p0[1], p1[0], p1[1])
    cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
    cur = proj.project(cur_v)
    if cur is not None:
        cxp, cyp = _map_pt(cur, bounds, px0, py0, px1, py1)
        p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        p.drawEllipse(cxp - 5, cyp - 5, 10, 10)
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    p.drawText(rect, 0x84, "no camera frame\n(PCA projection fallback)")


_REACHER_L1 = 0.42
_REACHER_L2 = 0.38
_REACHER_TRAIL_MAX = 80


def _draw_reacher_schematic(p: QtGui.QPainter, f: ObservabilityFrame,
                            rect: QtCore.QRect,
                            trail: Optional[Deque[Tuple[float, float]]] = None) -> bool:
    """2D arm schematic from obs cos/sin joints (no GPU). Returns True if drawn."""
    v = f.obs_vector if f.obs_vector is not None else f.sanitized_state
    kin = _reacher_kinematics_from_obs(v)
    if kin is None:
        return False
    ex, ey = kin["ex"], kin["ey"]
    fx, fy = kin["fx"], kin["fy"]
    tx, ty = kin["tx"], kin["ty"]
    dist = kin["dist"]
    pts = [(0.0, 0.0), (ex, ey), (fx, fy), (tx, ty)]
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    # Stable workspace for Reacher-v5 (avoid collapsed bounds).
    xlo = min(min(xs) - 0.12, -0.28)
    xhi = max(max(xs) + 0.12, 0.28)
    ylo = min(min(ys) - 0.12, -0.28)
    yhi = max(max(ys) + 0.12, 0.28)
    margin = 16
    px0 = rect.x() + margin
    py0 = rect.y() + margin + 22
    px1 = rect.right() - margin
    py1 = rect.bottom() - margin
    plot_w = max(px1 - px0, 40)
    plot_h = max(py1 - py0, 40)

    def _map(xv: float, yv: float) -> Tuple[int, int]:
        px = px0 + (xv - xlo) / (xhi - xlo) * plot_w
        py = py1 - (yv - ylo) / (yhi - ylo) * plot_h
        return int(px), int(py)

    # Dark panel — never leave default/green GL clear visible.
    spike = False
    try:
        spike = float(getattr(f, "prediction_error", 0.0) or 0.0) > 25.0
    except Exception:
        spike = False
    p.fillRect(rect, QtGui.QColor(10, 10, 14))
    p.fillRect(px0, py0, plot_w, plot_h, QtGui.QColor(18, 20, 28))
    p.setPen(QtGui.QPen(GRID_COL, 1))
    p.drawRect(px0, py0, plot_w, plot_h)
    if spike:
        p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60, 200), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(px0 + 1, py0 + 1, max(plot_w - 2, 1), max(plot_h - 2, 1))
    # Crosshair at base
    o = _map(0.0, 0.0)
    p.setPen(QtGui.QPen(QtGui.QColor(50, 55, 70), 1, QtCore.Qt.DashLine))
    p.drawLine(px0, o[1], px0 + plot_w, o[1])
    p.drawLine(o[0], py0, o[0], py0 + plot_h)
    e = _map(ex, ey)
    tip = _map(fx, fy)
    goal = _map(tx, ty)
    # Fingertip trail (motion history)
    trail_pts = list(trail or [])
    if len(trail_pts) >= 2:
        n = len(trail_pts)
        for i in range(1, n):
            a = int(20 + 80 * i / n)
            p0 = _map(trail_pts[i - 1][0], trail_pts[i - 1][1])
            p1 = _map(trail_pts[i][0], trail_pts[i][1])
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
            p.drawLine(p0[0], p0[1], p1[0], p1[1])
    # Ghost arm from G′ predicted next state (amber, translucent)
    pred = getattr(f, "predicted_state", None)
    pred_kin = _reacher_kinematics_from_obs(pred)
    if pred_kin is not None:
        pe = _map(pred_kin["ex"], pred_kin["ey"])
        pt = _map(pred_kin["fx"], pred_kin["fy"])
        p.setBrush(QtCore.Qt.NoBrush)
        ghost_col = QtGui.QColor(241, 196, 15, 140)
        if spike:
            ghost_col = QtGui.QColor(231, 76, 60, 170)
        p.setPen(QtGui.QPen(ghost_col, 2, QtCore.Qt.DashLine))
        p.drawLine(o[0], o[1], pe[0], pe[1])
        p.drawLine(pe[0], pe[1], pt[0], pt[1])
        p.setBrush(QtGui.QColor(ghost_col.red(), ghost_col.green(), ghost_col.blue(), 110))
        p.drawEllipse(pt[0] - 4, pt[1] - 4, 8, 8)
    # Links: blue shoulder, amber elbow
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 3))
    p.drawLine(o[0], o[1], e[0], e[1])
    p.setPen(QtGui.QPen(ACCENT, 3))
    p.drawLine(e[0], e[1], tip[0], tip[1])
    # Fingertip → target vector
    p.setPen(QtGui.QPen(QtGui.QColor(210, 210, 220), 1, QtCore.Qt.DashLine))
    p.drawLine(tip[0], tip[1], goal[0], goal[1])
    # Goal: yellow ring; fill when reached
    goal_pen = QtGui.QPen(ACCENT, 2)
    if getattr(f, "violations_count", 0) > 0:
        goal_pen = QtGui.QPen(QtGui.QColor(231, 76, 60), 3)
    p.setBrush(QtGui.QColor(241, 196, 15, 120) if getattr(f, "goal_reached", False)
               else QtCore.Qt.NoBrush)
    p.setPen(goal_pen)
    p.drawEllipse(goal[0] - 8, goal[1] - 8, 16, 16)
    # Fingertip
    p.setBrush(ACCENT)
    p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
    p.drawEllipse(tip[0] - 5, tip[1] - 5, 10, 10)
    # Base joint
    p.setBrush(QtGui.QColor(52, 152, 219))
    p.drawEllipse(o[0] - 4, o[1] - 4, 8, 8)
    # Labels
    p.setPen(TEXT_COL)
    p.setFont(_F_LABEL_B)
    title = f"Reacher 2D  dist={dist:.3f}" + ("  SPIKE" if spike else "")
    p.drawText(rect.x() + 8, rect.y() + 16, title)
    p.setFont(_F_AXIS)
    p.setPen(DIM_COL)
    p.drawText(px0 + 6, py0 + plot_h - 6, "● base   ● fingertip   ○ goal   ghost=G′")
    p.setPen(QtGui.QColor(52, 152, 219))
    p.setFont(_F_AXIS)
    p.drawText(rect.right() - 28, rect.y() + 16, "2D")
    return True


def _render_reacher_schematic_pixmap(
        f: ObservabilityFrame, w: int, h: int,
        trail: Optional[Deque[Tuple[float, float]]] = None) -> QtGui.QPixmap:
    """Offscreen 2D arm schematic (no GPU / no paintEvent camera blit)."""
    obs = f.obs_vector if f.obs_vector is not None else f.sanitized_state
    obs_bytes = np.asarray(obs).tobytes() if obs is not None else b""
    trail_len = len(trail) if trail else 0
    sig = hashlib.md5(obs_bytes + str((w, h, trail_len)).encode()).hexdigest()
    cached = _REACHER_CACHE.get(sig)
    if cached is not None:
        return cached
    pm = QtGui.QPixmap(max(w, 2), max(h, 2))
    pm.fill(QtGui.QColor(12, 12, 16))
    p = QtGui.QPainter(pm)
    _draw_reacher_schematic(p, f, QtCore.QRect(0, 0, w, h), trail=trail)
    p.end()
    _REACHER_CACHE[sig] = pm
    while len(_REACHER_CACHE) > 8:
        _REACHER_CACHE.pop(next(iter(_REACHER_CACHE)))
    return pm


def _apply_overview_camera_overlay(
        pm: QtGui.QPixmap, f: Optional[ObservabilityFrame], *,
        badge: str, camera_cycle_id: Optional[int] = None,
        stale: bool = False, schematic: bool = False) -> QtGui.QPixmap:
    """Draw cycle/err/conf badge on camera pixmap (offscreen, not in paintEvent)."""
    if pm.isNull():
        return pm
    out = pm.copy()
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    w, h = out.width(), out.height()
    # Mode badge (top-right)
    badge_col = QtGui.QColor(52, 152, 219) if schematic else QtGui.QColor(46, 204, 113)
    if stale:
        badge_col = QtGui.QColor(241, 196, 15)
    if badge == "SYNC?":
        badge_col = QtGui.QColor(231, 76, 60)
    p.setFont(_F_LABEL_B)
    fm = p.fontMetrics()
    bw = fm.horizontalAdvance(badge) + 12
    bx, by = w - bw - 6, 4
    p.fillRect(bx, by, bw, 18, QtGui.QColor(0, 0, 0, 160))
    p.setPen(badge_col)
    p.drawText(bx + 6, by + 14, badge)
    # Status line (bottom-left)
    if f is not None:
        kin = _reacher_kinematics_from_obs(
            f.obs_vector if f.obs_vector is not None else f.sanitized_state)
        dist_s = f"{kin['dist']:.3f}" if kin else "—"
        line = (f"cycle={f.cycle_id}  err={f.prediction_error:.2f}  "
                f"conf={f.prediction_confidence:.3f}  dist={dist_s}")
        if camera_cycle_id is not None and int(camera_cycle_id) != int(f.cycle_id):
            line += f"  cam@{camera_cycle_id}"
        p.setFont(_F_AXIS)
        fm2 = p.fontMetrics()
        p.setPen(QtGui.QColor(230, 230, 235))
        p.fillRect(4, h - 22, min(w - 8, fm2.horizontalAdvance(line) + 10), 18,
                   QtGui.QColor(0, 0, 0, 140))
        p.drawText(8, h - 8, line)
    p.end()
    return out


def _draw_overview_tau_bar(p: QtGui.QPainter, f: ObservabilityFrame, rect: QtCore.QRect) -> None:
    """Slim continuous-action bar below the camera QLabel."""
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    tau_h = min(36, max(28, h // 6))
    tau_rect = QtCore.QRect(x + 4, y + h - tau_h - 2, w - 8, tau_h)
    p.fillRect(rect, QtGui.QColor(12, 12, 16))
    act = f.continuous_action if f.continuous_action is not None else f.last_action_vector
    dim_names = list(getattr(f, "dim_names", []) or [])
    p.save()
    try:
        p.setClipRect(tau_rect)
        p.fillRect(tau_rect, QtGui.QColor(12, 12, 16))
        _draw_tau_bar(p, act, dim_names, tau_rect.x(), tau_rect.y(),
                      tau_rect.width(), tau_rect.height())
        p.setPen(DIM_COL)
        p.setFont(_F_AXIS)
        p.drawText(x + 8, tau_rect.y() + 12, "Flow → timings  |  Candidates → scores")
    finally:
        p.restore()


def _draw_phase_grid_base(
    p: QtGui.QPainter,
    f: ObservabilityFrame,
    rect: QtCore.QRect,
    trail: Deque[Any],
    *,
    show_confidence_heat: bool = False,
) -> Tuple[int, float, float, float]:
    """Grid base layer: walls, optional confidence heat, ghost, goal, trail, agent.

    Returns (n, ox, oy, cell) for overlay layers.
    """
    g = f.grid
    if g is None:
        p.setPen(DIM_COL)
        p.drawText(rect, 0x84, "GridWorld…")
        return 0, 0.0, 0.0, 0.0
    n = g.shape[0]
    cell = min(rect.width() - 8, rect.height() - 8) / n
    ox = rect.x() + (rect.width() - cell * n) / 2
    oy = rect.y() + (rect.height() - cell * n) / 2
    heat = _prediction_heatmap(f.predicted_state, n)
    pred_cell = None
    if heat is not None:
        pred_cell = np.unravel_index(int(np.argmax(heat)), heat.shape)
    for r in range(n):
        for c in range(n):
            v = g[r, c]
            col = QtGui.QColor(40, 40, 50)
            if v == 1:
                col = QtGui.QColor(90, 90, 100)
            elif v == 2:
                col = QtGui.QColor(39, 174, 96)
            p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell), col)
    if show_confidence_heat and heat is not None:
        for r in range(n):
            for c in range(n):
                a = float(np.clip(heat[r, c], 0, 1))
                if a <= 0.03:
                    continue
                p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell),
                           QtGui.QColor(231, 76, 60, int(150 * a)))
    if pred_cell is not None:
        gr, gc = int(pred_cell[0]), int(pred_cell[1])
        p.setBrush(QtGui.QColor(241, 196, 15, 40))
        p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 2))
        p.drawEllipse(int(ox + gc * cell + cell * 0.18),
                      int(oy + gr * cell + cell * 0.18),
                      int(cell * 0.64), int(cell * 0.64))
    if f.goal_pos is not None:
        gr, gc = f.goal_pos
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        p.drawRect(int(ox + gc * cell), int(oy + gr * cell), int(cell), int(cell))
    tl = list(trail)
    for i in range(1, len(tl)):
        (r0, c0), (r1, c1) = tl[i - 1], tl[i]
        a = int(60 + 195 * i / max(len(tl), 1))
        p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
        p.drawLine(int(ox + c0 * cell + cell / 2), int(oy + r0 * cell + cell / 2),
                   int(ox + c1 * cell + cell / 2), int(oy + r1 * cell + cell / 2))
    if tl:
        r, c = tl[-1]
        p.setBrush(QtGui.QColor(241, 196, 15))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        p.drawEllipse(int(ox + c * cell + cell * 0.25),
                      int(oy + r * cell + cell * 0.25),
                      int(cell * 0.5), int(cell * 0.5))
    return n, ox, oy, cell


def _draw_overview_grid(p: QtGui.QPainter, f: ObservabilityFrame, rect: QtCore.QRect,
                        trail: Deque[Any]) -> None:
    _draw_phase_grid_base(p, f, rect, trail, show_confidence_heat=True)


def _draw_overview_body(p: QtGui.QPainter, f: ObservabilityFrame, rect: QtCore.QRect,
                        proj: Optional["BeliefProjection"],
                        trail: Deque[Any], arena_trail: Deque[Tuple[float, float]],
                        ax: ScaleState, ay: ScaleState) -> None:
    """Body panel: env-adaptive world (camera / grid / arena / projection)."""
    kind = (getattr(f, "env_kind", "") or "").lower()
    sd = int(getattr(f, "state_dim", 0) or 0)
    if f.grid is not None:
        _draw_overview_grid(p, f, rect, trail)
    elif kind == "mujoco_rgb":
        p.fillRect(rect, QtGui.QColor(12, 12, 16))
        _draw_overview_tau_bar(p, f, rect)
    elif kind == "continuous" and 2 <= sd <= 4:
        margin = 8
        left, right = rect.x() + margin, rect.right() - margin
        top, bot = rect.y() + margin, rect.bottom() - margin
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(left, top, right - left, bot - top)
        trail_pts = list(arena_trail)
        if not trail_pts:
            p.setPen(DIM_COL); p.drawText(rect, 0x84, "2D arena (collecting…)"); return
        xs = [pt[0] for pt in trail_pts]; ys = [pt[1] for pt in trail_pts]
        xlo, xhi = ax.update(float(min(xs)), float(max(xs)))
        ylo, yhi = ay.update(float(min(ys)), float(max(ys)))
        if xhi - xlo < 1e-9: xhi = xlo + 1
        if yhi - ylo < 1e-9: yhi = ylo + 1

        def _mpt(xv, yv):
            px = left + (xv - xlo) / (xhi - xlo) * (right - left)
            py = bot - (yv - ylo) / (yhi - ylo) * (bot - top)
            return int(px), int(py)
        n = len(trail_pts)
        for i in range(1, n):
            a = int(40 + 180 * i / n)
            x0, y0 = _mpt(*trail_pts[i - 1]); x1, y1 = _mpt(*trail_pts[i])
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
            p.drawLine(x0, y0, x1, y1)
        gref = getattr(f, "goal_ref", None)
        if gref is not None:
            gv = np.asarray(gref, dtype=np.float32).reshape(-1)
            if gv.size >= 2:
                gx, gy = _mpt(float(gv[0]), float(gv[1]))
                p.setBrush(QtGui.QColor(46, 204, 113, 180))
                p.setPen(QtGui.QPen(QtGui.QColor(46, 204, 113), 1))
                p.drawEllipse(gx - 6, gy - 6, 12, 12)
        axp, ayp = _mpt(*trail_pts[-1])
        p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        p.drawEllipse(axp - 5, ayp - 5, 10, 10)
    elif proj is not None:
        px0, py0, px1, py1 = rect.x() + 8, rect.y() + 8, rect.right() - 8, rect.bottom() - 8
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
        bounds = proj.bounds()
        hist = proj.history
        if len(hist) >= 2:
            n = len(hist)
            for i in range(1, n):
                a = int(40 + 200 * i / n)
                p0 = _map_pt(hist[i - 1], bounds, px0, py0, px1, py1)
                p1 = _map_pt(hist[i], bounds, px0, py0, px1, py1)
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
                p.drawLine(p0[0], p0[1], p1[0], p1[1])
        cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
        cur = proj.project(cur_v)
        cx = cy = None
        if cur is not None:
            cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
            p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(cx - 5, cy - 5, 10, 10)
        pred = proj.project(f.predicted_state)
        if pred is not None and cx is not None:
            px, py = _map_pt(pred, bounds, px0, py0, px1, py1)
            _arrow(p, cx, cy, px, py, QtGui.QColor(231, 76, 60, 180), size=7)
    else:
        p.setPen(DIM_COL); p.drawText(rect, 0x84, "World (collecting…)")


def _window_session_incomplete(widget: QtWidgets.QWidget) -> bool:
    """True when the owning ObservatoryWindow marks the session incomplete/aborted."""
    win = widget.window() if widget is not None else None
    return bool(getattr(win, "_session_incomplete", False))


def _draw_data_contract_banner(
    p: QtGui.QPainter,
    w: int,
    *,
    replay: bool,
    panel_key: str,
    review: bool = False,
    multi_agent: bool = False,
    incomplete: bool = False,
    y: int = 2,
) -> int:
    """Draw LIVE/REVIEW/REPLAY/ABORTED data-contract strip; returns y offset."""
    text = data_contract_text(
        panel_key,
        replay=replay,
        review=review,
        multi_agent=multi_agent,
        incomplete=incomplete,
    )
    if not text:
        return 0
    if review and incomplete:
        p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 1))
        p.setBrush(QtGui.QColor(231, 76, 60, 50))
    elif review:
        p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 1))
        p.setBrush(QtGui.QColor(155, 89, 182, 40))
    elif replay:
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 1))
        p.setBrush(QtGui.QColor(52, 152, 219, 40))
    else:
        p.setPen(QtGui.QPen(QtGui.QColor(90, 90, 100), 1))
        p.setBrush(QtGui.QColor(40, 40, 48, 120))
    p.drawRoundedRect(8, y, w - 16, 14, 3, 3)
    p.setPen(TEXT_COL if (review or replay or incomplete) else DIM_COL)
    p.setFont(_F_AXIS)
    p.drawText(12, y + 11, text)
    return 16


def _overview_grid_chip(
    p: QtGui.QPainter,
    f: ObservabilityFrame,
    cx: int,
    cy: int,
) -> int:
    """GRID N/A for non-grid envs; kinematics chip for Reacher."""
    kind = (getattr(f, "env_kind", "") or "").lower()
    if kind == "grid" and f.agent_pos is not None:
        ap = tuple(f.agent_pos)
        return cx + _overview_chip(p, cx, cy, f"GRID {ap[0]},{ap[1]}",
                                   QtGui.QColor(149, 165, 166))
    if kind != "grid":
        kin = _reacher_kinematics_from_obs(
            f.obs_vector if f.obs_vector is not None else f.sanitized_state)
        if kin is not None:
            return cx + _overview_chip(
                p, cx, cy, f"tip {kin['dist']:.2f}",
                QtGui.QColor(52, 152, 219))
        return cx + _overview_chip(p, cx, cy, "GRID N/A", QtGui.QColor(90, 90, 100))
    return cx


def _overview_chip(p: QtGui.QPainter, x: int, y: int, text: str,
                   col: QtGui.QColor) -> int:
    p.setFont(QtGui.QFont("Monospace", 8, QtGui.QFont.Bold))
    fm = p.fontMetrics()
    w = fm.horizontalAdvance(text) + 10; h = 14
    p.setPen(QtGui.QPen(col, 1)); p.setBrush(col.darker(160))
    p.drawRoundedRect(x, y, w, h, 4, 4)
    p.setPen(col); p.drawText(x + 5, y + 10, text)
    return w + 6


def _draw_overview_header(p: QtGui.QPainter, f: ObservabilityFrame, rect: QtCore.QRect,
                          drive_change: Optional[Tuple[int, int]] = None,
                          flags: Optional[Dict[str, Any]] = None,
                          session_complete: bool = False) -> None:
    r = f.action_rationale or {}
    gid = _overview_goal_id(f)
    goal_lbl = DRIVE_NAMES.get(gid, _drive_short(gid)) if gid is not None else "—"
    tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
    p.setPen(TEXT_COL); p.setFont(_F_TITLE)
    p.drawText(rect.x() + 8, rect.y() + 20, f"AGENT · cycle {f.cycle_id} · {goal_lbl} · {tag}")
    cx = rect.x() + 8
    cy = rect.y() + 4
    if drive_change is not None:
        old_d, new_d = drive_change
        cx += _overview_chip(
            p, cx, cy, f"{_drive_short(old_d)}→{_drive_short(new_d)}",
            QtGui.QColor(52, 152, 219))
    rbta_ok = f.violations_count == 0
    cx += _overview_chip(p, cx, cy,
                         "RBTA " + ("OK" if rbta_ok else f"{f.violations_count}V"),
                         QtGui.QColor(46, 204, 113) if rbta_ok else QtGui.QColor(231, 76, 60))
    cx = _overview_grid_chip(p, f, cx, cy)
    if getattr(f, "goal_reached", False):
        cx += _overview_chip(p, cx, cy, "GOAL", QtGui.QColor(241, 196, 15))
    phi = float(getattr(f, "phi_criticality", 0.0) or 0.0)
    if phi > 0.01:
        phi_col = (QtGui.QColor(231, 76, 60) if phi > 0.5
                   else QtGui.QColor(241, 196, 15) if phi > 0.2
                   else QtGui.QColor(46, 204, 113))
        cx += _overview_chip(p, cx, cy, f"Φ={phi:.2f}", phi_col)
    recovery = bool(getattr(f, "recovery_active", False))
    if recovery:
        cx += _overview_chip(p, cx, cy, "RECOVERY", QtGui.QColor(155, 89, 182))
    ms = getattr(f, "meta_stable", None) or {}
    if ms.get("is_meta_stable", ms.get("stable", False)):
        _overview_chip(p, cx, cy, "META", QtGui.QColor(52, 152, 219))
    flags = flags or _overview_moment_flags(f, deque())
    if flags["spike"]:
        cx += _overview_chip(p, cx, cy, "SPIKE", QtGui.QColor(231, 76, 60))
    if flags.get("learn_burst", False):
        cx += _overview_chip(p, cx, cy, "G′lrn", QtGui.QColor(52, 152, 219))
    if flags.get("decision_shift", False):
        cx += _overview_chip(p, cx, cy, "DECISION", QtGui.QColor(155, 89, 182))
    if flags.get("near_bound"):
        nb = flags["near_bound"]
        nb_lbl = PIPELINE_LABEL.get(nb, str(nb))[:6]
        cx += _overview_chip(p, cx, cy, nb_lbl, _to_qcolor(MOMENT_COLORS["near_bound"]))
    if flags.get("violation"):
        cx += _overview_chip(p, cx, cy, "VIOL", _to_qcolor(MOMENT_COLORS["spike"]))
    if session_complete:
        _overview_chip(p, cx, cy, "SESSION COMPLETE", QtGui.QColor(46, 204, 113))


def _draw_session_results_panel(
    p: QtGui.QPainter,
    lines: List[str],
    rect: QtCore.QRect,
) -> None:
    """Post-run performance summary (replaces narrative + event log in review)."""
    p.setPen(QtGui.QPen(QtGui.QColor(46, 204, 113, 120), 1))
    p.setBrush(QtGui.QColor(46, 204, 113, 20))
    p.drawRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), 4, 4)
    p.setPen(TEXT_COL)
    p.setFont(_F_LABEL_B)
    p.drawText(rect.x() + 8, rect.y() + 12, "Session results")
    p.setFont(_F_CAPTION)
    p.setPen(_CAPTION_COL)
    note = _elide_line(p, "scrub prefix decimated to 2000 pts", rect.width() - 16)
    p.drawText(rect.x() + 8, rect.y() + 22, note)
    y = rect.y() + 34
    for line in lines[:8]:
        p.setPen(DIM_COL if line.startswith("  ") else TEXT_COL)
        p.drawText(rect.x() + 8, y, line)
        y += 11


def _execution_phase_segments(f: ObservabilityFrame) -> List[Tuple[str, float]]:
    timings = dict(getattr(f, "module_timings", {}) or {})
    return [(label, _phase_ms(timings, key)) for key, label in EXECUTION_PHASE_STEPS]


def _overview_phase_segments(f: ObservabilityFrame) -> List[Tuple[str, float]]:
    return _execution_phase_segments(f)


def _execution_dominant_phase(f: ObservabilityFrame) -> str:
    segs = _execution_phase_segments(f)
    if not segs:
        return ""
    return max(segs, key=lambda s: s[1])[0]


def _overview_dominant_phase(f: ObservabilityFrame) -> str:
    return _execution_dominant_phase(f)


def _draw_execution_phase_strip(p: QtGui.QPainter, f: ObservabilityFrame,
                                rect: QtCore.QRect, *, prefix: str = "exec: ",
                                learn_burst: bool = False) -> None:
    segs = _execution_phase_segments(f)
    if not segs:
        return
    total = sum(ms for _, ms in segs) or 1.0
    dominant = _execution_dominant_phase(f)
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    bar_y = y + 2
    bar_h = max(h - 14, 6)
    cx = x + 4
    learn_x0 = learn_x1 = 0
    for label, ms in segs:
        share = max(ms / total, 0.02)
        seg_w = max(int((w - 8) * share), 4)
        col = QtGui.QColor(52, 73, 94)
        if label == dominant and ms > 0.0:
            col = QtGui.QColor(52, 152, 219)
        p.setPen(QtGui.QPen(col.darker(130), 1))
        p.setBrush(col)
        p.drawRect(cx, bar_y, seg_w, bar_h)
        if learn_burst and label == "Learn" and ms > 0:
            learn_x0, learn_x1 = cx, cx + seg_w
        cx += seg_w + 1
    if learn_x1 > learn_x0:
        p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["learn_burst"]), 2))
        p.drawRect(learn_x0 - 1, bar_y - 1, learn_x1 - learn_x0 + 2, bar_h + 2)
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    trail = " → ".join(s[0] for s in segs)
    p.drawText(x + 4, y + h - 2, f"{prefix}{trail}" if prefix else trail)


def _draw_overview_phase_strip(p: QtGui.QPainter, f: ObservabilityFrame,
                               rect: QtCore.QRect, *,
                               learn_burst: bool = False) -> None:
    _draw_execution_phase_strip(p, f, rect, prefix="", learn_burst=learn_burst)


def _overview_plain_story(f: ObservabilityFrame, flags: Dict[str, Any],
                          err_hist: Deque[float],
                          dist_hist: Optional[Deque[float]] = None) -> str:
    r = f.action_rationale or {}
    mode = "exploring candidates" if r.get("explored") else "exploiting best action"
    errs = list(err_hist)
    if flags.get("spike"):
        err_part = "error spiked (surprise)"
    elif len(errs) >= 2:
        err_part = "error rose" if errs[-1] > errs[-2] else (
            "error fell" if errs[-1] < errs[-2] else "error stable")
    else:
        err_part = "error stable"
    parts = [f"Cycle {f.cycle_id}", mode, err_part]
    if flags.get("learn_burst"):
        parts.append("world model updated")
    if dist_hist is not None and len(dist_hist) >= 2:
        if dist_hist[-1] < dist_hist[-2]:
            parts.append("moving toward target")
        elif dist_hist[-1] > dist_hist[-2]:
            parts.append("drifting from target")
    if flags.get("decision_shift"):
        parts.append("decision shifted")
    if getattr(f, "goal_reached", False):
        parts.append("goal reached")
    return " · ".join(parts)


def _overview_metrics_line(f: ObservabilityFrame, flags: Dict[str, Any],
                           err_hist: Deque[float],
                           dist_hist: Optional[Deque[float]] = None) -> str:
    """Compact numeric metrics (secondary line under plain story)."""
    r = f.action_rationale or {}
    tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
    errs = list(err_hist)
    err_arrow = ""
    if len(errs) >= 2:
        err_arrow = "↘" if errs[-1] <= errs[-2] else "↗"
    kin = _reacher_kinematics_from_obs(
        f.obs_vector if f.obs_vector is not None else f.sanitized_state)
    dist_s = f"{kin['dist']:.3f}" if kin else "—"
    dist_arrow = ""
    if dist_hist is not None and len(dist_hist) >= 2:
        dist_arrow = "↘" if dist_hist[-1] <= dist_hist[-2] else "↗"
    score_s = f"{flags['score']:.2f}" if flags["score"] is not None else "—"
    learn_s = f"{flags['learn_ms']:.1f}ms" if flags["learn_ms"] > 0.0 else "—"
    peu_s = f" · PEŪ={flags['peu_mean']:.2f}" if flags["peu_mean"] is not None else ""
    gid = _overview_goal_id(f)
    drive_s = _drive_short(gid) if gid else "—"
    return (f"{tag} · err={f.prediction_error:.2f}{err_arrow}"
            f" · dist={dist_s}{dist_arrow} · score={score_s}"
            f" · G′lrn={learn_s}{peu_s} · MDIM:{drive_s}")


def _overview_new_events(f: ObservabilityFrame, flags: Dict[str, Any],
                         drive_change: Optional[Tuple[int, int]],
                         *, explore_entered: bool = False) -> List[str]:
    events: List[str] = []
    if flags.get("spike"):
        events.append("SPIKE: prediction error jumped")
    if flags.get("learn_burst"):
        events.append(f"LEARN: G′ updated ({flags['learn_ms']:.1f}ms)")
    if getattr(f, "goal_reached", False):
        events.append("GOAL: target reached")
    if drive_change is not None:
        old_d, new_d = drive_change
        events.append(f"DRIVE: {_drive_short(old_d)}→{_drive_short(new_d)}")
    if flags.get("decision_shift"):
        events.append("DECISION: best action changed")
    if explore_entered:
        events.append("EXPLORE: sampling candidates")
    if flags.get("near_bound"):
        nb = flags["near_bound"]
        lbl = PIPELINE_LABEL.get(nb, str(nb))
        events.append(f"NEAR-BOUND: {lbl}")
    if flags.get("violation"):
        events.append("VIOLATION: RBTA bound exceeded")
    return events


def _draw_overview_event_log(p: QtGui.QPainter, events: List[str],
                             rect: QtCore.QRect) -> None:
    if not events:
        return
    p.setPen(DIM_COL); p.setFont(_F_AXIS)
    y = rect.y() + 10
    for line in events[-5:]:
        p.drawText(rect.x() + 8, y, line)
        y += 12


def _draw_overview_narrative(p: QtGui.QPainter, f: ObservabilityFrame,
                             rect: QtCore.QRect,
                             err_hist: Deque[float],
                             dist_hist: Optional[Deque[float]] = None) -> None:
    """Goal → evidence → outcome narrative (three lines)."""
    flags = _overview_moment_flags(f, err_hist)
    intent = _overview_goal_intent_line(f, flags)
    evidence = _overview_evidence_line(f, flags, err_hist, dist_hist)
    outcome = _overview_outcome_line(f, flags, err_hist, dist_hist)
    p.setPen(TEXT_COL); p.setFont(_F_AXIS)
    p.drawText(rect.x() + 8, rect.y() + 11, intent)
    p.setPen(DIM_COL)
    p.drawText(rect.x() + 8, rect.y() + 24, evidence)
    p.drawText(rect.x() + 8, rect.y() + 37, outcome)


# Semantic map for Overview observability: used as a single audit reference.
_OVERVIEW_SEMANTIC_MAP = {
    "header.goal": "action_rationale.goal_id -> fallback active_drive_id",
    "header.mode": "action_rationale.explored",
    "header.safety": "violations_count / goal_reached / meta_stable",
    "narrative.error": "prediction_error + err trend from _err_hist",
    "narrative.learning": "module_timings.gprime_learn (burst >= 5ms)",
    "narrative.decision": "action_rationale.best_score/decision_shift (0.20 threshold)",
    "narrative.plain": "_overview_plain_story (legacy; outcome uses physical deltas)",
    "narrative.intent": "_overview_goal_intent_line (+ eps, k_candidates)",
    "narrative.evidence": "_overview_evidence_line (+ peu_mean)",
    "narrative.outcome": "_overview_outcome_line (dist, action, goal, rbta)",
    "narrative.phase": "_draw_overview_phase_strip",
    "narrative.events": "_overview_new_events + hold",
    "narrative.distance": "reacher dist from obs_vector[-2:]",
    "glyph.core": "prediction_confidence blended with active drive color",
    "glyph.limbs": "continuous_action (UI-smoothed for mujoco_rgb 2-DoF)",
    "world.schematic": "obs_vector -> reacher kinematics + predicted_state ghost",
}


def _draw_mini_drive_ring(p: QtGui.QPainter, f: ObservabilityFrame,
                          cx: int, cy: int, R: int) -> None:
    import math
    levels = list(getattr(f, "drive_levels", []) or [])
    n = _n_drives(f, levels)
    active = int(_overview_goal_id(f) or 0)
    for i in range(n):
        did = i + 1
        lvl = float(levels[i]) if i < len(levels) else 0.0
        lvl = max(0.0, min(1.0, lvl))
        a0 = -math.pi / 2 + i * 2 * math.pi / n
        a1 = a0 + 2 * math.pi / n - 0.12
        r_out = int(R * (0.55 + 0.4 * lvl))
        col = _drive_color(did)
        col_a = QtGui.QColor(col); col_a.setAlpha(int(100 + 120 * lvl))
        if did == active:
            p.setPen(QtGui.QPen(ACCENT, 2))
        else:
            p.setPen(QtGui.QPen(col.darker(140), 1))
        p.setBrush(col_a)
        path = QtGui.QPainterPath()
        path.moveTo(cx + R * 0.35 * math.cos(a0), cy + R * 0.35 * math.sin(a0))
        path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2,
                   math.degrees(a0), math.degrees(a1 - a0))
        path.lineTo(cx + R * 0.35 * math.cos(a1), cy + R * 0.35 * math.sin(a1))
        p.drawPath(path)


def _draw_vitals_ribbon(p: QtGui.QPainter, f: ObservabilityFrame, rect: QtCore.QRect,
                        err_hist: Deque[float], conf_hist: Deque[float]) -> None:
    """Single horizontal vitals strip (prediction, confidence, drives, attention, system)."""
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    p.setPen(QtGui.QPen(GRID_COL, 1))
    p.drawLine(x, y, x + w, y)
    cells = 5
    cw = w // cells
    # Prediction
    cx0 = x
    p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
    p.drawText(cx0 + 6, y + 14, "prediction")
    errs = list(err_hist)
    if errs:
        _draw_sparkline(p, errs, cx0 + 6, y + 18, cw - 12, h - 28,
                        QtGui.QColor(231, 76, 60), log=True)
        q = max(1, len(errs) // 4)
        arrow = "↘" if float(np.mean(errs[-q:])) <= float(np.mean(errs[:q])) else "↗"
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(cx0 + 6, y + h - 6, f"err={errs[-1]:.2f} {arrow}")
  # Confidence
    cx1 = x + cw
    p.setPen(GRID_COL); p.drawLine(cx1, y + 4, cx1, y + h - 4)
    p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
    p.drawText(cx1 + 6, y + 14, "confidence")
    confs = list(conf_hist)
    if confs:
        _draw_sparkline(p, confs, cx1 + 6, y + 18, cw - 12, h - 28,
                        QtGui.QColor(46, 204, 113))
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(cx1 + 6, y + h - 6, f"conf={confs[-1]:.3f}")
    # Drives mini ring
    cx2 = x + 2 * cw
    p.setPen(GRID_COL); p.drawLine(cx2, y + 4, cx2, y + h - 4)
    p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
    p.drawText(cx2 + 6, y + 14, "drives")
    _draw_mini_drive_ring(p, f, cx2 + cw // 2, y + h // 2 + 4, min(cw, h) // 2 - 8)
    # Attention
    cx3 = x + 3 * cw
    p.setPen(GRID_COL); p.drawLine(cx3, y + 4, cx3, y + h - 4)
    p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
    p.drawText(cx3 + 6, y + 14, "attention")
    p.setFont(_F_AXIS); p.setPen(DIM_COL)
    if f.attention_indices:
        parts = []
        for i in range(min(4, len(f.attention_indices))):
            sal = float(f.attention_saliences[i]) if i < len(f.attention_saliences) else 0.0
            parts.append(f"#{int(f.attention_indices[i])} {sal:.2g}")
        p.setPen(TEXT_COL)
        p.drawText(cx3 + 6, y + 30, "  ".join(parts))
    else:
        p.drawText(cx3 + 6, y + 30, "no salient chunks")
    # System
    cx4 = x + 4 * cw
    p.setPen(GRID_COL); p.drawLine(cx4, y + 4, cx4, y + h - 4)
    p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
    p.drawText(cx4 + 6, y + 14, "system")
    p.setFont(_F_AXIS)
    learn_ms = float((getattr(f, "module_timings", {}) or {}).get("gprime_learn", 0.0) or 0.0)
    p.drawText(cx4 + 6, y + 28,
               f"RSS {f.rss_bytes / 1e6:.0f}MB  lat {f.latency_ms:.1f}ms  lrn={learn_ms:.1f}ms")
    gid = _overview_goal_id(f)
    goal_lbl = DRIVE_NAMES.get(gid, gid) if gid is not None else "—"
    r = f.action_rationale or {}
    tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
    score = r.get("best_score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
    note = r.get("note", "")
    line = f"{goal_lbl} · {tag} · score={score_s}"
    if note:
        line += " · " + note
    fm = p.fontMetrics()
    p.drawText(cx4 + 6, y + h - 8, fm.elidedText(line, QtCore.Qt.ElideRight, cw - 12))


# ----- base canvas + shared chart infra --------------------------------------
class _BaseCanvas(QtWidgets.QWidget):
    """Base for QPainter canvases.

    v8 calm-render: set_frame only stores state + marks ``_dirty``; it does NOT
    call ``update()``. A single ``RenderPacer`` QTimer (owned by the window)
    calls :meth:`repaint_if_dirty` on the canvases of the *visible* tab at a
    calm ~6 Hz cadence. Paused/no-new-data → not dirty → 0 repaints. Subclasses
    keep their existing ``_draw``; heavy per-frame compute (sorts/argsort/polyfit)
    must live in ``_draw`` (runs only when visible) not in ``set_frame``."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 220)
        # v8: opaque paint + no system background → skip per-frame bg erase.
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, PANEL_BG)
        self.setPalette(pal)
        self.frame: Optional[ObservabilityFrame] = None
        # v6: repaint counter for the flicker-free acceptance gate.
        self.repaint_count: int = 0
        # v8: dirty-gating + backing-cache infra.
        self._dirty: bool = True            # paint at least once on show
        self._bg_cache: Optional[QtGui.QPixmap] = None
        self._bg_cache_key: tuple = ()
        self._multi_agent: bool = False

    def set_frame(self, f: ObservabilityFrame) -> None:
        """Feed state; do NOT repaint. Mark dirty so the RenderPacer picks it up."""
        self.frame = f
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def repaint_if_dirty(self) -> None:
        """Called by the RenderPacer on the visible tab. No-op if nothing changed."""
        if self._dirty:
            self._dirty = False
            self.update()

    def _invalidate_cache(self) -> None:
        self._bg_cache = None
        self._dirty = True

    def resizeEvent(self, _ev: QtCore.QEvent) -> None:
        self._invalidate_cache()
        super().resizeEvent(_ev)

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:
        self.repaint_count += 1
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), PANEL_BG)
        # v8: blit a cached static layer if the subclass populated one.
        if self._bg_cache is not None and not self._bg_cache.isNull():
            p.drawPixmap(0, 0, self._bg_cache)
        try:
            self._draw(p)
        except Exception as e:
            # A TypeError in a C++ paintEvent virtual makes PyQt5 call qFatal
            # (SIGABRT) — surface the error as text instead of crashing the app.
            import traceback
            p.setPen(QtGui.QColor(231, 76, 60))
            p.setFont(QtGui.QFont("Monospace", 8))
            msg = f"{type(e).__name__}: {e}\n" + "\n".join(traceback.format_exc().splitlines()[-3:])
            p.drawText(self.rect(), 0x84, f"{self.__class__.__name__} paint error:\n{msg}")
        finally:
            p.end()

    def _empty(self, p: QtGui.QPainter, text: str) -> None:
        p.setPen(DIM_COL); p.drawText(self.rect(), 0x84, text)

    def _title(self, p: QtGui.QPainter, text: str, y: int = 15, x: int = 10) -> None:
        p.setPen(TEXT_COL); p.setFont(_F_TITLE)
        p.drawText(x, y, text)

    def _caption(self, p: QtGui.QPainter, text: str, y: int = 28, x: int = 10) -> None:
        """One-line dim 'what this tells you' under a title."""
        p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
        p.drawText(x, y, _elide_line(p, text, self.width() - x - 8))

    def _footer_caption(self, p: QtGui.QPainter, text: str, *,
                        y: Optional[int] = None, x: int = 10) -> None:
        if y is None:
            y = self.height() - 10
        p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
        p.drawText(x, y, _elide_line(p, text, self.width() - x - 8))

    def _legend(self, p: QtGui.QPainter, items: List[Tuple[str, QtGui.QColor]],
                x: int = 10, y: Optional[int] = None) -> None:
        """Generic swatch+label legend (x-right progression)."""
        if y is None:
            y = self.height() - 14
        p.setFont(_F_LABEL)
        cx = x
        for lbl, col in items:
            c = col if isinstance(col, QtGui.QColor) else _to_qcolor(col)
            p.setPen(QtGui.QPen(c, 2)); p.drawLine(cx, y - 4, cx + 14, y - 4)
            p.setPen(TEXT_COL); p.drawText(cx + 18, y, lbl)
            cx += 18 + p.fontMetrics().boundingRect(lbl).width() + 10


class _ChartCanvas(_BaseCanvas):
    """Line-chart canvas: margins, cycle x-axis, gridlines, per-series
    adaptive y-normalization (so series with different units coexist), legend
    with current value + range. Replaces the ad-hoc drawLine loops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = ""
        self.xlabel = "cycle"
        self._series: List[Dict[str, Any]] = []

    def add_series(self, name: str, color: str, data: Any,
                   log: bool = False, fill: bool = False) -> None:
        # Each series carries its own ScaleState so the y-band is stabilised
        # across frames (no per-frame min/max jumps -> no line/bar twitching).
        self._series.append(dict(name=name, color=color, data=data, log=log,
                                 fill=fill, scale=ScaleState()))

    def _norm(self, s: Dict[str, Any]) -> Tuple[List[float], float, float]:
        """Stabilised normalisation for one series (uses its ScaleState)."""
        vals = [float(v) for v in s["data"] if v is not None]
        if not vals:
            return [], 0.0, 1.0
        log = s["log"]
        scale: ScaleState = s["scale"]
        if log:
            vals = [v if v > 0 else 1e-6 for v in vals]
            rlo, rhi = float(min(vals)), float(max(vals))
            if rhi <= rlo:
                rhi = rlo * 2 if rlo > 0 else 1.0
            lo, hi = scale.update(float(np.log(rlo)), float(np.log(rhi)))
            return [float(np.log(v)) for v in vals], lo, hi
        rlo, rhi = float(min(vals)), float(max(vals))
        lo, hi = scale.update(rlo, rhi)
        return vals, lo, hi

    def _plot_rect(self) -> Tuple[int, int, int, int]:
        ml, mr, mt, mb = 52, 16, 26, 30
        return ml, mt, self.width() - mr, self.height() - mb

    def _draw(self, p: QtGui.QPainter) -> None:
        px0, py0, px1, py1 = self._plot_rect()
        self._title(p, self.title)
        # axes
        p.setPen(QtGui.QPen(GRID_COL, 1))
        p.drawLine(px0, py0, px0, py1)
        p.drawLine(px0, py1, px1, py1)
        # gridlines (4 horizontal)
        for g in range(1, 4):
            y = py0 + g * (py1 - py0) // 4
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1))
            p.drawLine(px0, y, px1, y)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(4, py1 + 6, self.xlabel)
        any_data = False
        for s in self._series:
            vals = list(s["data"])
            if not vals:
                continue
            any_data = True
            nv, lo, hi = self._norm(s)
            n = len(nv)
            col = _to_qcolor(s["color"])
            if s["fill"]:
                poly = QtGui.QPolygonF()
                poly.append(QtCore.QPointF(px0, py1))
                for i, v in enumerate(nv):
                    x = px0 + i * (px1 - px0) / max(n - 1, 1)
                    y = py1 - (v - lo) / (hi - lo) * (py1 - py0)
                    poly.append(QtCore.QPointF(x, y))
                poly.append(QtCore.QPointF(px0 + (n - 1) * (px1 - px0) / max(n - 1, 1), py1))
                p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 60))
                p.setPen(QtGui.QPen(QtGui.QColor(col.red(), col.green(), col.blue(), 60), 1))
                p.drawPolygon(poly)
            p.setPen(QtGui.QPen(col, 2))
            for i in range(1, n):
                x0 = px0 + (i - 1) * (px1 - px0) / max(n - 1, 1)
                x1 = px0 + i * (px1 - px0) / max(n - 1, 1)
                y0 = py1 - (nv[i - 1] - lo) / (hi - lo) * (py1 - py0)
                y1 = py1 - (nv[i] - lo) / (hi - lo) * (py1 - py0)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        if not any_data:
            self._empty(p, "collecting…")
        self._legend(p, py0)

    def _legend(self, p: QtGui.QPainter, py0: int) -> None:
        x = self.width() - 14
        y = py0 + 12
        p.setFont(_F_LABEL)
        for s in self._series:
            vals = list(s["data"])
            col = _to_qcolor(s["color"])
            p.setPen(QtGui.QPen(col, 2)); p.drawLine(x - 14, y - 4, x - 4, y - 4)
            if vals:
                _, lo, hi = self._norm(s)
                cur = vals[-1]
                tag = f"{s['name']}={cur:.3g} [{lo:.2g}..{hi:.2g}]"
            else:
                tag = f"{s['name']}=—"
            p.setPen(TEXT_COL)
            rect = p.boundingRect(QtCore.QRect(0, 0, 400, 14), 0, tag)
            p.drawText(x - 18 - rect.width(), y, tag)
            y += 14

# ----- Overview tab canvases -------------------------------------------------

_OVERVIEW_HEADER_H = 32
_OVERVIEW_PHASE_H = 16
_OVERVIEW_NARRATIVE_H = 42
_OVERVIEW_EVENT_LOG_H = 28
_OVERVIEW_RIBBON_H = 56
_OVERVIEW_MARGIN = 8
_OVERVIEW_HZ = 4.0
_OVERVIEW_LEARN_MS_MIN = 25.0  # keep in sync with cognitive_panels.LEARN_MS_MIN
_OVERVIEW_EVENT_HOLD = 5
