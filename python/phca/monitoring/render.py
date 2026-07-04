"""PHCA v3.0 — Cognitive Dashboard renderer (Observability v2, shared).

Single source of truth for the live visualiser and the replay tool. Builds a
persistent-artist dashboard and mutates artists in place per tick (no
``ax.clear()``) so the live window stays responsive and flicker-free.

Layout (GridSpec 3x3, constrained_layout):
  world   | drives+targets | pred-error/confidence trend
  (tall)  | attention      | RBTA + retention status
  action-rationale (bottom strip, full width)

World panel:
  GridWorld  -> base grid (walls/goal) + G' predicted next-cell heatmap (with
                the same confidence guard the cycle uses) + path trail + agent.
  MuJoCo     -> predicted next-state vs goal-reference bar groups (the MPC
                alignment signal), not raw observation.

v3.0 trace: A4 (prediction made visible, honestly guarded), A5 (error/conf +
drive adaptation trends), A1 (RBTA reasons + retention caps visible), G5
(active goal drive highlighted).
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from .observability import ObservabilityFrame, normalize_observability_json

DRIVE_NAMES = {
    1: "D1 PredErr", 2: "D2 Critical", 3: "D3 Compet",
    4: "D4 Curious", 5: "D5 Energy", 6: "D6 Empower",
}
DRIVE_COLORS = {
    1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd",
    5: "#1f77b4", 6: "#17becf",
}
TREND_WINDOW = 200  # rolling window for error/confidence trend


class DashboardHandle:
    """Holds all persistent artists + per-panel state."""

    def __init__(self, fig, axes, artists, is_grid: bool):
        self.fig = fig
        self.axes = axes
        self.artists = artists
        self.is_grid = is_grid
        self.trail = deque(maxlen=400)          # agent path (gridworld)
        self.trend = deque(maxlen=TREND_WINDOW)  # recent frames for trend


def build_dashboard(fig, is_grid: bool) -> DashboardHandle:
    """Create the dashboard layout + persistent artists on ``fig``.

    ``fig`` must be a fresh Figure (this clears it). ``is_grid`` selects the
    world-panel mode (GridWorld grid vs MuJoCo prediction/goal-ref bars).
    """

    fig.clear()
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1, 0.55])
    ax_world = fig.add_subplot(gs[0:2, 0])
    ax_drive = fig.add_subplot(gs[0, 1])
    ax_trend = fig.add_subplot(gs[0, 2])
    ax_att = fig.add_subplot(gs[1, 1])
    ax_status = fig.add_subplot(gs[1, 2])
    ax_action = fig.add_subplot(gs[2, 0:3])

    ax_status.axis("off")
    ax_action.axis("off")
    fig.suptitle("PHCA v3.0 — Cognitive Dashboard", fontsize=13)

    artists: Dict[str, Any] = {}

    # --- world panel ---
    if is_grid:
        ax_world.set_title("External world — agent + G' predicted next-cell")
        # placeholder images; real data set on first update
        base_img = ax_world.imshow(np.zeros((2, 2)), cmap="RdGy", vmin=-1, vmax=1,
                                   origin="upper", aspect="equal")
        heat_img = ax_world.imshow(np.zeros((2, 2)), cmap="Blues", alpha=0.0,
                                   origin="upper", aspect="equal")
        (trail_line,) = ax_world.plot([], [], "-", color="orange", linewidth=1.5,
                                      alpha=0.6)
        (agent_dot,) = ax_world.plot([], [], "o", color="royalblue", markersize=14,
                                     markeredgecolor="white", zorder=5)
        (goal_star,) = ax_world.plot([], [], "*", color="lime", markersize=16, zorder=5)
        artists.update(base_img=base_img, heat_img=heat_img, trail_line=trail_line,
                       agent_dot=agent_dot, goal_star=goal_star)
    else:
        ax_world.set_title("MuJoCo — predicted next-state vs goal reference (MPC)")
        ax_world.axhline(0, color="grey", linewidth=0.5)
        pred_bars = ax_world.bar([0], [0], color="#1f77b4", label="predicted")
        ref_bars = ax_world.bar([0], [0], color="#2ca02c", alpha=0.5,
                                label="goal ref")
        ax_world.legend(loc="upper right", fontsize=7)
        ax_world.set_xlabel("state dim")
        artists.update(pred_bars=pred_bars, ref_bars=ref_bars)

    # --- drives + targets ---
    ax_drive.set_title("MDIM drives (value vs target)")
    ax_drive.set_ylim(0, 1.2)
    drive_bars = ax_drive.bar([], [])  # empty; populated on first update
    artists["drive_bars"] = drive_bars
    # target marker line (single hline reused) + active-drive highlight edge
    (target_line,) = ax_drive.plot([], [], "k_", markersize=10, mew=2,
                                   label="target")
    artists["target_line"] = target_line
    ax_drive.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax_drive.legend(loc="upper right", fontsize=7)

    # --- trend ---
    ax_trend.set_title("Prediction error / confidence (rolling)")
    ax_trend.set_ylim(0, 1.0)
    (err_line,) = ax_trend.plot([], [], "-", color="crimson", label="pred error")
    (conf_line,) = ax_trend.plot([], [], "-", color="darkgreen", label="confidence")
    ax_trend.set_xlabel("cycle")
    ax_trend.legend(loc="upper right", fontsize=7)
    artists.update(err_line=err_line, conf_line=conf_line)

    # --- attention ---
    ax_att.set_title("Attention focus (selected chunks)")
    ax_att.set_ylim(0, 1.0)
    att_bars = ax_att.bar([], [])
    artists["att_bars"] = att_bars

    # --- status (RBTA + retention + cycle) ---
    status_txt = ax_status.text(
        0.02, 0.98, "", transform=ax_status.transAxes, fontsize=8.5,
        family="monospace", verticalalignment="top")
    artists["status_txt"] = status_txt

    # --- action rationale ---
    action_txt = ax_action.text(
        0.02, 0.92, "", transform=ax_action.transAxes, fontsize=9,
        family="monospace", verticalalignment="top")
    artists["action_txt"] = action_txt

    axes = dict(world=ax_world, drive=ax_drive, trend=ax_trend, att=ax_att,
                status=ax_status, action=ax_action)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return DashboardHandle(fig, axes, artists, is_grid)


def _grid_base(grid: np.ndarray, goal_pos) -> np.ndarray:
    size = grid.shape[0]
    base = np.zeros((size, size), dtype=np.float32)
    g = np.asarray(grid)
    base[g == 1] = -1.0  # walls
    if goal_pos is not None:
        base[goal_pos[0], goal_pos[1]] = 0.6
    return base


def _prediction_heatmap(predicted: np.ndarray, size: int) -> Optional[np.ndarray]:
    """First size^2 dims as a (size,size) soft cell distribution, with the
    cycle's confidence guard (cycle.py:772): near-uniform predictions render
    as a dim/neutral map instead of a fake sharp peak."""
    if predicted is None or predicted.shape[0] < size * size:
        return None
    heat = predicted[:size * size].reshape(size, size).astype(np.float32)
    heat = heat - heat.min()
    mx = heat.max()
    if mx < 0.05:
        return None  # uniform/low-confidence -> no heatmap overlay
    return heat / mx


def _update_world(handle: DashboardHandle, f: ObservabilityFrame):
    ax = handle.axes["world"]
    a = handle.artists
    if handle.is_grid and f.grid is not None and f.agent_pos is not None:
        size = f.grid.shape[0]
        base = _grid_base(f.grid, f.goal_pos)
        a["base_img"].set_data(base)
        a["base_img"].set_extent((-0.5, size - 0.5, size - 0.5, -0.5))
        heat = _prediction_heatmap(f.predicted_state, size)
        if heat is not None:
            a["heat_img"].set_data(heat)
            a["heat_img"].set_alpha(0.45)
            a["heat_img"].set_extent((-0.5, size - 0.5, size - 0.5, -0.5))
        else:
            a["heat_img"].set_alpha(0.0)
        # trail
        if not handle.trail or handle.trail[-1] != f.agent_pos:
            handle.trail.append(f.agent_pos)
        rows = [p[0] for p in handle.trail]
        cols = [p[1] for p in handle.trail]
        a["trail_line"].set_data(cols, rows)
        a["agent_dot"].set_data([f.agent_pos[1]], [f.agent_pos[0]])
        if f.goal_pos is not None:
            a["goal_star"].set_data([f.goal_pos[1]], [f.goal_pos[0]])
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True, color="grey", linewidth=0.3, alpha=0.5)
    elif not handle.is_grid:
        # MuJoCo: predicted next-state vs goal reference
        pred = f.predicted_state
        ref = f.goal_ref
        if pred is None and ref is None:
            return
        n = 0
        if pred is not None:
            n = max(n, pred.shape[0])
        if ref is not None:
            n = max(n, ref.shape[0])
        x = np.arange(n)
        # rebuild bars (count may change once at start; stable afterwards)
        if len(a["pred_bars"]) != n:
            ax.clear()
            ax.set_title("MuJoCo — predicted next-state vs goal reference (MPC)")
            ax.axhline(0, color="grey", linewidth=0.5)
            a["pred_bars"] = ax.bar(x, np.zeros(n), color="#1f77b4", label="predicted")
            a["ref_bars"] = ax.bar(x, np.zeros(n), color="#2ca02c", alpha=0.5,
                                   label="goal ref")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_xlabel("state dim")
        for rect, i in zip(a["pred_bars"], range(n)):
            rect.set_height(float(pred[i]) if pred is not None and i < pred.shape[0] else 0.0)
        for rect, i in zip(a["ref_bars"], range(n)):
            rect.set_height(float(ref[i]) if ref is not None and i < ref.shape[0] else 0.0)
        allv = []
        if pred is not None:
            allv.extend(pred.tolist())
        if ref is not None:
            allv.extend(ref.tolist())
        if allv:
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.1 + 1e-6
            ax.set_ylim(lo - pad, hi + pad)


def _update_drives(handle: DashboardHandle, f: ObservabilityFrame):
    ax = handle.axes["drive"]
    a = handle.artists
    n = len(f.drive_levels)
    if n == 0:
        return
    labels = [DRIVE_NAMES.get(i + 1, f"D{i+1}") for i in range(n)]
    colors = [DRIVE_COLORS.get(i + 1, "teal") for i in range(n)]
    # highlight active goal drive
    edge = ["white"] * n
    if 1 <= f.active_drive_id <= n:
        edge[f.active_drive_id - 1] = "black"
    bars = a["drive_bars"]
    if len(bars) != n:
        ax.clear()
        ax.set_title("MDIM drives (value vs target)")
        ax.set_ylim(0, 1.2)
        a["drive_bars"] = ax.bar(range(n), f.drive_levels, color=colors,
                                 edgecolor=edge, linewidth=1.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=25, fontsize=8)
        (a["target_line"],) = ax.plot([], [], "k_", markersize=10, mew=2,
                                      label="target")
        ax.legend(loc="upper right", fontsize=7)
        bars = a["drive_bars"]
    else:
        for rect, v in zip(bars, f.drive_levels):
            rect.set_height(float(v))
        for rect, c, e in zip(bars, colors, edge):
            rect.set_color(c)
            rect.set_edgecolor(e)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=25, fontsize=8)
    # target markers (x positions, y = target)
    if f.drive_targets:
        xs = list(range(min(len(f.drive_targets), n)))
        ys = [f.drive_targets[i] for i in xs]
        a["target_line"].set_data(xs, ys)


def _update_trend(handle: DashboardHandle, f: ObservabilityFrame):
    a = handle.artists
    handle.trend.append(f)
    hist = list(handle.trend)
    xs = [h.cycle_id for h in hist]
    errs = [h.prediction_error for h in hist]
    confs = [h.prediction_confidence for h in hist]
    a["err_line"].set_data(xs, errs)
    a["conf_line"].set_data(xs, confs)
    ax = handle.axes["trend"]
    if xs:
        ax.set_xlim(min(xs), max(max(xs), min(xs) + 1))
    ymax = max([1.0] + errs + confs) if (errs or confs) else 1.0
    ax.set_ylim(0, ymax * 1.05 + 1e-6)


def _update_attention(handle: DashboardHandle, f: ObservabilityFrame):
    ax = handle.axes["att"]
    a = handle.artists
    sal = f.attention_saliences
    if not sal:
        return
    n = len(sal)
    labels = [str(f.attention_indices[i]) if i < len(f.attention_indices) else str(i)
              for i in range(n)]
    bars = a["att_bars"]
    if len(bars) != n:
        ax.clear()
        ax.set_title("Attention focus (selected chunks)")
        ax.set_ylim(0, max(1.0, max(sal) * 1.1) if sal else 1.0)
        a["att_bars"] = ax.bar(range(n), sal, color="#9467bd")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("chunk id")
        bars = a["att_bars"]
    else:
        for rect, v in zip(bars, sal):
            rect.set_height(float(v))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(1.0, max(sal) * 1.1))


def _update_status(handle: DashboardHandle, f: ObservabilityFrame):
    rbta_color = {"CONTINUE": "#2ca02c", "INTERRUPT": "#ff7f0e"}.get(
        f.rbta_action, "#d62728")
    # Build a multi-line status string; RBTA reasons listed when interrupting.
    active = DRIVE_NAMES.get(f.active_drive_id, f"D{f.active_drive_id}")
    lines = [
        f"cycle     : {f.cycle_id}",
        f"latency   : {f.latency_ms:.1f} ms",
        f"active    : {active}  (goal drive)",
        f"RBTA      : {f.rbta_action}",
    ]
    if f.rbta_action != "CONTINUE" and f.rbta_violations:
        for v in f.rbta_violations[:4]:
            lines.append(f"   {v['module_id']:>6}.{v['bound_type']:<6} "
                         f"{v['measured']:.2f}>{v['allowed']:.2f}")
    elif f.violations_count == 0:
        lines.append("   all bounds ok")
    lines.append(f"violations: {f.violations_count}")
    lines.append(f"goal_reached: {f.goal_reached}")
    lines.append(f"pred err  : {f.prediction_error:.4f}   conf: {f.prediction_confidence:.3f}")
    lines.append(f"M3 epis   : {f.episode_count}/{f.m3_cap or '-'}")
    lines.append(f"M4 facts  : {f.fact_count}/{f.m4_cap or '-'}  prune@{f.m4_prune_target}")
    lines.append(f"RSS       : {f.rss_bytes / 1e6:.1f} MB")
    # Render with the RBTA line colored.
    txt = "\n".join(lines)
    a = handle.artists["status_txt"]
    a.set_text(txt)
    # color the whole text by RBTA state for quick scanning
    a.set_color(rbta_color if f.rbta_action != "CONTINUE" else "black")


def _update_action(handle: DashboardHandle, f: ObservabilityFrame):
    r = f.action_rationale or {}
    explored = r.get("explored", False)
    eps = r.get("eps")
    cont = r.get("continuous", False)
    if cont:
        kind = "EXPLORE (random continuous)" if explored else "EXPLOIT (MPC argmax)"
        score = r.get("best_score")
        k = r.get("k_candidates")
        line = f"{kind}  eps={eps:.3f}  K={k}  score={('%.3f' % score) if score is not None else '-'}"
        if f.continuous_action is not None:
            vec = ", ".join(f"{x:+.2f}" for x in f.continuous_action.tolist())
            line += f"  action=[{vec}]"
    else:
        gid = r.get("goal_id")
        gname = DRIVE_NAMES.get(gid, f"D{gid}") if gid else "-"
        if explored:
            line = f"EXPLORE (e-greedy)  eps={eps:.3f}  goal={gname}"
        else:
            score = r.get("best_score")
            k = r.get("k_candidates")
            note = r.get("note")
            line = (f"EXPLOIT  eps={eps:.3f}  goal={gname}  "
                    f"K={k}  score={('%.3f' % score) if score is not None else '-'}"
                    + (f"  [{note}]" if note else ""))
        line += f"  action={f.action_name}"
    handle.artists["action_txt"].set_text(
        f"Action selection:\n  {line}")


def update_dashboard(handle: DashboardHandle, f: ObservabilityFrame) -> None:
    """Mutate all persistent artists with the latest frame (no clear/rebuild)."""
    _update_world(handle, f)
    _update_drives(handle, f)
    _update_trend(handle, f)
    _update_attention(handle, f)
    _update_status(handle, f)
    _update_action(handle, f)


def frame_from_json(obj: Dict[str, Any]) -> ObservabilityFrame:
    """Reconstruct an ObservabilityFrame from a JSONL line (for replay)."""
    obj = normalize_observability_json(obj)
    f = ObservabilityFrame()
    array_fields = {
        "predicted_state", "obs_vector", "goal_ref", "continuous_action",
        "sanitized_state", "state_precision", "goal_target",
        "prediction_precision", "gprime_uncertainty", "per_dim_peu",
        "attention_weights", "last_action_vector",
    }
    for k, v in obj.items():
        if k == "grid" and v is not None:
            setattr(f, k, np.asarray(v, dtype=np.int32))
        elif k in array_fields and v is not None:
            setattr(f, k, np.asarray(v, dtype=np.float32))
        elif isinstance(v, (dict, list)):
            setattr(f, k, copy.deepcopy(v))
        else:
            setattr(f, k, v)
    return f
