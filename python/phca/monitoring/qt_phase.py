"""Phase Space & Trajectory tab — trajectory view, drive radar, portrait."""
from __future__ import annotations
import time as _time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from phca.monitoring.qt_base import *  # noqa: F401, F403
from phca.monitoring.qt_base import (
    _BaseCanvas,
    _RolloutCloudCache, ScaleState, _Smoother, freeze_sig,
    _traj_canvas_layout, _draw_data_contract_banner, _window_session_incomplete,
    _phase_status_line, _phase_grid_caption, _grid_err_summary_line,
    _draw_belief_rollout_cloud, _draw_score_proxy_cloud, _draw_rollout_score_legend,
    _draw_moment_ticks, _draw_moment_chips, _elide_line, _dim_label,
    _map_pt, _ellipse_pixel_axes, _draw_phase_grid_base, _prediction_heatmap,
    _drive_color, _drive_short, _to_qcolor,
    MOMENT_COLORS,
    _CAPTION_COL, _FOOTER_COL, _F_CAPTION, _F_AXIS, _F_LABEL, _F_LABEL_B, _F_TITLE,
)


class TrajectoryView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail: Deque[Any] = deque(maxlen=500)
        self.errmap: Optional[np.ndarray] = None
        self.grid_n: int = 0
        self.is_grid: bool = True
        self.frame: Optional[ObservabilityFrame] = None
        self.proj: Optional[BeliefProjection] = None
        self._replay: bool = False
        self._review: bool = False
        self._low_conf: bool = False
        self._rollout_cache = _RolloutCloudCache()
        self._last_cycle_id: int = -1
        self._pred_err_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._moment_series: List[Dict[str, Any]] = []
        self._prev_drive_id: Optional[int] = None
        self._prev_best_score: Optional[float] = None
        self._prefix_len: int = 0

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

    def _clear_rollout_cache(self) -> None:
        self._rollout_cache.clear()

    def _draw_warming(self, p: QtGui.QPainter, n_hist: int, y0: int = 0) -> None:
        w, h = self.width(), self.height()
        mid = h // 2
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(10, mid - 24 + y0, f"Building belief projection ({n_hist}/4 frames)…")
        bx, by, bw, bh = 40, mid - 8 + y0, max(w - 80, 120), 12
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(bx, by, bw, bh, 4, 4)
        fill = int(bw * min(n_hist, 4) / 4)
        if fill > 0:
            p.setPen(QtCore.Qt.NoPen); p.setBrush(ACCENT)
            p.drawRoundedRect(bx, by, fill, bh, 4, 4)

    def _apply_grid_frame(self, f: ObservabilityFrame) -> None:
        self.is_grid = True
        self.grid_n = f.grid.shape[0]
        ap = tuple(f.agent_pos)
        if not self.trail or self.trail[-1] != ap:
            self.trail.append(ap)
        heat = _prediction_heatmap(f.predicted_state, self.grid_n)
        if heat is not None and ap is not None:
            self._low_conf = False
            actual = np.zeros_like(heat)
            r, c = ap
            if 0 <= r < actual.shape[0] and 0 <= c < actual.shape[1]:
                actual[r, c] = 1.0
            err = np.abs(heat - actual)
            self.errmap = err if self.errmap is None else 0.9 * self.errmap + 0.1 * err
        else:
            self._low_conf = True
            if self.errmap is not None:
                self.errmap = self.errmap * 0.85

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.trail.clear()
        self.errmap = None
        self._low_conf = False
        self._clear_rollout_cache()
        self._pred_err_hist.clear()
        self._moment_series = build_moment_series(frames)
        self._prev_drive_id = None
        self._prev_best_score = None
        for f in frames:
            self._pred_err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
            if f.grid is not None and f.agent_pos is not None:
                self._apply_grid_frame(f)
            else:
                self.is_grid = False
        if frames:
            from .cognitive_panels import goal_id_from_frame, _best_score
            gid = goal_id_from_frame(frames[-1])
            if gid:
                self._prev_drive_id = gid
            self._prev_best_score = _best_score(frames[-1])

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        cid = int(getattr(f, "cycle_id", -1))
        if cid != self._last_cycle_id:
            self._clear_rollout_cache()
            self._last_cycle_id = cid
        self.frame = f
        self._replay = replay
        self._review = review
        if f.grid is not None and f.agent_pos is not None:
            if not histories_done:
                self._apply_grid_frame(f)
        else:
            self.is_grid = False
        if not histories_done:
            self._pred_err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
            self._moment_series, self._prev_drive_id, self._prev_best_score = (
                append_cognitive_moment(
                    self._moment_series, f, self._pred_err_hist,
                    prev_drive_id=self._prev_drive_id,
                    prev_best_score=self._prev_best_score,
                    maxlen=TREND_WINDOW))
        self._dirty = True

    def _draw_pred_diag_strip(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 6, y + 12, "prediction diagnostics — spatial view → Overview tab")
        vals = list(self._pred_err_hist)
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                hi = lo + 1.0
            path = QtGui.QPainterPath()
            for i, v in enumerate(vals):
                px = x + 8 + i * (w - 16) / max(len(vals) - 1, 1)
                py = y + h - 6 - (v - lo) / (hi - lo) * (h - 22)
                (path.moveTo if i == 0 else path.lineTo)(px, py)
            p.setPen(QtGui.QPen(QtGui.QColor(230, 126, 34), 2))
            p.drawPath(path)
        emax = float(self.errmap.max()) if self.errmap is not None else 0.0
        p.setPen(DIM_COL)
        p.drawText(x + 6, y + h - 4, f"err trend · max cell err={emax:.2f} · trail={len(self.trail)}")

    def _draw_phase_drive_goals_inset(self, p: QtGui.QPainter, f: ObservabilityFrame,
                                      x: int, y: int, w: int, h: int) -> None:
        import math
        from phca.monitoring.cognitive_panels import frame_drive_goal_norms
        mags = frame_drive_goal_norms(f)
        if not mags or all(m <= 0 for m in mags):
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 10, "no drive targets")
            return
        nd = max(len(mags), _n_drives(f))
        while len(mags) < nd:
            mags.append(0.0)
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRect(x, y, w, h)
        pull_kind = str(getattr(f, "drive_pull_kind", "") or "")
        pull_label = "drive pull" if pull_kind == "goal_vector_norm" else "deficit proxy"
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + 10, pull_label)
        cx, cy = x + w // 2, y + h // 2 + 4
        R = min(w, h) // 2 - 8
        mx = max(mags) if mags else 1.0
        mx = mx if mx > 1e-6 else 1.0
        for i in range(nd):
            ang = -math.pi / 2 + i * 2 * math.pi / max(nd, 1)
            r = (mags[i] / mx) * R
            ex = int(cx + r * math.cos(ang)); ey = int(cy + r * math.sin(ang))
            col = _drive_color(i + 1)
            p.setPen(QtGui.QPen(col, 1)); p.drawLine(cx, cy, ex, ey)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Phase space…"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review,
            panel_key="phase", multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        show_radar = (not self.is_grid) and w >= 900
        status = _phase_status_line(
            f, replay=self._replay, review=self._review, prefix_len=self._prefix_len,
            is_grid=self.is_grid, proj=self.proj,
            show_radar=show_radar)
        cl = _traj_canvas_layout(w, h, y0, is_grid=self.is_grid)
        spark_w = cl.get("spark_w", 96)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(10, 26 + y0, _elide_line(p, status, w - spark_w - 20))
        if len(self._pred_err_hist) >= 2:
            self._pred_err_spark(p, w - spark_w - 6, 20 + y0, spark_w, 22)
        else:
            sx, sy, sw, sh = w - spark_w - 6, 20 + y0, spark_w, 22
            p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
            p.drawRoundedRect(sx, sy, sw, sh, 3, 3)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(sx + 4, sy + 14, "pred err —")
        if self.is_grid:
            n = self.grid_n or 5
            chart_top = cl["chart_top"]
            diag_h = 56
            footer_h = 18
            avail = cl["chart_bot"] - chart_top - footer_h - diag_h
            chart_h = min(avail, int((cl["chart_bot"] - chart_top) * 0.4))
            chart_h = max(int(chart_h), min(60, max(0, avail)))
            emax = float(self.errmap.max()) if self.errmap is not None else 0.0
            tl = list(self.trail)
            self._title(p, f"GridWorld trajectory + G′ cell error map  trail={len(tl)}  max err={emax:.2f}",
                        y=cl["title_y"])
            self._caption(p, _phase_grid_caption(low_conf=self._low_conf) +
                          " · spatial view → Overview", y=cl["caption_y"])
            if chart_h < 24:
                p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
                p.drawText(10, chart_top + 12, "chart area too small — widen window")
                return
            rect = QtCore.QRect(10, chart_top, w - 20, chart_h)
            p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
            p.drawRoundedRect(rect.x() - 2, rect.y() - 2, rect.width() + 4, rect.height() + 4, 6, 6)
            _draw_phase_grid_base(p, f, rect, self.trail, show_confidence_heat=False)
            cell = min(rect.width() - 8, rect.height() - 8) / n
            ox = rect.x() + (rect.width() - cell * n) / 2
            oy = rect.y() + (rect.height() - cell * n) / 2
            if self.errmap is not None:
                for r in range(n):
                    for c in range(n):
                        a = int(np.clip(self.errmap[r, c], 0, 1) * 200)
                        if a <= 0:
                            continue
                        p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell),
                                   QtGui.QColor(230, 126, 34, a))
            if self._low_conf:
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 1))
                p.setBrush(QtGui.QColor(241, 196, 15, 40))
                p.drawRoundedRect(rect.x() + 4, rect.y() + 4, min(rect.width() - 8, 220), 16, 3, 3)
                p.setPen(TEXT_COL); p.setFont(_F_AXIS)
                p.drawText(rect.x() + 8, rect.y() + 16, "low confidence — err heat decaying")
            diag_y = chart_top + chart_h + 6
            if diag_y + diag_h <= h - 6:
                self._draw_pred_diag_strip(p, 10, diag_y, w - 20, diag_h)
            p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
            p.drawText(10, min(cl["chart_bot"] - 4, h - 6),
                       _grid_err_summary_line(self.errmap, self.trail))
        else:
            n_hist = len(self.proj.history) if self.proj is not None else 0
            if self.proj is None or n_hist < 4:
                self._draw_warming(p, n_hist, y0); return
            bounds = self.proj.bounds()
            varexp = self.proj.variance_explained()
            raw2d = bool(getattr(self.proj, "_raw2d", False))
            kind = getattr(f, "gprime_kind", None) or "g′"
            ve_txt = f"  PCA {varexp:.0f}%" if varexp else ""
            proj_lbl = "raw-2D" if raw2d else "PCA-2D"
            rollouts = list(getattr(f, "candidate_rollouts", []) or [])
            r = f.action_rationale or {}
            ci = r.get("chosen_idx")
            chosen_s = (f" · chosen=#{int(ci)}"
                        if rollouts and isinstance(ci, (int, float)) else "")
            self._title(p, f"Belief-space projection ({proj_lbl}, dim={f.state_dim or '?'}) + {kind}{ve_txt}{chosen_s}",
                        y=cl["title_y"])
            cap = "belief trajectory · ellipse = G′ σ · score-colored rollouts"
            if self._replay and not rollouts:
                cap += " · rollouts unavailable in replay"
            self._caption(p, cap, y=cl["caption_y"])
            if getattr(f, "goal_ref", None) is not None:
                ref_lbl = "sanitized" if f.sanitized_state is not None else "obs_vector"
                chip_x, chip_y = w - 128, cl["title_y"] - 2
                p.setPen(QtGui.QPen(DIM_COL, 1))
                p.setBrush(QtGui.QColor(28, 28, 38, 180))
                p.drawRoundedRect(chip_x, chip_y, 118, 16, 3, 3)
                p.setPen(TEXT_COL); p.setFont(_F_AXIS)
                p.drawText(chip_x + 4, chip_y + 12, f"ref={ref_lbl}|goal_ref")
            px0 = 20
            py0 = cl["chart_top"]
            px1 = w - 20
            py1 = cl["chart_bot"]
            chart_w, chart_h = px1 - px0, py1 - py0
            if chart_w <= 8 or chart_h < 24:
                p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
                p.drawText(px0, py0 + 12, "chart area too small — widen window")
                return
            p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
            p.drawRoundedRect(px0 - 4, py0 - 4, chart_w + 8, chart_h + 8, 6, 6)
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, chart_w, chart_h)
            dim_names = list(getattr(f, "dim_names", []) or [])
            ax_x = dim_names[0] if (raw2d and dim_names) else "PC1"
            ax_y = dim_names[1] if (raw2d and dim_names and len(dim_names) > 1) else "PC2"
            p.save()
            try:
                p.setClipRect(px0, py0, chart_w, chart_h)
                for frac in (0.25, 0.5, 0.75):
                    tx = int(px0 + frac * (px1 - px0))
                    ty = int(py0 + frac * (py1 - py0))
                    p.setPen(QtGui.QPen(GRID_COL, 1))
                    p.drawLine(tx, py1, tx, py1 - 3)
                    p.drawLine(px0, ty, px0 + 3, ty)
                is_cont = bool((f.action_rationale or {}).get("continuous", False))
                drawn = _draw_belief_rollout_cloud(
                    p, f, self.proj, px0, py0, px1, py1,
                    self._rollout_cache,
                    replay=self._replay,
                    is_continuous=is_cont,
                    draw_anchor_label=False,
                )
                if not drawn and self._replay:
                    p.setPen(DIM_COL); p.setFont(_F_AXIS)
                    p.drawText(px0 + 6, py0 + 14, "rollout cloud unavailable (JSONL replay)")
                elif not drawn:
                    sc = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
                    ci = r.get("chosen_idx")
                    cidx = int(ci) if isinstance(ci, (int, float)) else (
                        int(np.argmax(sc)) if sc else -1)
                    if sc:
                        _draw_score_proxy_cloud(
                            p, f, self.proj, px0, py0, px1, py1, sc, cidx)
                tl = list(self.proj.history)
                for i in range(1, len(tl)):
                    a, b = tl[i - 1], tl[i]
                    if a is None or b is None:
                        continue
                    ax, ay = _map_pt(a, bounds, px0, py0, px1, py1)
                    bx, by = _map_pt(b, bounds, px0, py0, px1, py1)
                    aa = int(60 + 195 * i / max(len(tl), 1))
                    p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, aa), 2))
                    p.drawLine(ax, ay, bx, by)
                use_san = f.sanitized_state is not None
                cur_v = f.sanitized_state if use_san else f.obs_vector
                cur = self.proj.project(cur_v)
                pred = self.proj.project(f.predicted_state)
                goal_pt = f.goal_target if f.goal_target is not None else getattr(f, "goal_ref", None)
                goal = self.proj.project(goal_pt) if goal_pt is not None else None
                if not use_san and self._replay:
                    p.setPen(DIM_COL); p.setFont(_F_AXIS)
                    p.drawText(px0 + 6, py0 + 28, "anchor: obs_vector")
                ell = None
                if f.gprime_uncertainty is not None and len(f.gprime_uncertainty):
                    ell = self.proj.uncertainty_ellipse(np.asarray(f.gprime_uncertainty, dtype=np.float32))
                if ell is not None and ell[0] is not None and cur is not None:
                    axes, ang = ell
                    ax_p, ay_p = axes
                    if ax_p > 0 and ay_p > 0:
                        cx, cy, pix_ax, pix_ay = _ellipse_pixel_axes(
                            cur, ax_p, ay_p, bounds, px0, py0, px1, py1)
                        p.setBrush(QtGui.QColor(52, 152, 219, 40))
                        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, 160), 1))
                        p.translate(cx, cy); p.rotate(ang)
                        p.drawEllipse(QtCore.QRectF(-pix_ax, -pix_ay, pix_ax * 2, pix_ay * 2))
                        p.resetTransform()
                if pred is not None:
                    pxp, pyp = _map_pt(pred, bounds, px0, py0, px1, py1)
                    p.setBrush(QtGui.QColor(46, 204, 113)); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                    p.drawEllipse(pxp - 3, pyp - 3, 6, 6)
                if goal is not None:
                    gx, gy = _map_pt(goal, bounds, px0, py0, px1, py1)
                    p.setBrush(QtGui.QColor(0, 0, 0, 0)); p.setPen(QtGui.QPen(ACCENT, 2))
                    p.drawEllipse(gx - 6, gy - 6, 12, 12)
                elif self._replay:
                    p.setPen(DIM_COL); p.setFont(_F_AXIS)
                    p.drawText(px0 + 6, py1 - 8, "goal unavailable (replay)")
                if cur is not None:
                    cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
                    p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                    p.drawEllipse(cx - 4, cy - 4, 8, 8)
                if varexp is not None and not raw2d:
                    p.setPen(QtGui.QColor(52, 152, 219)); p.setFont(_F_LABEL_B)
                    p.drawText(px0 + 6, py0 + 14, f"PCA {varexp:.0f}% var")
                if not raw2d and getattr(self.proj, "basis_changed", False):
                    p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(_F_LABEL_B)
                    p.drawText(px0 + 6, py0 + 28, "⟳ PCA re-fit")
                self._draw_phase_drive_goals_inset(p, f, px1 - 88, py0 + 4, 80, 52)
                if self._moment_series:
                    _draw_moment_ticks(p, px0, py0, px1, py1,
                                       self._moment_series[-len(tl):])
            finally:
                p.restore()
            if drawn or rollouts:
                _draw_rollout_score_legend(p, px0 + 4, py1 - 16, chart_w - 8, 14)
            p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
            p.drawText(px0, cl["axis_y"], ax_y)
            p.drawText(px1 - 26, cl["axis_y"], ax_x)
            self._legend(p, [("● current", ACCENT), ("● predicted next", QtGui.QColor(46, 204, 113)),
                             ("○ goal", ACCENT), ("◐ candidates", QtGui.QColor(150, 150, 160)),
                             ("◯ ±σ ellipse", QtGui.QColor(52, 152, 219))],
                         y=cl["legend_y"])

    def _pred_err_spark(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 3, 3)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + 9, "pred err")
        vals = list(self._pred_err_hist)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.drawText(x + 3, y + h - 3, "—"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        n = len(vals)
        p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 1))
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + 3 + i * (w - 6) / max(n - 1, 1)
            py = (y + h - 2) - (v - lo) / (hi - lo) * (h - 12)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.drawPath(path)
        ms = self._moment_series[-n:] if self._moment_series else []
        _draw_moment_ticks(p, x + 3, y + 8, x + w - 3, y + h - 1, ms)


class DriveRadarView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist: Deque[List[float]] = deque(maxlen=200)
        self._active: int = 0
        self.frame: Optional[ObservabilityFrame] = None
        self._replay: bool = False
        self._review: bool = False
        self._prefix_len: int = 0

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.hist.clear()
        for f in frames:
            lv = list(f.drive_levels or [])
            if lv:
                self.hist.append([float(x) for x in lv])

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        if not histories_done:
            lv = list(f.drive_levels or [])
            if lv:
                self.hist.append([float(x) for x in lv])
        self._active = int(getattr(f, "active_drive_id", 0) or 0)
        self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        import math
        w, h = self.width(), self.height()
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(1, 1, w - 2, h - 2, 4, 4)
        cx, cy = w // 2, h // 2 + 4
        R = min(w, h) // 2 - 46
        hist = list(self.hist)
        if not hist:
            self._empty(p, "Drive radar (collecting…)"); return
        if self._review and self._prefix_len > 0:
            p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
            p.drawText(8, 14, f"drive radar · prefix {self._prefix_len} · window ≤200")
        f = self.frame
        if f is not None:
            gid = (f.action_rationale or {}).get("goal_id") or getattr(f, "active_drive_id", None)
            if gid:
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(8, 26, f"active drive D{int(gid)} (Action tab goal link)")
        n = len(hist[-1])
        n = max(3, n)
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            pts.append((cx + R * math.cos(ang), cy + R * math.sin(ang), ang))
        for ring in (0.25, 0.5, 0.75, 1.0):
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 60), 1))
            poly = QtGui.QPolygonF([QtCore.QPointF(cx + ring * R * math.cos(a),
                                                   cy + ring * R * math.sin(a))
                                    for _, _, a in pts])
            p.drawPolygon(poly)
        for i, (x, y, a) in enumerate(pts):
            did = i + 1
            is_active = (did == self._active)
            p.setPen(QtGui.QPen(_drive_color(did) if is_active else QtGui.QColor(50, 50, 60),
                                2 if is_active else 1))
            p.drawLine(cx, cy, int(x), int(y))
            lx = cx + (R + 16) * math.cos(a); ly = cy + (R + 16) * math.sin(a)
            p.setPen(_drive_color(did)); p.setFont(_F_LABEL_B)
            p.drawText(int(lx) - 12, int(ly) + 4, _drive_short(did))
        tgt = list(getattr(self.frame, "drive_targets", []) or []) if self.frame else []
        if tgt:
            tn = min(n, len(tgt))
            poly = QtGui.QPolygonF()
            for i in range(tn):
                ang = -math.pi / 2 + i * 2 * math.pi / n
                t = float(np.clip(tgt[i], 0, 1))
                poly.append(QtCore.QPointF(cx + t * R * math.cos(ang), cy + t * R * math.sin(ang)))
            p.setBrush(QtGui.QColor(0, 0, 0, 0)); p.setPen(QtGui.QPen(QtGui.QColor(46, 204, 113, 160), 1, QtCore.Qt.DashLine))
            p.drawPolygon(poly)
        for k, levels in enumerate(hist[:-1]):
            a = int(20 + 60 * k / max(len(hist), 1))
            poly = QtGui.QPolygonF()
            for i, lvl in enumerate(levels):
                ang = -math.pi / 2 + i * 2 * math.pi / n
                poly.append(QtCore.QPointF(cx + lvl * R * math.cos(ang),
                                           cy + lvl * R * math.sin(ang)))
            p.setBrush(QtGui.QColor(241, 196, 15, a // 3))
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 1))
            p.drawPolygon(poly)
        cur = hist[-1]
        poly = QtGui.QPolygonF()
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            poly.append(QtCore.QPointF(cx + lvl * R * math.cos(ang), cy + lvl * R * math.sin(ang)))
        p.setBrush(QtGui.QColor(241, 196, 15, 70)); p.setPen(QtGui.QPen(ACCENT, 2))
        p.drawPolygon(poly)
        defs = list(getattr(self.frame, "drive_deficits", []) or []) if self.frame else []
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            vx = int(cx + lvl * R * math.cos(ang)); vy = int(cy + lvl * R * math.sin(ang))
            p.setBrush(_drive_color(i + 1))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(vx - 3, vy - 3, 6, 6)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(vx + 5, vy - 4, f"{lvl:.2f}")
            if defs and i < len(defs) and defs[i] > 0.05:
                tx = int(cx + (lvl + 0.06) * R * math.cos(ang))
                ty = int(cy + (lvl + 0.06) * R * math.sin(ang))
                p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 2))
                p.drawLine(vx, vy, tx, ty)
        self._title(p, f"{n}-drive radar (solid=level · dashed=target · red tick=deficit)")
        cap = "radial drive levels 0..1 · faded = recent history · active spoke bold · numbers = raw level"
        if self._replay:
            cap += " · replay — drive pull from drive_goal_norms"
        self._caption(p, cap)


class _DimSelector(QtWidgets.QWidget):
    def __init__(self, view: "_PhasePortraitView", parent=None):
        super().__init__(parent)
        self.view = view
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.prev = QtWidgets.QPushButton("‹ prev")
        self.next = QtWidgets.QPushButton("next ›")
        self.label = QtWidgets.QLabel("")
        for b in (self.prev, self.next):
            b.setFixedWidth(70); lay.addWidget(b)
        lay.addWidget(self.label); lay.addStretch(1)
        lay.addWidget(QtWidgets.QLabel("jump:"))
        self.jump = QtWidgets.QSpinBox()
        self.jump.setRange(0, 9999); self.jump.setFixedWidth(72)
        lay.addWidget(self.jump)
        self.prev.clicked.connect(self._back)
        self.next.clicked.connect(self._fwd)
        self.jump.valueChanged.connect(self._jump)

    def _back(self):
        self.view.set_page(self.view.page - 1)

    def _fwd(self):
        self.view.set_page(self.view.page + 1)

    def _jump(self, dim: int):
        if self.view.page_size > 0:
            self.view.set_page(dim // self.view.page_size)

    def refresh(self):
        v = self.view
        n_pages = max(1, (v.n_dims + v.page_size - 1) // v.page_size)
        a = v.page * v.page_size
        b = min(v.n_dims, (v.page + 1) * v.page_size) - 1
        b = max(b, 0)
        dim_names = list(getattr(v.frame, "dim_names", []) or []) if getattr(v, "frame", None) else []
        def nm(i):
            return dim_names[i] if i < len(dim_names) else f"d{i}"
        self.label.setText(f"page {v.page + 1}/{n_pages}  {nm(a)}..{nm(b)} of {v.n_dims}")
        self.prev.setEnabled(v.page > 0)
        self.next.setEnabled(v.page + 1 < n_pages)
        self.jump.setMaximum(max(0, v.n_dims - 1))


class _PhasePortraitView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.page: int = 0
        self.page_size: int = 24
        self.n_dims: int = 0
        self.mi_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.be_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._scale = ScaleState(contract=0.05, head=0.06)
        self._mi_scale = ScaleState(contract=0.05, head=0.06)
        self._be_scale = ScaleState(contract=0.05, head=0.06)
        self._offenders: List[int] = []
        self._offender_t: float = 0.0
        self._offender_errs: Optional[np.ndarray] = None
        self._replay: bool = False
        self._review: bool = False
        self._ref_prefer_goal: bool = False
        self._moment_series: List[Dict[str, Any]] = []
        self._ref_chip_rect: Optional[QtCore.QRect] = None
        self._prefix_len: int = 0

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if (self._ref_chip_rect is not None
                and self._ref_chip_rect.contains(ev.pos())
                and getattr(self.frame, "goal_ref", None) is not None):
            self._ref_prefer_goal = not self._ref_prefer_goal
            self._dirty = True
            self.update()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def _append_trends(self, f: ObservabilityFrame) -> None:
        mi = getattr(f, "gprime_mutual_info", None)
        if mi is None and f.belief_entropies:
            mi = float(list(f.belief_entropies.values())[0])
        if mi is not None:
            self.mi_hist.append(float(mi))
        be = float(list(f.belief_entropies.values())[0]) if f.belief_entropies else None
        if be is not None:
            self.be_hist.append(float(be))

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.mi_hist.clear()
        self.be_hist.clear()
        self._offenders = []
        self._offender_errs = None
        self._offender_t = 0.0
        self._moment_series = build_moment_series(frames)
        self._scale.reset()
        self._mi_scale.reset()
        self._be_scale.reset()
        for f in frames:
            self._append_trends(f)
        if frames:
            last = frames[-1]
            if getattr(last, "goal_ref", None) is not None:
                self._ref_prefer_goal = False
            if last.predicted_state is not None:
                ref_a, _ = belief_reference(last, prefer_goal=self._ref_prefer_goal)
                if ref_a is not None:
                    pred = np.asarray(last.predicted_state, dtype=np.float32).reshape(-1)
                    d = int(min(len(pred), len(ref_a)))
                    if d > 0:
                        errs = np.abs(pred[:d] - ref_a[:d])
                        self._offenders = [int(x) for x in np.argsort(-errs)[:5]]
                        self._offender_errs = errs
                        self._offender_t = _time.monotonic()
                        if self._offenders:
                            self.page = self._offenders[0] // self.page_size

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        if not histories_done:
            self._append_trends(f)
        self._dirty = True

    def set_page(self, p: int) -> None:
        n_pages = max(1, (self.n_dims + self.page_size - 1) // self.page_size)
        self.page = max(0, min(n_pages - 1, int(p)))
        self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None or f.predicted_state is None:
            self._empty(p, "Phase portrait (collecting…)"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review, panel_key="phase",
            multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        pred = np.asarray(f.predicted_state, dtype=np.float32).reshape(-1)
        ref_a, ref_src = belief_reference(f, prefer_goal=self._ref_prefer_goal)
        if ref_a is None:
            self._empty(p, "Phase portrait (no reference)"); return
        ref = ref_a
        d = int(min(len(pred), len(ref)))
        if d == 0:
            self._empty(p, "Phase portrait (empty)"); return
        from phca.monitoring.cognitive_panels import frame_has_full_peu, frame_per_dim_peu
        peu = frame_per_dim_peu(f)
        full_peu = frame_has_full_peu(f)
        sparse_peu_indices = {
            int(item.get("idx", -1))
            for item in (getattr(f, "per_dim_peu_top", None) or [])
            if isinstance(item, dict)
        }
        peu_a = None
        if peu is not None:
            pa = np.asarray(peu, dtype=np.float32).reshape(-1)
            peu_a = pa[:d] if len(pa) >= d else pa
        errs = np.abs(pred[:d] - ref[:d])
        std = None
        if f.gprime_uncertainty is not None and len(f.gprime_uncertainty):
            us = np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1)
            std = us[:d] if len(us) >= d else us
        self.n_dims = d
        n_pages = max(1, (d + self.page_size - 1) // self.page_size)
        self.page = max(0, min(n_pages - 1, self.page))
        lo_i = self.page * self.page_size
        hi_i = min(d, lo_i + self.page_size)
        idx = np.arange(lo_i, hi_i)
        errs_s = errs[idx]
        std_s = None
        if std is not None:
            std_idx = [int(i) for i in idx if int(i) < len(std)]
            if len(std_idx) == len(idx):
                std_s = std[idx]
        n = len(idx)
        bw = (w - 40) / max(n, 1)
        hdr = y0 + 2
        top, bot = hdr + 52, min(max(h // 2, h - 72), h - 28)
        kind = getattr(f, "gprime_kind", "g′") or "g′"
        mi = getattr(f, "gprime_mutual_info", None)
        mi_txt = f"  mi={float(mi):.3f}" if mi is not None else ""
        self._title(p, f"Phase portrait  |pred−{ref_src}| ▮ + {kind} ±σ ░  dims {n}/{d}{mi_txt}",
                    y=hdr + 14)
        if getattr(f, "goal_ref", None) is not None:
            chip_x, chip_y = w - 118, hdr + 2
            chip_w, chip_h = 108, 16
            self._ref_chip_rect = QtCore.QRect(chip_x, chip_y, chip_w, chip_h)
            active = self._ref_prefer_goal
            p.setPen(QtGui.QPen(ACCENT if active else DIM_COL, 1))
            p.setBrush(QtGui.QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 50 if active else 20))
            p.drawRoundedRect(chip_x, chip_y, chip_w, chip_h, 3, 3)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(chip_x + 4, chip_y + 12, f"ref={'goal_ref' if active else ref_src}")
        else:
            self._ref_chip_rect = None
        st_line = _phase_status_line(
            f, replay=self._replay, review=self._review,
            prefix_len=self._prefix_len, is_grid=False)
        defs = list(getattr(f, "drive_deficits", []) or [])
        if defs:
            ds = ",".join(f"{v:.2f}" for v in defs[:6])
            st_line += f" · deficits=[{ds}]"
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(10, hdr + 30, _elide_line(p, st_line, w - 20))
        cap = "red = prediction error · blue whisker = ±σ"
        if peu_a is None and not self._replay:
            cap += " · stable order (no re-sort)"
        elif peu_a is None and self._replay:
            cap += " · replay · PEU top-K missing"
        elif not full_peu:
            cap += " · ▒ = PEU top-K only; non-top dimensions unknown"
        else:
            cap += " · ▒ = full PEU layer"
        self._caption(p, cap, y=hdr + 44)
        if (self._replay
                and (f.gprime_uncertainty is None
                     or len(np.asarray(f.gprime_uncertainty).reshape(-1)) == 0)):
            p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(_F_AXIS)
            p.drawText(w - 148, hdr + 16, "σ unavailable (replay)")
        chart_x, chart_w = 16, w - 32
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(chart_x - 4, top - 8, chart_w + 8, bot - top + 16, 6, 6)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
        raw_max = float(errs_s.max()) if errs_s.size else 1.0
        if std_s is not None:
            band_top = (errs_s + std_s).max() if std_s.size else 0.0
            raw_max = max(raw_max, float(band_top))
        _, mx = self._scale.update(0.0, float(raw_max if raw_max > 1e-6 else 1.0))
        mx = mx if mx > 1e-6 else 1.0
        p.save()
        try:
            p.setClipRect(chart_x, top, chart_w, bot - top)
            for i in range(n):
                x = 20 + i * bw
                eh = int(errs_s[i] / mx * (bot - top))
                if std_s is not None and i < len(std_s):
                    y_e = bot - eh
                    s_pix = float(std_s[i]) / mx * (bot - top)
                    p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, 140), 1))
                    p.drawLine(int(x + bw / 2), int(y_e - s_pix), int(x + bw / 2), int(y_e + s_pix))
                    p.drawLine(int(x + bw / 2 - 3), int(y_e - s_pix), int(x + bw / 2 + 3), int(y_e - s_pix))
                    p.drawLine(int(x + bw / 2 - 3), int(y_e + s_pix), int(x + bw / 2 + 3), int(y_e + s_pix))
                # Do not treat action chosen_idx as a state-dim highlight.
                p.fillRect(int(x + 2), int(bot - eh), int(bw - 8), eh, QtGui.QColor(231, 76, 60, 220))
                dim_i = int(idx[i])
                if (peu_a is not None and dim_i < len(peu_a)
                        and (full_peu or dim_i in sparse_peu_indices)):
                    pe = float(peu_a[dim_i])
                    ph = int(pe / mx * (bot - top) * 0.5)
                    p.fillRect(int(x + bw / 2 - 2), int(bot - ph), 4, ph,
                               QtGui.QColor(150, 150, 160, 140))
        finally:
            p.restore()
        if bot + 14 <= h - 14:
            p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
            for i in range(n):
                x = 20 + i * bw
                lbl = _dim_label(int(idx[i]), f)
                p.drawText(int(x), min(bot + 11, h - 18), lbl if n <= 24 else str(int(idx[i])))
        p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
        scale_y = h - 14 if d > 8 else h - 4
        p.drawText(20, scale_y, f"scale 0..{mx:.3g}  (named dims · pager below)")
        now = _time.monotonic()
        if now - self._offender_t >= 0.5 or not self._offenders:
            self._offender_t = now
            self._offenders = [int(x) for x in np.argsort(-errs)[:5]]
            self._offender_errs = errs
        inset_h = h // 2 - 44
        if h >= 120 and inset_h >= 24:
            off_title = "top error dims"
            if self._review and self._prefix_len > 0 and self.frame is not None:
                off_title += f" · cycle {int(self.frame.cycle_id)}"
            self._offenders_inset(p, self._offender_errs if self._offender_errs is not None else errs,
                                  f, w - 150, top + 4, 140, inset_h, title=off_title)
        if self._review and self._prefix_len > 0 and self.frame is not None:
            self._footer_caption(
                p, f"top error dims · through cycle {int(self.frame.cycle_id)}",
                y=h - 2)
        self._legend(p, [("▮ error", QtGui.QColor(231, 76, 60)),
                         ("├─┤ ±σ whisker", QtGui.QColor(52, 152, 219))],
                     y=h // 2 + 4, x=20)
        t_top, t_bot = h // 2 + 22, h - 16
        if t_bot > t_top + 8:
            self._trend(p, self.mi_hist, t_top, t_bot, QtGui.QColor(155, 89, 182),
                        "mutual_info trend", left=40, right=w // 2 - 6, scale=self._mi_scale)
            self._trend(p, self.be_hist, t_top, t_bot, QtGui.QColor(46, 204, 113),
                        "belief-entropy trend", left=w // 2 + 6, right=w - 16, scale=self._be_scale)
            if self._moment_series:
                ms = self._moment_series[-len(self.mi_hist):] if self.mi_hist else []
                _draw_moment_ticks(p, 40, t_top, w // 2 - 6, t_bot, ms)
                _draw_moment_ticks(p, w // 2 + 6, t_top, w - 16, t_bot, ms)
        if d > 8:
            pc_top, pc_bot = t_bot + 8, h - 4
            pc_h = pc_bot - pc_top
            if pc_h >= 16 and pc_bot <= h - 4:
                self._parallel_residual(p, pred[:d], ref[:d], f, 20, pc_top, w - 40, pc_h)

    def _parallel_residual(self, p: QtGui.QPainter, pred: np.ndarray, ref: np.ndarray,
                           f: ObservabilityFrame, x: int, y: int, w: int, h: int) -> None:
        if h < 12 or w < 40:
            return
        resid = np.abs(pred - ref)
        n = min(len(resid), 48)
        if n < 2:
            return
        mx = float(resid[:n].max()) or 1.0
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y - 2, f"residual |pred−actual| parallel-coords (top {n} dims)")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(x, y, w, h)
        step = w / max(n - 1, 1)
        path = QtGui.QPainterPath()
        for i in range(n):
            frac = float(resid[i]) / mx
            px = x + i * step
            py = y + h - frac * (h - 4) - 2
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.save()
        try:
            p.setClipRect(x, y, w, h)
            p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60, 200), 1))
            p.drawPath(path)
        finally:
            p.restore()
        p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
        for i in range(0, n, max(1, n // 8)):
            lx = int(x + i * step) - 8
            ly = min(y + h + 10, y + h - 2)
            p.drawText(lx, ly, _dim_label(i, f)[:5])

    def _offenders_inset(self, p: QtGui.QPainter, errs: np.ndarray,
                         f: ObservabilityFrame, x: int, y: int, w: int, h: int,
                         *, title: str = "top error dims") -> None:
        if w < 1 or h < 1:
            return
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 12, title[:28])
        mx = float(errs.max()) if errs.size else 1.0
        mx = mx if mx > 1e-6 else 1.0
        ry = y + 18
        for rank, di in enumerate(self._offenders):
            if ry + 14 > y + h:
                break
            lbl = _dim_label(int(di), f)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            jump = f"↪{lbl}" if rank == 0 else lbl
            p.drawText(x + 4, ry + 10, f"#{rank + 1} {jump}")
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(231, 76, 60, 200))
            p.fillRect(x + 78, ry + 4, int((w - 84) * float(errs[di]) / mx), 6, QtGui.QColor(231, 76, 60, 200))
            ry += 14

    def _trend(self, p: QtGui.QPainter, s: Deque[float], top: int, bot: int,
               col: QtGui.QColor, label: str, left: int, right: int,
               scale: Optional[ScaleState] = None) -> None:
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(left, top - 4, label)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(left, bot, right, bot)
        vals = list(s)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(left + 4, top + 14, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if scale is not None:
            lo, hi = scale.update(float(lo), float(hi))
        if hi - lo < 1e-9:
            hi = lo + 1
        n = len(vals)
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            x = left + i * (right - left) / (n - 1)
            y = bot - (v - lo) / (hi - lo) * (bot - top - 4)
            (path.moveTo if i == 0 else path.lineTo)(x, y)
        p.setPen(QtGui.QPen(col, 2)); p.drawPath(path)
