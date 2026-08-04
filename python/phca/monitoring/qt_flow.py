"""Cognitive Flow tab — timing topology, radial pipeline clock-face, module heatmap."""

from __future__ import annotations
import hashlib
import json
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui

from phca.monitoring.qt_base import *  # noqa: F401, F403
from phca.monitoring.qt_base import (
    _BaseCanvas, _Smoother,
    _FLOW_ALL_MODULES,
    PANEL_BG_ALT, PANEL_BORDER, CHIP_FILL_ALPHA,
    _F_AXIS, _F_LABEL_B,
    _flow_layout, _radial_sankey_link, _heatmap_cell_color, _heatmap_cell_color_precomputed,
    _to_qcolor, _cost_color,
    _draw_moment_ticks, _draw_moment_chips,
    _draw_execution_phase_strip,
    _draw_data_contract_banner, _window_session_incomplete,
    _flow_update_active_idx, _flow_status_line, _action_score_margin, _elide_line,
)


class CognitiveFlowView(_BaseCanvas):
    """v7 focal: radial clock-face pipeline. 7 PIPELINE nodes on a ring (active
    node enlarged), 6 SIDE_MODULES as outer satellites. Per-node cost-coloured
    disc + measured-vs-bound bar + status badge; prediction/action nodes get a
    dim-agnostic scalar gauge instead of a 12-bar thumb. Composite RBTA envelope
    + legend consolidated into one top-right panel; module×cycle heatmap is a
    thin bottom strip. EMA-smoothed timings + hysteresis-held active node (no
    strobe). Additive — same fields, no controller change."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.heat: Deque[Dict[str, float]] = deque(maxlen=80)
        self.viol_mods: set = set()
        self.last_viol: Dict[str, str] = {}
        self.cycle_id: int = 0
        self._ms_smooth: Dict[str, _Smoother] = {}
        # v7: EMA-smoothed timing deltas + hold counter so the active node
        # doesn't strobe between cycles.
        self._delta_smooth: Dict[str, _Smoother] = {m: _Smoother(0.15) for m in PIPELINE}
        self._active_hold: int = 0
        self._active_idx: int = 0
        # slow wall-clock animation (decoupled from cycle rate → no strobe)
        self._anim_t: float = 0.0
        self._last_wall: float = time.monotonic()
        # v8: heatmap is repaint-heavy (≈80×13 cells) → cache to a QPixmap and
        # regenerate at most every 0.5 s (2 Hz sub-cadence).
        self._heat_pm: Optional[QtGui.QPixmap] = None
        self._heat_t: float = 0.0
        self._heat_key: tuple = ()
        # v8 B2: EMA-smoothed link widths (Sankey activation mass share).
        self._link_smooth: Dict[str, _Smoother] = {}
        self._topo_sig: tuple = ()
        self._replay: bool = False
        self._review: bool = False
        self._moment_series: List[Dict[str, Any]] = []
        self._moment_err_hist: Deque[float] = deque(maxlen=80)
        self._prev_drive_id: Optional[int] = None
        self._prev_best_score: Optional[float] = None
        self._heat_cycle_ids: Deque[int] = deque(maxlen=80)

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.heat.clear()
        self._heat_cycle_ids.clear()
        self._ms_smooth.clear()
        self._link_smooth.clear()
        self._heat_pm = None
        self._heat_key = ()
        self._moment_err_hist.clear()
        self._prev_drive_id = None
        self._prev_best_score = None
        self._moment_series = build_moment_series(frames)
        self._anim_t = 0.0
        for f in frames:
            self.heat.append(dict(getattr(f, "module_timings", {}) or {}))
            self._heat_cycle_ids.append(int(getattr(f, "cycle_id", 0) or 0))
        if frames:
            from .cognitive_panels import goal_id_from_frame, _best_score
            gid = goal_id_from_frame(frames[-1])
            if gid:
                self._prev_drive_id = gid
            self._prev_best_score = _best_score(frames[-1])
        if len(self.heat) >= 2:
            prev, cur = self.heat[-2], self.heat[-1]
            try:
                self._active_idx, self._active_hold = _flow_update_active_idx(
                    prev, cur, self._delta_smooth, self._active_idx, self._active_hold)
            except Exception:
                pass

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        self.cycle_id = int(f.cycle_id)
        if not histories_done:
            self.heat.append(dict(f.module_timings))
            self._heat_cycle_ids.append(int(f.cycle_id))
            self._moment_series, self._prev_drive_id, self._prev_best_score = (
                append_cognitive_moment(
                    self._moment_series, f, self._moment_err_hist,
                    prev_drive_id=self._prev_drive_id,
                    prev_best_score=self._prev_best_score,
                    maxlen=self.heat.maxlen or 80))
            if len(self._moment_series) > len(self.heat):
                self._moment_series = self._moment_series[-len(self.heat):]
        now = time.monotonic()
        dt = now - self._last_wall; self._last_wall = now
        self._anim_t = (self._anim_t + dt / 0.5) % len(PIPELINE)
        # active node = most-recently-active by EMA'd timing delta + hysteresis
        if not histories_done and len(self.heat) >= 2:
            prev, cur = self.heat[-2], self.heat[-1]
            try:
                self._active_idx, self._active_hold = _flow_update_active_idx(
                    prev, cur, self._delta_smooth, self._active_idx, self._active_hold)
            except Exception:
                pass
        self.viol_mods = {RBTA_TO_FLOW.get(v.get("module_id", v.get("module", "")),
                                          v.get("module_id", v.get("module", "")))
                          for v in f.rbta_violations}
        self.last_viol = {RBTA_TO_FLOW.get(v.get("module_id", v.get("module", "")),
                                           v.get("module_id", v.get("module", ""))):
                          f"{v.get('bound_type','?')} {v.get('measured','?'):.4g}>{v.get('allowed','?'):.4g}"
                          for v in f.rbta_violations}
        self._dirty = True

    def _node_pos(self, w: int, graph_h: int, y_offset: int = 0, *, scale: float = 1.0):
        import math
        cx, cy = w // 2, y_offset + graph_h // 2
        R = int(max(50, min(w, graph_h) // 2 - 50) * scale)
        pos = {}
        n = len(PIPELINE)
        for i, mod in enumerate(PIPELINE):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            pos[mod] = (int(cx + R * math.cos(ang)), int(cy + R * math.sin(ang)))
        ns = len(SIDE_MODULES)
        for j, mod in enumerate(SIDE_MODULES):
            ang = -math.pi / 2 + (j + 0.5) * 2 * math.pi / ns
            pos[mod] = (int(cx + (R + 36) * math.cos(ang)),
                        int(cy + (R + 36) * math.sin(ang)))
        return pos, (cx, cy), R

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Cognitive flow…"); return
        active_idx = self._active_idx
        lay = _flow_layout(w, h, len(_FLOW_ALL_MODULES), replay=self._replay)
        y0 = lay["y0"]
        _draw_data_contract_banner(p, self.width(), replay=self._replay,
                                   review=self._review, panel_key="flow", y=2,
                                   multi_agent=self._multi_agent,
                                   incomplete=_window_session_incomplete(self))
        self._title(p, "Cognitive flow — timing topology + execution phases", y=18 + y0)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(10, 40 + y0, _elide_line(
            p, _flow_status_line(f, active_idx=active_idx), w - 20))
        moment = self._moment_series[-1] if self._moment_series else None
        _draw_moment_chips(p, 10, 48 + y0, moment)
        strip_y = 58 + y0
        learn_burst = bool(moment and moment.get("learn_burst"))
        _draw_execution_phase_strip(
            p, f, QtCore.QRect(8, strip_y, w - 16, 14), learn_burst=learn_burst)
        self._caption(p, "disc = timing cost · red = RBTA violation · "
                         "link width = adjacent timing share · gold arc = Δ edge · "
                         "POST-CYCLE = inter-cycle housekeeping", y=104 + y0)
        pos, (cx, cy), R = self._node_pos(
            w, lay["graph_h"], lay["graph_top"], scale=lay.get("node_scale", 1.0))
        n = len(PIPELINE)
        tot = sum(float(f.module_timings.get(m, 0.0)) for m in PIPELINE) or 1.0
        side_max = max((float(f.module_timings.get(m, 0.0)) for m in SIDE_MODULES),
                       default=1.0) or 1.0
        for i in range(n - 1):
            a, b = pos[PIPELINE[i]], pos[PIPELINE[i + 1]]
            m0, m1 = PIPELINE[i], PIPELINE[i + 1]
            share = (float(f.module_timings.get(m0, 0.0)) + float(f.module_timings.get(m1, 0.0))) / tot
            sm = self._link_smooth.setdefault(f"{m0}>{m1}", _Smoother(0.2))
            w_share = sm.value(share)
            col = QtGui.QColor(241, 196, 15, 230) if i == active_idx else QtGui.QColor(160, 160, 180, 200)
            lw = 1.5 + w_share * 14.0
            _radial_sankey_link(p, a[0], a[1], b[0], b[1], cx, cy, col, width=lw)
        if 0 <= active_idx < n - 1:
            a, b = pos[PIPELINE[active_idx]], pos[PIPELINE[active_idx + 1]]
            t = self._anim_t - active_idx
            t = t - int(t)
            px = int(a[0] + t * (b[0] - a[0])); py = int(a[1] + t * (b[1] - a[1]))
            p.setBrush(QtGui.QColor(255, 255, 255)); p.setPen(QtGui.QPen(ACCENT, 1))
            p.drawEllipse(px - 4, py - 4, 8, 8)
        for sm in SIDE_MODULES:
            ms_side = float(f.module_timings.get(sm, 0.0))
            if ms_side <= 0:
                continue
            a = pos[sm]
            anchor = "prediction" if sm in ("gprime_learn", "attn") else "memory_write"
            b = pos[anchor]
            lw = 1.0 + 3.0 * (ms_side / side_max)
            alpha = max(40, int(40 + 120 * (ms_side / side_max)))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, alpha), lw, QtCore.Qt.DashLine))
            p.drawLine(a[0], a[1], b[0], b[1])
        for mod in PIPELINE + SIDE_MODULES:
            x, y = pos[mod]
            raw_ms = float(f.module_timings.get(mod, 0.0))
            sm = self._ms_smooth.setdefault(mod, _Smoother(alpha=0.25))
            ms = sm.value(raw_ms)
            viol = mod in self.viol_mods
            is_consol = mod == "consolidation"
            if viol:
                col = QtGui.QColor(231, 76, 60)
            elif is_consol and raw_ms <= 0:
                col = QtGui.QColor(80, 80, 90, 140)
            else:
                col = _to_qcolor(_cost_color(ms))
            is_active = (PIPELINE.index(mod) == active_idx) if mod in PIPELINE else False
            r = int(18 + min(ms / 20.0, 1.0) * 8 + (6 if is_active else 0))
            pen_w = 2 if is_active else 1
            if is_consol and raw_ms <= 0:
                p.setBrush(col)
                p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 110), pen_w, QtCore.Qt.DashLine))
            else:
                p.setBrush(col)
                p.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), pen_w))
            p.drawEllipse(x - r, y - r, r * 2, r * 2)
            p.setFont(_F_AXIS); p.setPen(TEXT_COL)
            p.drawText(x - 22, y + 4, 44, 11, 0x84, f"{ms:.1f}ms")
            lbl_y = y + r + 10
            p.setFont(_F_LABEL_B); p.setPen(TEXT_COL)
            node_lbl = PIPELINE_LABEL.get(mod, mod)
            if mod == "gprime_learn":
                from phca.monitoring.cognitive_panels import frame_is_geometry_control
                if frame_is_geometry_control(f):
                    node_lbl = "G′WM"
            p.drawText(x - 30, lbl_y, 60, 12, 0x84, node_lbl)
            self._mb_bar(p, mod, ms, x - 30, lbl_y + 14, 60)
            if mod in ("prediction", "action_selection"):
                self._scalar_gauge_thumb(p, mod, f, x - 30, y - r - 30, 60, 22)
            elif mod == "peu":
                self._peu_gauge_thumb(p, f, x - 30, y - r - 30, 60, 22)
            else:
                self._status_badge(p, mod, ms, x - 30, y - r - 24, 60)
            if viol and mod in self.last_viol:
                p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(_F_AXIS)
                p.drawText(x - 30, lbl_y + 32, 60, 10, 0x84, self.last_viol[mod])
            sub = self._flow_node_subcaption(mod, f)
            sub_y = lbl_y + 44
            if sub:
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(x - 30, sub_y, 60, 10, 0x84, sub[:12])
            if mod == "attn":
                self._attn_salience_bars(p, f, x - 30, sub_y + 12, 60)
        self._side_panel(p, lay["graph_top"], compact=lay.get("compact", False))
        self._heatmap(p, lay)

    def _bound_for(self, mod: str, bounds: dict) -> Optional[float]:
        return rbta_time_bound_ms(mod, bounds)

    def _mb_bar(self, p: QtGui.QPainter, mod: str, measured: float,
                x: int, y: int, w: int) -> None:
        """Measured-vs-bound bar (v7: sparkline dropped — the bottom heatmap
        already carries timing history, so one bar per node is enough)."""
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        b_ms = self._bound_for(mod, bounds)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRect(x, y, w, 7)
        scale = b_ms if b_ms else 25.0
        ratio = max(0.0, min(measured / scale, 1.5))
        fw = int(min(ratio, 1.0) * w)
        col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else _to_qcolor(_cost_color(measured))
        p.fillRect(x, y, fw, 7, col)
        if b_ms:
            bx = x + w
            p.setPen(QtGui.QPen(ACCENT, 2)); p.drawLine(bx, y - 2, bx, y + 9)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 16, f"{measured:.1f}/{b_ms:.1f}ms")
        else:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 16, f"{measured:.1f}ms")

    def _flow_node_subcaption(self, mod: str, f: ObservabilityFrame) -> str:
        if mod == "prediction":
            err = float(getattr(f, "prediction_error", 0.0) or 0.0)
            return f"err={err:.2f}" if err > 0 else ""
        if mod == "peu":
            from phca.monitoring.cognitive_panels import frame_peu_mean
            mean = frame_peu_mean(f)
            if mean is not None:
                return f"PEŪ={mean:.2f}"
            if self._replay:
                return "PEU —"
            return ""
        if mod in ("tspl", "mdim"):
            acc = float(getattr(f, "tspl_skill_accuracy", 0.0) or 0.0)
            if acc > 0:
                txt = f"skill={acc:.2f}"
                if getattr(f, "tspl_skill_compiled", False):
                    txt += " compiled"
                return txt
            return ""
        if mod == "rbta":
            act = str(getattr(f, "rbta_action", "") or "CONTINUE")
            vc = int(getattr(f, "violations_count", 0) or 0)
            return f"{act}" + (f" {vc}V" if vc else "")
        if mod == "consolidation":
            ml = dict(getattr(f, "memory_log", {}) or {})
            el = dict(getattr(f, "energy_log", {}) or {})
            parts = []
            if ml:
                parts.append(f"M:{sum(ml.values()):.0f}ms")
            if el:
                parts.append(f"E:{sum(el.values()):.0f}ms")
            return " ".join(parts)
        return ""

    def _status_badge(self, p: QtGui.QPainter, mod: str, measured: float,
                      x: int, y: int, w: int) -> None:
        if mod == "consolidation" and measured <= 0:
            txt, c = "POST-CYCLE", QtGui.QColor(120, 120, 130)
            p.setPen(QtGui.QPen(c, 1)); p.setBrush(PANEL_BG_ALT)
            p.drawRoundedRect(x, y, w, 14, 4, 4)
            p.setBrush(QtGui.QColor(c.red(), c.green(), c.blue(), CHIP_FILL_ALPHA))
            p.drawRoundedRect(x + 1, y + 1, w - 2, 12, 3, 3)
            p.setPen(c); p.setFont(_F_AXIS)
            p.drawText(x, y, w, 14, 0x84, txt)
            return
        viol = mod in self.viol_mods
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        b_ms = self._bound_for(mod, bounds)
        if viol:
            # VIOLATION only from rbta_violations / viol_mods — never invent from ms.
            txt, c = "VIOLATION", QtGui.QColor(231, 76, 60)
        elif not b_ms or b_ms <= 0:
            # No fake 25ms scale — do not invent OK/VIOLATION without a bound.
            txt, c = "NO-BOUND", QtGui.QColor(120, 120, 130)
        else:
            ratio = measured / b_ms if b_ms else 0.0
            if ratio > 1.0:
                txt, c = "OVER", QtGui.QColor(231, 76, 60)
            elif ratio > 0.8:
                txt, c = "NEAR-BOUND", QtGui.QColor(241, 196, 15)
            else:
                txt, c = "OK", QtGui.QColor(46, 204, 113)
        p.setPen(QtGui.QPen(c, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, 14, 4, 4)
        p.setBrush(QtGui.QColor(c.red(), c.green(), c.blue(), CHIP_FILL_ALPHA))
        p.drawRoundedRect(x + 1, y + 1, w - 2, 12, 3, 3)
        p.setPen(c); p.setFont(_F_AXIS)
        p.drawText(x, y, w, 14, 0x84, txt)

    def _scalar_gauge_value(self, mod: str, f: ObservabilityFrame) -> Tuple[float, str, str]:
        """Scalar for node gauge with action-specific semantics on Act node."""
        if mod == "prediction":
            from phca.monitoring.cognitive_panels import frame_display_confidence
            v = float(frame_display_confidence(f))
            return v, "conf", ""
        from phca.monitoring.cognitive_panels import geometry_score_label
        score_lbl = geometry_score_label(f)
        r = f.action_rationale or {}
        scores = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
        bs = r.get("best_score")
        if isinstance(bs, (int, float)):
            return float(bs), score_lbl, ""
        margin = _action_score_margin(scores)
        if margin is not None:
            return max(0.0, min(1.0, margin)), "margin", ""
        eps = r.get("eps")
        if isinstance(eps, (int, float)) and not r.get("explored"):
            return max(0.0, min(1.0, 1.0 - float(eps))), "conf", ""
        if scores:
            return float(np.mean(scores)), score_lbl, ""
        return 0.0, score_lbl, ""

    def _scalar_gauge_thumb(self, p: QtGui.QPainter, mod: str,
                            f: ObservabilityFrame, x: int, y: int, w: int, h: int) -> None:
        """Dim-agnostic scalar gauge (prediction confidence or action score/MI)."""
        v, label, unit = self._scalar_gauge_value(mod, f)
        vc = max(0.0, min(1.0, v))
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        col = (QtGui.QColor(46, 204, 113) if vc > 0.66 else
               QtGui.QColor(241, 196, 15) if vc > 0.33 else QtGui.QColor(231, 76, 60))
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 180))
        p.drawRoundedRect(x + 2, y + 2, int((w - 4) * vc), h - 4, 3, 3)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + h - 5, f"{label}={v:.2f}{unit}")

    def _peu_gauge_thumb(self, p: QtGui.QPainter, f: ObservabilityFrame,
                         x: int, y: int, w: int, h: int) -> None:
        """PEU mean gauge on the PEU node (high = warmer)."""
        from phca.monitoring.cognitive_panels import frame_peu_mean
        mean = frame_peu_mean(f)
        if mean is None:
            self._status_badge(p, "peu", float(f.module_timings.get("peu", 0.0)),
                               x, y + 6, w)
            return
        v = float(mean)
        # Scale display: typical PEU mean on grid ≈0–few; clamp for bar fill.
        vc = max(0.0, min(1.0, v / 5.0))
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        col = (QtGui.QColor(46, 204, 113) if vc < 0.33 else
               QtGui.QColor(241, 196, 15) if vc < 0.66 else QtGui.QColor(231, 76, 60))
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 180))
        p.drawRoundedRect(x + 2, y + 2, int((w - 4) * vc), h - 4, 3, 3)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + h - 5, f"PEŪ={v:.2f}")

    def _attn_salience_bars(self, p: QtGui.QPainter, f: ObservabilityFrame,
                            x: int, y: int, w: int) -> None:
        """Mini salience strip under the Attn node (selected attention pairs)."""
        from phca.monitoring.cognitive_panels import frame_attention_pairs
        pairs = frame_attention_pairs(f)
        if not pairs:
            return
        vals = [float(s) for _, s in pairs[:4]]
        if not vals:
            return
        mx = max(abs(v) for v in vals) or 1.0
        bar_w = max(3, (w - 8) // max(len(vals), 1))
        for i, val in enumerate(vals):
            bh = max(1, int(abs(val) / mx * 10))
            p.fillRect(x + 2 + i * (bar_w + 1), y + (10 - bh), bar_w, bh,
                       QtGui.QColor(52, 152, 219, 200))

    def _side_panel(self, p: QtGui.QPainter, y_offset: int = 12,
                    *, compact: bool = False) -> None:
        """Composite bounds + near-bound list + latency check (top-right)."""
        f = self.frame
        w = self.width()
        x, y, tw, th = w - 168, y_offset, 158, 142
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRect(x, y, tw, th)
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 12, "composite bounds")
        bounds = getattr(f, "rbta_bounds", None) or {} if f else {}
        pipe_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in PIPELINE]
        side_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in SIDE_MODULES]
        pb = pipeline_time_budget_ms(bounds, PIPELINE)
        lat = float(getattr(f, "latency_ms", 0.0) or 0.0) if f else 0.0
        p.setPen(ACCENT); p.setFont(_F_AXIS)
        lat_s = f"  lat={lat:.0f}ms" if lat > 0 else ""
        p.drawText(x + 4, y + 28, f"Σpipe={pb:.0f}ms{lat_s}")
        if f and lat > 0 and pb > 0 and abs(lat - pb) / max(lat, 1e-6) > 0.2:
            p.setPen(QtGui.QColor(241, 196, 15))
            p.drawText(x + 4, y + 40, f"⚠ pipe≠latency Δ{lat - pb:+.0f}ms")
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 54, f"({len(pipe_ids)}+{len(side_ids)} bound mods)")
        if f:
            ms = getattr(f, "meta_stable", None) or {}
            stable = bool(ms.get("is_meta_stable", ms.get("stable", False)))
            chip_col = QtGui.QColor(46, 204, 113) if stable else QtGui.QColor(231, 76, 60)
            chip_lbl = "meta stable" if stable else "meta unstable"
            p.setFont(_F_AXIS)
            fm = p.fontMetrics()
            cw = fm.horizontalAdvance(chip_lbl) + 8
            p.setPen(QtGui.QPen(chip_col, 1))
            p.setBrush(QtGui.QColor(chip_col.red(), chip_col.green(), chip_col.blue(), CHIP_FILL_ALPHA))
            p.drawRoundedRect(x + 4, y + 58, cw, 14, 3, 3)
            p.setPen(chip_col)
            p.drawText(x + 8, y + 69, chip_lbl)
            near = flow_near_bound_modules(f, top_k=3)
            if near:
                p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
                p.drawText(x + 4, y + 82, "near-bound")
                ly = y + 96
                for mod, ratio in near:
                    p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(_F_AXIS)
                    p.drawText(x + 4, ly, f"{PIPELINE_LABEL.get(mod, mod)[:6]} {ratio:.0%}")
                    ly += 12
        if compact:
            return
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + th - 14, "gold=Δ · purple=side")

    def _heatmap(self, p: QtGui.QPainter, lay: Dict[str, Any]) -> None:
        if not self.heat:
            return
        mods = _FLOW_ALL_MODULES
        w = self.width()
        top = lay["heatmap_top"]
        strip_h = lay["heatmap_h"]
        label_w = lay["label_w"]
        row_h = lay["row_h"]
        col_hist: Dict[str, List[float]] = {m: [] for m in mods}
        for mts in self.heat:
            for mod in mods:
                col_hist[mod].append(float(mts.get(mod, 0.0)))
        # C2+H3: hash-based key (avoids 1040-element tuple); precomputed percentiles
        heat_json = json.dumps(
            [[round(col_hist[m][i], 4) for m in mods] for i in range(len(self.heat))],
            sort_keys=True, default=str)
        key = (w, strip_h, label_w, hashlib.md5(heat_json.encode()).hexdigest())
        now = time.monotonic()
        if (self._heat_pm is None or key != self._heat_key
                or now - self._heat_t >= 0.5):
            self._heat_t = now; self._heat_key = key
            pm = QtGui.QPixmap(w, strip_h + 8)
            pm.fill(QtCore.Qt.transparent)
            rp = QtGui.QPainter(pm)
            rp.setRenderHint(QtGui.QPainter.Antialiasing, False)
            data_x = 10 + label_w
            band_w = w - data_x - 10
            local_top = 4
            rp.setPen(QtGui.QPen(PANEL_BORDER, 1)); rp.setBrush(PANEL_BG_ALT)
            rp.drawRect(8, local_top - 2, w - 16, strip_h + 4)
            rmax = max((max((mt.get(m, 0.0) for m in mods), default=0.0) for mt in self.heat), default=1.0)
            rmax = rmax if rmax > 1e-6 else 25.0
            cw = band_w / max(len(self.heat), 1)
            for j, mod in enumerate(mods):
                hist = col_hist[mod]
                lbl = PIPELINE_LABEL.get(mod, mod)[:5]
                if mod == "consolidation" and all(v <= 0 for v in hist):
                    lbl = "—"
                rp.setPen(TEXT_COL); rp.setFont(_F_AXIS)
                rp.drawText(10, int(local_top + j * row_h + row_h * 0.75), lbl)
            col_ranks: Dict[str, List[float]] = {}
            for mod in mods:
                hist = col_hist[mod]
                nz = sorted([float(v) for v in hist if float(v) > 0])
                if not nz:
                    col_ranks[mod] = [0.0] * len(hist)
                else:
                    col_ranks[mod] = [
                        sum(1 for v_nz in nz if v_nz <= float(ms)) / len(nz)
                        if float(ms) > 0 else 0.0
                        for ms in hist
                    ]
            for i, mts in enumerate(self.heat):
                for j, mod in enumerate(mods):
                    ms = float(mts.get(mod, 0.0))
                    cell = _heatmap_cell_color_precomputed(ms, col_ranks[mod][i])
                    if cell.alpha() <= 0:
                        continue
                    rp.fillRect(int(data_x + i * cw), int(local_top + j * row_h),
                                max(int(cw), 1), max(int(row_h) - 1, 1), cell)
            rp.setPen(DIM_COL); rp.setFont(_F_AXIS)
            rp.drawText(data_x, local_top - 4, f"module×cycle cost heatmap (max={rmax:.1f}ms)")
            rp.end()
            self._heat_pm = pm
        p.drawPixmap(0, top - 4, self._heat_pm)
        data_x = 10 + label_w
        band_w = w - data_x - 10
        learn_row = None
        for j, mod in enumerate(mods):
            if mod == "gprime_learn":
                learn_row = j
                break
        if learn_row is not None and any(
                m.get("learn_burst") for m in self._moment_series[-len(self.heat):]):
            ry = int(top - 4 + 4 + learn_row * row_h)
            p.setPen(QtGui.QPen(_to_qcolor(MOMENT_COLORS["learn_burst"]), 2))
            p.drawRect(data_x - 1, ry - 1, int(band_w) + 2, max(int(row_h), 1) + 1)
        f = self.frame
        rbta_act = str(getattr(f, "rbta_action", "") or "CONTINUE") if f else "CONTINUE"
        if rbta_act != "CONTINUE":
            rbta_row = next((j for j, m in enumerate(mods) if m == "rbta"), None)
            if rbta_row is not None:
                ry = int(top - 4 + 4 + rbta_row * row_h)
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 2))
                p.drawRect(data_x - 1, ry - 1, int(band_w) + 2, max(int(row_h), 1) + 1)
        tick_top = top - 4
        tick_bot = top - 4 + strip_h
        _draw_moment_ticks(
            p, data_x, tick_top, data_x + int(band_w), tick_bot,
            self._moment_series[-len(self.heat):] if self._moment_series else [])
        ncols = len(self.heat)
        if ncols >= 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            cids = list(self._heat_cycle_ids)[-ncols:]
            cw = band_w / max(ncols, 1)
            for i, cid in enumerate(cids):
                if i % 5 == 0 or i == ncols - 1:
                    tx = int(data_x + i * cw + 2)
                    p.drawText(tx, tick_bot + 10, str(cid))
