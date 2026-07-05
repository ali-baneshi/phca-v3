"""Goals & Motivation panel (extracted from qt_dashboard monolith)."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.belief_projection import ScaleState
from phca.monitoring.overview_narrative import TREND_WINDOW
from phca.monitoring.playback import _Smoother

# Shared chrome / helpers from the panel module (already loaded when we import).
from phca.monitoring.qt_dashboard import (
    PANEL_BG,
    GRID_COL,
    TEXT_COL,
    DIM_COL,
    ACCENT,
    _FOOTER_COL,
    _F_AXIS,
    _F_LABEL,
    _F_LABEL_B,
    _BaseCanvas,
    _arrow,
    _draw_data_contract_banner,
    _window_session_incomplete,
    _drive_color,
    _drive_short,
    _n_drives,
)

class GoalsMotivationView(_BaseCanvas):
    """v7: focal 'homeostasis tanks' (6 vertical tanks: level vs target setpoint,
    deficit gap, active highlight, Pareto ring). Goal stack → indented tree with
    per-node progress bars; goal_history → step-strip; deficit heatmap gets a
    correct caption + colorbar; deficits EMA-smoothed; trend y-scale snapped to
    a ScaleState; drive_goals radial 'where each drive pulls' inset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drive_hist: Deque[List[float]] = deque(maxlen=TREND_WINDOW)
        self.goal_hist: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.temp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.emp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        # v7: EMA-smoothed per-drive deficit + snapped trend scales.
        self._def_smooth: List[_Smoother] = []
        self._temp_scale = ScaleState(contract=0.05, head=0.06)
        self._emp_scale = ScaleState(contract=0.05, head=0.06)
        self._replay: bool = False
        self._review: bool = False
        self._prefix_len: int = 0

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.drive_hist.clear()
        self.goal_hist.clear()
        self.temp_hist.clear()
        self.emp_hist.clear()
        self._def_smooth = []
        self._temp_scale.reset()
        self._emp_scale.reset()
        for f in frames:
            self.set_frame(f, histories_done=False)
        self._dirty = True

    def _ensure_def_smooth(self, n: int) -> None:
        while len(self._def_smooth) < n:
            self._def_smooth.append(_Smoother(0.2))

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self._replay = replay
        self._review = review
        if not histories_done:
            if f.drive_deficits is not None:
                defs = np.asarray(f.drive_deficits, dtype=np.float32).reshape(-1)
                nd = len(defs)
                self._ensure_def_smooth(nd)
                self.drive_hist.append([float(x) for x in defs])
                for i in range(nd):
                    self._def_smooth[i].value(float(defs[i]))
            gh = getattr(f, "goal_history", None)
            if gh:
                self.goal_hist.append(int(gh[-1]))
            t = getattr(f, "cr_temperature", None)
            if t is not None:
                self.temp_hist.append(float(t))
            e = getattr(f, "empowerment", None)
            if e is not None:
                self.emp_hist.append(float(e))
        super().set_frame(f)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Goals & motivation…"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review,
            panel_key="goals", multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        levels = f.drive_levels or []
        targets = f.drive_targets or []
        nd = _n_drives(f, list(levels))
        self._ensure_def_smooth(nd)
        defs = [self._def_smooth[i]._v if i < len(self._def_smooth) and self._def_smooth[i]._have else 0.0
                for i in range(nd)]
        pareto = {int(x) for x in (getattr(f, "pareto_front", None) or [])}
        active = int(getattr(f, "active_drive_id", 0) or 0)
        footer_top = h - 18
        # ---- v7 focal: homeostasis tanks (top, full width) ----
        tank_h = max(min(h // 3, footer_top - 120), 80)
        self._title(p, "Homeostasis tanks — setpoint band · deficit arrow · ◯ = Pareto · ▮ = active",
                    x=8, y=14 + y0)
        self._caption(p, "shaded band = target setpoint · red gap + arrow = deficit toward setpoint",
                      x=8, y=26 + y0)
        tank_y = max(40 + y0, 32)
        self._tanks(p, levels, targets, defs, pareto, active, 8, tank_y, w - 16, tank_h)
        # ---- bottom-left: goal stack tree + goal_history step-strip ----
        by = tank_y + tank_h + 20
        lw = max(w // 2 - 8, 120)
        self._title(p, "Goal stack (tree, deepest active first)", x=8, y=by)
        stack = getattr(f, "goal_stack", None) or []
        gy = by + 12
        gy = self._goal_tree(p, stack, 8, gy, lw)
        self._goal_history_strip(p, 8, gy + 6, lw, 26)
        # ---- bottom-right: drive_goals radial inset + heatmap + trends ----
        rx = lw + 16
        rw = max(w - rx - 8, 160)
        heatmap_w = max(80, rw - 140)
        heatmap_h = max(48, h // 4)
        self._drive_goals_inset(p, f, rx, by, min(130, rw), min(130, heatmap_h + 80))
        self._title(p, "Drive deficit history heatmap", x=rx + 140, y=by)
        self._heatmap(p, rx + 140, by + 12, heatmap_w, heatmap_h)
        trend_y = by + heatmap_h + 24
        trend_h = max(48, min(footer_top - trend_y, h - trend_y - 8))
        if trend_h >= 48 and trend_y + trend_h <= footer_top:
            self._trend_pair(p, rx, trend_y, rw, trend_h)

    def _tanks(self, p, levels, targets, defs, pareto, active, x, y, w, h) -> None:
        """v7 focal: vertical homeostasis tanks (dim-agnostic N drives)."""
        n = _n_drives(self.frame, list(levels))
        tw = w // max(n, 1) - 8
        lvl_max = max([float(x) for x in levels] + [1.0]) if levels else 1.0
        tgt_max = max([float(x) for x in targets if x is not None] + [1.0]) if targets else 1.0
        scale = max(lvl_max, tgt_max, 1.0)
        for i in range(n):
            did = i + 1
            tx = x + i * (w / max(n, 1)) + 4
            col = _drive_color(did)
            lvl = float(levels[i]) if i < len(levels) else 0.0
            tgt = float(targets[i]) if i < len(targets) and targets[i] is not None else None
            dfc = float(defs[i]) if i < len(defs) else 0.0
            is_active = (did == active)
            is_pareto = did in pareto
            # tank frame
            frame_col = col if is_active else GRID_COL
            p.setPen(QtGui.QPen(frame_col, 2 if is_active else 1))
            p.setBrush(PANEL_BG); p.drawRect(int(tx), y, tw, h - 18)
            # level fill (normalized when levels exceed 1.0)
            disp = min(1.0, max(0.0, lvl / scale))
            lh = int(disp * (h - 20))
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 200))
            p.fillRect(int(tx) + 2, y + (h - 18) - lh, tw - 4, lh, col)
            # v8 B7: target setpoint band (lower–upper) + dashed centre line
            if tgt is not None:
                band = 0.05 * scale
                t_lo = int(min(1.0, max(0.0, (tgt - band) / scale)) * (h - 20))
                t_hi = int(min(1.0, max(0.0, (tgt + band) / scale)) * (h - 20))
                ty = y + (h - 18) - int(min(1.0, max(0.0, tgt / scale)) * (h - 20))
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 35))
                p.fillRect(int(tx) + 2, y + (h - 18) - t_hi, tw - 4, t_hi - t_lo, QtGui.QColor(ACCENT.red(), ACCENT.green(), ACCENT.blue(), 35))
                p.setPen(QtGui.QPen(ACCENT, 1, QtCore.Qt.DashLine))
                p.drawLine(int(tx), ty, int(tx) + tw, ty)
            # deficit gap shading + arrow toward setpoint
            if tgt is not None and dfc > 0.02:
                ly = y + (h - 18) - lh
                p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(231, 76, 60, 90))
                gap_top = min(ly, ty); gap_bot = max(ly, ty)
                p.fillRect(int(tx) + 2, gap_top, tw - 4, gap_bot - gap_top, QtGui.QColor(231, 76, 60, 90))
                # deficit arrow (level → setpoint)
                ax = int(tx + tw // 2); ay = ly
                _arrow(p, ax, ay, ax, ty, QtGui.QColor(231, 76, 60, 200), size=5)
            # Pareto ring
            if is_pareto:
                p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
                p.setBrush(QtGui.QColor(0, 0, 0, 0))
                p.drawEllipse(int(tx) + tw // 2 - 6, y + (h - 18) - lh - 6, 12, 12)
            # labels
            p.setPen(col if is_active else TEXT_COL); p.setFont(_F_LABEL_B)
            p.drawText(int(tx), y + h - 14, f"{_drive_short(did)} {lvl:.2f}")
            if tgt is not None:
                p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
                p.drawText(int(tx), y + h - 4, f"Δ{dfc:.2f}")

    def _goal_tree(self, p, stack, x, y, w) -> int:
        """v7: indented goal-stack tree with a per-node completion-progress bar."""
        if not stack:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 10, "(no active goals)"); return y + 14
        for i, g in enumerate(stack[:8]):
            did = g.get("drive_id", "?")
            tnorm = g.get("target_norm")
            pri = g.get("priority")
            comp = bool(g.get("completed", False))
            depth = int(g.get("depth", 0) or 0)
            prog = float(g.get("progress", 1.0 if comp else 0.0) or 0.0)
            indent = depth * 14
            col = QtGui.QColor(46, 204, 113) if comp else ACCENT
            # v8 B7: connector/label line-weight ∝ goal priority (drive dominance proxy)
            pri_f = float(pri) if isinstance(pri, (int, float)) else 1.0
            lw = max(1, min(4, int(1 + pri_f)))
            # tree connector
            p.setPen(QtGui.QPen(DIM_COL, 1)); p.setFont(_F_AXIS)
            if depth > 0:
                p.drawText(x + indent - 10, y + 10, "└")
            p.setPen(QtGui.QPen(col, lw)); p.setFont(_F_LABEL_B)
            tgt_s = f"|tgt|={float(tnorm):.2f}" if tnorm is not None else "no tgt"
            p.drawText(x + indent, y + 10, f"#{i+1} d{did} {tgt_s}")
            # progress bar
            bx = x + indent + 150; bw = max(20, w - indent - 210)
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRect(bx, y + 3, bw, 8)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(bx, y + 3, int(bw * np.clip(prog, 0, 1)), 8, col)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            tag = "✓done" if comp else "active"
            p.drawText(bx + bw + 4, y + 10, f"{tag} p{pri}")
            y += 18
        return y

    def _goal_history_strip(self, p, x, y, w, h) -> None:
        """v7: goal_history as a coloured step-strip (which drive held the goal
        over time). Wires up the previously-undrawn goal_hist deque."""
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x, y, "goal history (which drive held the goal)")
        gh = list(self.goal_hist)
        if not gh:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 16, "(collecting…)"); return
        n = len(gh); cw = w / max(n, 1)
        sy = y + 4; sh = h - 4
        for i, did in enumerate(gh):
            did = max(1, int(did))
            col = _drive_color(did)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(int(x + i * cw), sy, max(int(cw), 1), sh, col)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + h + 10, f"{n} cycles · colour = drive id")

    def _drive_goals_inset(self, p, f, x, y, w, h) -> None:
        """v7: radial 'where each drive pulls' — spoke length = ||drive_goals[i]||
        (magnitude of each drive's pull on the goal vector)."""
        import math
        dgs = getattr(f, "drive_goals", None) or []
        if not dgs and self._replay:
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 3, y + 11, "drive_goals unavailable (replay)")
            return
        nd = max(len(dgs), _n_drives(f))
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + 11, "drive pull ‖·‖")
        cx, cy = x + w // 2, y + h // 2 + 6
        R = min(w, h) // 2 - 16
        mags = []
        for i in range(nd):
            dg = dgs[i] if i < len(dgs) else None
            m = float(np.linalg.norm(np.asarray(dg, dtype=np.float32))) if dg is not None else 0.0
            mags.append(m)
        mx = max(mags) if mags else 1.0
        mx = mx if mx > 1e-6 else 1.0
        for i in range(nd):
            ang = -math.pi / 2 + i * 2 * math.pi / max(nd, 1)
            r = (mags[i] / mx) * R
            ex = int(cx + r * math.cos(ang)); ey = int(cy + r * math.sin(ang))
            col = _drive_color(i + 1)
            p.setPen(QtGui.QPen(col, 2)); p.drawLine(cx, cy, ex, ey)
            p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
            p.drawEllipse(ex - 2, ey - 2, 4, 4)

    def _heatmap(self, p, x, y, w, h) -> None:
        hist = list(self.drive_hist)
        if not hist:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 10, "drive×cycle deficit heatmap (collecting…)"); return
        n = len(hist); nd = max((len(d) for d in hist), default=6)
        cw = w / max(n, 1); rh = h / max(nd, 1)
        for i, defs in enumerate(hist):
            for j in range(min(nd, len(defs))):
                a = int(np.clip(defs[j], 0, 1) * 230)
                if a < 8:
                    continue
                col = _drive_color(j + 1); col.setAlpha(a)
                p.fillRect(int(x + i * cw), int(y + j * rh), int(cw) + 1, int(rh) - 1, col)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + h + 8, f"{nd} drives × {n} cycles · more opaque = higher deficit")
        # v7: labeled colorbar (integer coords — fillRect rejects floats)
        cbw = 6; cbh = int(h); cbx = int(x + w - cbw - 2); cby = int(y)
        steps = 12
        for k in range(steps):
            a = int((steps - 1 - k) / (steps - 1) * 230)
            c = QtGui.QColor(200, 200, 210, a)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(c)
            p.fillRect(cbx, cby + k * cbh // steps, cbw, cbh // steps + 1, c)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(cbx - 16, cby + 8, "hi")
        p.drawText(cbx - 16, cby + cbh - 2, "lo")

    def _trend_pair(self, p, x, y, w, h) -> None:
        half = h // 2
        self._mini_trend(p, self.temp_hist, x, y, w, half, QtGui.QColor(231, 126, 34), "CR temperature", self._temp_scale)
        self._mini_trend(p, self.emp_hist, x, y + half + 4, w, half, QtGui.QColor(52, 152, 219), "empowerment", self._emp_scale)

    def _mini_trend(self, p, s, x, y, w, h, col, label, scale: Optional[ScaleState] = None) -> None:
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x, y + 10, label)
        top, bot = y + 16, y + h - 2
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, bot, x + w, bot)
        vals = list(s)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, top + 12, "collecting…"); return
        lo, hi = min(vals), max(vals)
        # v7: snap y-scale to a ScaleState (EMA of recent max) — no per-frame jumps
        if scale is not None:
            lo, hi = scale.update(float(lo), float(hi))
        if hi - lo < 1e-9: hi = lo + 1
        n = len(vals); path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + i * w / (n - 1); py = bot - (v - lo) / (hi - lo) * (bot - top - 2)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.setPen(QtGui.QPen(col, 2)); p.drawPath(path)
