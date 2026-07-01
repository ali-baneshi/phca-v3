"""PHCA v3.0 — Cognitive Observatory (Observability v3.1, PyQt5).

Multi-tab dashboard that visualizes the cognition as it happens. Reuses the
v2 data layer (``ObservabilityFrame``/``ObservabilityStore``/``SessionRecorder``)
unchanged; only the rendering is a real Qt GUI. The same ``DashboardController``
drives both the live launcher and the ``--qt`` replay.

v3.1 hardening (this revision): fixed the error-scale clipping, the invisible
flow heatmap, the blank Action tab on D5/explore, drive-label overlap, missing
arrowheads, and the MuJoCo phase-space placeholder; added shared chart infra
(``_ChartCanvas``), a QSS dark theme, per-node sparklines + animated data
packet on the flow graph, a 6-drive radar, predicted-next-cell ghost, retention
cap-engagement / event markers / leak-rate, and a replay scrubber.

Tabs:
  1. Overview         — world (grid + G′ heatmap + predicted-cell ghost + trail)
                        + drives (value vs target, active highlighted) + adaptive
                        error/confidence trend + attention + status (gauges).
  2. Cognitive Flow   — animated ASI→M2→G′→PE→PEU→TSPL→Action→RBTA pipeline;
                        per-node ms + sparkline, cost-colored, radius-by-cost,
                        arrowheads, travelling data packet, red-on-violation,
                        module×cycle cost heatmap (rolling-max scaled), legend.
  3. Action Selection — candidate-score bars (chosen annotated w/ action name),
                        D5/explore state banner, ε-greedy decay, explore/exploit
                        dots, continuous-action torque dial.
  4. Phase Space      — GridWorld trajectory + G′ error map; 6-drive radar
                        portrait; MuJoCo per-dim predicted-vs-actual error bars.
  5. Retention        — M3/M4/RSS/latency with current values + cap-engagement
                        % + prune/VACUUM markers + latency histogram + leak-rate;
                        scrolling RBTA violation table.

All polling-side: read-only on the frame; the cycle thread never touches Qt.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .observability import ObservabilityFrame
from .render import _grid_base, _prediction_heatmap

from PyQt5 import QtWidgets, QtCore, QtGui

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
TREND_WINDOW = 200
PIPELINE = ["sanitize", "memory_write", "prediction", "peu", "tspl",
            "action_selection", "rbta"]
SIDE_MODULES = ["gprime_learn", "mdim", "attn", "hpm", "cr", "consolidation"]
PIPELINE_LABEL = {
    "sanitize": "ASI", "memory_write": "M2", "prediction": "G′",
    "peu": "PEU", "tspl": "TSPL", "action_selection": "Act", "rbta": "RBTA",
    "gprime_learn": "G′lrn", "mdim": "MDIM", "attn": "Att", "hpm": "HPM",
    "cr": "CR", "consolidation": "Cons",
}
# Verified via env.get_action_names(): ['MOVE_N','MOVE_S','MOVE_E','MOVE_W','STAY']
GRID_ACTIONS = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY"]

# RBTA module_id codes (e.g. "ACTION", "G'", "WM") -> flow-graph module keys.
RBTA_TO_FLOW = {
    "ASI": "sanitize", "WM": "memory_write", "G'": "prediction", "PEU": "peu",
    "TSPL": "tspl", "ACTION": "action_selection", "RBTA": "rbta",
    "G'LEARN": "gprime_learn", "GPRIME_LEARN": "gprime_learn", "MDIM": "mdim",
    "ATTN": "attn", "HPM": "hpm", "CR": "cr",
    "CONSOLID": "consolidation", "CONS": "consolidation",
}

PANEL_BG = QtGui.QColor(18, 18, 24)
GRID_COL = QtGui.QColor(60, 60, 70)
TEXT_COL = QtGui.QColor(210, 210, 220)
DIM_COL = QtGui.QColor(150, 150, 160)
ACCENT = QtGui.QColor(241, 196, 15)


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


# ----- base canvas + shared chart infra --------------------------------------

class _BaseCanvas(QtWidgets.QWidget):
    """Base for QPainter canvases — stores state, repaints on update()."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 220)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, PANEL_BG)
        self.setPalette(pal)

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), PANEL_BG)
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

    def _title(self, p: QtGui.QPainter, text: str, y: int = 15) -> None:
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Bold))
        p.drawText(10, y, text)
        p.setFont(QtGui.QFont("Sans", 9))


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
        self._series.append(dict(name=name, color=color, data=data, log=log, fill=fill))

    @staticmethod
    def _norm(vals: List[float], log: bool) -> Tuple[List[float], float, float]:
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return [], 0.0, 1.0
        if log:
            vals = [v if v > 0 else 1e-6 for v in vals]
            lo, hi = min(vals), max(vals)
            if hi <= lo:
                hi = lo * 2 if lo > 0 else 1.0
            return [float(np.log(v)) for v in vals], float(np.log(lo)), float(np.log(hi))
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        return vals, lo, hi

    def _plot_rect(self) -> Tuple[int, int, int, int]:
        ml, mr, mt, mb = 52, 16, 26, 30
        return ml, mt, self.width() - mr, self.height() - mb

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
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
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(4, py1 + 6, self.xlabel)
        any_data = False
        for s in self._series:
            vals = list(s["data"])
            if not vals:
                continue
            any_data = True
            nv, lo, hi = self._norm(vals, s["log"])
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
        p.setFont(QtGui.QFont("Sans", 8))
        for s in self._series:
            vals = list(s["data"])
            col = _to_qcolor(s["color"])
            p.setPen(QtGui.QPen(col, 2)); p.drawLine(x - 14, y - 4, x - 4, y - 4)
            if vals:
                _, lo, hi = self._norm(vals, s["log"])
                cur = vals[-1]
                tag = f"{s['name']}={cur:.3g} [{lo:.2g}..{hi:.2g}]"
            else:
                tag = f"{s['name']}=—"
            p.setPen(TEXT_COL)
            rect = p.boundingRect(QtCore.QRect(0, 0, 400, 14), 0, tag)
            p.drawText(x - 18 - rect.width(), y, tag)
            y += 14


# ----- Overview tab canvases -------------------------------------------------

class WorldCanvas(_BaseCanvas):
    """GridWorld grid + G′ heatmap + predicted-cell ghost + trail + agent,
    or MuJoCo predicted-vs-goal-ref bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.trail: Deque[Any] = deque(maxlen=160)

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if f.grid is not None and f.agent_pos is not None:
            ap = tuple(f.agent_pos)
            if not self.trail or self.trail[-1] != ap:
                self.trail.append(ap)
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            self._empty(p, "Awaiting cycle…"); return
        w, h = self.width(), self.height()
        if f.grid is not None:
            g = f.grid
            n = g.shape[0]
            cell = min(w - 20, h - 40) / n
            ox = (w - cell * n) / 2
            oy = 28
            heat = _prediction_heatmap(f.predicted_state, n)
            pred_cell = None
            if heat is not None:
                pred_cell = np.unravel_index(int(np.argmax(heat)), heat.shape)
            # cells
            for r in range(n):
                for c in range(n):
                    v = g[r, c]
                    col = QtGui.QColor(40, 40, 50)
                    if v == 1:
                        col = QtGui.QColor(90, 90, 100)
                    elif v == 2:
                        col = QtGui.QColor(39, 174, 96)
                    p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell), col)
            # G′ heatmap
            if heat is not None:
                for r in range(n):
                    for c in range(n):
                        a = float(np.clip(heat[r, c], 0, 1))
                        if a <= 0.03:
                            continue
                        p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell),
                                   QtGui.QColor(231, 76, 60, int(150 * a)))
            # predicted-next-cell ghost (hollow amber ring)
            if pred_cell is not None:
                gr, gc = int(pred_cell[0]), int(pred_cell[1])
                p.setBrush(QtGui.QColor(241, 196, 15, 40))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 2))
                p.drawEllipse(int(ox + gc * cell + cell * 0.18),
                              int(oy + gr * cell + cell * 0.18),
                              int(cell * 0.64), int(cell * 0.64))
            # goal marker outline
            if f.goal_pos is not None:
                gr, gc = f.goal_pos
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                p.drawRect(int(ox + gc * cell), int(oy + gr * cell), int(cell), int(cell))
            # trail
            tl = list(self.trail)
            for i in range(1, len(tl)):
                (r0, c0), (r1, c1) = tl[i - 1], tl[i]
                a = int(60 + 195 * i / max(len(tl), 1))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
                p.drawLine(int(ox + c0 * cell + cell / 2), int(oy + r0 * cell + cell / 2),
                           int(ox + c1 * cell + cell / 2), int(oy + r1 * cell + cell / 2))
            # agent
            if tl:
                r, c = tl[-1]
                p.setBrush(QtGui.QColor(241, 196, 15))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(int(ox + c * cell + cell * 0.25),
                              int(oy + r * cell + cell * 0.25),
                              int(cell * 0.5), int(cell * 0.5))
            self._title(p, f"GridWorld {n}×{n}  conf={f.prediction_confidence:.2f}  "
                           f"err={f.prediction_error:.2f}  (amber ghost = G′ predicted next cell)")
        else:
            pred = f.predicted_state
            ref = f.goal_ref
            if pred is None:
                self._empty(p, "MuJoCo — no prediction"); return
            n = min(len(pred), 12)
            ref = ref[:n] if ref is not None else None
            bw = (w - 40) / n
            self._title(p, f"MuJoCo predicted (red) vs goal-ref (green)  dims={n}")
            midy = h // 2 + 10
            for i in range(n):
                x = 20 + i * bw
                pv = float(np.clip(pred[i], -2, 2))
                p.fillRect(int(x), int(midy - abs(pv) * 28), int(bw - 4), int(abs(pv) * 56),
                           QtGui.QColor(231, 76, 60, 180))
                if ref is not None:
                    rv = float(np.clip(ref[i], -2, 2))
                    _arrow(p, int(x + bw / 2), int(midy),
                           int(x + bw / 2), int(midy - rv * 60), QtGui.QColor(39, 174, 96), size=6)
            p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1))
            p.drawLine(20, midy, w - 20, midy)


class DrivesCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None or not f.drive_levels:
            self._empty(p, "Drives…"); return
        w, h = self.width(), self.height()
        n = len(f.drive_levels)
        left, right, top, bot = 44, w - 12, 44, h - 34
        # y-axis ticks 0/0.5/1
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        for tv, lbl in ((0.0, "0"), (0.5, ".5"), (1.0, "1")):
            y = int(bot - tv * (bot - top))
            p.drawLine(left, y, right, y)
            p.drawText(6, y + 3, lbl)
        barw = (right - left) / n
        for i in range(n):
            did = i + 1
            v = float(np.clip(f.drive_levels[i], 0, 1))
            t = float(np.clip(f.drive_targets[i], 0, 1)) if i < len(f.drive_targets) else v
            x = left + i * barw
            active = (f.active_drive_id == did)
            # track
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1))
            p.setBrush(QtGui.QColor(30, 30, 38))
            p.drawRect(int(x + 4), int(top), int(barw - 10), int(bot - top))
            # value bar
            col = _to_qcolor(DRIVE_COLORS[did])
            p.setBrush(col)
            p.setPen(QtGui.QPen(col.darker(140), 1))
            bh = int(v * (bot - top))
            p.drawRect(int(x + 4), int(bot - bh), int(barw - 10), bh)
            # target marker
            ty = bot - t * (bot - top)
            p.setPen(QtGui.QPen(ACCENT, 2))
            p.drawLine(int(x + 2), int(ty), int(x + barw - 6), int(ty))
            # label (abbreviated)
            lbl = DRIVE_SHORT[did]
            p.setPen(ACCENT if active else TEXT_COL)
            if active:
                p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(int(x + 4), h - 16, lbl)
            p.setFont(QtGui.QFont("Sans", 8))
            if active:
                p.setPen(TEXT_COL)
                p.drawText(int(x + 4), h - 4, DRIVE_NAMES[did].split()[1])
        self._title(p, f"Drives  value vs ◆target  active={DRIVE_SHORT.get(f.active_drive_id,'?')}")
        # color legend
        p.setFont(QtGui.QFont("Sans", 7))
        lx = left + 4
        for did in (1, 3, 5):
            p.setPen(_to_qcolor(DRIVE_COLORS[did])); p.drawLine(lx, 30, lx + 10, 30)
            p.setPen(DIM_COL); p.drawText(lx + 13, 33, DRIVE_NAMES[did])
            lx += 92


class TrendCanvas(_ChartCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.err: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.conf: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.add_series("err", "#e74c3c", self.err, log=True, fill=True)
        self.add_series("conf", "#2ecc71", self.conf, log=False, fill=False)

    def push(self, f: ObservabilityFrame) -> None:
        self.err.append(float(f.prediction_error))
        self.conf.append(float(f.prediction_confidence))
        self.title = (f"Prediction error (red, log) & confidence (green)  "
                      f"err={f.prediction_error:.2f}  conf={f.prediction_confidence:.3f}")
        self.update()


class AttentionCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None or not f.attention_indices:
            self._empty(p, "Attention…"); return
        self._title(p, "Attention focus (attended chunk ids, by salience)")
        w, h = self.width(), self.height()
        n = len(f.attention_indices)
        top, bot = 34, h - 24
        bw = (w - 40) / n
        sal_max = max((float(s) for s in f.attention_saliences), default=1.0) or 1.0
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        for tv in (0.0, 0.5, 1.0):
            y = int(bot - tv * (bot - top))
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1)); p.drawLine(20, y, w - 20, y)
        for i in range(n):
            x = 20 + i * bw
            raw = float(f.attention_saliences[i]) if i < len(f.attention_saliences) else 0.0
            sal = float(np.clip(raw / sal_max, 0, 1))
            bh = int(sal * (bot - top))
            p.setBrush(QtGui.QColor(155, 89, 182, 210))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 1))
            p.drawRect(int(x), int(bot - bh), int(bw - 8), bh)
            p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(int(x), h - 8, f"#{int(f.attention_indices[i])}")
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(w - 90, 16, f"sal max={sal_max:.2g}")


class StatusPanel(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.cycle_error: Optional[str] = None

    def set_state(self, f: Optional[ObservabilityFrame], err: Optional[str]) -> None:
        self.frame = f; self.cycle_error = err; self.update()

    def _gauge(self, p: QtGui.QPainter, x: int, y: int, label: str,
               val: float, vmax: float, unit: str, col: QtGui.QColor) -> int:
        gw = 150; gh = 8
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Monospace", 9))
        p.drawText(x, y, f"{label} {val:.1f}{unit}")
        frac = float(np.clip(val / vmax, 0, 1)) if vmax > 0 else 0.0
        p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1))
        p.setBrush(QtGui.QColor(30, 30, 38))
        p.drawRect(x, y + 4, gw, gh)
        p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
        p.fillRect(x, y + 4, int(gw * frac), gh, col)
        return gw + 12

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None and self.cycle_error is None:
            self._empty(p, "Status…"); return
        if self.cycle_error:
            p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Monospace", 10, QtGui.QFont.Bold))
            p.drawText(self.rect(), 0x84, f"CYCLE ERROR:\n{self.cycle_error}")
            return
        p.setFont(QtGui.QFont("Monospace", 9))
        y = 20
        rbta_ok = (f.violations_count == 0)
        p.setPen(QtGui.QColor(46, 204, 113) if rbta_ok else QtGui.QColor(231, 76, 60))
        p.drawText(10, y, f"Cycle {f.cycle_id}   RBTA: {'OK' if rbta_ok else str(f.violations_count)+' VIOLATIONS'}")
        y += 15
        p.setPen(TEXT_COL)
        for v in f.rbta_violations[:3]:
            mid = v.get("module_id", v.get("module", "?"))
            p.drawText(20, y, f"↳ {mid}/{v.get('bound_type','?')}: "
                              f"{v.get('measured','?'):.4g} > {v.get('allowed','?'):.4g}")
            y += 14
        # Retention caps with engagement
        m3e = f.episode_count / f.m3_cap if f.m3_cap else 0.0
        m4e = f.fact_count / f.m4_cap if f.m4_cap else 0.0
        p.drawText(10, y, f"M3 {f.episode_count}/{f.m3_cap} ({m3e*100:.0f}%)   "
                          f"M4 {f.fact_count}/{f.m4_cap} ({m4e*100:.0f}%)  prune→{f.m4_prune_target}")
        y += 16
        # gauges
        x = self._gauge(p, 10, y, "RSS", f.rss_bytes / 1e6, 1024.0, "MB", QtGui.QColor(230, 126, 34))
        self._gauge(p, x, y, "lat", f.latency_ms, 50.0, "ms", QtGui.QColor(46, 204, 113))
        y += 24
        # drives: MDIM goal vs action goal (can differ — display honestly)
        ad = f.active_drive_id
        p.setPen(ACCENT)
        p.drawText(10, y, f"MDIM goal: {DRIVE_NAMES.get(ad, ad)}")
        y += 14
        r = f.action_rationale or {}
        gid = r.get("goal_id")
        goal_lbl = DRIVE_NAMES.get(gid, gid) if gid is not None else "—"
        p.setPen(TEXT_COL)
        tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
        note = r.get("note", "")
        score = r.get("best_score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        k = r.get("k_candidates")
        k_s = str(k) if k is not None else "—"
        p.drawText(10, y, f"Action goal: {goal_lbl}   {tag}  ε={r.get('eps',0):.3f}  K={k_s}  score={score_s}"
                          + (f"   [{note}]" if note else ""))


# ----- Cognitive Flow tab ----------------------------------------------------

class CognitiveFlowView(_BaseCanvas):
    """Animated directed pipeline graph: per-node ms + sparkline, cost-colored,
    radius-by-cost, arrowheads, travelling data packet, red-on-violation,
    module×cycle cost heatmap (rolling-max scaled), legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.heat: Deque[Dict[str, float]] = deque(maxlen=80)
        self.viol_mods: set = set()
        self.last_viol: Dict[str, str] = {}
        self.cycle_id: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.cycle_id = int(f.cycle_id)
        self.heat.append(dict(f.module_timings))
        self.viol_mods = {RBTA_TO_FLOW.get(v.get("module_id", v.get("module", "")),
                                          v.get("module_id", v.get("module", "")))
                          for v in f.rbta_violations}
        self.last_viol = {RBTA_TO_FLOW.get(v.get("module_id", v.get("module", "")),
                                           v.get("module_id", v.get("module", ""))):
                          f"{v.get('bound_type','?')} {v.get('measured','?'):.4g}>{v.get('allowed','?'):.4g}"
                          for v in f.rbta_violations}
        self.update()

    def _node_pos(self, w: int, h: int) -> Dict[str, Tuple[int, int]]:
        n = len(PIPELINE); ns = len(SIDE_MODULES)
        band_y = h // 3 + 10
        side_y = 2 * h // 3 + 10
        pos = {}
        for i, mod in enumerate(PIPELINE):
            pos[mod] = (int((i + 0.5) * w / n), band_y)
        for j, mod in enumerate(SIDE_MODULES):
            pos[mod] = (int((j + 0.5) * w / ns), side_y)
        return pos

    def _spark(self, p: QtGui.QPainter, mod: str, x: int, y: int, w: int = 50) -> None:
        vals = [float(mt.get(mod, 0.0)) for mt in self.heat]
        if not vals:
            return
        mx = max(vals) if vals else 1.0
        mx = mx if mx > 1e-6 else 1.0
        col = _to_qcolor(_cost_color(max(vals)))
        p.setPen(QtGui.QPen(col, 1))
        n = len(vals)
        for i in range(1, n):
            x0 = x + (i - 1) * w / max(n - 1, 1)
            x1 = x + i * w / max(n - 1, 1)
            y0 = y + 16 - (vals[i - 1] / mx) * 14
            y1 = y + 16 - (vals[i] / mx) * 14
            p.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Cognitive flow…"); return
        pos = self._node_pos(w, h)
        n = len(PIPELINE)
        # main chain edges with arrowheads + travelling packet
        active_idx = self.cycle_id % n
        for i in range(n - 1):
            a, b = pos[PIPELINE[i]], pos[PIPELINE[i + 1]]
            col = QtGui.QColor(241, 196, 15, 230) if i == active_idx else QtGui.QColor(120, 120, 140, 150)
            _arrow(p, a[0], a[1], b[0], b[1], col, size=10)
        # travelling data packet on the active edge
        if 0 <= active_idx < n - 1:
            a, b = pos[PIPELINE[active_idx]], pos[PIPELINE[active_idx + 1]]
            t = (self.cycle_id % 4) / 4.0
            px = int(a[0] + t * (b[0] - a[0])); py = int(a[1] + t * (b[1] - a[1]))
            p.setBrush(QtGui.QColor(255, 255, 255)); p.setPen(QtGui.QPen(ACCENT, 1))
            p.drawEllipse(px - 4, py - 4, 8, 8)
        # side-module routed (elbow) connectors — anchor under the main node, drop, then horizontal
        anchor_x = {m: pos[m][0] for m in PIPELINE}
        for sm in SIDE_MODULES:
            a = pos[sm]
            anchor = "prediction" if sm in ("gprime_learn", "attn") else "memory_write"
            b = pos[anchor]
            midx = (a[0] + b[0]) // 2
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 110), 1, QtCore.Qt.DashLine))
            p.drawLine(b[0], b[1] + 26, b[0], a[1] - 30)
            p.drawLine(b[0], a[1] - 30, a[0], a[1] - 30)
            p.drawLine(a[0], a[1] - 30, a[0], a[1])
        # nodes
        for mod in PIPELINE + SIDE_MODULES:
            x, y = pos[mod]
            ms = float(f.module_timings.get(mod, 0.0))
            viol = mod in self.viol_mods
            col = QtGui.QColor(231, 76, 60) if viol else _to_qcolor(_cost_color(ms))
            # radius scales with cost (clamped)
            r = int(20 + min(ms / 20.0, 1.0) * 10)
            p.setBrush(col); p.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 2))
            p.drawEllipse(x - r, y - r, r * 2, r * 2)
            p.setPen(QtGui.QColor(20, 20, 24))
            p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(x - 26, y - 3, 52, 12, 0x84, PIPELINE_LABEL.get(mod, mod))
            p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(x - 22, y + 9, 44, 11, 0x84, f"{ms:.1f}ms")
            # sparkline under node
            self._spark(p, mod, x - 25, y + r - 2, w=50)
            # violation bound below
            if viol and mod in self.last_viol:
                p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Sans", 7))
                p.drawText(x - 30, y + r + 16, 60, 10, 0x84, self.last_viol[mod])
        self._heatmap(p)
        # legend
        self._legend(p)

    def _legend(self, p: QtGui.QPainter) -> None:
        p.setFont(QtGui.QFont("Sans", 8))
        x = 12; y = self.height() - 50
        p.setPen(TEXT_COL); p.drawText(x, y, "Legend:")
        x += 52
        for lbl, col in (("solid=main pipeline", "#f1c40f"), ("dashed=side coupling", "#9b59b6"),
                         ("green<5ms", "#2ca02c"), ("orange<20ms", "#ff7f0e"), ("red≥20ms/viol", "#e74c3c")):
            p.setPen(_to_qcolor(col)); p.drawLine(x, y - 4, x + 14, y - 4)
            p.setPen(TEXT_COL); p.drawText(x + 18, y, lbl)
            x += 16 + p.fontMetrics().boundingRect(lbl).width() + 10

    def _heatmap(self, p: QtGui.QPainter) -> None:
        if not self.heat:
            return
        mods = PIPELINE + SIDE_MODULES
        h = self.height(); strip_h = 26; top = h - strip_h - 6
        # rolling max for adaptive scale (fixed 25ms was the bug — sub-5ms was invisible)
        rmax = max((max((mt.get(m, 0.0) for m in mods), default=0.0) for mt in self.heat), default=1.0)
        rmax = rmax if rmax > 1e-6 else 25.0
        cw = (self.width() - 20) / max(len(self.heat), 1)
        for i, mts in enumerate(self.heat):
            for j, mod in enumerate(mods):
                ms = float(mts.get(mod, 0.0))
                a = int(np.clip(ms / rmax, 0, 1) * 230)
                if a < 6:
                    continue
                p.fillRect(int(10 + i * cw), int(top + j * (strip_h / len(mods))),
                           int(cw) + 1, int(strip_h / len(mods)) - 1,
                           QtGui.QColor(231, 76, 60, a))
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(10, top - 2, f"module×cycle cost heatmap (rolling-max={rmax:.1f}ms)")


# ----- Action Selection tab --------------------------------------------------

class CandidateScoreView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.last_scores: List[float] = []
        self.last_chosen: int = -1
        self.last_continuous: Optional[np.ndarray] = None
        self.eps_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.ee_hist: Deque[bool] = deque(maxlen=TREND_WINDOW)
        self.score_hist: Deque[float] = deque(maxlen=TREND_WINDOW)

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        r = f.action_rationale or {}
        if r:
            self.eps_hist.append(float(r.get("eps", 0.0)))
            self.ee_hist.append(bool(r.get("explored", False)))
            bs = r.get("best_score")
            if isinstance(bs, (int, float)):
                self.score_hist.append(float(bs))
        if f.candidate_scores:  # keep last non-empty for fallback context
            self.last_scores = [float(x) for x in f.candidate_scores]
            self.last_chosen = int(np.argmax(self.last_scores))
        if f.continuous_action is not None:
            self.last_continuous = f.continuous_action
        self.update()

    def _bars(self, p: QtGui.QPainter, scores: List[float], chosen: int,
              faded: bool, top: int, bot: int, w: int) -> None:
        n = len(scores)
        bw = (w - 40) / n
        mx = max(scores) if scores else 1.0
        mx = mx if mx > 1e-6 else 1.0
        for i, s in enumerate(scores):
            x = 20 + i * bw
            bh = int(max(s, 0.0) / mx * (bot - top))
            if faded:
                col = QtGui.QColor(241, 196, 15, 70) if i == chosen else QtGui.QColor(52, 152, 219, 60)
            else:
                col = QtGui.QColor(241, 196, 15) if i == chosen else QtGui.QColor(52, 152, 219, 200)
            p.setBrush(col); p.setPen(QtGui.QPen(col.darker(140), 1))
            p.drawRect(int(x), int(bot - bh), int(bw - 8), bh)
            # action name label (gridworld)
            lbl = GRID_ACTIONS[i] if i < len(GRID_ACTIONS) else str(i)
            p.setPen(TEXT_COL if i == chosen else DIM_COL)
            p.setFont(QtGui.QFont("Sans", 7, QtGui.QFont.Bold if i == chosen else 0))
            p.drawText(int(x), bot + 12, lbl)
        # chosen annotation
        if 0 <= chosen < n:
            x = 20 + chosen * bw
            p.setPen(ACCENT); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            name = GRID_ACTIONS[chosen] if chosen < len(GRID_ACTIONS) else f"a{chosen}"
            p.drawText(int(x), bot + 24, f"←{name}")

    def _torque_dial(self, p: QtGui.QPainter, cx: int, cy: int, R: int, vec: np.ndarray) -> None:
        import math
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 70), 1))
        p.setBrush(QtGui.QColor(24, 24, 30))
        p.drawEllipse(cx - R, cy - R, R * 2, R * 2)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 80, 90), 1))
        p.drawLine(cx - R, cy, cx + R, cy); p.drawLine(cx, cy - R, cx, cy + R)
        n = min(len(vec), 2)
        if n == 2:
            vx, vy = float(np.clip(vec[0], -1, 1)), float(np.clip(vec[1], -1, 1))
            ex = cx + vx * R; ey = cy + vy * R
            _arrow(p, cx, cy, int(ex), int(ey), QtGui.QColor(155, 89, 182), size=8)
            p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(cx - R, cy + R + 14, 2 * R, 14, 0x84, f"τ=[{vx:+.2f},{vy:+.2f}]")
        else:
            self._signed_bars(p, vec, cx - R, cy - R // 2, 2 * R, R, horizontal=False)

    def _signed_bars(self, p: QtGui.QPainter, vec: np.ndarray, x: int, y: int,
                     w: int, h: int, horizontal: bool = True) -> None:
        n = min(len(vec), 16)
        if horizontal:
            bw = w / n; mid = y + h // 2
            for i in range(n):
                v = float(np.clip(vec[i], -1, 1))
                bx = x + i * bw
                col = QtGui.QColor(155, 89, 182) if v >= 0 else QtGui.QColor(231, 76, 60)
                p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
                p.fillRect(int(bx + bw / 2 - 3), int(mid - abs(v) * h / 2), 6, int(abs(v) * h), col)
            p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1)); p.drawLine(x, mid, x + w, mid)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            self._empty(p, "Action selection…"); return
        w, h = self.width(), self.height()
        r = f.action_rationale or {}
        scores = [float(x) for x in f.candidate_scores]
        chosen = int(np.argmax(scores)) if scores else -1
        is_continuous = bool(r.get("continuous", f.continuous_action is not None))
        goal_id = r.get("goal_id")
        goal_lbl = DRIVE_NAMES.get(goal_id, goal_id) if goal_id is not None else "—"
        explored = bool(r.get("explored"))
        note = r.get("note", "")
        # header
        hdr = f"goal={goal_lbl}  {'EXPLORE' if explored else 'EXPLOIT'}  ε={r.get('eps',0):.3f}"
        bs = r.get("best_score")
        if isinstance(bs, (int, float)):
            hdr += f"  score={bs:.3f}"
        self._title(p, hdr)
        # candidate bars region
        bar_top, bar_bot = 36, h // 2 - 10
        if scores:
            self._bars(p, scores, chosen, faded=False, top=bar_top, bot=bar_bot, w=w)
            # faded last-scores context line
            if self.last_scores and len(self.last_scores) == len(scores) and self.last_chosen != chosen:
                pass
        else:
            # D5/explore banner — no candidates this cycle
            tag = "EXPLORE (random action)" if explored else (note or "D5 ENERGY → STAY (no candidates)")
            p.setPen(QtGui.QColor(231, 76, 60) if not explored else QtGui.QColor(52, 152, 219))
            p.setFont(QtGui.QFont("Sans", 12, QtGui.QFont.Bold))
            p.drawText(20, bar_top + 20, f"● {tag}")
            p.setFont(QtGui.QFont("Sans", 9)); p.setPen(DIM_COL)
            p.drawText(20, bar_top + 42, "(candidate scores are not computed for this branch)")
            # faded last non-empty scores for context
            if self.last_scores:
                p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 8))
                p.drawText(20, bar_top + 60, f"last non-empty scores (faded):")
                self._bars(p, self.last_scores, self.last_chosen, faded=True,
                           top=bar_top + 70, bot=bar_bot, w=w)
        # continuous action: dial for 2D, signed bars otherwise
        if is_continuous and self.last_continuous is not None:
            if len(self.last_continuous) <= 2:
                self._torque_dial(p, w // 4, bar_bot + 60, 48, self.last_continuous)
                p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
                p.drawText(w // 4 - 60, bar_bot + 130, "continuous torque dial")
            else:
                self._signed_bars(p, self.last_continuous, 20, bar_bot + 30, w - 40, 60, horizontal=True)
                p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
                p.drawText(20, bar_bot + 100, "continuous action vector (signed)")
        # ε-decay + explore/exploit dots
        bot_y = h - 46
        if self.eps_hist:
            n = len(self.eps_hist)
            left, right = 20, w - 20
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(left, bot_y - 16, "ε-greedy decay (blue) + explore(red)/exploit(green) dots")
            p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
            for i in range(1, n):
                x0 = left + (i - 1) * (right - left) / max(n - 1, 1)
                x1 = left + i * (right - left) / max(n - 1, 1)
                p.drawLine(int(x0), int(bot_y - self.eps_hist[i - 1] * 30),
                           int(x1), int(bot_y - self.eps_hist[i] * 30))
            for i, e in enumerate(self.ee_hist):
                x = int(left + i * (right - left) / max(n - 1, 1))
                p.setBrush(QtGui.QColor(231, 76, 60) if e else QtGui.QColor(46, 204, 113))
                p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60) if e else QtGui.QColor(46, 204, 113), 1))
                p.drawEllipse(x - 2, bot_y + 6, 4, 4)


# ----- Phase Space tab -------------------------------------------------------

class TrajectoryView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail: Deque[Any] = deque(maxlen=500)
        self.errmap: Optional[np.ndarray] = None
        self.grid_n: int = 0
        self.is_grid: bool = True
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if f.grid is not None and f.agent_pos is not None:
            self.is_grid = True
            self.grid_n = f.grid.shape[0]
            ap = tuple(f.agent_pos)
            if not self.trail or self.trail[-1] != ap:
                self.trail.append(ap)
            heat = _prediction_heatmap(f.predicted_state, self.grid_n)
            if heat is not None and ap is not None:
                actual = np.zeros_like(heat)
                r, c = ap
                if 0 <= r < actual.shape[0] and 0 <= c < actual.shape[1]:
                    actual[r, c] = 1.0
                err = np.abs(heat - actual)
                self.errmap = err if self.errmap is None else 0.9 * self.errmap + 0.1 * err
        else:
            self.is_grid = False
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Phase space…"); return
        if self.is_grid:
            n = self.grid_n or 5
            cell = min(w - 20, h - 40) / n
            ox = (w - cell * n) / 2; oy = 30
            if self.errmap is not None:
                for r in range(n):
                    for c in range(n):
                        a = int(np.clip(self.errmap[r, c], 0, 1) * 200)
                        p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell),
                                   QtGui.QColor(231, 76, 60, a))
            tl = list(self.trail)
            for i in range(1, len(tl)):
                (r0, c0), (r1, c1) = tl[i - 1], tl[i]
                a = int(60 + 195 * i / max(len(tl), 1))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
                p.drawLine(int(ox + c0 * cell + cell / 2), int(oy + r0 * cell + cell / 2),
                           int(ox + c1 * cell + cell / 2), int(oy + r1 * cell + cell / 2))
            if tl:
                r, c = tl[-1]
                p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(int(ox + c * cell + cell * 0.3), int(oy + r * cell + cell * 0.3),
                              int(cell * 0.4), int(cell * 0.4))
            self._title(p, f"GridWorld trajectory (trail={len(tl)}) + G′ |pred−actual| heatmap")
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(8, h - 6, "red = where world model is wrong")
        else:
            # MuJoCo: per-dim predicted-vs-actual error bars
            pred = f.predicted_state
            obs = f.obs_vector
            if pred is None:
                self._empty(p, "MuJoCo phase-space…"); return
            ref = obs if obs is not None else f.goal_ref
            n = min(len(pred), 12)
            bw = (w - 40) / n
            self._title(p, "MuJoCo predicted vs actual (per-dim error)")
            top, bot = 40, h - 30
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
            mx = 0.0
            for i in range(n):
                if ref is not None and i < len(ref):
                    mx = max(mx, abs(float(pred[i]) - float(ref[i])))
            mx = mx if mx > 1e-6 else 1.0
            for i in range(n):
                x = 20 + i * bw
                if ref is not None and i < len(ref):
                    err = abs(float(pred[i]) - float(ref[i]))
                    bh = int(err / mx * (bot - top))
                    p.setBrush(QtGui.QColor(231, 76, 60, 200))
                    p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 1))
                    p.fillRect(int(x), int(bot - bh), int(bw - 6), bh, QtGui.QColor(231, 76, 60, 200))
                p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
                p.drawText(int(x), h - 8, f"d{i}")


class DriveRadarView(_BaseCanvas):
    """6-drive radar/hexagon portrait over the rolling window — far more
    revealing than the old arbitrary D1/D3/D5 scatter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist: Deque[List[float]] = deque(maxlen=200)

    def set_frame(self, f: ObservabilityFrame) -> None:
        if len(f.drive_levels) >= 6:
            self.hist.append([float(f.drive_levels[i]) for i in range(6)])
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 4
        R = min(w, h) // 2 - 40
        import math
        pts = []
        for i in range(6):
            ang = -math.pi / 2 + i * math.pi / 3
            pts.append((cx + R * math.cos(ang), cy + R * math.sin(ang), ang))
        # grid rings
        for ring in (0.25, 0.5, 0.75, 1.0):
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 60), 1))
            poly = QtGui.QPolygonF([QtCore.QPointF(cx + ring * R * math.cos(a),
                                                   cy + ring * R * math.sin(a))
                                    for _, _, a in pts])
            p.drawPolygon(poly)
        # spokes + labels
        for i, (x, y, a) in enumerate(pts):
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 60), 1))
            p.drawLine(cx, cy, int(x), int(y))
            did = i + 1
            lx = cx + (R + 16) * math.cos(a); ly = cy + (R + 16) * math.sin(a)
            p.setPen(_to_qcolor(DRIVE_COLORS[did])); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(int(lx) - 12, int(ly) + 4, DRIVE_SHORT[did])
        hist = list(self.hist)
        if not hist:
            self._empty(p, "Drive radar (collecting…)"); return
        # faded history
        for k, levels in enumerate(hist[:-1]):
            a = int(20 + 60 * k / max(len(hist), 1))
            poly = QtGui.QPolygonF()
            for i, lvl in enumerate(levels):
                ang = -math.pi / 2 + i * math.pi / 3
                poly.append(QtCore.QPointF(cx + lvl * R * math.cos(ang),
                                           cy + lvl * R * math.sin(ang)))
            p.setBrush(QtGui.QColor(241, 196, 15, a // 3))
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 1))
            p.drawPolygon(poly)
        # current polygon (filled, per-drive colored vertices)
        cur = hist[-1]
        poly = QtGui.QPolygonF()
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * math.pi / 3
            poly.append(QtCore.QPointF(cx + lvl * R * math.cos(ang), cy + lvl * R * math.sin(ang)))
        p.setBrush(QtGui.QColor(241, 196, 15, 70))
        p.setPen(QtGui.QPen(ACCENT, 2))
        p.drawPolygon(poly)
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * math.pi / 3
            p.setBrush(_to_qcolor(DRIVE_COLORS[i + 1]))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(int(cx + lvl * R * math.cos(ang)) - 3,
                          int(cy + lvl * R * math.sin(ang)) - 3, 6, 6)
        self._title(p, "6-drive radar (current solid + faded history)")


# ----- Retention tab ---------------------------------------------------------

class RetentionView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m3: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.m4: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.rss: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.lat: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.m3_cap: int = 0; self.m4_cap: int = 0
        self.m3_events: List[int] = []   # cycle indices where count dropped (prune/VACUUM)
        self.m4_events: List[int] = []
        self.cycle_base: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        if not self.m3:
            self.cycle_base = int(f.cycle_id)
        prev_m3 = self.m3[-1] if self.m3 else None
        prev_m4 = self.m4[-1] if self.m4 else None
        self.m3.append(int(f.episode_count)); self.m4.append(int(f.fact_count))
        self.rss.append(float(f.rss_bytes)); self.lat.append(float(f.latency_ms))
        self.m3_cap = int(f.m3_cap); self.m4_cap = int(f.m4_cap)
        if prev_m3 is not None and int(f.episode_count) < prev_m3:
            self.m3_events.append(len(self.m3) - 1)
        if prev_m4 is not None and int(f.fact_count) < prev_m4:
            self.m4_events.append(len(self.m4) - 1)
        self.update()

    def _leak_rate(self) -> float:
        if len(self.rss) < 30:
            return 0.0
        xs = np.arange(30, dtype=float)
        ys = np.array(list(self.rss)[-30:], dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])  # B/cyc
        return slope

    def _panel(self, p: QtGui.QPainter, top: int, bot: int, title: str,
               series: Deque[float], cap: float, col: QtGui.QColor,
               events: List[int], unit: str = "") -> None:
        w = self.width()
        left, right = 60, w - 16
        self._title(p, title, y=top + 12)
        vals = list(series)
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        if vals:
            cur = vals[-1]
            if cap and cap > 0:
                eng = cur / cap * 100 if cap else 0
                vtxt = f"now={cur:.1f}{unit}  cap={cap:.0f}{unit}  eng={eng:.0f}%"
            else:
                vtxt = f"now={cur:.0f}{unit}"
            rect = p.boundingRect(QtCore.QRect(0, 0, 1, 1), 0, vtxt)
            p.drawText(right - rect.width(), top + 12, vtxt)
        lo = min(vals) if vals else 0
        hi = max(vals) if vals else 1
        hi = max(hi, cap) if cap else hi
        if hi - lo < 1e-9:
            hi = lo + 1
        n = len(vals)
        pt0, pt1 = top + 22, bot - 6
        # gridlines
        for g in range(1, 3):
            y = pt0 + g * (pt1 - pt0) // 3
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1)); p.drawLine(left, y, right, y)
        # fill + line
        if n:
            poly = QtGui.QPolygonF([QtCore.QPointF(left, pt1)])
            for i, v in enumerate(vals):
                x = left + i * (right - left) / max(n - 1, 1)
                y = pt1 - (v - lo) / (hi - lo) * (pt1 - pt0)
                poly.append(QtCore.QPointF(x, y))
            poly.append(QtCore.QPointF(left + (n - 1) * (right - left) / max(n - 1, 1), pt1))
            p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 50))
            p.setPen(QtGui.QPen(QtGui.QColor(col.red(), col.green(), col.blue(), 50), 1))
            p.drawPolygon(poly)
            p.setPen(QtGui.QPen(col, 2))
            for i in range(1, n):
                x0 = left + (i - 1) * (right - left) / max(n - 1, 1)
                x1 = left + i * (right - left) / max(n - 1, 1)
                y0 = pt1 - (vals[i - 1] - lo) / (hi - lo) * (pt1 - pt0)
                y1 = pt1 - (vals[i] - lo) / (hi - lo) * (pt1 - pt0)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
            # event markers
            for ev in events:
                if 0 <= ev < n:
                    x = int(left + ev * (right - left) / max(n - 1, 1))
                    p.setPen(QtGui.QPen(ACCENT, 1)); p.drawLine(x, pt0, x, pt1)
        # cap line
        if cap and cap > 0 and hi > 0:
            cy = pt1 - (cap - lo) / (hi - lo) * (pt1 - pt0)
            p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 1, QtCore.Qt.DashLine))
            p.drawLine(left, int(cy), right, int(cy))
            p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(right - 60, int(cy) - 3, "cap")

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        if not self.m3:
            self._empty(p, "Retention…"); return
        # latency histogram is drawn inline in the latency panel; to keep it
        # simple, render 4 stacked panels + a leak-rate header.
        leak = self._leak_rate()
        self._title(p, f"Retention & resources   RSS leak-rate (last 30) ≈ {leak:.1f} B/cyc", y=15)
        ph = (h - 30) // 4
        self._panel(p, 30, 30 + ph, "M3 episodes (memory)", self.m3, float(self.m3_cap),
                    QtGui.QColor(52, 152, 219), self.m3_events)
        self._panel(p, 30 + ph, 30 + 2 * ph, "M4 facts (consolidated)", self.m4, float(self.m4_cap),
                    QtGui.QColor(155, 89, 182), self.m4_events)
        self._panel(p, 30 + 2 * ph, 30 + 3 * ph, "RSS (bytes)", self.rss, 0.0,
                    QtGui.QColor(230, 126, 34), [], unit="")
        # latency as a small histogram in the last panel
        self._latency_panel(p, 30 + 3 * ph, h - 4)

    def _latency_panel(self, p: QtGui.QPainter, top: int, bot: int) -> None:
        w = self.width()
        left, right = 60, w - 16
        self._title(p, "Latency (ms) — histogram", y=top + 12)
        vals = list(self.lat)
        if not vals:
            return
        cur = vals[-1]
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        vtxt = f"now={cur:.1f}ms  max={max(vals):.1f}ms  med={float(np.median(vals)):.1f}ms"
        rect = p.boundingRect(QtCore.QRect(0, 0, 1, 1), 0, vtxt)
        p.drawText(right - rect.width(), top + 12, vtxt)
        # histogram
        lo, hi = 0.0, max(vals) * 1.05 if vals else 1.0
        if hi <= 0:
            hi = 1.0
        nb = 24
        bins = np.linspace(lo, hi, nb + 1)
        counts, _ = np.histogram(vals, bins=bins)
        cm = int(counts.max()) if counts.size else 1
        cm = cm if cm > 0 else 1
        bw = (right - left) / nb
        pt0, pt1 = top + 22, bot - 6
        for i, c in enumerate(counts):
            bh = int(c / cm * (pt1 - pt0))
            p.setBrush(QtGui.QColor(46, 204, 113, 180))
            p.setPen(QtGui.QPen(QtGui.QColor(46, 204, 113), 1))
            p.fillRect(int(left + i * bw), int(pt1 - bh), int(bw) - 1, bh,
                       QtGui.QColor(46, 204, 113, 180))


class ViolationTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["cycle", "module", "bound_type", "measured", "allowed"])
        self.setAlternatingRowColors(True)
        self._seen: int = 0

    def add_frame(self, f: ObservabilityFrame) -> None:
        for v in f.rbta_violations:
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(f.cycle_id)))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(str(v.get("module_id") or v.get("module", ""))))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(v.get("bound_type", ""))))
            self.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{v.get('measured',''):.4f}"
                                                            if isinstance(v.get("measured"), (int, float))
                                                            else str(v.get("measured", ""))))
            self.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{v.get('allowed',''):.4f}"
                                                            if isinstance(v.get("allowed"), (int, float))
                                                            else str(v.get("allowed", ""))))
            it = self.item(row, 0)
            if it:
                it.setForeground(QtGui.QColor(231, 76, 60))
            self._seen += 1
        while self.rowCount() > 200:
            self.removeRow(0)
        if self._seen > 0 and self.rowCount() > 0:
            self.resizeColumnsToContents()
            self.scrollToBottom()

    def violations_seen(self) -> int:
        return self._seen


# ----- Controller + main window ---------------------------------------------

class DashboardController:
    """Mutates persistent widget state from frames (no widget rebuild)."""

    def __init__(self, window: "ObservatoryWindow"):
        self.w = window

    def update(self, frame: Optional[ObservabilityFrame],
               rolling: Optional[List[ObservabilityFrame]] = None,
               cycle_error: Optional[str] = None) -> None:
        if frame is None and cycle_error is None:
            return
        f = frame
        if f is not None:
            self.w.world.set_frame(f)
            self.w.drives.set_frame(f)
            self.w.trend.push(f)
            self.w.attention.set_frame(f)
            self.w.status.set_state(f, cycle_error)
            self.w.flow.set_frame(f)
            self.w.cand.set_frame(f)
            self.w.traj.set_frame(f)
            self.w.radar.set_frame(f)
            self.w.retention.set_frame(f)
            self.w.viol.add_frame(f)
            self.w.setWindowTitle(f"PHCA Cognitive Observatory — cycle {f.cycle_id}")
        else:
            self.w.status.set_state(None, cycle_error)
        if cycle_error:
            self.w.status.set_state(f, cycle_error)


class ObservatoryWindow(QtWidgets.QMainWindow):
    def __init__(self, title: str = "PHCA Cognitive Observatory"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1320, 840)
        self.setStyleSheet(_qss())
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        self._tabs = tabs

        # Overview
        ov = QtWidgets.QWidget(); ov_lay = QtWidgets.QGridLayout(ov)
        ov_lay.setContentsMargins(6, 6, 6, 6); ov_lay.setSpacing(6)
        self.world = WorldCanvas(); self.drives = DrivesCanvas()
        self.trend = TrendCanvas(); self.attention = AttentionCanvas()
        self.status = StatusPanel()
        ov_lay.addWidget(self.world, 0, 0, 2, 2)
        ov_lay.addWidget(self.drives, 0, 2)
        ov_lay.addWidget(self.trend, 1, 2)
        ov_lay.addWidget(self.attention, 2, 0)
        ov_lay.addWidget(self.status, 2, 1, 1, 2)
        tabs.addTab(ov, "Overview")

        # Cognitive Flow
        self.flow = CognitiveFlowView()
        tabs.addTab(self.flow, "Cognitive Flow")

        # Action Selection
        self.cand = CandidateScoreView()
        tabs.addTab(self.cand, "Action Selection")

        # Phase Space
        ps = QtWidgets.QWidget(); ps_lay = QtWidgets.QHBoxLayout(ps)
        ps_lay.setContentsMargins(6, 6, 6, 6); ps_lay.setSpacing(6)
        self.traj = TrajectoryView(); self.radar = DriveRadarView()
        ps_lay.addWidget(self.traj); ps_lay.addWidget(self.radar)
        tabs.addTab(ps, "Phase Space & Trajectory")

        # Retention
        ret = QtWidgets.QWidget(); ret_lay = QtWidgets.QVBoxLayout(ret)
        ret_lay.setContentsMargins(6, 6, 6, 6); ret_lay.setSpacing(6)
        self.retention = RetentionView()
        self.viol = ViolationTable()
        ret_lay.addWidget(self.retention)
        ret_lay.addWidget(QtWidgets.QLabel("RBTA violation log (scrolling, last 200):"))
        ret_lay.addWidget(self.viol, 1)
        tabs.addTab(ret, "Retention & Resources")

        self.controller = DashboardController(self)


def make_app() -> "QtWidgets.QApplication":
    """Construct (but do not exec) the QApplication. Caller owns it."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(_qss())
    return app
