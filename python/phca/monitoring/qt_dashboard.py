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

import time
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

# v5: cached QFont instances (no per-frame QFont construction → stable metrics).
_F_TITLE = QtGui.QFont("Sans", 10, QtGui.QFont.Bold)
_F_AXIS = QtGui.QFont("Sans", 7)
_F_LABEL = QtGui.QFont("Sans", 8)
_F_LABEL_B = QtGui.QFont("Sans", 8, QtGui.QFont.Bold)
_F_CAPTION = QtGui.QFont("Sans", 7)
_F_DIMSEL = QtGui.QFont("Sans", 8, QtGui.QFont.Bold)


def _dim_label(i: int, f: Optional[ObservabilityFrame]) -> str:
    """Named dimension label from frame.dim_names, else d{i} fallback."""
    if f is not None and f.dim_names and i < len(f.dim_names):
        nm = f.dim_names[i]
        return nm if nm else f"d{i}"
    return f"d{i}"


class ScaleState:
    """Stable autoscale bounds with hysteresis + EMA contraction.

    Eliminates per-frame scale jitter: bounds expand *immediately* when a new
    extreme appears, but contract *slowly* (EMA toward the rolling min/max at
    ``contract`` per update) so the y-axis / projection range never twitches
    frame-to-frame. Add ``head`` fractional headroom on top. Pure read on
    frames; never mutates the cycle.
    """

    __slots__ = ("lo", "hi", "_ema_lo", "_ema_hi", "contract", "head", "_have")

    def __init__(self, contract: float = 0.05, head: float = 0.05):
        self.lo: float = 0.0
        self.hi: float = 1.0
        self._ema_lo: float = 0.0
        self._ema_hi: float = 1.0
        self.contract = contract
        self.head = head
        self._have: bool = False

    def update(self, rlo: float, rhi: float) -> Tuple[float, float]:
        if not self._have:
            self._ema_lo = float(rlo); self._ema_hi = float(rhi)
            self.lo = float(rlo); self.hi = float(rhi)
            self._have = True
        else:
            # EMA of the rolling extremes (smoothed target).
            self._ema_lo += (float(rlo) - self._ema_lo) * 0.2
            self._ema_hi += (float(rhi) - self._ema_hi) * 0.2
            # Expand immediately to real extremes; contract slowly toward EMA.
            self.lo = min(float(rlo), self.lo + (self._ema_lo - self.lo) * self.contract)
            self.hi = max(float(rhi), self.hi + (self._ema_hi - self.hi) * self.contract)
        span = (self.hi - self.lo) or 1.0
        self.lo -= span * self.head
        self.hi += span * self.head
        if self.hi - self.lo < 1e-9:
            self.hi = self.lo + 1.0
        return self.lo, self.hi


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


# ----- Dimension-agnostic state-space projection (v4 scaling heart) ----------

class BeliefProjection:
    """Rolling PCA 2-D projection of the state space — the single abstraction
    that makes every spatial tab work for grid, 2-D, 3-D and arbitrary-dim
    continuous environments.

    Maintains a rolling window of observation/sanitized-state vectors and fits a
    2-component PCA via SVD on the centered window. For ``state_dim > 64`` the
    top-32 variance dims are kept first (so SVD stays cheap). Falls back to the
    top-2 variance dims if SVD fails. For ``state_dim <= 3`` the raw first two
    dims are used (no PCA) so low-dim envs keep their natural geometry.

    Shared across the World / Phase-Space / Action tabs so the projection is
    consistent. Pure read on frames — never mutates the cycle.
    """

    def __init__(self, window: int = 256):
        self.window = window
        self._buf: Deque[np.ndarray] = deque(maxlen=window)
        self._mean: Optional[np.ndarray] = None
        self._comp: Optional[np.ndarray] = None      # (2, d') basis
        self._top_dims: Optional[np.ndarray] = None
        self._fallback: Tuple[int, int] = (0, min(1, 0))
        self._raw2d: bool = False
        self._hist: Deque[Tuple[float, float]] = deque(maxlen=window)
        # stable per-axis bounds (no per-frame rescale jitter on projection views)
        self._bx = ScaleState(contract=0.04, head=0.08)
        self._by = ScaleState(contract=0.04, head=0.08)

    def update(self, f: ObservabilityFrame) -> None:
        v = f.sanitized_state
        if v is None:
            v = f.obs_vector
        if v is None:
            return
        try:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            return
        if v.size == 0 or not np.all(np.isfinite(v)):
            return
        self._buf.append(v)
        if len(self._buf) >= 4:
            self._recompute()

    def _recompute(self) -> None:
        M = np.array(list(self._buf), dtype=np.float64)
        d = M.shape[1]
        mean = M.mean(axis=0)
        if d <= 3:
            self._raw2d = True
            self._mean = mean
            self._comp = None
            self._top_dims = None
            return
        self._raw2d = False
        X = M - mean
        top = None
        if d > 64:
            var = X.var(axis=0)
            top = np.argsort(var)[-32:]
            X = X[:, top]
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            if Vt.shape[0] >= 2:
                self._comp = Vt[:2].astype(np.float64)
                self._top_dims = top
                self._mean = mean
                return
        except Exception:
            pass
        # Fallback: top-2 variance dims of the full vector.
        var = (M - mean).var(axis=0)
        idx = np.argsort(var)[-2:]
        self._fallback = (int(idx[0]), int(idx[1]))
        self._comp = None
        self._top_dims = None
        self._mean = mean

    def project(self, v: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if v is None:
            return None
        try:
            v = np.asarray(v, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if v.size == 0 or not np.all(np.isfinite(v)):
            return None
        if self._raw2d:
            return (float(v[0]), float(v[1]) if v.size > 1 else 0.0)
        if self._mean is None:
            return None
        x = v - self._mean
        if self._comp is not None:
            if self._top_dims is not None and x.size > self._top_dims.size:
                x = x[self._top_dims]
            if x.size != self._comp.shape[1]:
                return None
            c = x @ self._comp.T
            return (float(c[0]), float(c[1]))
        # fallback dims
        i0, i1 = self._fallback
        if i1 >= v.size:
            i1 = 0
        return (float(v[i0] - self._mean[i0]), float(v[i1] - self._mean[i1]))

    def push_history(self, pt: Optional[Tuple[float, float]]) -> None:
        if pt is not None:
            self._hist.append(pt)

    @property
    def history(self) -> List[Tuple[float, float]]:
        return list(self._hist)

    def bounds(self) -> Tuple[float, float, float, float]:
        """Stable autoscale bounds (ScaleState per axis — no per-frame jumps)."""
        pts = self._hist
        if not pts:
            return (-1.0, 1.0, -1.0, 1.0)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        xlo, xhi = self._bx.update(float(min(xs)), float(max(xs)))
        ylo, yhi = self._by.update(float(min(ys)), float(max(ys)))
        return (xlo, xhi, ylo, yhi)

    def uncertainty_ellipse(self, per_dim_std: Optional[np.ndarray]
                            ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Project the diagonal posterior covariance into the 2-D plane.

        Returns ((axis0_std, axis1_std), angle_deg) or None if no basis.
        """
        if per_dim_std is None or self._raw2d:
            if per_dim_std is None:
                return None
            std = np.asarray(per_dim_std, dtype=np.float64).reshape(-1)
            if std.size >= 2:
                return ((float(std[0]), float(std[1])), 0.0)
            return None
        if self._comp is None:
            return None
        std = np.asarray(per_dim_std, dtype=np.float64).reshape(-1)
        if self._top_dims is not None and std.size > self._top_dims.size:
            std = std[self._top_dims]
        if std.size != self._comp.shape[1]:
            return None
        cov2 = (self._comp * (std ** 2)) @ self._comp.T
        try:
            eigval, eigvec = np.linalg.eigh(cov2)
        except Exception:
            return None
        order = np.argsort(eigval)[::-1]
        eigval = eigval[order]; eigvec = eigvec[:, order]
        import math
        axes = (float(np.sqrt(max(eigval[0], 0.0))),
                float(np.sqrt(max(eigval[1], 0.0))))
        angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
        return (axes, angle)


def _map_pt(pt: Tuple[float, float], bounds: Tuple[float, float, float, float],
            px0: int, py0: int, px1: int, py1: int) -> Tuple[int, int]:
    xlo, xhi, ylo, yhi = bounds
    xr = (xhi - xlo) or 1.0; yr = (yhi - ylo) or 1.0
    x = px0 + (pt[0] - xlo) / xr * (px1 - px0)
    y = py1 - (pt[1] - ylo) / yr * (py1 - py0)
    return (int(x), int(y))


def _draw_qimage(p: QtGui.QPainter, frame: np.ndarray, x: int, y: int,
                 max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    """Fit an RGB uint8 array into a box; return the placed rect."""
    fh, fw = frame.shape[0], frame.shape[1]
    scale = min(max_w / fw, max_h / fh)
    dw, dh = int(fw * scale), int(fh * scale)
    dx = x + (max_w - dw) // 2; dy = y + (max_h - dh) // 2
    qimg = QtGui.QImage(frame.tobytes(), fw, fh, 3 * fw, QtGui.QImage.Format_RGB888)
    p.drawImage(QtCore.QRect(dx, dy, dw, dh), qimg)
    return (dx, dy, dw, dh)


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
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.update()

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

    def _title(self, p: QtGui.QPainter, text: str, y: int = 15, x: int = 10) -> None:
        p.setPen(TEXT_COL); p.setFont(_F_TITLE)
        p.drawText(x, y, text)

    def _caption(self, p: QtGui.QPainter, text: str, y: int = 28, x: int = 10) -> None:
        """One-line dim 'what this tells you' under a title."""
        p.setPen(DIM_COL); p.setFont(_F_CAPTION)
        p.drawText(x, y, text)

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

class WorldCanvas(_BaseCanvas):
    """Env-adaptive world view: grid / MuJoCo RGB camera + overlay / PCA
    projection. The single spatial widget that scales grid→2D→3D→n-D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.trail: Deque[Any] = deque(maxlen=160)
        self.proj: Optional[BeliefProjection] = None

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if f.grid is not None and f.agent_pos is not None:
            ap = tuple(f.agent_pos)
            if not self.trail or self.trail[-1] != ap:
                self.trail.append(ap)
        # projection history is pushed once by the controller (single source)
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            self._empty(p, "Awaiting cycle…"); return
        w, h = self.width(), self.height()
        kind = (getattr(f, "env_kind", "") or "").lower()
        if f.grid is not None:
            self._draw_grid(p, f, w, h)
        elif kind == "mujoco_rgb":
            self._draw_rgb(p, f, w, h)
        else:
            self._draw_projection(p, f, w, h)

    def _draw_grid(self, p, f, w, h):
        g = f.grid
        n = g.shape[0]
        cell = min(w - 20, h - 40) / n
        ox = (w - cell * n) / 2
        oy = 28
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
        if heat is not None:
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
        tl = list(self.trail)
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
        self._title(p, f"GridWorld {n}×{n}  conf={f.prediction_confidence:.2f}  "
                       f"err={f.prediction_error:.2f}  (amber ghost = G′ predicted next cell)")
        self._caption(p, "agent ● + trail · amber ghost = predicted next cell · env=grid (dim-adaptive)")

    def _draw_rgb(self, p, f, w, h):
        frame = getattr(f, "env_frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._draw_projection(p, f, w, h); return
        dx, dy, dw, dh = _draw_qimage(p, frame, 10, 28, int(w * 0.62) - 10, h - 40)
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 70), 1))
        p.drawRect(dx - 1, dy - 1, dw + 2, dh + 2)
        self._title(p, f"MuJoCo camera  cycle={f.cycle_id}  conf={f.prediction_confidence:.2f}  "
                       f"err={f.prediction_error:.2f}")
        self._caption(p, "live RGB camera frame · env=mujoco_rgb (dim-adaptive; high-dim → see Phase Space)")
        cx, cy = dx + dw // 2, dy + dh // 2
        pred = f.predicted_state
        if pred is not None and pred.size >= 2:
            ex = int(np.clip(cx + (pred[0] - cx) * 0.3, dx, dx + dw))
            ey = int(np.clip(cy + (pred[1] - cy) * 0.3, dy, dy + dh))
            _arrow(p, cx, cy, ex, ey, QtGui.QColor(39, 174, 96), size=8)
        act = f.continuous_action if f.continuous_action is not None else f.last_action_vector
        if act is not None and act.size >= 2:
            ax = int(np.clip(cx + float(np.clip(act[0], -1, 1)) * 40, dx, dx + dw))
            ay = int(np.clip(cy + float(np.clip(act[1], -1, 1)) * 40, dy, dy + dh))
            _arrow(p, cx, cy, ax, ay, ACCENT, size=8)
        if self.proj is not None:
            inset_x = int(w * 0.62) + 6
            self._draw_projection_inset(p, f, inset_x, 28, w - inset_x - 8, h - 40)
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(dx, dy + dh + 12, "green=predicted goal  amber=chosen action")

    def _draw_projection(self, p, f, w, h):
        if self.proj is None:
            self._empty(p, "Projection (collecting…)"); return
        px0, py0, px1, py1 = 30, 30, w - 16, h - 30
        p.setPen(QtGui.QPen(GRID_COL, 1))
        p.drawRect(px0, py0, px1 - px0, py1 - py0)
        bounds = self.proj.bounds()
        hist = self.proj.history
        if len(hist) >= 2:
            n = len(hist)
            for i in range(1, n):
                a = int(40 + 200 * i / n)
                p0 = _map_pt(hist[i - 1], bounds, px0, py0, px1, py1)
                p1 = _map_pt(hist[i], bounds, px0, py0, px1, py1)
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
                p.drawLine(p0[0], p0[1], p1[0], p1[1])
        cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
        cur = self.proj.project(cur_v)
        cx = cy = None
        if cur is not None:
            cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
            p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(cx - 5, cy - 5, 10, 10)
        pred = self.proj.project(f.predicted_state)
        if pred is not None:
            px, py = _map_pt(pred, bounds, px0, py0, px1, py1)
            p.setBrush(QtGui.QColor(231, 76, 60, 160))
            p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 2))
            p.drawEllipse(px - 4, py - 4, 8, 8)
            if cx is not None:
                _arrow(p, cx, cy, px, py, QtGui.QColor(231, 76, 60, 180), size=7)
        g = self.proj.project(f.goal_target)
        if g is not None:
            gx, gy = _map_pt(g, bounds, px0, py0, px1, py1)
            p.setBrush(QtGui.QColor(39, 174, 96, 120))
            p.setPen(QtGui.QPen(QtGui.QColor(39, 174, 96), 2))
            p.drawRect(gx - 6, gy - 6, 12, 12)
        self._title(p, f"State-space projection (PCA-2D)  dim={f.state_dim}  "
                       f"kind={f.gprime_kind or '?'}  ●now ◆goal ●pred")
        self._caption(p, "PCA-2D of belief state · scales grid→2D→3D→n-D · stable bounds (no jitter)")
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(8, h - 6, "dimension-agnostic — scales grid/2D/3D/n-D")

    def _draw_projection_inset(self, p, f, x, y, w, h):
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(x + 6, y + 14, "PCA projection")
        bounds = self.proj.bounds()
        hist = self.proj.history
        ix0, iy0, ix1, iy1 = x + 6, y + 22, x + w - 6, y + h - 6
        if len(hist) >= 2:
            n = len(hist)
            for i in range(1, n):
                a = int(40 + 180 * i / n)
                p0 = _map_pt(hist[i - 1], bounds, ix0, iy0, ix1, iy1)
                p1 = _map_pt(hist[i], bounds, ix0, iy0, ix1, iy1)
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
                p.drawLine(p0[0], p0[1], p1[0], p1[1])
        cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
        cur = self.proj.project(cur_v)
        if cur is not None:
            cxp, cyp = _map_pt(cur, bounds, ix0, iy0, ix1, iy1)
            p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(cxp - 3, cyp - 3, 6, 6)


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
        self._caption(p, "fixed 0..1 scale (no per-frame rescale) · ◆ = target · 6 homeostatic drives")
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
        self._caption(p, "attended state chunks ranked by precision/salience · gumbel-τ shown")
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
        # slow wall-clock animation (decoupled from cycle rate → no strobe)
        self._anim_t: float = 0.0
        self._last_wall: float = time.monotonic()
        self._active_idx: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.cycle_id = int(f.cycle_id)
        self.heat.append(dict(f.module_timings))
        # advance the slow animation clock (~0.5s per pipeline step)
        now = time.monotonic()
        dt = now - self._last_wall; self._last_wall = now
        self._anim_t = (self._anim_t + dt / 0.5) % len(PIPELINE)
        # active node = most-recently-active by timing delta (stable, meaningful)
        try:
            if len(self.heat) >= 2:
                prev, cur = self.heat[-2], self.heat[-1]
                deltas = {m: float(cur.get(m, 0.0)) - float(prev.get(m, 0.0))
                          for m in PIPELINE}
                self._active_idx = int(max(range(len(PIPELINE)),
                                           key=lambda i: deltas.get(PIPELINE[i], 0.0)))
        except Exception:
            pass
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
        self._title(p, "Cognitive flow — node cost · measured-vs-bound (▮ + sparkline) · active edge")
        self._caption(p, "node radius/colour = timing cost · red = RBTA violation · packet = most-recently-active edge")
        pos = self._node_pos(w, h)
        n = len(PIPELINE)
        # main chain edges with arrowheads + travelling packet (slow wall-clock)
        active_idx = self._active_idx
        for i in range(n - 1):
            a, b = pos[PIPELINE[i]], pos[PIPELINE[i + 1]]
            col = QtGui.QColor(241, 196, 15, 230) if i == active_idx else QtGui.QColor(120, 120, 140, 150)
            _arrow(p, a[0], a[1], b[0], b[1], col, size=10)
        # travelling data packet on the active edge — slow, smooth, off wall-clock
        if 0 <= active_idx < n - 1:
            a, b = pos[PIPELINE[active_idx]], pos[PIPELINE[active_idx + 1]]
            t = self._anim_t - active_idx
            t = t - int(t)  # fractional position along the active edge
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
            p.setFont(_F_LABEL_B)
            p.drawText(x - 26, y - 3, 52, 12, 0x84, PIPELINE_LABEL.get(mod, mod))
            p.setFont(_F_AXIS)
            p.drawText(x - 22, y + 9, 44, 11, 0x84, f"{ms:.1f}ms")
            # v5: consolidated measured-vs-bound bar WITH overlaid sparkline
            # (replaces the separate _spark mini-chart — one chart, not two).
            self._mb_bar(p, mod, ms, x - 30, y + r + 12, 60)
            # v5: content thumbnail only for the 2 most informative nodes;
            # others get a clear status badge (OK / NEAR-BOUND / VIOLATION).
            if mod in ("prediction", "action_selection"):
                self._content_thumb(p, mod, f, x - 30, y - r - 26, 60, 20)
            else:
                self._status_badge(p, mod, ms, x - 30, y - r - 24, 60)
            # violation bound below
            if viol and mod in self.last_viol:
                p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(_F_AXIS)
                p.drawText(x - 30, y + r + 30, 60, 10, 0x84, self.last_viol[mod])
        self._composite_bounds(p)
        self._heatmap(p)
        # legend
        self._legend(p)

    def _bound_for(self, mod: str, bounds: dict) -> Optional[float]:
        for k, v in bounds.items():
            if RBTA_TO_FLOW.get(k, "") == mod and isinstance(v, dict):
                t = v.get("time")
                if t is not None and t > 0:
                    return float(t)
        return None

    def _mb_bar(self, p: QtGui.QPainter, mod: str, measured: float,
                x: int, y: int, w: int) -> None:
        """Measured-vs-bound bar with a faint overlaid timing sparkline
        (one consolidated mini-chart per node, not two)."""
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        b = self._bound_for(mod, bounds)
        # track
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, y, w, 7)
        # measured fill (scaled to bound, or to 25ms if no bound)
        scale = b if b else 25.0
        ratio = max(0.0, min(measured / scale, 1.5))
        fw = int(min(ratio, 1.0) * w)
        col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else (
            _to_qcolor(_cost_color(measured)))
        p.fillRect(x, y, fw, 7, col)
        # faint timing sparkline overlaid on the bar (consolidates _spark)
        vals = [float(mt.get(mod, 0.0)) for mt in self.heat]
        if len(vals) >= 2:
            mx = max(vals) or 1.0
            sc = QtGui.QColor(255, 255, 255, 120)
            p.setPen(QtGui.QPen(sc, 1)); n = len(vals)
            for i in range(1, n):
                x0 = x + (i - 1) * w / max(n - 1, 1)
                x1 = x + i * w / max(n - 1, 1)
                y0 = y + 7 - (vals[i - 1] / mx) * 6
                y1 = y + 7 - (vals[i] / mx) * 6
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        # bound tick
        if b:
            bx = x + w
            p.setPen(QtGui.QPen(ACCENT, 2)); p.drawLine(bx, y - 2, bx, y + 9)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 16, f"{measured:.1f}/{b:.0f}ms")
        else:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 16, f"{measured:.1f}ms")

    def _status_badge(self, p: QtGui.QPainter, mod: str, measured: float,
                      x: int, y: int, w: int) -> None:
        """Compact OK / NEAR-BOUND / VIOLATION badge (replaces per-node thumbnail
        for nodes that don't carry a directly visualisable payload)."""
        viol = mod in self.viol_mods
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        b = self._bound_for(mod, bounds)
        scale = b if b else 25.0
        ratio = measured / scale if scale else 0.0
        if viol or ratio > 1.0:
            txt, c = "VIOLATION", QtGui.QColor(231, 76, 60)
        elif ratio > 0.8:
            txt, c = "NEAR-BOUND", QtGui.QColor(241, 196, 15)
        else:
            txt, c = "OK", QtGui.QColor(46, 204, 113)
        p.setPen(QtGui.QPen(c, 1)); p.setBrush(QtGui.QColor(c.red(), c.green(), c.blue(), 40))
        p.drawRoundedRect(x, y, w, 14, 4, 4)
        p.setPen(c); p.setFont(_F_AXIS)
        p.drawText(x, y, w, 14, 0x84, txt)

    def _content_thumb(self, p: QtGui.QPainter, mod: str, f: ObservabilityFrame,
                       x: int, y: int, w: int, h: int) -> None:
        """Tiny per-node content preview: prediction → predicted-state mini,
        action → chosen-action mini, mdim → deficits mini, attn → precision mini."""
        vec = None; col = QtGui.QColor(155, 89, 182, 200)
        if mod == "prediction" and f.predicted_state is not None:
            vec = np.asarray(f.predicted_state, dtype=np.float32).reshape(-1)[:12]; col = QtGui.QColor(52, 152, 219, 200)
        elif mod == "action_selection":
            av = f.last_action_vector if f.last_action_vector is not None else f.continuous_action
            if av is not None:
                vec = np.asarray(av, dtype=np.float32).reshape(-1)[:12]; col = ACCENT
        elif mod == "mdim" and f.drive_deficits is not None:
            vec = np.asarray(f.drive_deficits, dtype=np.float32).reshape(-1)[:6]; col = QtGui.QColor(230, 126, 34, 200)
        elif mod == "attn" and f.attention_precisions is not None:
            vec = np.asarray(f.attention_precisions, dtype=np.float32).reshape(-1)[:12]; col = QtGui.QColor(155, 89, 182, 200)
        if vec is None or vec.size == 0:
            return
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
        mx = float(np.max(np.abs(vec))) or 1.0
        n = vec.size; bw = w / n
        mid = y + h // 2
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, mid, x + w, mid)
        for i, val in enumerate(vec):
            bh = int(abs(float(val)) / mx * (h / 2 - 1))
            bx = int(x + i * bw)
            if float(val) >= 0:
                p.fillRect(bx + 1, mid - bh, max(int(bw) - 2, 1), bh, col)
            else:
                p.fillRect(bx + 1, mid, max(int(bw) - 2, 1), bh, col)

    def _composite_bounds(self, p: QtGui.QPainter) -> None:
        """Small bound-tree: regulation_block (PARALLEL of side modules) +
        root SEQUENCE of the pipeline — shows the composite RBTA envelope."""
        w, h = self.width(), self.height()
        x, y, tw, th = w - 150, 12, 140, 70
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, y, tw, th)
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(x + 4, y + 12, "composite bounds")
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(x + 4, y + 26, "root: SEQUENCE(pipeline)")
        p.drawText(x + 4, y + 38, "reg: PARALLEL(sides)")
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        pipe_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in PIPELINE]
        side_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in SIDE_MODULES]
        pb = sum(float(bounds[k].get("time", 0.0)) for k in pipe_ids if isinstance(bounds[k], dict))
        sb = max((float(bounds[k].get("time", 0.0)) for k in side_ids if isinstance(bounds[k], dict)), default=0.0)
        p.setPen(ACCENT); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(x + 4, y + 54, f"Σpipe={pb:.0f}ms")
        p.drawText(x + 70, y + 54, f"maxpar={sb:.0f}ms")
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 6))
        p.drawText(x + 4, y + 66, f"({len(pipe_ids)}+{len(side_ids)} bound mods)")

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
        self.proj: Optional[BeliefProjection] = None

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

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

    def _rollout_cloud(self, p: QtGui.QPainter, f: ObservabilityFrame,
                       w: int, top: int, bot: int, is_continuous: bool) -> None:
        """Render candidate rollouts in the shared PCA plane.

        Discrete: per-action predicted-next thumbnails (dot per action, colored
        by score, chosen circled). Continuous: MPC K-candidate trajectory cloud
        (each candidate's predicted next state, chosen highlighted). Falls back
        to a per-action predicted-bar mini when no projection basis yet.
        """
        rollouts = list(getattr(f, "candidate_rollouts", []) or [])
        label = ("MPC candidate cloud (chosen ◆)" if is_continuous
                 else "per-action predicted-next (chosen ◆)")
        self._title(p, label, y=top + 12)
        if not rollouts:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(20, top + 30, "(no rollouts this cycle — explore/D5 branch)")
            return
        # Score-normalised colour: green=high, red=low.
        scores = [float(r.get("score", 0.0)) for r in rollouts]
        smin, smax = (min(scores), max(scores)) if scores else (0.0, 1.0)
        srange = (smax - smin) or 1.0
        if self.proj is not None and self.proj.history:
            bounds = self.proj.bounds()
            px0, py0, px1, py1 = 20, top + 20, w - 20, bot - 4
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
            # current state anchor
            cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            cur = self.proj.project(cur_v)
            if cur is not None:
                cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
                p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)
                for r in rollouts:
                    pt = self.proj.project(r.get("predicted"))
                    if pt is None:
                        continue
                    rx, ry = _map_pt(pt, bounds, px0, py0, px1, py1)
                    s = float(r.get("score", 0.0))
                    t = (s - smin) / srange
                    col = QtGui.QColor(int(231 - 180 * t), int(60 + 140 * t), int(60 + 60 * t), 220)
                    p.setBrush(col); p.setPen(QtGui.QPen(col.darker(140), 1))
                    rad = 6 if r.get("chosen") else 4
                    p.drawEllipse(rx - rad, ry - rad, rad * 2, rad * 2)
                    if r.get("chosen"):
                        p.setBrush(QtGui.QColor(241, 196, 15, 60))
                        p.setPen(QtGui.QPen(ACCENT, 2))
                        p.drawEllipse(rx - 8, ry - 8, 16, 16)
                    if is_continuous and cur is not None:
                        p.setPen(QtGui.QPen(col, 1))
                        p.drawLine(cx, cy, rx, ry)
            return
        # Fallback: per-action predicted-bar mini (no projection basis yet).
        n = len(rollouts)
        bw = (w - 40) / max(n, 1)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot - 6, w - 20, bot - 6)
        for i, r in enumerate(rollouts):
            pred = r.get("predicted")
            if pred is None:
                continue
            head = np.asarray(pred, dtype=np.float32).reshape(-1)[:12]
            mx = float(np.max(np.abs(head))) or 1.0
            x = 20 + i * bw
            for j, val in enumerate(head):
                bh = int(abs(float(val)) / mx * (bot - top - 30))
                col = QtGui.QColor(52, 152, 219, 180)
                if r.get("chosen"):
                    col = QtGui.QColor(241, 196, 15, 220)
                p.fillRect(int(x + j * 2), int(bot - 6 - bh), 2, bh, col)
            p.setPen(ACCENT if r.get("chosen") else DIM_COL)
            p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(int(x), bot + 0, f"{float(r.get('score',0)):.2f}")

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
        kind = "continuous τ" if is_continuous else "discrete action"
        self._caption(p, f"candidate {kind} scores · amber = chosen · rollout cloud + torque dial below · stable scale")
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
        # ── v4: rollout candidate cloud (discrete per-action predicted + continuous MPC) ──
        cloud_top = bar_bot + 8
        self._rollout_cloud(p, f, w, cloud_top, h - 46, is_continuous)
        # continuous action: dial for 2D, signed bars otherwise
        if is_continuous and self.last_continuous is not None:
            if len(self.last_continuous) <= 2:
                self._torque_dial(p, w - 120, cloud_top + 50, 40, self.last_continuous)
            else:
                self._signed_bars(p, self.last_continuous, 20, cloud_top + 4, w - 40, 28, horizontal=True)
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
        self.proj: Optional[BeliefProjection] = None

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

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
            # projection history is pushed once by the controller (single source)
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
            self._caption(p, "agent trail over time · cell heat = per-dim prediction error magnitude")
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(8, h - 6, "red = where world model is wrong")
        else:
            # env-adaptive: 2D PCA projection trajectory + goal + predicted-next
            # + candidate cloud + G′ uncertainty ellipse (scales to any dim).
            if self.proj is None or not self.proj.history:
                self._empty(p, "Building belief projection (PCA)…"); return
            px0, py0, px1, py1 = 20, 36, w - 20, h - 30
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
            bounds = self.proj.bounds()
            # trail
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
            cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            cur = self.proj.project(cur_v)
            pred = self.proj.project(f.predicted_state)
            goal = self.proj.project(f.goal_target) if f.goal_target is not None else None
            # uncertainty ellipse from G′ per-dim std
            ell = None
            if f.gprime_uncertainty is not None and len(f.gprime_uncertainty):
                ell = self.proj.uncertainty_ellipse(np.asarray(f.gprime_uncertainty, dtype=np.float32))
            if ell is not None and ell[0] is not None:
                axes, ang = ell
                ax_p, ay_p = axes
                if ax_p > 0 and ay_p > 0 and cur is not None:
                    cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
                    p.setBrush(QtGui.QColor(52, 152, 219, 40))
                    p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, 160), 1))
                    p.translate(cx, cy); p.rotate(ang)
                    p.drawEllipse(QtCore.QRectF(-ax_p, -ay_p, ax_p * 2, ay_p * 2))
                    p.resetTransform()
            # candidate cloud (predicted-next per rollout)
            for r in (f.candidate_rollouts or []):
                pt = self.proj.project(r.get("predicted"))
                if pt is None:
                    continue
                rx, ry = _map_pt(pt, bounds, px0, py0, px1, py1)
                col = ACCENT if r.get("chosen") else QtGui.QColor(150, 150, 160, 140)
                p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
                p.drawEllipse(rx - 3, ry - 3, 6, 6)
            if pred is not None:
                pxp, pyp = _map_pt(pred, bounds, px0, py0, px1, py1)
                p.setBrush(QtGui.QColor(46, 204, 113)); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(pxp - 3, pyp - 3, 6, 6)
            if goal is not None:
                gx, gy = _map_pt(goal, bounds, px0, py0, px1, py1)
                p.setBrush(QtGui.QColor(0, 0, 0, 0)); p.setPen(QtGui.QPen(ACCENT, 2))
                p.drawEllipse(gx - 6, gy - 6, 12, 12)
            if cur is not None:
                cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
                p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(cx - 4, cy - 4, 8, 8)
            kind = getattr(f, "gprime_kind", None) or "g′"
            self._title(p, f"Belief-space projection (PCA-2D, dim={f.state_dim or '?'}) + {kind} uncertainty ellipse")
            self._caption(p, "belief trajectory in PCA-2D · ellipse = G′ posterior σ projected · candidate cloud = rollouts")
            self._legend(p, [("● current", ACCENT), ("● predicted next", QtGui.QColor(46, 204, 113)),
                             ("○ goal", ACCENT), ("◐ candidates", QtGui.QColor(150, 150, 160)),
                             ("◯ ±σ ellipse", QtGui.QColor(52, 152, 219))],
                         y=h - 14)


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
        self._caption(p, "radial drive levels 0..1 · faded = recent history · shows drive balance at a glance")


class PerDimErrorView(_BaseCanvas):
    """Per-dimension predicted-vs-actual error bars with G′ uncertainty bands.
    Subsamples to the top-K highest-error dims when state_dim > 24 so it scales
    to arbitrary continuous environments (Reacher, Cartpole, high-dim)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scroll: int = 0

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None or f.predicted_state is None:
            self._empty(p, "Per-dim prediction error…"); return
        pred = np.asarray(f.predicted_state, dtype=np.float32).reshape(-1)
        ref = f.obs_vector if f.obs_vector is not None else f.goal_ref
        if ref is None:
            self._empty(p, "Per-dim error (no reference)"); return
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        d = min(len(pred), len(ref))
        if d == 0:
            self._empty(p, "Per-dim error (empty)"); return
        errs = np.abs(pred[:d] - ref[:d])
        std = None
        if f.gprime_uncertainty is not None and len(f.gprime_uncertainty):
            us = np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1)
            std = us[:d] if len(us) >= d else us
        MAX = 24
        if d > MAX:
            # keep the MAX highest-error dimensions (scale to high-dim)
            idx = np.argsort(-errs)[:MAX]
            idx.sort()
        else:
            idx = np.arange(d)
        errs_s = errs[idx]; std_s = std[idx] if std is not None else None
        n = len(idx)
        bw = (w - 40) / n
        top, bot = 40, h - 30
        self._title(p, f"Per-dim |pred−actual| error ± G′ σ band  (showing {n}/{d} dims)")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
        mx = float(errs_s.max()) if errs_s.size else 1.0
        if std_s is not None:
            mx = max(mx, float(std_s.max()) * 1.0)
        mx = mx if mx > 1e-6 else 1.0
        for i in range(n):
            x = 20 + i * bw
            # uncertainty band (±σ) as a translucent vertical bar
            if std_s is not None:
                sb = float(std_s[i]) / mx * (bot - top)
                p.fillRect(int(x + 1), int(bot - sb), int(bw - 6), int(sb * 2),
                           QtGui.QColor(52, 152, 219, 50))
            eh = int(errs_s[i] / mx * (bot - top))
            p.fillRect(int(x + 2), int(bot - eh), int(bw - 8), eh, QtGui.QColor(231, 76, 60, 220))
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(int(x), h - 10, f"d{int(idx[i])}")
        self._legend(p, [("█ error", QtGui.QColor(231, 76, 60)), ("░ ±σ band", QtGui.QColor(52, 152, 219))],
                     y=h - 4, x=20)


class UncertaintyPortraitView(_BaseCanvas):
    """G′ uncertainty portrait: per-dim std (Gaussian posterior or MC variance)
    + mutual-info + belief-entropy trend. The 'how sure is the world model' view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mi_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.be_hist: Deque[float] = deque(maxlen=TREND_WINDOW)

    def set_frame(self, f: ObservabilityFrame) -> None:
        mi = getattr(f, "gprime_mutual_info", None)
        if mi is None and f.belief_entropies:
            mi = float(list(f.belief_entropies.values())[0])
        if mi is not None:
            self.mi_hist.append(float(mi))
        be = f.belief_entropies.get("total") if f.belief_entropies else None
        if be is None and f.belief_entropies:
            be = float(list(f.belief_entropies.values())[0])
        if be is not None:
            self.be_hist.append(float(be))
        super().set_frame(f)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None or f.gprime_uncertainty is None:
            self._empty(p, "G′ uncertainty portrait…"); return
        std = np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1)
        d = len(std)
        if d == 0:
            self._empty(p, "G′ uncertainty (empty)"); return
        MAX = 32
        if d > MAX:
            idx = np.argsort(-std)[:MAX]; idx.sort()
        else:
            idx = np.arange(d)
        std_s = std[idx]
        n = len(idx)
        bw = (w - 40) / n
        top, bot = 36, h // 2
        kind = getattr(f, "gprime_kind", "g′") or "g′"
        mi = getattr(f, "gprime_mutual_info", None)
        mi_txt = f"  mutual_info={float(mi):.3f}" if mi is not None else ""
        self._title(p, f"{kind} per-dim uncertainty (σ)  top-{n}/{d}{mi_txt}")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
        mx = float(std_s.max()) if std_s.size else 1.0
        mx = mx if mx > 1e-6 else 1.0
        for i in range(n):
            x = 20 + i * bw
            bh = int(std_s[i] / mx * (bot - top))
            col = QtGui.QColor(155, 89, 182, 220) if kind == "mlp" else QtGui.QColor(52, 152, 219, 220)
            p.fillRect(int(x + 2), int(bot - bh), int(bw - 8), bh, col)
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 6))
            if n <= 24:
                p.drawText(int(x), bot + 10, f"d{int(idx[i])}")
        # mutual-info / belief-entropy trend (lower half)
        t_top, t_bot = h // 2 + 16, h - 14
        self._trend(p, self.mi_hist, t_top, t_bot, QtGui.QColor(155, 89, 182),
                    "mutual_info trend", left=40, right=w // 2 - 6)
        self._trend(p, self.be_hist, t_top, t_bot, QtGui.QColor(46, 204, 113),
                    "belief-entropy trend", left=w // 2 + 6, right=w - 16)

    def _trend(self, p: QtGui.QPainter, s: Deque[float], top: int, bot: int,
               col: QtGui.QColor, label: str, left: int, right: int) -> None:
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(left, top - 4, label)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(left, bot, right, bot)
        vals = list(s)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(left + 4, top + 14, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1
        n = len(vals)
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            x = left + i * (right - left) / (n - 1)
            y = bot - (v - lo) / (hi - lo) * (bot - top - 4)
            (path.moveTo if i == 0 else path.lineTo)(x, y)
        p.setPen(QtGui.QPen(col, 2)); p.drawPath(path)


class _DimSelector(QtWidgets.QWidget):
    """Prev/next range pager for high-dimensional per-dim views (10..1000 dims)."""

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
        self.prev.clicked.connect(self._back)
        self.next.clicked.connect(self._fwd)

    def _back(self):
        self.view.set_page(self.view.page - 1)

    def _fwd(self):
        self.view.set_page(self.view.page + 1)

    def refresh(self):
        v = self.view
        n_pages = max(1, (v.n_dims + v.page_size - 1) // v.page_size)
        a = v.page * v.page_size
        b = min(v.n_dims, (v.page + 1) * v.page_size) - 1
        b = max(b, 0)
        self.label.setText(f"page {v.page + 1}/{n_pages}  dims {a}..{b} of {v.n_dims}")
        self.prev.setEnabled(v.page > 0)
        self.next.setEnabled(v.page + 1 < n_pages)


class _PhasePortraitView(_BaseCanvas):
    """Consolidated Phase-Space portrait: per-dim |pred−actual| error bar with
    the ±σ uncertainty band overlaid on the SAME axis (one view, not two),
    named dims, a shared stable scale, and a _DimSelector pager so it scales
    from 10 → 1000 dims. Mutual-info / belief-entropy trend strip below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.page: int = 0
        self.page_size: int = 24
        self.n_dims: int = 0
        self.mi_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.be_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._scale = ScaleState(contract=0.05, head=0.06)

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        mi = getattr(f, "gprime_mutual_info", None)
        if mi is None and f.belief_entropies:
            mi = float(list(f.belief_entropies.values())[0])
        if mi is not None:
            self.mi_hist.append(float(mi))
        be = f.belief_entropies.get("total") if f.belief_entropies else None
        if be is None and f.belief_entropies:
            be = float(list(f.belief_entropies.values())[0])
        if be is not None:
            self.be_hist.append(float(be))
        self.update()

    def set_page(self, p: int) -> None:
        n_pages = max(1, (self.n_dims + self.page_size - 1) // self.page_size)
        self.page = max(0, min(n_pages - 1, int(p)))
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None or f.predicted_state is None:
            self._empty(p, "Phase portrait (collecting…)"); return
        pred = np.asarray(f.predicted_state, dtype=np.float32).reshape(-1)
        ref = f.obs_vector if f.obs_vector is not None else f.goal_ref
        if ref is None:
            self._empty(p, "Phase portrait (no reference)"); return
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        d = int(min(len(pred), len(ref)))
        if d == 0:
            self._empty(p, "Phase portrait (empty)"); return
        errs = np.abs(pred[:d] - ref[:d])
        std = None
        if f.gprime_uncertainty is not None and len(f.gprime_uncertainty):
            us = np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1)
            std = us[:d] if len(us) >= d else us
        self.n_dims = d
        # page selection (top-K highest-error within the current page window)
        n_pages = max(1, (d + self.page_size - 1) // self.page_size)
        self.page = max(0, min(n_pages - 1, self.page))
        lo_i = self.page * self.page_size
        hi_i = min(d, lo_i + self.page_size)
        if d > self.page_size:
            # within the page, show the highest-error dims of that window
            window_err = errs[lo_i:hi_i]
            k = min(self.page_size, len(window_err))
            local = np.argsort(-window_err)[:k]
            idx = local + lo_i
            idx.sort()
        else:
            idx = np.arange(lo_i, hi_i)
        errs_s = errs[idx]
        std_s = std[idx] if std is not None else None
        n = len(idx)
        bw = (w - 40) / max(n, 1)
        top, bot = 44, h // 2
        kind = getattr(f, "gprime_kind", "g′") or "g′"
        mi = getattr(f, "gprime_mutual_info", None)
        mi_txt = f"  mi={float(mi):.3f}" if mi is not None else ""
        self._title(p, f"Phase portrait  |pred−actual| ▮ + {kind} ±σ ░  dims {n}/{d}{mi_txt}")
        self._caption(p, "red = prediction error · blue band = model uncertainty σ · shared stable scale")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
        raw_max = float(errs_s.max()) if errs_s.size else 1.0
        if std_s is not None:
            raw_max = max(raw_max, float(std_s.max()))
        _, mx = self._scale.update(0.0, float(raw_max if raw_max > 1e-6 else 1.0))
        mx = mx if mx > 1e-6 else 1.0
        for i in range(n):
            x = 20 + i * bw
            if std_s is not None:
                sb = float(std_s[i]) / mx * (bot - top)
                p.fillRect(int(x + 1), int(bot - sb), int(bw - 6), int(sb * 2),
                           QtGui.QColor(52, 152, 219, 60))
            eh = int(errs_s[i] / mx * (bot - top))
            p.fillRect(int(x + 2), int(bot - eh), int(bw - 8), eh, QtGui.QColor(231, 76, 60, 220))
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            if n <= 24:
                p.drawText(int(x), bot + 11, _dim_label(int(idx[i]), f))
            else:
                p.drawText(int(x), bot + 11, str(int(idx[i])))
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(20, h - 4, f"scale 0..{mx:.3g}  (named dims · pager below)")
        self._legend(p, [("▮ error", QtGui.QColor(231, 76, 60)),
                         ("░ ±σ band", QtGui.QColor(52, 152, 219))],
                     y=h // 2 + 4, x=20)
        # mutual-info / belief-entropy trend strip (lower half)
        t_top, t_bot = h // 2 + 22, h - 16
        self._trend(p, self.mi_hist, t_top, t_bot, QtGui.QColor(155, 89, 182),
                    "mutual_info trend", left=40, right=w // 2 - 6)
        self._trend(p, self.be_hist, t_top, t_bot, QtGui.QColor(46, 204, 113),
                    "belief-entropy trend", left=w // 2 + 6, right=w - 16)

    def _trend(self, p: QtGui.QPainter, s: Deque[float], top: int, bot: int,
               col: QtGui.QColor, label: str, left: int, right: int) -> None:
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(left, top - 4, label)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(left, bot, right, bot)
        vals = list(s)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(left + 4, top + 14, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1
        n = len(vals)
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            x = left + i * (right - left) / (n - 1)
            y = bot - (v - lo) / (hi - lo) * (bot - top - 4)
            (path.moveTo if i == 0 else path.lineTo)(x, y)
        p.setPen(QtGui.QPen(col, 2)); p.drawPath(path)


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
        self._panel_scales: Dict[str, ScaleState] = {}

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
        # stable per-panel scale (no per-frame rescale jitter); cap kept in range
        sc = self._panel_scales.setdefault(title, ScaleState(contract=0.05, head=0.05))
        lo, hi = sc.update(float(lo), float(max(hi, cap) if cap else hi))
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
        leak = self._leak_rate()
        self._title(p, f"Retention & resources   RSS leak-rate (last 30) ≈ {leak:.1f} B/cyc", y=15)
        self._caption(p, "M3 episodic · M4 consolidated facts · resources = RSS (orange) + latency (green) on one shared stable scale")
        # v5: 3 panels — M3, M4, and a consolidated Resources panel (RSS+latency)
        ph = (h - 36) // 3
        self._panel(p, 30, 30 + ph, "M3 episodes (memory)", self.m3, float(self.m3_cap),
                    QtGui.QColor(52, 152, 219), self.m3_events)
        self._panel(p, 30 + ph, 30 + 2 * ph, "M4 facts (consolidated)", self.m4, float(self.m4_cap),
                    QtGui.QColor(155, 89, 182), self.m4_events)
        self._resources_panel(p, 30 + 2 * ph, h - 4)

    def _resources_panel(self, p: QtGui.QPainter, top: int, bot: int) -> None:
        """Consolidated resources panel: RSS + latency on one shared stable
        scale (normalised 0..1 of each series' own ScaleState) with dual labels
        — replaces the separate RSS line + latency histogram panels."""
        w = self.width()
        left, right = 60, w - 16
        self._title(p, "Resources — RSS (B, orange) + latency (ms, green)", y=top + 12)
        rss = list(self.rss); lat = list(self.lat)
        if not rss and not lat:
            return
        # dual labels (right-aligned)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        rss_txt = f"RSS now={rss[-1]/1e6:.1f}MB" if rss else "RSS —"
        lat_txt = f"lat now={lat[-1]:.1f}ms med={float(np.median(lat)):.1f}ms" if lat else "lat —"
        r1 = p.boundingRect(QtCore.QRect(0, 0, 1, 1), 0, rss_txt)
        r2 = p.boundingRect(QtCore.QRect(0, 0, 1, 1), 0, lat_txt)
        p.drawText(right - r1.width(), top + 12, rss_txt)
        p.drawText(right - r2.width(), top + 24, lat_txt)
        pt0, pt1 = top + 30, bot - 6
        # gridlines
        for g in range(1, 3):
            y = pt0 + g * (pt1 - pt0) // 3
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1)); p.drawLine(left, y, right, y)
        rss_sc = self._panel_scales.setdefault("rss_res", ScaleState())
        lat_sc = self._panel_scales.setdefault("lat_res", ScaleState())
        for vals, col, sc in ((rss, QtGui.QColor(230, 126, 34), rss_sc),
                              (lat, QtGui.QColor(46, 204, 113), lat_sc)):
            if not vals:
                continue
            lo, hi = sc.update(float(min(vals)), float(max(vals)))
            if hi - lo < 1e-9:
                hi = lo + 1
            n = len(vals)
            p.setPen(QtGui.QPen(col, 2))
            for i in range(1, n):
                x0 = left + (i - 1) * (right - left) / max(n - 1, 1)
                x1 = left + i * (right - left) / max(n - 1, 1)
                y0 = pt1 - (vals[i - 1] - lo) / (hi - lo) * (pt1 - pt0)
                y1 = pt1 - (vals[i] - lo) / (hi - lo) * (pt1 - pt0)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(left, bot - 1, "shared 0..1 normalised scale · stable bounds")

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


# ----- RBTA bound-envelope (Retention tab addition) --------------------------

class RBTABoundsView(_BaseCanvas):
    """Per-module measured TIME vs B_time bars (12 modules) with violation red.
    The 'are we inside the resource envelope' portrait."""

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "RBTA bounds…"); return
        bounds = getattr(f, "rbta_bounds", None) or {}
        timings = f.module_timings or {}
        items = []
        for mid, b in bounds.items():
            if not isinstance(b, dict):
                continue
            t = b.get("time")
            if t is None or t <= 0:
                continue
            flow = RBTA_TO_FLOW.get(mid, mid)
            meas = float(timings.get(flow, 0.0))
            items.append((mid, flow, meas, float(t)))
        if not items:
            self._empty(p, "RBTA bound envelope (no bounds)"); return
        items.sort(key=lambda r: r[2] / r[3], reverse=True)
        n = len(items)
        rowh = (h - 30) / max(n, 1)
        self._title(p, f"RBTA bound envelope — measured vs B_time  ({n} modules)")
        self._caption(p, "▮ measured timing vs │ bound · red = violation · stable shared time scale")
        bw_max = w - 160
        for i, (mid, flow, meas, b) in enumerate(items):
            y = 28 + int(i * rowh)
            p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(8, y + 12, f"{mid:>10}")
            ratio = meas / b
            col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else _to_qcolor(_cost_color(meas))
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRect(100, y + 4, bw_max, 12)
            fw = int(min(ratio, 1.0) * bw_max)
            p.fillRect(100, y + 4, fw, 12, col)
            # bound tick
            p.setPen(QtGui.QPen(ACCENT, 2)); p.drawLine(100 + bw_max, y + 2, 100 + bw_max, y + 16)
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            over = "  OVER" if ratio > 1.0 else ""
            p.drawText(100 + bw_max + 6, y + 14, f"{meas:.1f}/{b:.0f}ms{over}")


# ----- NEW Memory & Belief tab ------------------------------------------------

class MemoryBeliefView(_BaseCanvas):
    """M3 recent/high-error episodes + M4 relevant/top facts + belief/precision
    portrait + G′ distribution thumbnail. The 'what the agent remembers' view."""

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Memory & belief…"); return
        # ---- left column: M3 episodes (recent + high-error) ----
        col_w = w // 2 - 8
        self._title(p, "M3 episodic memory (recent / high-error)", x=10, y=14)
        self._caption(p, "recent episodes + highest prediction-error replays · mini state preview", x=10, y=26)
        m3r = getattr(f, "m3_recent", None) or []
        m3e = getattr(f, "m3_top_error", None) or []
        y = 30
        p.setPen(QtGui.QColor(46, 204, 113)); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(10, y, f"recent ({len(m3r)})"); y += 12
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 7))
        for ep in m3r[:4]:
            y = self._ep_line(p, ep, 10, y, col_w)
        y += 6
        p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(10, y, f"high-error ({len(m3e)})"); y += 12
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 7))
        for ep in m3e[:4]:
            y = self._ep_line(p, ep, 10, y, col_w)
        # ---- right-top: M4 facts (relevant + top) ----
        rx = col_w + 16
        self._title(p, "M4 facts (relevant to state / top by support)", x=rx, y=14)
        self._caption(p, "consolidated statistical facts · relevance to current state + support rank", x=rx, y=26)
        m4r = getattr(f, "m4_relevant", None) or []
        m4t = getattr(f, "m4_top", None) or []
        y = 30
        p.setPen(QtGui.QColor(155, 89, 182)); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(rx, y, f"relevant ({len(m4r)})"); y += 12
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 7))
        for fac in m4r[:4]:
            y = self._fact_line(p, fac, rx, y)
        y += 4
        p.setPen(ACCENT); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(rx, y, f"top ({len(m4t)})"); y += 12
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 7))
        for fac in m4t[:4]:
            y = self._fact_line(p, fac, rx, y)
        # ---- bottom: belief portrait (sanitized vs raw diff + precision) ----
        bt = h // 2 + 20
        self._title(p, "Belief portrait — |sanitized−raw| per-dim + state-precision", x=10, y=bt)
        self._caption(p, "per-dim salience (sanitised vs raw) · precision weights · named dims", x=10, y=bt + 12)
        self._belief_portrait(p, f, 10, bt + 10, w - 20, h - bt - 30)

    def _ep_line(self, p, ep: dict, x: int, y: int, w: int) -> int:
        did = ep.get("drive_id", "?"); pe = ep.get("prediction_error")
        conf = ep.get("confidence"); ts = ep.get("timestamp", "")
        pe_s = f"{float(pe):.3f}" if pe is not None else "?"
        conf_s = f"{float(conf):.2f}" if conf is not None else "?"
        line = f"d{did}  err={pe_s}  conf={conf_s}  t={str(ts)[-7:]}"
        p.drawText(x, y, line)
        # state-before → state-after mini preview (sb_head / sa_head from _episode_to_dict)
        sb = ep.get("sb_head"); sa = ep.get("sa_head")
        if sb is not None and sa is not None:
            try:
                sbv = np.asarray(sb, dtype=np.float32).reshape(-1)[:8]
                sav = np.asarray(sa, dtype=np.float32).reshape(-1)[:8]
                mx = max(float(np.max(np.abs(sbv)) if sbv.size else 1.0),
                         float(np.max(np.abs(sav)) if sav.size else 1.0)) or 1.0
                bx = x + 200; bw = 8
                for i in range(sbv.size):
                    bh = int(abs(float(sbv[i])) / mx * 8)
                    p.fillRect(bx + i * bw, y - bh, bw - 1, bh, QtGui.QColor(150, 150, 160, 160))
                    bh2 = int(abs(float(sav[i])) / mx * 8)
                    p.fillRect(bx + i * bw, y - bh2, bw - 1, bh2, QtGui.QColor(46, 204, 113, 200))
            except Exception:
                pass
        return y + 12

    def _fact_line(self, p, fac: dict, x: int, y: int) -> int:
        ftype = str(fac.get("fact_type", fac.get("predicate", fac.get("type", "?"))))
        summary = str(fac.get("summary", ""))
        freq = fac.get("frequency", fac.get("support", "?"))
        conf = fac.get("confidence", "?")
        head = f"{ftype}  f={freq} c={conf}"
        p.drawText(x, y, head[:40])
        if summary:
            p.setPen(DIM_COL); p.drawText(x + 40 if len(head) < 38 else x,
                                          y + 10, summary[:50])
            p.setPen(TEXT_COL)
            return y + 22
        return y + 12

    def _belief_portrait(self, p, f, x, y, w, h) -> None:
        raw = f.obs_vector; san = f.sanitized_state; prec = f.state_precision
        if raw is None or san is None:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(x, y + 14, "(no sanitized/raw state this frame)"); return
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        san = np.asarray(san, dtype=np.float32).reshape(-1)
        d = min(len(raw), len(san))
        MAX = 32
        idx = np.arange(d)
        if d > MAX:
            diff = np.abs(san[:d] - raw[:d])
            idx = np.argsort(-diff)[:MAX]; idx.sort()
        diff = np.abs(san[idx] - raw[idx])
        n = len(idx); bw = w / n
        mx = float(diff.max()) if diff.size else 1.0
        mx = mx if mx > 1e-6 else 1.0
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, y + h - 14, x + w, y + h - 14)
        for i in range(n):
            bx = int(x + i * bw)
            bh = int(diff[i] / mx * (h - 20))
            p.fillRect(bx + 1, y + h - 14 - bh, max(int(bw) - 2, 1), bh, QtGui.QColor(231, 76, 60, 200))
            # precision overlay (amber ticks)
            if prec is not None and i < len(prec):
                ph = int(float(prec[idx[i]]) * (h - 20))
                p.setPen(QtGui.QPen(ACCENT, 1))
                p.drawLine(bx + int(bw / 2), y + h - 14, bx + int(bw / 2), y + h - 14 - ph)
        self._legend(p, [("█ |sanitized−raw|", QtGui.QColor(231, 76, 60)),
                         ("| precision", ACCENT)], y=y + h - 2, x=x)


# ----- NEW Goals & Motivation tab ---------------------------------------------

class GoalsMotivationView(_BaseCanvas):
    """MDIM goal stack + drives full (value/deficit/target) + Pareto + meta-stable
    + drive/goal history heatmaps + temperature/empowerment. The 'why' portrait."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drive_hist: Deque[List[float]] = deque(maxlen=120)
        self.goal_hist: Deque[int] = deque(maxlen=120)
        self.temp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.emp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)

    def set_frame(self, f: ObservabilityFrame) -> None:
        if f.drive_deficits is not None:
            self.drive_hist.append([float(x) for x in np.asarray(f.drive_deficits, dtype=np.float32).reshape(-1)[:6]])
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
        # ---- left: goal stack ----
        lw = w // 3
        self._title(p, "Goal stack (deepest active first)", x=8, y=14)
        self._caption(p, "active sub-goals by depth · ✓done = completed · target magnitude shown", x=8, y=26)
        stack = getattr(f, "goal_stack", None) or []
        y = 30
        for i, g in enumerate(stack[:8]):
            did = g.get("drive_id", "?")
            tnorm = g.get("target_norm")
            tol = g.get("tolerance"); pri = g.get("priority")
            comp = g.get("completed", False)
            depth = g.get("depth", "?")
            tgt_s = f"|tgt|={float(tnorm):.2f}" if tnorm is not None else "no target"
            col = QtGui.QColor(46, 204, 113) if comp else ACCENT
            p.setPen(col); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(8, y, f"#{i+1} d{did} {tgt_s}")
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(8, y + 11, f"depth={depth} tol={tol} pri={pri} {'✓done' if comp else 'active'}")
            y += 26
        if not stack:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(8, y, "(no active goals)")
        # ---- middle: drives compact strip + Pareto + meta-stable ----
        # v5: replaced the per-drive deficit BAR ROW (duplicated the heatmap)
        # with a compact "deficit now" numeric strip + Pareto ● markers.
        mx = lw + 8
        mw = w // 3
        self._title(p, "Drives — deficit now + Pareto● (heatmap → right)", x=mx, y=14)
        self._caption(p, "one clear deficit view (strip here, history heatmap right); ● = on Pareto front", x=mx, y=26)
        defs = f.drive_deficits; levels = f.drive_levels
        targets = getattr(f, "drive_goals", None)
        pareto = set(getattr(f, "pareto_front", None) or [])
        y = 42
        # compact numeric strip header
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(mx, y, "drive   value  deficit  target")
        y += 16
        for i in range(6):
            did = i + 1
            val = float(levels[i]) if levels is not None and i < len(levels) else 0.0
            dfc = float(defs[i]) if defs is not None and i < len(defs) else 0.0
            tgt = None
            if targets is not None and i < len(targets) and targets[i] is not None:
                try:
                    tgt = float(targets[i])
                except Exception:
                    tgt = None
            col = _to_qcolor(DRIVE_COLORS[did])
            mark = "●" if i in pareto else " "
            p.setPen(col); p.setFont(_F_LABEL_B)
            p.drawText(mx, y, f"{mark} {DRIVE_SHORT[did]}")
            p.setPen(TEXT_COL); p.setFont(_F_LABEL)
            tgt_s = f"{tgt:.2f}" if tgt is not None else "—"
            p.drawText(mx + 40, y, f"{val:.2f}   {dfc:.2f}    {tgt_s}")
            y += 16
        ms = getattr(f, "meta_stable", None) or {}
        stable = bool(ms.get("is_meta_stable", ms.get("stable", False)))
        p.setPen(ACCENT if stable else QtGui.QColor(231, 76, 60))
        p.setFont(_F_LABEL_B)
        cse = ms.get("cycles_since_entry", "?")
        p.drawText(mx, y + 6, f"meta-stable: {'YES' if stable else 'NO'}  ({cse} cyc)")
        # ---- right: heatmaps + temperature/empowerment ----
        rx = mx + mw + 8
        self._title(p, "Drive history heatmap + temperature/empowerment", x=rx, y=14)
        self._caption(p, "deficit-over-time (6 drives × cycles) · temperature/empowerment trend below", x=rx, y=26)
        self._heatmap(p, rx, 28, w - rx - 8, h // 2 - 30)
        self._trend_pair(p, rx, h // 2 + 10, w - rx - 8, h - h // 2 - 20)

    def _heatmap(self, p, x, y, w, h) -> None:
        hist = list(self.drive_hist)
        if not hist:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(x, y + 10, "drive×cycle deficit heatmap (collecting…)"); return
        n = len(hist); cw = w / max(n, 1); rh = h / 6
        for i, defs in enumerate(hist):
            for j in range(min(6, len(defs))):
                a = int(np.clip(defs[j], 0, 1) * 230)
                if a < 8:
                    continue
                col = _to_qcolor(DRIVE_COLORS[j + 1]); col.setAlpha(a)
                p.fillRect(int(x + i * cw), int(y + j * rh), int(cw) + 1, int(rh) - 1, col)
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(x, y + h + 8, f"6 drives × {n} cycles (darker=deficit higher)")

    def _trend_pair(self, p, x, y, w, h) -> None:
        half = h // 2
        self._mini_trend(p, self.temp_hist, x, y, w, half, QtGui.QColor(231, 126, 34), "CR temperature")
        self._mini_trend(p, self.emp_hist, x, y + half + 4, w, half, QtGui.QColor(52, 152, 219), "empowerment")

    def _mini_trend(self, p, s, x, y, w, h, col, label) -> None:
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
        p.drawText(x, y + 10, label)
        top, bot = y + 16, y + h - 2
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, bot, x + w, bot)
        vals = list(s)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(x + 4, top + 12, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9: hi = lo + 1
        n = len(vals); path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + i * w / (n - 1); py = bot - (v - lo) / (hi - lo) * (bot - top - 2)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.setPen(QtGui.QPen(col, 2)); p.drawPath(path)


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
            # single source of truth for the shared belief projection + history
            self.w.proj.update(f)
            v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            self.w.proj.push_history(self.w.proj.project(v))
            self.w.world.set_frame(f)
            self.w.drives.set_frame(f)
            self.w.trend.push(f)
            self.w.attention.set_frame(f)
            self.w.status.set_state(f, cycle_error)
            self.w.flow.set_frame(f)
            self.w.cand.set_frame(f)
            self.w.traj.set_frame(f)
            self.w.radar.set_frame(f)
            self.w.perdim.set_frame(f)
            self.w.dim_selector.refresh()
            self.w.retention.set_frame(f)
            self.w.rbta_bounds.set_frame(f)
            self.w.viol.add_frame(f)
            self.w.memory.set_frame(f)
            self.w.goals.set_frame(f)
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

        # shared dimension-agnostic projection (World / Phase-Space / Action)
        self.proj = BeliefProjection(window=256)

        # Overview
        ov = QtWidgets.QWidget(); ov_lay = QtWidgets.QGridLayout(ov)
        ov_lay.setContentsMargins(6, 6, 6, 6); ov_lay.setSpacing(6)
        self.world = WorldCanvas(); self.world.set_projection(self.proj)
        self.drives = DrivesCanvas()
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
        self.cand = CandidateScoreView(); self.cand.set_projection(self.proj)
        tabs.addTab(self.cand, "Action Selection")

        # Phase Space & Belief Uncertainty
        ps = QtWidgets.QWidget(); ps_lay = QtWidgets.QGridLayout(ps)
        ps_lay.setContentsMargins(6, 6, 6, 6); ps_lay.setSpacing(6)
        self.traj = TrajectoryView(); self.traj.set_projection(self.proj)
        self.radar = DriveRadarView()
        # v5: consolidated per-dim error + uncertainty (one view, not two) + pager
        self.perdim = _PhasePortraitView()
        self.dim_selector = _DimSelector(self.perdim)
        ps_lay.addWidget(self.traj, 0, 0, 2, 2)
        ps_lay.addWidget(self.radar, 0, 2, 1, 1)
        ps_lay.addWidget(self.perdim, 2, 0, 1, 3)
        ps_lay.addWidget(self.dim_selector, 3, 0, 1, 3)
        tabs.addTab(ps, "Phase Space & Trajectory")

        # Retention & Resources
        ret = QtWidgets.QWidget(); ret_lay = QtWidgets.QVBoxLayout(ret)
        ret_lay.setContentsMargins(6, 6, 6, 6); ret_lay.setSpacing(6)
        self.retention = RetentionView()
        self.rbta_bounds = RBTABoundsView()
        self.rbta_bounds.setMaximumHeight(220)
        self.viol = ViolationTable()
        ret_lay.addWidget(self.retention)
        ret_lay.addWidget(self.rbta_bounds)
        ret_lay.addWidget(QtWidgets.QLabel("RBTA violation log (scrolling, last 200):"))
        ret_lay.addWidget(self.viol, 1)
        tabs.addTab(ret, "Retention & Resources")

        # NEW: Memory & Belief
        self.memory = MemoryBeliefView()
        tabs.addTab(self.memory, "Memory & Belief")

        # NEW: Goals & Motivation
        self.goals = GoalsMotivationView()
        tabs.addTab(self.goals, "Goals & Motivation")

        self.controller = DashboardController(self)


def make_app() -> "QtWidgets.QApplication":
    """Construct (but do not exec) the QApplication. Caller owns it."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(_qss())
    return app
