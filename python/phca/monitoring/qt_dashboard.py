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
from .playback import _Smoother, freeze_sig
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
        # v7: cache singular values + previous basis for variance-% and a
        # basis-stability flag (so views can badge "PCA re-fit" instead of
        # letting the portrait silently drift as the window slides).
        self._sing: Optional[np.ndarray] = None
        self._prev_comp: Optional[np.ndarray] = None
        self._basis_changed: bool = False

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
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            if Vt.shape[0] >= 2:
                new_comp = Vt[:2].astype(np.float64)
                # v7: basis-stability flag — did the principal directions flip/rotate?
                if self._prev_comp is not None and self._prev_comp.shape == new_comp.shape:
                    diff = float(np.max(np.abs(np.abs(new_comp) - np.abs(self._prev_comp))))
                    self._basis_changed = diff > 0.25
                else:
                    self._basis_changed = True
                self._prev_comp = new_comp
                self._comp = new_comp
                self._sing = np.asarray(S, dtype=np.float64)
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
        self._sing = None

    def variance_explained(self) -> Optional[float]:
        """v7: % of variance captured by PC1+PC2 (None if no SVD basis)."""
        if self._sing is None or self._sing.size == 0:
            return None
        tot = float(self._sing.sum())
        if tot <= 0:
            return None
        return float(self._sing[:2].sum() / tot * 100.0)

    @property
    def basis_changed(self) -> bool:
        """v7: True on the frame the PCA basis rotated/flipped (for a re-fit badge)."""
        return self._basis_changed

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
        # v6: repaint counter for the flicker-free acceptance gate.
        self.repaint_count: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.update()

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:
        self.repaint_count += 1
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

class AgentPortraitView(_BaseCanvas):
    """v6 focal 'picture of the cognitive agent' — a single composite organism
    whose morphology encodes internal state, surrounded by semantic gauges and
    one convergence trend. Dimension-agnostic (scalars only) → works for any
    future 2D / n-D environment. Additive; does not replace any existing view.

    Glyph encoding:
      - body halo : 6 drive segments (radius = level, alpha = level, gap = deficit)
      - core disc : colour = prediction confidence (green→red), brightness = empowerment
      - breath    : slow ~0.4 Hz wall-clock ring (calm, not a strobe)
      - head      : marker on the active-drive segment
      - limbs     : 2 limbs encoding the action vector direction/magnitude
    Gauges (radial): PEU/free-energy, confidence, empowerment, meta-stability,
      mean attention precision, RBTA time-headroom.
    Trend: composite free-energy (prediction_error) sparkline + ↘/↗ health arrow.
    """

    GAUGES = ("free-energy", "mutual-info", "cr-temp",
              "mean-PEU", "RBTA-room", "goal-pri")
    # v7: gauge units + human meaning (for the legend/tooltip under each arc).
    GAUGE_META = {
        "free-energy": ("", "prediction error (free energy)"),
        "mutual-info": ("bit", "G′ mutual information"),
        "cr-temp": ("", "CR cooling temperature"),
        "mean-PEU": ("", "mean predictive empowerment"),
        "RBTA-room": ("%", "RBTA time headroom"),
        "goal-pri": ("", "active goal priority"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self._sm = {k: _Smoother(alpha=0.25) for k in self.GAUGES}
        self._raw: Dict[str, float] = {k: 0.0 for k in self.GAUGES}
        self._fe_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._t0 = time.monotonic()

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        # feed smoothers + free-energy history + v7 unclamped raw values
        fe = float(getattr(f, "prediction_error", 0.0) or 0.0)
        self._fe_hist.append(fe)
        self._sm["free-energy"].value(fe); self._raw["free-energy"] = fe
        mi = float(getattr(f, "gprime_mutual_info", 0.0) or 0.0)
        self._sm["mutual-info"].value(mi); self._raw["mutual-info"] = mi
        ct = float(getattr(f, "cr_temperature", 0.0) or 0.0)
        self._sm["cr-temp"].value(ct); self._raw["cr-temp"] = ct
        emp = float(getattr(f, "empowerment", 0.0) or 0.0)
        self._sm["mean-PEU"].value(emp); self._raw["mean-PEU"] = emp
        room = self._rbta_room(f)
        self._sm["RBTA-room"].value(room); self._raw["RBTA-room"] = room * 100.0
        gp = float(getattr(f, "goal_priority", 0.0) or 0.0)
        self._sm["goal-pri"].value(gp); self._raw["goal-pri"] = gp
        self.update()

    @staticmethod
    def _rbta_room(f: ObservabilityFrame) -> float:
        bounds = getattr(f, "rbta_bounds", None) or {}
        timings = getattr(f, "module_timings", None) or {}
        worst = 0.0
        for k, v in bounds.items():
            if not isinstance(v, dict):
                continue
            t = v.get("time")
            if not t or t <= 0:
                continue
            m = 0.0
            for mod in (k,):
                mv = timings.get(mod)
                if mv is None:
                    # rbta keys may be module ids; try matching via flow map
                    mv = timings.get(RBTA_TO_FLOW.get(mod, mod))
                if mv is not None:
                    m = max(m, float(mv))
            if m > 0:
                worst = max(worst, m / float(t))
        return max(0.0, 1.0 - worst)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Agent portrait (collecting…)"); return
        self._title(p, "Cognitive agent — portrait")
        self._caption(p, "halo=drives · core colour=confidence · brightness=empowerment · "
                         "head=active drive · core dot=meta-stable · limbs=action · "
                         "gauges=vitals (raw+units) · under-halo=Δ deficits")
        # layout: glyph on the left, gauges grid on the right, trend at bottom
        glyph_cx = w // 4
        glyph_cy = h // 2 + 6
        R = min(w // 5, h // 3)
        self._draw_glyph(p, f, glyph_cx, glyph_cy, R)
        # gauges grid (3 cols x 2 rows) on the right half
        gx0 = w // 2
        gw = (w - gx0 - 8) // 3
        gh = (h // 2 - 30) // 2
        for i, name in enumerate(self.GAUGES):
            col = i % 3; row = i // 3
            x = gx0 + col * gw; y = 36 + row * gh
            self._draw_gauge(p, name, self._sm[name]._v, x + 4, y + 4, gw - 8, gh - 8)
        # free-energy convergence trend at the bottom-right
        self._draw_trend(p, gx0, h // 2 + 14, w - gx0 - 8, h // 2 - 28)

    def _draw_glyph(self, p, f, cx, cy, R):
        import math
        levels = list(getattr(f, "drive_levels", []) or [0.0] * 6)
        deficits = list(getattr(f, "drive_deficits", []) or [0.0] * 6)
        active = int(getattr(f, "active_drive_id", 0) or 0)
        conf = float(getattr(f, "prediction_confidence", 0.0) or 0.0)
        emp = self._sm["mean-PEU"]._v
        ms = getattr(f, "meta_stable", None) or {}
        stable = bool(ms.get("is_meta_stable", ms.get("stable", False)))
        # v7: calm breath ring — fixed radius, narrow alpha band 60→90 (no pulse)
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * (time.monotonic() - self._t0) * 0.4)
        br = int(R * 1.20)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, int(60 + 30 * breath)), 2))
        p.drawEllipse(cx - br, cy - br, br * 2, br * 2)
        # drive halo: 6 segments
        for i in range(6):
            did = i + 1
            lvl = float(levels[i]) if i < len(levels) else 0.0
            dfc = float(deficits[i]) if i < len(deficits) else 0.0
            lvl = max(0.0, min(1.0, lvl))
            a0 = -math.pi / 2 + i * math.pi / 3
            a1 = a0 + math.pi / 3 - 0.18 * (dfc)  # gap widens with deficit
            r_out = int(R * (0.55 + 0.45 * lvl))
            col = _to_qcolor(DRIVE_COLORS[did]); col.setAlpha(int(80 + 160 * lvl))
            p.setBrush(col); p.setPen(QtGui.QPen(col.darker(140), 1))
            from PyQt5 import QtGui as _QtGui
            path = _QtGui.QPainterPath()
            path.moveTo(cx + R * 0.45 * math.cos(a0), cy + R * 0.45 * math.sin(a0))
            path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2,
                       math.degrees(a0), math.degrees(a1 - a0))
            path.lineTo(cx + R * 0.45 * math.cos(a1), cy + R * 0.45 * math.sin(a1))
            p.drawPath(path)
        # core: colour by confidence (green→amber→red), brightness by empowerment
        cval = max(0.0, min(1.0, conf))
        if cval > 0.66:
            core = QtGui.QColor(46, 204, 113)
        elif cval > 0.33:
            core = QtGui.QColor(241, 196, 15)
        else:
            core = QtGui.QColor(231, 76, 60)
        b = int(120 + 110 * max(0.0, min(1.0, emp)))
        core.setAlpha(min(255, b))
        cr = int(R * 0.4)
        p.setBrush(core); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 2))
        p.drawEllipse(cx - cr, cy - cr, cr * 2, cr * 2)
        # v7: meta-stable as a binary dot inside the core (not a 270° arc)
        if stable:
            p.setBrush(QtGui.QColor(241, 196, 15)); p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            p.drawEllipse(cx + cr - 12, cy - cr + 4, 8, 8)
        # active-drive head marker
        if 1 <= active <= 6:
            i = active - 1
            a = -math.pi / 2 + i * math.pi / 3 + math.pi / 6
            hx = cx + (R * 1.02) * math.cos(a); hy = cy + (R * 1.02) * math.sin(a)
            p.setBrush(_to_qcolor(DRIVE_COLORS[active]))
            p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            p.drawEllipse(int(hx) - 5, int(hy) - 5, 10, 10)
        # limbs: action vector (continuous) or chosen action_names label (discrete)
        self._draw_limbs(p, f, cx, cy, cr, R)
        # v7: deficits under the halo (top-3 Δ D#=…)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        order = sorted(range(6), key=lambda i: -(deficits[i] if i < len(deficits) else 0.0))[:3]
        dy = cy + br + 12
        for i in order:
            if i < len(deficits) and deficits[i] > 0.02:
                p.setPen(_to_qcolor(DRIVE_COLORS[i + 1]))
                p.drawText(cx - br, dy, f"Δ D{i+1}={deficits[i]:.2f}")
                dy += 11
        # core label
        p.setPen(QtGui.QColor(20, 20, 24)); p.setFont(_F_LABEL_B)
        p.drawText(cx - cr, cy - cr, cr * 2, cr * 2, 0x84,
                   f"conf\n{conf:.2f}")

    def _draw_limbs(self, p, f, cx, cy, cr, R):
        import math
        ca = getattr(f, "continuous_action", None)
        names = list(getattr(f, "action_names", []) or [])
        if ca is not None:
            vec = np.asarray(ca, dtype=np.float32).reshape(-1)
            if vec.size >= 2:
                vx = float(np.clip(vec[0], -1, 1)); vy = float(np.clip(vec[1], -1, 1))
                mag = float(min(1.0, math.hypot(vx, vy)))
                ang = math.atan2(vy, vx)
                for off in (0.0, math.pi):
                    a = ang + off
                    ex = cx + (cr + R * 0.5 * mag) * math.cos(a)
                    ey = cy + (cr + R * 0.5 * mag) * math.sin(a)
                    p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
                    p.drawLine(cx, cy, int(ex), int(ey))
                    p.setBrush(QtGui.QColor(155, 89, 182)); p.setPen(QtCore.Qt.white)
                    p.drawEllipse(int(ex) - 4, int(ey) - 4, 8, 8)
                p.setPen(ACCENT); p.setFont(_F_AXIS)
                p.drawText(cx - R, cy + cr + 16, f"τ‖·‖={mag:.2f}")
                return
            # continuous but 1-D → magnitude arc
            mag = float(min(1.0, abs(float(vec[0]))))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
            p.drawArc(cx - cr - 6, cy - cr - 6, (cr + 6) * 2, (cr + 6) * 2, 90 * 16, int(-mag * 360 * 16))
            return
        # v7 discrete: one limb in the chosen action's direction + action_names label
        scores = list(getattr(f, "candidate_scores", []) or [0])
        chosen = int(np.argmax(scores)) if scores else 0
        a = -math.pi / 2
        ex = cx + (cr + R * 0.4) * math.cos(a); ey = cy + (cr + R * 0.4) * math.sin(a)
        p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 220), 3))
        p.drawLine(cx, cy, int(ex), int(ey))
        lbl = names[chosen] if chosen < len(names) else f"a{chosen}"
        p.setPen(ACCENT); p.setFont(_F_LABEL_B)
        p.drawText(cx - R, cy + cr + 16, f"act: {lbl}")

    def _draw_gauge(self, p, name, v, x, y, w, h):
        import math
        v = max(0.0, min(1.0, float(v)))
        cx = x + w // 2; cy = y + h - 6; R = min(w // 2 - 4, h - 16)
        # track arc (270° sweep)
        p.setPen(QtGui.QPen(GRID_COL, 4)); p.setBrush(QtCore.Qt.NoBrush)
        p.drawArc(cx - R, cy - R, R * 2, R * 2, 225 * 16, -270 * 16)
        # value arc
        col = QtGui.QColor(46, 204, 113) if v > 0.66 else (
            QtGui.QColor(241, 196, 15) if v > 0.33 else QtGui.QColor(231, 76, 60))
        p.setPen(QtGui.QPen(col, 4)); p.drawArc(cx - R, cy - R, R * 2, R * 2,
                                               225 * 16, int(-270 * 16 * v))
        # needle
        ang = math.radians(225 + 270 * v)
        nx = cx + (R - 2) * math.cos(ang); ny = cy - (R - 2) * math.sin(ang)
        p.setPen(QtGui.QPen(QtCore.Qt.white, 2)); p.drawLine(cx, cy, int(nx), int(ny))
        p.setBrush(col); p.setPen(QtCore.Qt.white); p.drawEllipse(cx - 3, cy - 3, 6, 6)
        # v7: label + UNCLAMPED raw value + units (arc is 0..1 normalised; the
        # number beneath shows the true magnitude so free-energy/MI saturating
        # the arc still reads).
        unit, meaning = self.GAUGE_META.get(name, ("", ""))
        raw = self._raw.get(name, v)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + 10, name)
        p.setPen(DIM_COL)
        p.drawText(x, y + h - 2, f"{raw:.2f}{unit}")
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + 22, meaning[:22])

    def _draw_trend(self, p, x, y, w, h):
        self._title(p, "free-energy convergence (↘ healthy)", x=x, y=y + 12)
        vals = list(self._fe_hist)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, y + h - 4, x + w, y + h - 4)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, y + 30, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9: hi = lo + 1
        n = len(vals)
        # health arrow: compare last quarter mean to first quarter
        q = max(1, n // 4)
        early = float(np.mean(vals[:q])); late = float(np.mean(vals[-q:]))
        arrow = "↘ healthy" if late <= early else "↗ rising"
        col = QtGui.QColor(46, 204, 113) if late <= early else QtGui.QColor(231, 76, 60)
        p.setPen(col); p.setFont(_F_LABEL_B)
        p.drawText(x + w - 110, y + 12, arrow)
        # sparkline
        p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + i * w / max(n - 1, 1)
            py = (y + h - 4) - (v - lo) / (hi - lo) * (h - 22)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.drawPath(path)


class WorldCanvas(_BaseCanvas):
    """Env-adaptive world view: grid / MuJoCo RGB camera + overlay / PCA
    projection. The single spatial widget that scales grid→2D→3D→n-D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.trail: Deque[Any] = deque(maxlen=160)
        self.proj: Optional[BeliefProjection] = None
        # v6: 2D arena path for low-dim continuous envs (future 2D phases).
        self.arena_trail: Deque[Tuple[float, float]] = deque(maxlen=256)
        self._ax = ScaleState(contract=0.04, head=0.08)
        self._ay = ScaleState(contract=0.04, head=0.08)

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if f.grid is not None and f.agent_pos is not None:
            ap = tuple(f.agent_pos)
            if not self.trail or self.trail[-1] != ap:
                self.trail.append(ap)
        # v6: 2D arena trail for low-dim continuous envs (x,y from first 2 dims).
        kind = (getattr(f, "env_kind", "") or "").lower()
        sd = int(getattr(f, "state_dim", 0) or 0)
        if kind == "continuous" and 2 <= sd <= 4 and f.obs_vector is not None:
            v = np.asarray(f.obs_vector, dtype=np.float32).reshape(-1)
            if v.size >= 2:
                pt = (float(v[0]), float(v[1]))
                if not self.arena_trail or self.arena_trail[-1] != pt:
                    self.arena_trail.append(pt)
        # projection history is pushed once by the controller (single source)
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            self._empty(p, "Awaiting cycle…"); return
        w, h = self.width(), self.height()
        kind = (getattr(f, "env_kind", "") or "").lower()
        sd = int(getattr(f, "state_dim", 0) or 0)
        if f.grid is not None:
            self._draw_grid(p, f, w, h)
        elif kind == "mujoco_rgb":
            self._draw_rgb(p, f, w, h)
        elif kind == "continuous" and 2 <= sd <= 4:
            self._draw_arena(p, f, w, h)
        else:
            self._draw_projection(p, f, w, h)

    def _draw_arena(self, p, f, w, h):
        """2D arena: agent dot + goal + trail in the x-y plane (low-dim
        continuous). Stable bounds (no jitter). Forward-looking for 2D phases."""
        margin = 24
        left, right, top, bot = margin, w - margin, 30, h - 24
        self._title(p, f"2D arena (x,y)  dim={f.state_dim}  ●agent ◆goal · trail")
        self._caption(p, "low-dim continuous env → planar view (higher-dim → PCA projection)")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(left, top, right - left, bot - top)
        trail = list(self.arena_trail)
        if not trail:
            self._empty(p, "2D arena (collecting…)"); return
        xs = [pt[0] for pt in trail]; ys = [pt[1] for pt in trail]
        xlo, xhi = self._ax.update(float(min(xs)), float(max(xs)))
        ylo, yhi = self._ay.update(float(min(ys)), float(max(ys)))
        if xhi - xlo < 1e-9: xhi = xlo + 1
        if yhi - ylo < 1e-9: yhi = ylo + 1

        def _m(x, y):
            px = left + (x - xlo) / (xhi - xlo) * (right - left)
            py = bot - (y - ylo) / (yhi - ylo) * (bot - top)
            return int(px), int(py)
        # trail
        n = len(trail)
        for i in range(1, n):
            a = int(40 + 180 * i / n)
            x0, y0 = _m(*trail[i - 1]); x1, y1 = _m(*trail[i])
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, a), 2))
            p.drawLine(x0, y0, x1, y1)
        # goal
        gref = getattr(f, "goal_ref", None)
        if gref is not None:
            gv = np.asarray(gref, dtype=np.float32).reshape(-1)
            if gv.size >= 2:
                gx, gy = _m(float(gv[0]), float(gv[1]))
                p.setBrush(QtGui.QColor(46, 204, 113, 180))
                p.setPen(QtGui.QPen(QtGui.QColor(46, 204, 113), 1))
                p.drawEllipse(gx - 6, gy - 6, 12, 12)
        # agent
        ax, ay = _m(*trail[-1])
        p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        p.drawEllipse(ax - 5, ay - 5, 10, 10)

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
               val: float, vmax: float, unit: str, col: QtGui.QColor,
               gw: int = 150) -> int:
        gh = 8
        p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Monospace", 9))
        p.drawText(x, y, f"{label} {val:.1f}{unit}")
        frac = float(np.clip(val / vmax, 0, 1)) if vmax > 0 else 0.0
        p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1))
        p.setBrush(QtGui.QColor(30, 30, 38))
        p.drawRect(x, y + 4, gw, gh)
        p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
        p.fillRect(x, y + 4, int(gw * frac), gh, col)
        return gw + 12

    def _chip(self, p, x, y, text, col):
        p.setFont(QtGui.QFont("Monospace", 8, QtGui.QFont.Bold))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 10; h = 14
        p.setPen(QtGui.QPen(col, 1)); p.setBrush(col.darker(160))
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(col); p.drawText(x + 5, y + 10, text)
        return w + 6

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None and self.cycle_error is None:
            self._empty(p, "Status…"); return
        if self.cycle_error:
            p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Monospace", 10, QtGui.QFont.Bold))
            p.drawText(self.rect(), 0x84, f"CYCLE ERROR:\n{self.cycle_error}")
            return
        W = self.width()
        p.setFont(QtGui.QFont("Monospace", 9))
        y = 20
        # v7: status chips (RBTA, goal_reached, meta_stable)
        rbta_ok = (f.violations_count == 0)
        cx = 10
        cx += self._chip(p, cx, y - 10, "RBTA " + ("OK" if rbta_ok else f"{f.violations_count}V"),
                         QtGui.QColor(46, 204, 113) if rbta_ok else QtGui.QColor(231, 76, 60))
        if getattr(f, "goal_reached", False):
            cx += self._chip(p, cx, y - 10, "GOAL", QtGui.QColor(241, 196, 15))
        ms = getattr(f, "meta_stable", None) or {}
        if ms.get("is_meta_stable", ms.get("stable", False)):
            cx += self._chip(p, cx, y - 10, "META-STABLE", QtGui.QColor(52, 152, 219))
        p.setPen(TEXT_COL)
        p.drawText(cx + 4, y, f"Cycle {f.cycle_id}")
        y += 15
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
        # v7: proportional-width gauges (fit the panel)
        gw = max(80, (W - 40) // 2)
        x = self._gauge(p, 10, y, "RSS", f.rss_bytes / 1e6, 1024.0, "MB",
                        QtGui.QColor(230, 126, 34), gw=gw)
        self._gauge(p, x, y, "lat", f.latency_ms, 50.0, "ms",
                    QtGui.QColor(46, 204, 113), gw=gw)
        y += 24
        # v7: 6-segment drive-deficit micro-strip (replaces the MDIM-goal text line)
        deficits = list(getattr(f, "drive_deficits", []) or [0.0] * 6)
        active = int(getattr(f, "active_drive_id", 0) or 0)
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Monospace", 8))
        p.drawText(10, y, "Δ drives:")
        sx = 70; sw = (W - sx - 10) // 6
        for i in range(6):
            d = float(deficits[i]) if i < len(deficits) else 0.0
            col = _to_qcolor(DRIVE_COLORS[i + 1])
            col.setAlpha(int(60 + 180 * min(1.0, d)))
            p.setBrush(col)
            p.setPen(QtGui.QPen(QtCore.Qt.white, 2 if (i + 1) == active else 1))
            p.drawRect(sx + i * sw, y - 8, sw - 2, 12)
            p.setPen(TEXT_COL)
            p.drawText(sx + i * sw + 2, y + 2, f"D{i+1}:{d:.2f}")
        y += 16
        r = f.action_rationale or {}
        gid = r.get("goal_id")
        goal_lbl = DRIVE_NAMES.get(gid, gid) if gid is not None else "—"
        tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
        note = r.get("note", "")
        score = r.get("best_score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        k = r.get("k_candidates")
        k_s = str(k) if k is not None else "—"
        head = f"Action goal: {goal_lbl}   {tag}  ε={r.get('eps',0):.3f}  K={k_s}  score={score_s}"
        p.setPen(TEXT_COL)
        p.drawText(10, y, head)
        y += 13
        # v7: wrap the rationale note inside the panel (was clipped on one line)
        if note:
            p.setPen(DIM_COL)
            p.setFont(QtGui.QFont("Monospace", 8))
            fm = p.fontMetrics()
            rect = QtCore.QRect(10, y, W - 20, 60)
            p.drawText(rect, 0x81, f"note: {note}")


# ----- Cognitive Flow tab ----------------------------------------------------

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

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.cycle_id = int(f.cycle_id)
        self.heat.append(dict(f.module_timings))
        now = time.monotonic()
        dt = now - self._last_wall; self._last_wall = now
        self._anim_t = (self._anim_t + dt / 0.5) % len(PIPELINE)
        # active node = most-recently-active by EMA'd timing delta + hysteresis
        try:
            if len(self.heat) >= 2:
                prev, cur = self.heat[-2], self.heat[-1]
                for m in PIPELINE:
                    d = float(cur.get(m, 0.0)) - float(prev.get(m, 0.0))
                    self._delta_smooth[m].value(d)
                if self._active_hold > 0:
                    self._active_hold -= 1
                else:
                    sm = {m: self._delta_smooth[m]._v for m in PIPELINE}
                    new = int(max(range(len(PIPELINE)), key=lambda i: sm[PIPELINE[i]]))
                    cur_v = sm[PIPELINE[self._active_idx]]
                    if new != self._active_idx and sm[PIPELINE[new]] > cur_v * 1.3 + 0.05:
                        self._active_idx = new
                        self._active_hold = 3
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

    def _node_pos(self, w: int, h: int):
        import math
        cx, cy = w // 2, h // 2 + 8
        R = max(70, min(w, h) // 2 - 78)
        pos = {}
        n = len(PIPELINE)
        for i, mod in enumerate(PIPELINE):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            pos[mod] = (int(cx + R * math.cos(ang)), int(cy + R * math.sin(ang)))
        ns = len(SIDE_MODULES)
        for j, mod in enumerate(SIDE_MODULES):
            ang = -math.pi / 2 + (j + 0.5) * 2 * math.pi / ns
            pos[mod] = (int(cx + (R + 40) * math.cos(ang)),
                        int(cy + (R + 40) * math.sin(ang)))
        return pos, (cx, cy), R

    def _draw(self, p: QtGui.QPainter) -> None:
        import math
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Cognitive flow…"); return
        self._title(p, "Cognitive flow — radial pipeline (active node enlarged)")
        self._caption(p, "disc radius/colour = timing cost · red = RBTA violation · "
                         "gold arc = active edge · satellites = side modules")
        pos, (cx, cy), R = self._node_pos(w, h)
        n = len(PIPELINE)
        active_idx = self._active_idx
        # main-chain chords with arrowheads + travelling packet on the active edge
        for i in range(n - 1):
            a, b = pos[PIPELINE[i]], pos[PIPELINE[i + 1]]
            col = QtGui.QColor(241, 196, 15, 230) if i == active_idx else QtGui.QColor(120, 120, 140, 150)
            _arrow(p, a[0], a[1], b[0], b[1], col, size=10)
        if 0 <= active_idx < n - 1:
            a, b = pos[PIPELINE[active_idx]], pos[PIPELINE[active_idx + 1]]
            t = self._anim_t - active_idx
            t = t - int(t)
            px = int(a[0] + t * (b[0] - a[0])); py = int(a[1] + t * (b[1] - a[1]))
            p.setBrush(QtGui.QColor(255, 255, 255)); p.setPen(QtGui.QPen(ACCENT, 1))
            p.drawEllipse(px - 4, py - 4, 8, 8)
        # side-module satellite connectors (thin dashed to their anchor node)
        for sm in SIDE_MODULES:
            a = pos[sm]
            anchor = "prediction" if sm in ("gprime_learn", "attn") else "memory_write"
            b = pos[anchor]
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 110), 1, QtCore.Qt.DashLine))
            p.drawLine(a[0], a[1], b[0], b[1])
        # nodes
        for mod in PIPELINE + SIDE_MODULES:
            x, y = pos[mod]
            raw_ms = float(f.module_timings.get(mod, 0.0))
            sm = self._ms_smooth.setdefault(mod, _Smoother(alpha=0.25))
            ms = sm.value(raw_ms)
            viol = mod in self.viol_mods
            col = QtGui.QColor(231, 76, 60) if viol else _to_qcolor(_cost_color(ms))
            is_active = (PIPELINE.index(mod) == active_idx) if mod in PIPELINE else False
            r = int(18 + min(ms / 20.0, 1.0) * 8 + (6 if is_active else 0))
            p.setBrush(col); p.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 2 if is_active else 1))
            p.drawEllipse(x - r, y - r, r * 2, r * 2)
            p.setPen(QtGui.QColor(20, 20, 24)); p.setFont(_F_LABEL_B)
            p.drawText(x - 26, y - 3, 52, 12, 0x84, PIPELINE_LABEL.get(mod, mod))
            p.setFont(_F_AXIS); p.setPen(TEXT_COL)
            p.drawText(x - 22, y + 9, 44, 11, 0x84, f"{ms:.1f}ms")
            # measured-vs-bound bar (no sparkline — heatmap carries history)
            self._mb_bar(p, mod, ms, x - 30, y + r + 12, 60)
            # focal nodes get a dim-agnostic scalar gauge; others a status badge
            if mod in ("prediction", "action_selection"):
                self._scalar_gauge_thumb(p, mod, f, x - 30, y - r - 30, 60, 22)
            else:
                self._status_badge(p, mod, ms, x - 30, y - r - 24, 60)
            if viol and mod in self.last_viol:
                p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(_F_AXIS)
                p.drawText(x - 30, y + r + 30, 60, 10, 0x84, self.last_viol[mod])
        # consolidated composite-bounds + legend panel (top-right)
        self._side_panel(p)
        # thin module×cycle heatmap (bottom strip)
        self._heatmap(p)

    def _bound_for(self, mod: str, bounds: dict) -> Optional[float]:
        for k, v in bounds.items():
            if RBTA_TO_FLOW.get(k, "") == mod and isinstance(v, dict):
                t = v.get("time")
                if t is not None and t > 0:
                    return float(t)
        return None

    def _mb_bar(self, p: QtGui.QPainter, mod: str, measured: float,
                x: int, y: int, w: int) -> None:
        """Measured-vs-bound bar (v7: sparkline dropped — the bottom heatmap
        already carries timing history, so one bar per node is enough)."""
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        b = self._bound_for(mod, bounds)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, y, w, 7)
        scale = b if b else 25.0
        ratio = max(0.0, min(measured / scale, 1.5))
        fw = int(min(ratio, 1.0) * w)
        col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else _to_qcolor(_cost_color(measured))
        p.fillRect(x, y, fw, 7, col)
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

    def _scalar_gauge_thumb(self, p: QtGui.QPainter, mod: str,
                            f: ObservabilityFrame, x: int, y: int, w: int, h: int) -> None:
        """v7: dim-agnostic scalar gauge replacing the 12-bar content thumb.
        prediction → prediction_confidence; action_selection → gprime_mutual_info
        (a proxy for action-information value). One arc + label + caption."""
        if mod == "prediction":
            v = float(getattr(f, "prediction_confidence", 0.0) or 0.0)
            label, unit = "conf", ""
        else:
            v = float(getattr(f, "gprime_mutual_info", 0.0) or 0.0)
            label, unit = "MI", "bit"
        vc = max(0.0, min(1.0, v))
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRoundedRect(x, y, w, h, 4, 4)
        # horizontal fill
        col = (QtGui.QColor(46, 204, 113) if vc > 0.66 else
               QtGui.QColor(241, 196, 15) if vc > 0.33 else QtGui.QColor(231, 76, 60))
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 120))
        p.drawRoundedRect(x + 2, y + 2, int((w - 4) * vc), h - 4, 3, 3)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + h - 5, f"{label}={v:.2f}{unit}")

    def _side_panel(self, p: QtGui.QPainter) -> None:
        """Consolidated composite-bounds + legend panel (top-right). Replaces the
        old _composite_bounds + _legend so nothing collides with the heatmap."""
        w, h = self.width(), self.height()
        x, y, tw, th = w - 168, 12, 158, 118
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, y, tw, th)
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 12, "composite bounds")
        bounds = getattr(self.frame, "rbta_bounds", None) or {}
        pipe_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in PIPELINE]
        side_ids = [k for k in bounds if RBTA_TO_FLOW.get(k, "") in SIDE_MODULES]
        pb = sum(float(bounds[k].get("time", 0.0)) for k in pipe_ids if isinstance(bounds[k], dict))
        sb = max((float(bounds[k].get("time", 0.0)) for k in side_ids if isinstance(bounds[k], dict)), default=0.0)
        p.setPen(ACCENT); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 28, f"Σpipe={pb:.0f}ms  maxpar={sb:.0f}ms")
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 42, f"root SEQUENCE(pipe) · reg PARALLEL(sides)")
        p.drawText(x + 4, y + 54, f"({len(pipe_ids)}+{len(side_ids)} bound mods)")
        # legend
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 70, "legend")
        ly = y + 84
        for lbl, col in (("gold arc = active edge", "#f1c40f"),
                         ("dashed = side coupling", "#9b59b6"),
                         ("red = RBTA violation", "#e74c3c")):
            p.setPen(_to_qcolor(col)); p.drawLine(x + 4, ly - 4, x + 18, ly - 4)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(x + 22, ly, lbl); ly += 12

    def _heatmap(self, p: QtGui.QPainter) -> None:
        if not self.heat:
            return
        mods = PIPELINE + SIDE_MODULES
        h = self.height(); strip_h = 18; top = h - strip_h - 4
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
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
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
        # v7: stable score range so ranked bar heights don't twitch frame-to-frame.
        self._score_scale = ScaleState(contract=0.05, head=0.10)

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

    def _rollout_cloud(self, p: QtGui.QPainter, f: ObservabilityFrame,
                       x0: int, top: int, x1: int, bot: int, is_continuous: bool) -> None:
        """Render candidate rollouts in the shared PCA plane.

        Discrete: per-action predicted-next thumbnails (dot per action, colored
        by score, chosen circled). Continuous: MPC K-candidate trajectory cloud
        (each candidate's predicted next state, chosen highlighted). Falls back
        to a per-action predicted-bar mini when no projection basis yet.
        v7: takes an explicit left edge (x0) so it sits beside the ranked list,
        and labels PCA variance-explained when the projection exposes it.
        """
        rollouts = list(getattr(f, "candidate_rollouts", []) or [])
        label = ("MPC candidate cloud (chosen ◆)" if is_continuous
                 else "per-action predicted-next (chosen ◆)")
        varexp_fn = getattr(self.proj, "variance_explained", None)
        varexp = varexp_fn() if callable(varexp_fn) else None
        if varexp:
            label += f"  PCA {varexp:.0f}%"
        self._title(p, label, y=top + 12, x=x0)
        if not rollouts:
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(x0, top + 30, "(no rollouts this cycle — explore/D5 branch)")
            return
        # Score-normalised colour: green=high, red=low.
        scores = [float(r.get("score", 0.0)) for r in rollouts]
        smin, smax = (min(scores), max(scores)) if scores else (0.0, 1.0)
        srange = (smax - smin) or 1.0
        if self.proj is not None and self.proj.history:
            bounds = self.proj.bounds()
            px0, py0, px1, py1 = x0, top + 20, x1, bot - 4
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
            # current state anchor
            cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            cur = self.proj.project(cur_v)
            if cur is not None:
                cx, cy = _map_pt(cur, bounds, px0, py0, px1, py1)
                p.setBrush(ACCENT); p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(cx + 6, cy + 3, "now")
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
        bw = (x1 - x0 - 20) / max(n, 1)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x0 + 10, bot - 6, x1 - 10, bot - 6)
        for i, r in enumerate(rollouts):
            pred = r.get("predicted")
            if pred is None:
                continue
            head = np.asarray(pred, dtype=np.float32).reshape(-1)[:12]
            mx = float(np.max(np.abs(head))) or 1.0
            x = x0 + 10 + i * bw
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
        cr_t = float(getattr(f, "cr_temperature", 0.0) or 0.0)
        pareto = set(int(x) for x in (getattr(f, "pareto_front", []) or []))
        names = list(getattr(f, "action_names", []) or [])
        # header
        hdr = f"goal={goal_lbl}  {'EXPLORE' if explored else 'EXPLOIT'}  ε={r.get('eps',0):.3f}  T={cr_t:.2f}"
        bs = r.get("best_score")
        if isinstance(bs, (int, float)):
            hdr += f"  score={bs:.3f}"
        self._title(p, hdr)
        kind = "continuous τ" if is_continuous else "discrete action"
        extra = f" · pareto={len(pareto)} cand · {note}" if note else f" · pareto={len(pareto)} cand"
        self._caption(p, f"ranked candidate {kind} scores · amber►=chosen · ringed=Pareto{extra}")
        # best-score sparkline (top-right) — wires up the previously-dead score_hist
        self._best_score_spark(p, w - 190, 6, 180, 26)
        # layout: ranked list (focal, left) + supporting stack (right)
        left_w = w // 2 - 6
        list_x, list_y, list_w, list_h = 10, 40, left_w, h - 70
        rx0 = w // 2 + 6
        if scores:
            self._ranked_list(p, scores, chosen, pareto, names, is_continuous,
                             list_x, list_y, list_w, list_h)
        else:
            tag = "EXPLORE (random action)" if explored else (note or "D5 ENERGY → STAY (no candidates)")
            p.setPen(QtGui.QColor(231, 76, 60) if not explored else QtGui.QColor(52, 152, 219))
            p.setFont(QtGui.QFont("Sans", 12, QtGui.QFont.Bold))
            p.drawText(list_x, list_y + 20, f"● {tag}")
            p.setFont(QtGui.QFont("Sans", 9)); p.setPen(DIM_COL)
            p.drawText(list_x, list_y + 42, "(candidate scores are not computed for this branch)")
            if self.last_scores:
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(list_x, list_y + 60, "last non-empty scores (faded):")
                self._bars(p, self.last_scores, self.last_chosen, faded=True,
                           top=list_y + 70, bot=list_y + list_h, w=list_w)
        # right stack: rollout cloud (top) + action heatmap (mid) + ε strip (bottom)
        cloud_top = 40
        self._rollout_cloud(p, f, rx0, cloud_top, w - 10, h // 2 + 10, is_continuous)
        # dim-agnostic continuous_action heatmap (replaces the 2D-only torque dial)
        if is_continuous and self.last_continuous is not None:
            self._action_heatmap(p, self.last_continuous, list(getattr(f, "dim_names", []) or []),
                                 rx0, h // 2 + 16, w - rx0 - 10, 36)
        # ε-decay + explore/exploit dots (reserved 30px bottom strip)
        self._epsilon_strip(p, rx0, h - 34, w - rx0 - 10, 28)

    def _ranked_list(self, p: QtGui.QPainter, scores: List[float], chosen: int,
                     pareto: set, names: List[str], is_continuous: bool,
                     x: int, y: int, w: int, h: int) -> None:
        """v7 focal: candidates ranked by score (desc). Each row = rank + name +
        score bar + value + margin-to-2nd (top row) + chosen marker + Pareto ring.
        ScaleState-stabilised range so bar widths glide, not twitch."""
        n = len(scores)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        smax = max(scores)
        smin = min(scores) if scores else 0.0
        lo, hi = self._score_scale.update(smin, smax)
        rng = (hi - lo) or 1.0
        rh = max(14, min(34, h // max(n, 1)))
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y - 4, f"rank · name · score (Δ to 2nd on top row)")
        for rank, idx in enumerate(order):
            ry = y + 6 + rank * rh
            if ry + rh > y + h:
                break
            s = scores[idx]
            is_chosen = (idx == chosen)
            is_pareto = idx in pareto
            name = (names[idx] if idx < len(names) else (f"cand {idx}" if is_continuous else f"a{idx}"))
            # row background
            if is_chosen:
                p.setPen(QtGui.QPen(ACCENT, 1)); p.setBrush(QtGui.QColor(241, 196, 15, 40))
                p.drawRoundedRect(x, ry, w, rh - 3, 4, 4)
            elif is_pareto:
                p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 120), 1))
                p.setBrush(QtGui.QColor(155, 89, 182, 18))
                p.drawRoundedRect(x, ry, w, rh - 3, 4, 4)
            # rank
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, ry + rh - 8, f"#{rank + 1}")
            # name
            p.setPen(TEXT_COL if is_chosen else DIM_COL)
            p.setFont(_F_LABEL_B if is_chosen else _F_AXIS)
            p.drawText(x + 34, ry + rh - 8, str(name)[:14])
            # score bar
            bar_x = x + 110; bar_w = w - 110 - 78
            frac = max(0.0, min(1.0, (s - lo) / rng))
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRect(bar_x, ry + 4, bar_w, rh - 12)
            col = ACCENT if is_chosen else QtGui.QColor(52, 152, 219, 200)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(bar_x, ry + 4, int(bar_w * frac), rh - 12, col)
            # value + margin-to-2nd
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            txt = f"{s:.3f}"
            if rank == 0 and n > 1:
                txt += f"  Δ{(s - scores[order[1]]):+.3f}"
            p.drawText(bar_x + bar_w + 6, ry + rh - 8, txt)
            # chosen marker
            if is_chosen:
                p.setPen(ACCENT); p.setFont(_F_LABEL_B)
                p.drawText(x + w - 14, ry + rh - 8, "►")

    def _action_heatmap(self, p: QtGui.QPainter, vec: np.ndarray,
                        dim_names: List[str], x: int, y: int, w: int, h: int) -> None:
        """v7 dim-agnostic signed heatmap of the chosen continuous action, indexed
        by dim_names (replaces the 2D-only torque dial). Diverging purple/red."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = v.size
        if n == 0:
            return
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y, "chosen action τ (signed, per-dim)")
        gy = y + 6
        bw = (w - 4) / n
        mid = gy + h // 2
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, mid, x + w, mid)
        for i in range(n):
            val = float(np.clip(v[i], -1, 1))
            bh = int(abs(val) * (h // 2 - 2))
            bx = int(x + i * bw)
            col = QtGui.QColor(155, 89, 182) if val >= 0 else QtGui.QColor(231, 76, 60)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            if val >= 0:
                p.fillRect(bx + 1, mid - bh, max(int(bw) - 2, 1), bh, col)
            else:
                p.fillRect(bx + 1, mid, max(int(bw) - 2, 1), bh, col)
        # dim labels (every k-th to avoid crowding)
        step = max(1, n // 12)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        for i in range(0, n, step):
            lbl = dim_names[i] if i < len(dim_names) else str(i)
            p.drawText(int(x + i * bw), gy + h + 8, lbl[:6])

    def _best_score_spark(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        """v7: best-score-over-cycles sparkline (wires up the previously-dead
        score_hist). Stable ScaleState-free mini (bounded 0..1-ish)."""
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 10, "best score")
        vals = list(self.score_hist)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.drawText(x + 4, y + h - 4, "—"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9: hi = lo + 1
        n = len(vals)
        p.setPen(QtGui.QPen(ACCENT, 2))
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + 4 + i * (w - 8) / max(n - 1, 1)
            py = (y + h - 3) - (v - lo) / (hi - lo) * (h - 16)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.drawPath(path)

    def _epsilon_strip(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        """v7: reserved ε-decay + explore/exploit strip (moved out of the cloud
        region so nothing collides)."""
        if not self.eps_hist:
            return
        n = len(self.eps_hist)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y, "ε-decay (blue) · explore(red)/exploit(green)")
        base_y = y + h - 6
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
        for i in range(1, n):
            x0 = x + (i - 1) * w / max(n - 1, 1)
            x1 = x + i * w / max(n - 1, 1)
            p.drawLine(int(x0), int(base_y - self.eps_hist[i - 1] * (h - 14)),
                       int(x1), int(base_y - self.eps_hist[i] * (h - 14)))
        for i, e in enumerate(self.ee_hist):
            xx = int(x + i * w / max(n - 1, 1))
            c = QtGui.QColor(231, 76, 60) if e else QtGui.QColor(46, 204, 113)
            p.setBrush(c); p.setPen(QtGui.QPen(c, 1))
            p.drawEllipse(xx - 2, base_y + 1, 4, 4)


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
            # + candidate cloud (faint underlay) + G′ uncertainty ellipse (scales
            # to any dim). v7: PC1/PC2 tick frame + variance-% + basis-stability
            # badge + dim_names axes when raw-2D.
            if self.proj is None or not self.proj.history:
                self._empty(p, "Building belief projection (PCA)…"); return
            px0, py0, px1, py1 = 20, 36, w - 20, h - 30
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawRect(px0, py0, px1 - px0, py1 - py0)
            bounds = self.proj.bounds()
            varexp = self.proj.variance_explained()
            raw2d = bool(getattr(self.proj, "_raw2d", False))
            dim_names = list(getattr(f, "dim_names", []) or [])
            # PC1/PC2 tick frame (4 light ticks per axis)
            for frac in (0.25, 0.5, 0.75):
                tx = int(px0 + frac * (px1 - px0))
                ty = int(py0 + frac * (py1 - py0))
                p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(tx, py1, tx, py1 + 3)
                p.drawLine(px0, ty, px0 - 3, ty)
            ax_x = dim_names[0] if (raw2d and dim_names) else "PC1"
            ax_y = dim_names[1] if (raw2d and dim_names and len(dim_names) > 1) else "PC2"
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(px1 - 26, py1 + 14, ax_x)
            p.drawText(px0 - 14, py0 + 4, ax_y)
            # candidate cloud — faint underlay (drawn first so trail/ellipse sit on top)
            for r in (f.candidate_rollouts or []):
                pt = self.proj.project(r.get("predicted"))
                if pt is None:
                    continue
                rx, ry = _map_pt(pt, bounds, px0, py0, px1, py1)
                col = ACCENT if r.get("chosen") else QtGui.QColor(150, 150, 160, 70)
                p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
                p.drawEllipse(rx - 3, ry - 3, 6, 6)
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
            ve_txt = f"  PCA {varexp:.0f}%" if varexp else ""
            proj_lbl = "raw-2D" if raw2d else "PCA-2D"
            self._title(p, f"Belief-space projection ({proj_lbl}, dim={f.state_dim or '?'}) + {kind} uncertainty ellipse{ve_txt}")
            self._caption(p, "belief trajectory · ellipse = G′ posterior σ projected · faint cloud = rollouts")
            # v7: basis-stability badge (only meaningful for PCA, not raw-2D)
            if not raw2d and getattr(self.proj, "basis_changed", False):
                p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(_F_LABEL_B)
                p.drawText(px0 + 6, py0 + 14, "⟳ PCA re-fit")
            self._legend(p, [("● current", ACCENT), ("● predicted next", QtGui.QColor(46, 204, 113)),
                             ("○ goal", ACCENT), ("◐ candidates", QtGui.QColor(150, 150, 160)),
                             ("◯ ±σ ellipse", QtGui.QColor(52, 152, 219))],
                         y=h - 14)


class DriveRadarView(_BaseCanvas):
    """6-drive radar/hexagon portrait over the rolling window — far more
    revealing than the old arbitrary D1/D3/D5 scatter. v7: drive_targets ghost
    polygon + drive_deficits red ticks + numeric vertex labels + active-drive
    spoke highlight; dim count derived from the frame (not hardcoded 6)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist: Deque[List[float]] = deque(maxlen=200)
        self._active: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        lv = list(f.drive_levels or [])
        if lv:
            n = min(6, len(lv))
            self.hist.append([float(lv[i]) for i in range(n)])
        self._active = int(getattr(f, "active_drive_id", 0) or 0)
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        import math
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 4
        R = min(w, h) // 2 - 46
        hist = list(self.hist)
        if not hist:
            self._empty(p, "Drive radar (collecting…)"); return
        n = len(hist[-1])
        n = max(3, min(6, n))
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            pts.append((cx + R * math.cos(ang), cy + R * math.sin(ang), ang))
        # grid rings
        for ring in (0.25, 0.5, 0.75, 1.0):
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 60), 1))
            poly = QtGui.QPolygonF([QtCore.QPointF(cx + ring * R * math.cos(a),
                                                   cy + ring * R * math.sin(a))
                                    for _, _, a in pts])
            p.drawPolygon(poly)
        # spokes + labels + active highlight
        for i, (x, y, a) in enumerate(pts):
            did = i + 1
            is_active = (did == self._active)
            p.setPen(QtGui.QPen(_to_qcolor(DRIVE_COLORS[did]) if is_active else QtGui.QColor(50, 50, 60),
                                2 if is_active else 1))
            p.drawLine(cx, cy, int(x), int(y))
            lx = cx + (R + 16) * math.cos(a); ly = cy + (R + 16) * math.sin(a)
            p.setPen(_to_qcolor(DRIVE_COLORS[did])); p.setFont(_F_LABEL_B)
            p.drawText(int(lx) - 12, int(ly) + 4, DRIVE_SHORT[did])
        # drive_targets ghost polygon (the setpoints the agent is pulled toward)
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
        # faded history
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
        # current polygon (filled, per-drive colored vertices + numeric labels)
        cur = hist[-1]
        poly = QtGui.QPolygonF()
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            poly.append(QtCore.QPointF(cx + lvl * R * math.cos(ang), cy + lvl * R * math.sin(ang)))
        p.setBrush(QtGui.QColor(241, 196, 15, 70)); p.setPen(QtGui.QPen(ACCENT, 2))
        p.drawPolygon(poly)
        # deficit red ticks (gap from target) + numeric vertex labels
        defs = list(getattr(self.frame, "drive_deficits", []) or []) if self.frame else []
        for i, lvl in enumerate(cur):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            vx = int(cx + lvl * R * math.cos(ang)); vy = int(cy + lvl * R * math.sin(ang))
            p.setBrush(_to_qcolor(DRIVE_COLORS[i + 1]))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(vx - 3, vy - 3, 6, 6)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(vx + 5, vy - 4, f"{lvl:.2f}")
            if defs and i < len(defs) and defs[i] > 0.05:
                # red deficit tick just outside the vertex along the spoke
                tx = int(cx + (lvl + 0.06) * R * math.cos(ang))
                ty = int(cy + (lvl + 0.06) * R * math.sin(ang))
                p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 2))
                p.drawLine(vx, vy, tx, ty)
        self._title(p, "6-drive radar (solid=level · dashed=target · red tick=deficit)")
        self._caption(p, "radial drive levels 0..1 · faded = recent history · active spoke bold · numbers = level")


class _DimSelector(QtWidgets.QWidget):
    """Prev/next range pager for high-dimensional per-dim views (10..1000 dims).
    v7: page label shows dim_names (not just indices) + a jump-to-dim spinbox."""

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
        # v7: per-trend stable scales (raw min/max jittered every frame) +
        # cached top-K offender dims so the bars stop re-sorting per page.
        self._mi_scale = ScaleState(contract=0.05, head=0.06)
        self._be_scale = ScaleState(contract=0.05, head=0.06)
        self._offenders: List[int] = []

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
        # v7: STABLE page selection — natural dim order within the page window
        # (the old per-page "highest-error" re-sort flickered every frame).
        # Top-K offenders are surfaced as a separate ranked inset instead.
        n_pages = max(1, (d + self.page_size - 1) // self.page_size)
        self.page = max(0, min(n_pages - 1, self.page))
        lo_i = self.page * self.page_size
        hi_i = min(d, lo_i + self.page_size)
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
        self._caption(p, "red = prediction error · blue whisker = ±σ around the error · stable order (no re-sort)")
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(20, bot, w - 20, bot)
        raw_max = float(errs_s.max()) if errs_s.size else 1.0
        if std_s is not None:
            # bracket the error: max(error+σ) defines the scale
            band_top = (errs_s + std_s).max() if std_s.size else 0.0
            raw_max = max(raw_max, float(band_top))
        _, mx = self._scale.update(0.0, float(raw_max if raw_max > 1e-6 else 1.0))
        mx = mx if mx > 1e-6 else 1.0
        for i in range(n):
            x = 20 + i * bw
            eh = int(errs_s[i] / mx * (bot - top))
            # v7: σ whisker brackets the error bar (error-σ .. error+σ), drawn as
            # a thin vertical band centred on the error height — semantically
            # "the model's error is this, ± this much uncertainty".
            if std_s is not None:
                y_e = bot - eh
                s_pix = float(std_s[i]) / mx * (bot - top)
                p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, 140), 1))
                p.drawLine(int(x + bw / 2), int(y_e - s_pix), int(x + bw / 2), int(y_e + s_pix))
                p.drawLine(int(x + bw / 2 - 3), int(y_e - s_pix), int(x + bw / 2 + 3), int(y_e - s_pix))
                p.drawLine(int(x + bw / 2 - 3), int(y_e + s_pix), int(x + bw / 2 + 3), int(y_e + s_pix))
            p.fillRect(int(x + 2), int(bot - eh), int(bw - 8), eh, QtGui.QColor(231, 76, 60, 220))
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            lbl = _dim_label(int(idx[i]), f)
            p.drawText(int(x), bot + 11, lbl if n <= 24 else str(int(idx[i])))
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(20, h - 4, f"scale 0..{mx:.3g}  (named dims · pager below)")
        # v7: top-K offenders inset (the ranked info the old re-sort tried to show,
        # now as a stable side list instead of destabilising the bars).
        self._offenders = [int(x) for x in np.argsort(-errs)[:5]]
        self._offenders_inset(p, errs, f, w - 150, 40, 140, h // 2 - 44)
        self._legend(p, [("▮ error", QtGui.QColor(231, 76, 60)),
                         ("├─┤ ±σ whisker", QtGui.QColor(52, 152, 219))],
                     y=h // 2 + 4, x=20)
        # mutual-info / belief-entropy trend strip (lower half, v7: ScaleState)
        t_top, t_bot = h // 2 + 22, h - 16
        self._trend(p, self.mi_hist, t_top, t_bot, QtGui.QColor(155, 89, 182),
                    "mutual_info trend", left=40, right=w // 2 - 6, scale=self._mi_scale)
        self._trend(p, self.be_hist, t_top, t_bot, QtGui.QColor(46, 204, 113),
                    "belief-entropy trend", left=w // 2 + 6, right=w - 16, scale=self._be_scale)

    def _offenders_inset(self, p: QtGui.QPainter, errs: np.ndarray,
                         f: ObservabilityFrame, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 12, "top error dims")
        mx = float(errs.max()) if errs.size else 1.0
        mx = mx if mx > 1e-6 else 1.0
        ry = y + 18
        for rank, di in enumerate(self._offenders):
            if ry + 14 > y + h:
                break
            lbl = _dim_label(int(di), f)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, ry + 10, f"#{rank + 1} {lbl}")
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
        self.m3_reasons: Dict[int, str] = {}  # v7: prune-reason per event index
        self.m4_reasons: Dict[int, str] = {}
        self.cycle_base: int = 0
        self._panel_scales: Dict[str, ScaleState] = {}
        # v7: EMA-smoothed leak rate (raw polyfit jittered every frame) + frame
        # cache so the focal gauge / prune reasons can read RBTA action state.
        self._leak_smooth = _Smoother(0.08)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if not self.m3:
            self.cycle_base = int(f.cycle_id)
        prev_m3 = self.m3[-1] if self.m3 else None
        prev_m4 = self.m4[-1] if self.m4 else None
        self.m3.append(int(f.episode_count)); self.m4.append(int(f.fact_count))
        self.rss.append(float(f.rss_bytes)); self.lat.append(float(f.latency_ms))
        self.m3_cap = int(f.m3_cap); self.m4_cap = int(f.m4_cap)
        reason = self._prune_reason(f)
        if prev_m3 is not None and int(f.episode_count) < prev_m3:
            self.m3_events.append(len(self.m3) - 1)
            self.m3_reasons[len(self.m3) - 1] = reason
        if prev_m4 is not None and int(f.fact_count) < prev_m4:
            self.m4_events.append(len(self.m4) - 1)
            self.m4_reasons[len(self.m4) - 1] = reason
        self.update()

    def _prune_reason(self, f: ObservabilityFrame) -> str:
        """v7: human-readable prune reason from rbta_action + meta_stable."""
        act = str(getattr(f, "rbta_action", "CONTINUE") or "CONTINUE")
        ms = getattr(f, "meta_stable", {}) or {}
        flag = ms.get("is_meta_stable")
        if act not in ("CONTINUE", "OK", "NONE", ""):
            return act
        if flag:
            return "meta-stable"
        return "prune"

    def _leak_rate(self) -> float:
        if len(self.rss) < 30:
            return 0.0
        xs = np.arange(30, dtype=float)
        ys = np.array(list(self.rss)[-30:], dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])  # B/cyc
        return slope

    def _panel(self, p: QtGui.QPainter, top: int, bot: int, title: str,
               series: Deque[float], cap: float, col: QtGui.QColor,
               events: List[int], unit: str = "",
               reasons: Optional[Dict[int, str]] = None) -> None:
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
            # v7: event markers + prune-reason annotations
            for ev in events:
                if 0 <= ev < n:
                    x = int(left + ev * (right - left) / max(n - 1, 1))
                    p.setPen(QtGui.QPen(ACCENT, 1)); p.drawLine(x, pt0, x, pt1)
                    if reasons and ev in reasons:
                        p.setPen(ACCENT); p.setFont(QtGui.QFont("Sans", 7))
                        p.drawText(x + 2, pt0 + 8, reasons[ev])
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
        leak = self._leak_smooth.value(self._leak_rate())
        # v7 focal: "inside envelope?" gauge across M3/M4/RSS/latency vs caps/bounds
        env_ok, env_seg = self._envelope_status()
        self._title(p, f"Retention & resources   RSS leak-rate ≈ {leak:+.1f} B/cyc", y=15)
        self._envelope_gauge(p, 10, 24, w - 20, 26, env_seg, env_ok)
        self._caption(p, "M3 episodic · M4 consolidated facts · resources = RSS (orange) + latency (green) · prune ticks labelled", y=56)
        ph = (h - 64) // 3
        self._panel(p, 64, 64 + ph, "M3 episodes (memory)", self.m3, float(self.m3_cap),
                    QtGui.QColor(52, 152, 219), self.m3_events, reasons=self.m3_reasons)
        self._panel(p, 64 + ph, 64 + 2 * ph, "M4 facts (consolidated)", self.m4, float(self.m4_cap),
                    QtGui.QColor(155, 89, 182), self.m4_events, reasons=self.m4_reasons)
        self._resources_panel(p, 64 + 2 * ph, h - 4)

    def _envelope_status(self) -> Tuple[bool, List[Tuple[str, float, bool]]]:
        """v7: per-resource engagement vs cap/bound → (all_ok, [(label, ratio, over)])."""
        segs = []
        m3 = float(self.m3[-1]) if self.m3 else 0.0
        m4 = float(self.m4[-1]) if self.m4 else 0.0
        rss = float(self.rss[-1]) if self.rss else 0.0
        lat = float(self.lat[-1]) if self.lat else 0.0
        c3 = float(self.m3_cap) if self.m3_cap else 1.0
        c4 = float(self.m4_cap) if self.m4_cap else 1.0
        # RSS / latency bounds: largest mem / time bound across modules (RBTA).
        bounds = getattr(self.frame, "rbta_bounds", {}) or {} if self.frame else {}
        mem_b = max((float(b.get("mem", 0.0)) for b in bounds.values() if isinstance(b, dict) and b.get("mem")), default=0.0)
        time_b = max((float(b.get("time", 0.0)) for b in bounds.values() if isinstance(b, dict) and b.get("time")), default=0.0)
        rss_b = mem_b if mem_b > 0 else (max(self.rss) * 1.25 if self.rss else 1.0)
        lat_b = time_b if time_b > 0 else (max(self.lat) * 1.25 if self.lat else 1.0)
        for lbl, val, cap in (("M3", m3, c3), ("M4", m4, c4), ("RSS", rss, rss_b), ("lat", lat, lat_b)):
            r = val / cap if cap > 0 else 0.0
            segs.append((lbl, min(r, 1.5), r > 1.0))
        return all(not s[2] for s in segs), segs

    def _envelope_gauge(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int,
                        segs: List[Tuple[str, float, bool]], ok: bool) -> None:
        """v7 focal gauge: 4 resource segments (M3/M4/RSS/lat) vs bound; red if
        any over, green headroom otherwise. One glance 'inside envelope?'."""
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(TEXT_COL if ok else QtGui.QColor(231, 76, 60)); p.setFont(_F_LABEL_B)
        p.drawText(x + 6, y + h - 8, "inside envelope?" + ("" if ok else "  ✗ OVER"))
        gx = x + 150; gw = w - 160
        segw = gw // max(len(segs), 1)
        for i, (lbl, ratio, over) in enumerate(segs):
            sx = gx + i * segw
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(QtGui.QColor(30, 30, 38))
            p.drawRect(sx, y + 6, segw - 6, h - 12)
            fw = int(min(ratio, 1.0) * (segw - 6))
            col = QtGui.QColor(231, 76, 60) if over else QtGui.QColor(46, 204, 113)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(sx, y + 6, fw, h - 12, col)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(sx + 2, y + h - 8, f"{lbl} {ratio*100:.0f}%")

    def _resources_panel(self, p: QtGui.QPainter, top: int, bot: int) -> None:
        """Consolidated resources panel: RSS + latency on one shared stable
        scale (normalised 0..1 of each series' own ScaleState) with dual labels
        — replaces the separate RSS line + latency histogram panels. v7: adds a
        stacked resource breakdown from the runtime/memory/energy logs."""
        w = self.width()
        left, right = 60, w - 180
        self._title(p, "Resources — RSS (B, orange) + latency (ms, green)", y=top + 12)
        rss = list(self.rss); lat = list(self.lat)
        if not rss and not lat:
            return
        # dual labels (right-aligned)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        rss_txt = f"RSS now={rss[-1]/1e6:.1f}MB" if rss else "RSS —"
        lat_txt = f"lat now={lat[-1]:.1f}ms med={float(np.median(lat)):.1f}ms" if lat else "lat —"
        p.drawText(right - 90, top + 12, rss_txt)
        p.drawText(right - 90, top + 24, lat_txt)
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
        # v7: stacked resource breakdown from runtime/memory/energy logs
        self._resource_breakdown(p, w - 170, top + 30, 160, bot - top - 36)

    def _resource_breakdown(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        """v7: stacked bar of runtime_log / memory_log / energy_log totals (wires
        up the previously-undrawn *_log frame fields)."""
        f = self.frame
        if f is None:
            return
        logs = [("runtime", getattr(f, "runtime_log", {}) or {}, QtGui.QColor(52, 152, 219)),
                ("memory", getattr(f, "memory_log", {}) or {}, QtGui.QColor(155, 89, 182)),
                ("energy", getattr(f, "energy_log", {}) or {}, QtGui.QColor(230, 126, 34))]
        totals = [(name, float(sum(d.values())) if d else 0.0, col) for name, d, col in logs]
        gtot = sum(t for _, t, _ in totals) or 1.0
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
        p.drawText(x + 4, y + 12, "resource logs (stacked)")
        cy = y + 18; ch = h - 44
        for name, t, col in totals:
            bh = int(t / gtot * ch)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(x + 4, cy, w - 8, bh, col)
            cy += bh
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        ly = y + h - 22
        for name, t, col in totals:
            p.setPen(col); p.drawLine(x + 4, ly - 4, x + 16, ly - 4)
            p.setPen(TEXT_COL); p.drawText(x + 20, ly, f"{name}={t:.2g}")
            ly += 11


class ViolationTable(QtWidgets.QTableWidget):
    """v7: rows coloured by measured/allowed ratio buckets; only auto-scrolls when
    already at the bottom (no yanking the view); column resize throttled; exposes
    a 'N violations · worst module' summary string for the tab header."""

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["cycle", "module", "bound_type", "measured", "allowed"])
        self.setAlternatingRowColors(True)
        self._seen: int = 0
        self._worst: str = ""
        self._last_resize: int = 0
        self._at_bottom: bool = True

    def add_frame(self, f: ObservabilityFrame) -> None:
        # remember scroll position before inserting so we don't yank the user
        sb = self.verticalScrollBar()
        self._at_bottom = (sb.value() >= sb.maximum() - 2)
        worst_ratio = 0.0
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
            meas = float(v.get("measured", 0.0) or 0.0)
            allowed = float(v.get("allowed", 0.0) or 0.0)
            ratio = (meas / allowed) if allowed > 0 else 0.0
            # ratio-bucket row colour (pale red→deep red as severity grows)
            if ratio >= 2.0:
                c = QtGui.QColor(231, 76, 60, 90)
            elif ratio >= 1.5:
                c = QtGui.QColor(231, 76, 60, 55)
            else:
                c = QtGui.QColor(241, 196, 15, 45)
            for col in range(5):
                it = self.item(row, col)
                if it:
                    it.setBackground(c)
            it = self.item(row, 0)
            if it:
                it.setForeground(QtGui.QColor(231, 76, 60))
            if ratio > worst_ratio:
                worst_ratio = ratio
                self._worst = str(v.get("module_id") or v.get("module", ""))
            self._seen += 1
        while self.rowCount() > 200:
            self.removeRow(0)
        # throttle column resize to once per ~250ms (avoid per-frame layout churn)
        now = time.monotonic()
        if self._seen > 0 and self.rowCount() > 0 and (now - self._last_resize) > 0.25:
            self.resizeColumnsToContents()
            self._last_resize = now
        # smart scroll: only follow new rows if the user was already at the bottom
        if self._seen > 0 and self.rowCount() > 0 and self._at_bottom:
            self.scrollToBottom()

    def violations_seen(self) -> int:
        return self._seen

    def summary(self) -> str:
        if self._seen == 0:
            return "0 violations"
        return f"{self._seen} violations · worst {self._worst}"


# ----- RBTA bound-envelope (Retention tab addition) --------------------------

class RBTABoundsView(_BaseCanvas):
    """v7: per-module measured vs bound for ALL bound types (time/mem/energy) as
    small-multiples. Bars overflow past the bound tick in red so violation
    magnitude reads; a faint per-module EMA sparkline sits behind each bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist: Dict[str, Deque[float]] = {}
        self._smooth: Dict[str, _Smoother] = {}

    def _measured(self, f: ObservabilityFrame, flow: str, btype: str) -> float:
        if btype == "time":
            return float((f.module_timings or {}).get(flow, 0.0))
        if btype == "mem":
            return float((getattr(f, "memory_log", {}) or {}).get(flow, 0.0))
        if btype == "energy":
            return float((getattr(f, "energy_log", {}) or {}).get(flow, 0.0))
        return 0.0

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "RBTA bounds…"); return
        bounds = getattr(f, "rbta_bounds", None) or {}
        btypes = [("time", "B_time (ms)", QtGui.QColor(52, 152, 219)),
                  ("mem", "B_mem (B)", QtGui.QColor(155, 89, 182)),
                  ("energy", "B_energy", QtGui.QColor(230, 126, 34))]
        # collect modules that have any bound
        mods = sorted({mid for mid, b in bounds.items() if isinstance(b, dict)
                       and any(b.get(t) for t, _, _ in btypes)})
        if not mods:
            self._empty(p, "RBTA bound envelope (no bounds)"); return
        self._title(p, f"RBTA bound envelope — measured vs B_time/B_mem/B_energy  ({len(mods)} mods)")
        self._caption(p, "▮ measured vs │ bound · overflow past tick = red violation · faint line = EMA history")
        col_w = w // max(len(btypes), 1)
        top, bot = 40, h - 8
        for ci, (bt, blbl, bcol) in enumerate(btypes):
            cx = ci * col_w + 8
            p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
            p.drawText(cx, top - 6, blbl)
            items = []
            for mid in mods:
                b = bounds.get(mid, {})
                if not isinstance(b, dict):
                    continue
                bv = b.get(bt)
                if bv is None or bv <= 0:
                    continue
                flow = RBTA_TO_FLOW.get(mid, mid)
                meas = self._measured(f, flow, bt)
                key = f"{mid}:{bt}"
                sm = self._smooth.setdefault(key, _Smoother(0.25))
                meas_s = sm.value(meas)
                self._hist.setdefault(key, deque(maxlen=60)).append(meas_s)
                items.append((mid, flow, meas_s, float(bv), key))
            if not items:
                continue
            items.sort(key=lambda r: r[2] / r[3], reverse=True)
            n = len(items)
            rowh = (bot - top - 8) / max(n, 1)
            bw_max = col_w - 170
            for i, (mid, flow, meas, b, key) in enumerate(items):
                y = top + int(i * rowh)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(cx, y + 12, f"{mid[:10]:>10}")
                ratio = meas / b
                col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else bcol
                # faint EMA sparkline behind the bar
                vals = list(self._hist.get(key, []))
                if len(vals) >= 2:
                    mx = max(vals) or 1.0
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 50), 1))
                    nn = len(vals)
                    for k in range(1, nn):
                        x0 = cx + 100 + (k - 1) * bw_max / max(nn - 1, 1)
                        x1 = cx + 100 + k * bw_max / max(nn - 1, 1)
                        y0 = y + 14 - (vals[k - 1] / mx) * 10
                        y1 = y + 14 - (vals[k] / mx) * 10
                        p.drawLine(int(x0), int(y0), int(x1), int(y1))
                p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
                p.drawRect(cx + 100, y + 4, bw_max, 12)
                # v7: overflow past the tick in red (don't clamp — show magnitude)
                fw = int(min(ratio, 1.0) * bw_max)
                p.fillRect(cx + 100, y + 4, fw, 12, col)
                if ratio > 1.0:
                    over = int(min((ratio - 1.0) * bw_max, bw_max * 0.5))
                    p.fillRect(cx + 100 + bw_max, y + 4, over, 12, QtGui.QColor(231, 76, 60, 200))
                # bound tick
                p.setPen(QtGui.QPen(ACCENT, 2))
                p.drawLine(cx + 100 + bw_max, y + 2, cx + 100 + bw_max, y + 16)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                over_t = "  OVER" if ratio > 1.0 else ""
                p.drawText(cx + 100 + bw_max + 4, y + 14, f"{meas:.1f}/{b:.0f}{over_t}")


# ----- NEW Memory & Belief tab ------------------------------------------------

class MemoryBeliefView(_BaseCanvas):
    """v7: focal 'belief geography map' (per-dim belief_entropies heat-strip +
    gprime_uncertainty band + dim_names) — the actual 'what the agent believes'.
    M3/M4 become ranked bars (swatch+bar+caption) frozen via content hash; the
    |sanitized−raw| diff is demoted to a small EMA-smoothed inset with a
    retention-cap engagement gauge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._m3_sig: str = ""
        self._m4_sig: str = ""
        self._m3_cache: List[dict] = []
        self._m4_cache: List[dict] = []
        self._diff_smooth = _Smoother(0.2)

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Memory & belief…"); return
        # ---- v7 focal: belief geography map (full width, top) ----
        self._title(p, "Belief geography — per-dim entropy heat-strip + G′ uncertainty band", x=10, y=14)
        self._caption(p, "heat = belief entropy per dim (dim_names) · blue band = G′ posterior σ · the agent's current belief shape", x=10, y=26)
        self._belief_geography(p, f, 10, 32, w - 20, h // 3)
        # ---- left: M3 ranked bars (frozen via content hash) ----
        col_w = w // 2 - 8
        my = h // 3 + 40
        self._title(p, "M3 episodic memory — ranked by prediction error", x=10, y=my)
        m3 = list(getattr(f, "m3_recent", None) or []) + list(getattr(f, "m3_top_error", None) or [])
        sig = freeze_sig(m3)
        if sig != self._m3_sig:
            self._m3_cache = m3; self._m3_sig = sig
        self._ranked_bars(p, self._m3_cache, key="prediction_error",
                          label_fn=lambda ep: f"d{ep.get('drive_id','?')} conf={ep.get('confidence','?')}",
                          x=10, y=my + 12, w=col_w, h=h - my - 14,
                          col=QtGui.QColor(52, 152, 219))
        # ---- right: M4 ranked bars + retention-cap gauge ----
        rx = col_w + 16
        self._title(p, "M4 consolidated facts — ranked by support", x=rx, y=my)
        m4 = list(getattr(f, "m4_relevant", None) or []) + list(getattr(f, "m4_top", None) or [])
        sig4 = freeze_sig(m4)
        if sig4 != self._m4_sig:
            self._m4_cache = m4; self._m4_sig = sig4
        self._ranked_bars(p, self._m4_cache, key="support",
                          label_fn=lambda fac: f"{fac.get('fact_type', fac.get('predicate','?'))} {str(fac.get('summary',''))[:18]}",
                          x=rx, y=my + 12, w=w - rx - 10, h=h - my - 40,
                          col=QtGui.QColor(155, 89, 182))
        # retention-cap gauge (top-right of the M4 column)
        self._cap_gauge(p, rx, my - 14, 120, 16, int(f.fact_count), int(f.m4_cap), "M4 cap")
        # v7: demoted |sanitized−raw| diff as a thin EMA-smoothed inset (bottom strip)
        self._diff_inset(p, f, 10, h - 26, w - 20, 22)

    def _belief_geography(self, p, f, x, y, w, h) -> None:
        """v7 focal: per-dim belief entropy heat-strip + gprime_uncertainty band."""
        dim_names = list(getattr(f, "dim_names", []) or [])
        be = getattr(f, "belief_entropies", {}) or {}
        # extract per-dim entropy values (keys like d0/i0/0 or dim_names)
        ent = self._per_dim_entropy(be, dim_names, f)
        unc = np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1) if f.gprime_uncertainty is not None else None
        n = len(ent)
        if n == 0:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 14, "(no per-dim belief entropy collected yet)")
            return
        emx = max(ent) or 1.0
        umx = float(unc.max()) if (unc is not None and unc.size) else 1.0
        umx = umx if umx > 1e-6 else 1.0
        bw = w / n
        strip_y = y + 8; strip_h = h - 28
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, strip_y, w, strip_h)
        for i in range(n):
            bx = int(x + i * bw)
            e = float(np.clip(ent[i] / emx, 0, 1)) if emx > 0 else 0.0
            # heat colour: low entropy=blue (certain), high=red (uncertain belief)
            col = QtGui.QColor(int(52 + 179 * e), int(152 - 92 * e), int(219 - 159 * e), 210)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(bx + 1, strip_y, max(int(bw) - 2, 1), strip_h, col)
            # gprime_uncertainty band overlay (translucent blue, height = σ)
            if unc is not None and i < unc.size:
                uh = int(float(np.clip(unc[i] / umx, 0, 1)) * strip_h)
                p.fillRect(bx + 1, strip_y + strip_h - uh, max(int(bw) - 2, 1), uh,
                           QtGui.QColor(46, 204, 113, 70))
        # dim labels (every k-th)
        step = max(1, n // 16)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        for i in range(0, n, step):
            lbl = dim_names[i] if i < len(dim_names) else f"d{i}"
            p.drawText(int(x + i * bw), strip_y + strip_h + 12, lbl[:6])
        self._legend(p, [("█ belief entropy", QtGui.QColor(231, 76, 60)),
                         ("▒ G′ σ band", QtGui.QColor(46, 204, 113))],
                     y=y + h - 2, x=x)

    def _per_dim_entropy(self, be: dict, dim_names: list, f) -> List[float]:
        """Pull per-dim entropy values out of the belief_entropies dict (keys may
        be 'd0'/'0'/dim_names; falls back to broadcasting 'total' or to gprime
        uncertainty length)."""
        if not be:
            return []
        # try integer/d-prefixed keys
        per = {}
        for k, v in be.items():
            try:
                per[float(k)] = float(v)
            except (TypeError, ValueError):
                if isinstance(k, str) and k.lower().startswith("d") and k[1:].replace(".", "", 1).isdigit():
                    per[float(k[1:])] = float(v)
                elif k in dim_names:
                    per[float(dim_names.index(k))] = float(v)
        if per:
            n = int(max(per.keys())) + 1
            return [per.get(float(i), 0.0) for i in range(n)]
        # fallback: gprime_uncertainty length broadcast of total
        tot = be.get("total")
        if tot is None and be:
            tot = float(list(be.values())[0])
        if tot is None:
            return []
        unc = f.gprime_uncertainty
        n = int(unc.size) if (unc is not None and unc.size) else 0
        return [float(tot)] * n if n else [float(tot)]

    def _ranked_bars(self, p, items: list, key: str, label_fn, x, y, w, h, col) -> None:
        """v7: ranked swatch+bar+caption list (M3 by prediction_error, M4 by support)."""
        if not items:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 12, "(empty)"); return
        ranked = sorted(items, key=lambda d: float(d.get(key, 0) or 0), reverse=True)[:8]
        mx = max(float(d.get(key, 0) or 0) for d in ranked) or 1.0
        rh = max(14, min(26, h // max(len(ranked), 1)))
        for i, d in enumerate(ranked):
            ry = y + i * rh
            if ry + rh > y + h:
                break
            v = float(d.get(key, 0) or 0)
            frac = v / mx if mx > 0 else 0.0
            # swatch
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.drawRect(x, ry + 3, 6, rh - 8)
            # bar
            bar_x = x + 12; bar_w = w - 12 - 96
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRect(bar_x, ry + 4, bar_w, rh - 12)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 200))
            p.fillRect(bar_x, ry + 4, int(bar_w * frac), rh - 12, col)
            # caption
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(bar_x + bar_w + 4, ry + rh - 8, f"{v:.2f}")
            p.setPen(DIM_COL)
            p.drawText(x + 12, ry + rh - 8, label_fn(d)[:30])

    def _cap_gauge(self, p, x, y, w, h, val, cap, label) -> None:
        """v7: small retention-cap engagement gauge (fact_count/m4_cap)."""
        if cap <= 0:
            return
        frac = min(val / cap, 1.0)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRoundedRect(x, y, w, h, 3, 3)
        col = QtGui.QColor(231, 76, 60) if frac > 0.9 else QtGui.QColor(46, 204, 113)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
        p.fillRect(x + 1, y + 1, int((w - 2) * frac), h - 2, col)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + h - 4, f"{label} {val}/{cap}")

    def _diff_inset(self, p, f, x, y, w, h) -> None:
        """v7: demoted |sanitized−raw| diff — a thin EMA-smoothed per-dim strip
        with dim labels + a y-scale caption (was the big bottom portrait)."""
        raw = f.obs_vector; san = f.sanitized_state
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y - 2, "|sanitized−raw| (EMA) · dim salience")
        if raw is None or san is None:
            return
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        san = np.asarray(san, dtype=np.float32).reshape(-1)
        d = min(len(raw), len(san))
        if d == 0:
            return
        diff = np.abs(san[:d] - raw[:d])
        mx = float(diff.max()) or 1.0
        sm = self._diff_smooth.value(mx)
        bw = w / d
        base_y = y + h - 4
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, base_y, x + w, base_y)
        for i in range(d):
            bx = int(x + i * bw)
            bh = int(diff[i] / max(sm, mx) * (h - 8))
            p.fillRect(bx + 1, base_y - bh, max(int(bw) - 2, 1), bh, QtGui.QColor(231, 76, 60, 160))
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + w - 90, y + 10, f"max={sm:.3g}")


# ----- NEW Goals & Motivation tab ---------------------------------------------

class GoalsMotivationView(_BaseCanvas):
    """v7: focal 'homeostasis tanks' (6 vertical tanks: level vs target setpoint,
    deficit gap, active highlight, Pareto ring). Goal stack → indented tree with
    per-node progress bars; goal_history → step-strip; deficit heatmap gets a
    correct caption + colorbar; deficits EMA-smoothed; trend y-scale snapped to
    a ScaleState; drive_goals radial 'where each drive pulls' inset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drive_hist: Deque[List[float]] = deque(maxlen=120)
        self.goal_hist: Deque[int] = deque(maxlen=120)
        self.temp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.emp_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        # v7: EMA-smoothed per-drive deficit + snapped trend scales.
        self._def_smooth = [_Smoother(0.2) for _ in range(6)]
        self._temp_scale = ScaleState(contract=0.05, head=0.06)
        self._emp_scale = ScaleState(contract=0.05, head=0.06)

    def set_frame(self, f: ObservabilityFrame) -> None:
        if f.drive_deficits is not None:
            defs = np.asarray(f.drive_deficits, dtype=np.float32).reshape(-1)[:6]
            self.drive_hist.append([float(x) for x in defs])
            for i in range(min(6, len(defs))):
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
        levels = f.drive_levels or []
        targets = f.drive_targets or []
        defs = [self._def_smooth[i]._v if self._def_smooth[i]._have else 0.0 for i in range(6)]
        pareto = set(getattr(f, "pareto_front", None) or [])
        active = int(getattr(f, "active_drive_id", 0) or 0)
        # ---- v7 focal: homeostasis tanks (top, full width) ----
        tank_h = h // 3
        self._title(p, "Homeostasis tanks — level vs target · gap = deficit · ◯ = Pareto · ▮ = active", x=8, y=14)
        self._caption(p, "each tank fills to drive_level; dashed line = drive_targets setpoint; red gap = deficit", x=8, y=26)
        self._tanks(p, levels, targets, defs, pareto, active, 8, 32, w - 16, tank_h)
        # ---- bottom-left: goal stack tree + goal_history step-strip ----
        by = tank_h + 40
        lw = w // 2 - 8
        self._title(p, "Goal stack (tree, deepest active first)", x=8, y=by)
        stack = getattr(f, "goal_stack", None) or []
        gy = by + 12
        gy = self._goal_tree(p, stack, 8, gy, lw)
        # goal_history step-strip under the stack
        self._goal_history_strip(p, 8, gy + 6, lw, 26)
        # ---- bottom-right: drive_goals radial inset + heatmap + trends ----
        rx = lw + 16
        rw = w - rx - 8
        self._drive_goals_inset(p, f, rx, by, 130, 130)
        self._title(p, "Drive deficit history heatmap", x=rx + 140, y=by)
        self._heatmap(p, rx + 140, by + 12, rw - 140, h // 4)
        self._trend_pair(p, rx, by + h // 4 + 24, rw, h - by - h // 4 - 30)

    def _tanks(self, p, levels, targets, defs, pareto, active, x, y, w, h) -> None:
        """v7 focal: 6 vertical homeostasis tanks."""
        n = 6
        tw = w // n - 8
        for i in range(n):
            did = i + 1
            tx = x + i * (w / n) + 4
            col = _to_qcolor(DRIVE_COLORS[did])
            lvl = float(levels[i]) if i < len(levels) else 0.0
            tgt = float(targets[i]) if i < len(targets) and targets[i] is not None else None
            dfc = float(defs[i]) if i < len(defs) else 0.0
            is_active = (did == active)
            is_pareto = i in pareto
            # tank frame
            frame_col = col if is_active else GRID_COL
            p.setPen(QtGui.QPen(frame_col, 2 if is_active else 1))
            p.setBrush(PANEL_BG); p.drawRect(int(tx), y, tw, h - 18)
            # level fill
            lh = int(np.clip(lvl, 0, 1) * (h - 20))
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 200))
            p.fillRect(int(tx) + 2, y + (h - 18) - lh, tw - 4, lh, col)
            # target setpoint dashed line
            if tgt is not None:
                ty = y + (h - 18) - int(np.clip(tgt, 0, 1) * (h - 20))
                p.setPen(QtGui.QPen(ACCENT, 1, QtCore.Qt.DashLine))
                p.drawLine(int(tx), ty, int(tx) + tw, ty)
            # deficit gap shading (red between level and target)
            if tgt is not None and dfc > 0.02:
                ly = y + (h - 18) - lh
                p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(231, 76, 60, 80))
                gap_top = min(ly, ty); gap_bot = max(ly, ty)
                p.fillRect(int(tx) + 2, gap_top, tw - 4, gap_bot - gap_top, QtGui.QColor(231, 76, 60, 80))
            # Pareto ring
            if is_pareto:
                p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
                p.setBrush(QtGui.QColor(0, 0, 0, 0))
                p.drawEllipse(int(tx) + tw // 2 - 6, y + (h - 18) - lh - 6, 12, 12)
            # labels
            p.setPen(col if is_active else TEXT_COL); p.setFont(_F_LABEL_B)
            p.drawText(int(tx), y + h - 4, f"{DRIVE_SHORT[did]} {lvl:.2f}")
            if tgt is not None:
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(int(tx) + tw - 40, y + h - 4, f"Δ{dfc:.2f}")

    def _goal_tree(self, p, stack, x, y, w) -> int:
        """v7: indented goal-stack tree with a per-node completion-progress bar."""
        if not stack:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 10, "(no active goals)"); return y + 14
        for i, g in enumerate(stack[:8]):
            did = g.get("drive_id", "?")
            tnorm = g.get("target_norm")
            tol = g.get("tolerance"); pri = g.get("priority")
            comp = bool(g.get("completed", False))
            depth = int(g.get("depth", 0) or 0)
            prog = float(g.get("progress", 1.0 if comp else 0.0) or 0.0)
            indent = depth * 14
            col = QtGui.QColor(46, 204, 113) if comp else ACCENT
            # tree connector
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            if depth > 0:
                p.drawText(x + indent - 10, y + 10, "└")
            p.setPen(col); p.setFont(_F_LABEL_B)
            tgt_s = f"|tgt|={float(tnorm):.2f}" if tnorm is not None else "no tgt"
            p.drawText(x + indent, y + 10, f"#{i+1} d{did} {tgt_s}")
            # progress bar
            bx = x + indent + 150; bw = w - indent - 210
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
            did = max(1, min(6, int(did)))
            col = _to_qcolor(DRIVE_COLORS[did])
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(int(x + i * cw), sy, max(int(cw), 1), sh, col)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + h + 10, f"{n} cycles · colour = drive id")

    def _drive_goals_inset(self, p, f, x, y, w, h) -> None:
        """v7: radial 'where each drive pulls' — spoke length = ||drive_goals[i]||
        (magnitude of each drive's pull on the goal vector)."""
        import math
        dgs = getattr(f, "drive_goals", None) or []
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG); p.drawRect(x, y, w, h)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x + 3, y + 11, "drive pull ‖·‖")
        cx, cy = x + w // 2, y + h // 2 + 6
        R = min(w, h) // 2 - 16
        mags = []
        for i in range(6):
            dg = dgs[i] if i < len(dgs) else None
            m = float(np.linalg.norm(np.asarray(dg, dtype=np.float32))) if dg is not None else 0.0
            mags.append(m)
        mx = max(mags) if mags else 1.0
        mx = mx if mx > 1e-6 else 1.0
        for i in range(6):
            ang = -math.pi / 2 + i * math.pi / 3
            r = (mags[i] / mx) * R
            ex = int(cx + r * math.cos(ang)); ey = int(cy + r * math.sin(ang))
            col = _to_qcolor(DRIVE_COLORS[i + 1])
            p.setPen(QtGui.QPen(col, 2)); p.drawLine(cx, cy, ex, ey)
            p.setBrush(col); p.setPen(QtGui.QPen(col, 1))
            p.drawEllipse(ex - 2, ey - 2, 4, 4)

    def _heatmap(self, p, x, y, w, h) -> None:
        hist = list(self.drive_hist)
        if not hist:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 10, "drive×cycle deficit heatmap (collecting…)"); return
        n = len(hist); cw = w / max(n, 1); rh = h / 6
        for i, defs in enumerate(hist):
            for j in range(min(6, len(defs))):
                a = int(np.clip(defs[j], 0, 1) * 230)
                if a < 8:
                    continue
                col = _to_qcolor(DRIVE_COLORS[j + 1]); col.setAlpha(a)
                p.fillRect(int(x + i * cw), int(y + j * rh), int(cw) + 1, int(rh) - 1, col)
        # v7: correct caption (alpha=more opaque=higher deficit, not 'darker')
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + h + 8, f"6 drives × {n} cycles · more opaque = higher deficit")
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


# ----- Controller + main window ---------------------------------------------

class _TransportBar(QtWidgets.QFrame):
    """v6 transport: pause / speed dial / step / scrub / cycle-throttle.

    Governs a ``PlaybackClock`` (view pause + slow-mo + scrub) and, when
    "throttle cycles" is checked, a ``CyclePacer`` (slow/stop the real cycle
    thread). Additive — only shown when installed on a window.
    """

    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(self, clock, pacer=None, parent=None):
        super().__init__(parent)
        self.clock = clock
        self.pacer = pacer
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setText("⏸ Pause"); self.play_btn.setCheckable(True)
        self.step_btn = QtWidgets.QToolButton(); self.step_btn.setText("⏭ Step")
        self.follow_btn = QtWidgets.QToolButton(); self.follow_btn.setText("⤓ Live")
        self.speed = QtWidgets.QComboBox()
        for s in self.SPEEDS:
            self.speed.addItem(f"{s:g}×")
        self.speed.setCurrentIndex(2)  # 1×
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(0); self.slider.setValue(0)
        self.label = QtWidgets.QLabel("0 / 0")
        self.label.setMinimumWidth(90)
        self.throttle = QtWidgets.QCheckBox("throttle cycles")
        self.throttle.setToolTip("When on, the speed dial + pause also slow/stop the "
                                 "actual cognitive cycle thread (not just the view).")
        for w in (self.play_btn, self.step_btn, self.follow_btn, self.speed,
                  self.slider, self.label):
            lay.addWidget(w)
        lay.addWidget(self.throttle, 0)
        # wire
        self.play_btn.toggled.connect(self._on_play)
        self.step_btn.clicked.connect(self._on_step)
        self.follow_btn.clicked.connect(self._on_follow)
        self.speed.currentIndexChanged.connect(self._on_speed)
        self.slider.valueChanged.connect(self._on_seek)
        if pacer is None:
            self.throttle.hide()
        else:
            self.throttle.toggled.connect(self._on_throttle)

    def set_range(self, n: int) -> None:
        self.slider.setMaximum(max(0, n - 1))

    def _on_play(self, checked: bool) -> None:
        # checked == paused
        self.play_btn.setText("▶ Play" if checked else "⏸ Pause")
        self.clock.set_paused(checked)
        if self.pacer is not None and self.throttle.isChecked():
            self.pacer.set_paused(checked)

    def _on_step(self) -> None:
        self.clock.step()
        self._sync_slider()

    def _on_follow(self) -> None:
        self.clock.follow_live()
        self._sync_slider()

    def _on_speed(self, _idx: int) -> None:
        s = self.SPEEDS[self.speed.currentIndex()]
        self.clock.set_speed(s)
        if self.pacer is not None and self.throttle.isChecked():
            from .playback import throttle_period
            self.pacer.set_period(throttle_period(s))

    def _on_seek(self, val: int) -> None:
        self.clock.seek(val)
        self.label.setText(f"{val} / {max(0, self.clock.n - 1)}")

    def _on_throttle(self, on: bool) -> None:
        if self.pacer is None:
            return
        if on:
            from .playback import throttle_period
            s = self.SPEEDS[self.speed.currentIndex()]
            self.pacer.set_period(throttle_period(s))
            self.pacer.set_paused(self.play_btn.isChecked())
        else:
            self.pacer.set_period(0.0)
            self.pacer.set_paused(False)

    def _sync_slider(self) -> None:
        i = self.clock.cursor_int
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.label.setText(f"{i} / {max(0, self.clock.n - 1)}")


class DashboardController:
    """Mutates persistent widget state from frames (no widget rebuild)."""

    def __init__(self, window: "ObservatoryWindow"):
        self.w = window
        # v6 flicker-free: skip the whole widget update when the frame is the
        # same cycle as the last one we rendered (idle/no-new-data → 0 repaints).
        self._last_cycle: int = -1

    def update(self, frame: Optional[ObservabilityFrame],
               rolling: Optional[List[ObservabilityFrame]] = None,
               cycle_error: Optional[str] = None) -> None:
        if frame is None and cycle_error is None:
            return
        f = frame
        if f is not None:
            # v6 flicker-free: drop redundant updates for an unchanged cycle
            # (heartbeat emitting the same cursor frame while production stalls).
            cid = int(getattr(f, "cycle_id", -1))
            if cycle_error is None and cid == self._last_cycle:
                return
            self._last_cycle = cid
            # single source of truth for the shared belief projection + history
            self.w.proj.update(f)
            v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            self.w.proj.push_history(self.w.proj.project(v))
            self.w.world.set_frame(f)
            self.w.portrait.set_frame(f)
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
            self.w._viol_summary.setText(self.w.viol.summary())
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
        # v6: top transport slot (hidden until install_transport is called).
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)
        self._top_frame = QtWidgets.QFrame()
        self._top_frame.hide()
        self._top_layout = QtWidgets.QHBoxLayout(self._top_frame)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self._top_frame)
        cv.addWidget(tabs, 1)
        self.setCentralWidget(central)
        self._tabs = tabs

        # shared dimension-agnostic projection (World / Phase-Space / Action)
        self.proj = BeliefProjection(window=256)

        # Overview
        ov = QtWidgets.QWidget(); ov_lay = QtWidgets.QGridLayout(ov)
        ov_lay.setContentsMargins(6, 6, 6, 6); ov_lay.setSpacing(6)
        self.world = WorldCanvas(); self.world.set_projection(self.proj)
        self.portrait = AgentPortraitView()   # v6 focal cognitive portrait
        self.drives = DrivesCanvas()
        self.trend = TrendCanvas(); self.attention = AttentionCanvas()
        self.status = StatusPanel()
        # v7: portrait owns the full top row (focal); world is demoted to a
        # compact inset in the bottom vitals strip (it's focal on Phase Space).
        ov_lay.addWidget(self.portrait, 0, 0, 1, 4)
        ov_lay.addWidget(self.world, 1, 0)
        ov_lay.addWidget(self.drives, 1, 1)
        ov_lay.addWidget(self.trend, 1, 2)
        ov_lay.addWidget(self.attention, 1, 3)
        ov_lay.addWidget(self.status, 1, 4)
        ov_lay.setRowStretch(0, 3)
        ov_lay.setRowStretch(1, 1)
        for c in range(5):
            ov_lay.setColumnStretch(c, 1)
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
        self._viol_summary = QtWidgets.QLabel("0 violations")
        self._viol_summary.setStyleSheet("color:#e74c3c; font-weight:bold; padding:2px 6px;")
        ret_lay.addWidget(self.retention)
        ret_lay.addWidget(self.rbta_bounds)
        ret_lay.addWidget(self._viol_summary)
        ret_lay.addWidget(self.viol, 1)
        tabs.addTab(ret, "Retention & Resources")

        # NEW: Memory & Belief
        self.memory = MemoryBeliefView()
        tabs.addTab(self.memory, "Memory & Belief")

        # NEW: Goals & Motivation
        self.goals = GoalsMotivationView()
        tabs.addTab(self.goals, "Goals & Motivation")

        self.controller = DashboardController(self)

    def install_transport(self, bar: QtWidgets.QWidget) -> None:
        """Dock a transport bar at the top of the window (additive)."""
        self._top_layout.addWidget(bar, 1)
        self._top_frame.show()


def make_app() -> "QtWidgets.QApplication":
    """Construct (but do not exec) the QApplication. Caller owns it."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(_qss())
    return app
