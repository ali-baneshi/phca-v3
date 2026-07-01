"""PHCA v3.0 — Cognitive Observatory (Observability v3, PyQt5).

Multi-tab dashboard that visualizes the cognition as it happens. Reuses the
v2 data layer (``ObservabilityFrame``/``ObservabilityStore``/``SessionRecorder``)
unchanged; only the rendering becomes a real Qt GUI. The same
``DashboardController`` drives both the live launcher and the ``--qt`` replay.

Tabs (the "worthy of such a system" surface):
  1. Overview         — world + drives + trend + attention + status
  2. Cognitive Flow   — animated ASI→M2→G′→PE→PEU→TSPL→Action→RBTA pipeline
                        with per-module ms (cost-colored), pulse on run, red on
                        violation; bottom strip = module×cycle cost heatmap.
  3. Action Selection — candidate-score bars (chosen highlighted), ε-greedy
                        decay, explore/exploit dots, continuous-action signed
                        bars, rationale text.
  4. Phase Space      — GridWorld trajectory w/ time-colored trail + G′ spatial
                        error map; drives Pareto scatter; MuJoCo dim error bars.
  5. Retention        — M3 episodes + cap, M4 facts + cap, RSS trend, latency
                        histogram, scrolling RBTA violation table.

All polling-side: read-only on the frame; the cycle thread never touches Qt.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .observability import ObservabilityFrame
from .render import _grid_base, _prediction_heatmap

from PyQt5 import QtWidgets, QtCore, QtGui


# Reuse the v2 constants so live + replay stay consistent with the matplotlib
# dashboard's colour/driver naming.
DRIVE_NAMES = {
    1: "D1 PredErr", 2: "D2 Critical", 3: "D3 Compet",
    4: "D4 Curious", 5: "D5 Energy", 6: "D6 Empower",
}
DRIVE_COLORS = {
    1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd",
    5: "#1f77b4", 6: "#17becf",
}
TREND_WINDOW = 200
# PHCA cognitive pipeline (order matters for the flow graph). The main chain
# runs left→right; side modules attach below.
PIPELINE = ["sanitize", "memory_write", "prediction", "peu", "tspl",
            "action_selection", "rbta"]
SIDE_MODULES = ["gprime_learn", "mdim", "attn", "hpm", "cr", "consolidation"]
PIPELINE_LABEL = {
    "sanitize": "ASI/Sanitize", "memory_write": "M2 Write",
    "prediction": "G′ Predict", "peu": "PEU", "tspl": "TSPL",
    "action_selection": "Action", "rbta": "RBTA",
    "gprime_learn": "G′ Learn", "mdim": "MDIM", "attn": "Attention",
    "hpm": "HPM", "cr": "CR", "consolidation": "Consolid.",
}


def _cost_color(ms: float) -> str:
    if ms < 5.0:
        return "#2ca02c"   # green
    if ms < 20.0:
        return "#ff7f0e"   # orange
    return "#d62728"       # red


def _to_qcolor(hex_or_rgb: Any) -> "QtGui.QColor":
    from PyQt5.QtGui import QColor
    if isinstance(hex_or_rgb, str):
        return QColor(hex_or_rgb)
    r, g, b = hex_or_rgb
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class _BaseCanvas(QtWidgets.QWidget):
    """Base for all QPainter canvases — stores state, repaints on update()."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(420, 240)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(18, 18, 24))
        self.setPalette(pal)

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:  # pragma: no cover (GUI)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(18, 18, 24))
        try:
            self._draw(p)
        finally:
            p.end()


# ----- Overview tab canvases -------------------------------------------------

class WorldCanvas(_BaseCanvas):
    """GridWorld grid + G′ heatmap + trail, or MuJoCo predicted-vs-goal bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.trail: Deque[Any] = deque(maxlen=120)

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        if f.grid is not None and f.agent_pos is not None:
            if not self.trail or self.trail[-1] != tuple(f.agent_pos):
                self.trail.append(tuple(f.agent_pos))
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Awaiting cycle…")
            return
        w, h = self.width(), self.height()
        is_grid = f.grid is not None
        if is_grid:
            g = f.grid
            n = g.shape[0]
            cell = min(w, h - 20) / n
            ox = (w - cell * n) / 2
            oy = 18
            for r in range(n):
                for c in range(n):
                    v = g[r, c]
                    col = QtGui.QColor(40, 40, 50)
                    if v == 1:
                        col = QtGui.QColor(120, 120, 130)   # wall
                    elif v == 2:
                        col = QtGui.QColor(39, 174, 96)     # goal
                    p.fillRect(int(ox + c * cell), int(oy + r * cell),
                               int(cell), int(cell), col)
            # G′ heatmap (confidence-guarded, derived from predicted_state)
            heat = _prediction_heatmap(f.predicted_state, n)
            if heat is not None:
                for r in range(n):
                    for c in range(n):
                        a = float(np.clip(heat[r, c], 0, 1))
                        if a <= 0.02:
                            continue
                        p.fillRect(int(ox + c * cell), int(oy + r * cell),
                                   int(cell), int(cell),
                                   QtGui.QColor(231, 76, 60, int(140 * a)))
            # trail
            tl = list(self.trail)
            for i in range(1, len(tl)):
                (r0, c0), (r1, c1) = tl[i - 1], tl[i]
                alpha = int(60 + 195 * i / max(len(tl), 1))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, alpha), 2))
                p.drawLine(int(ox + c0 * cell + cell / 2), int(oy + r0 * cell + cell / 2),
                           int(ox + c1 * cell + cell / 2), int(oy + r1 * cell + cell / 2))
            # agent
            if tl:
                r, c = tl[-1]
                p.setBrush(QtGui.QColor(241, 196, 15))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 1))
                p.drawEllipse(int(ox + c * cell + cell / 4),
                              int(oy + r * cell + cell / 4),
                              int(cell / 2), int(cell / 2))
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(8, 14, f"GridWorld {n}×{n}  conf={f.prediction_confidence:.2f}")
        else:
            # MuJoCo: predicted vs goal-ref bars
            pred = f.predicted_state
            ref = f.goal_ref
            if pred is None:
                p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "MuJoCo — no prediction"); return
            n = min(len(pred), 12)
            ref = ref[:n] if ref is not None else None
            bw = (w - 40) / n
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(8, 14, f"MuJoCo predicted vs goal-ref  (dims={n})")
            for i in range(n):
                x = 20 + i * bw
                pv = float(np.clip(pred[i], -2, 2))
                p.fillRect(int(x), int(h / 2 - abs(pv) * 30), int(bw - 4), int(abs(pv) * 60),
                           QtGui.QColor(231, 76, 60, 180))
                if ref is not None:
                    rv = float(np.clip(ref[i], -2, 2))
                    p.setPen(QtGui.QPen(QtGui.QColor(39, 174, 96), 2))
                    p.drawLine(int(x + bw / 2), int(h / 2 - rv * 30),
                               int(x + bw / 2), int(h / 2 + rv * 30))
                    p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 210), 1))
            p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1))
            p.drawLine(20, int(h / 2), w - 20, int(h / 2))


class DrivesCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None or not f.drive_levels:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Drives…"); return
        n = len(f.drive_levels)
        barw = (self.width() - 40) / n
        for i in range(n):
            did = i + 1
            v = float(np.clip(f.drive_levels[i], 0, 1))
            t = float(np.clip(f.drive_targets[i], 0, 1)) if i < len(f.drive_targets) else v
            x = 20 + i * barw
            active = (f.active_drive_id == did)
            # value bar
            p.fillRect(int(x), int(self.height() - 30 - v * (self.height() - 60)),
                       int(barw - 6), int(v * (self.height() - 60)),
                       _to_qcolor(DRIVE_COLORS[did]))
            # target marker
            ty = int(self.height() - 30 - t * (self.height() - 60))
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15), 2))
            p.drawLine(int(x), ty, int(x + barw - 6), ty)
            p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 210), 1))
            lbl = DRIVE_NAMES.get(did, f"D{did}")
            if active:
                p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(int(x), self.height() - 12, lbl)
            p.setPen(QtGui.QColor(200, 200, 210)); p.setFont(QtGui.QFont("Sans", 8))


class TrendCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.err: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.conf: Deque[float] = deque(maxlen=TREND_WINDOW)

    def push(self, f: ObservabilityFrame) -> None:
        self.err.append(float(f.prediction_error)); self.conf.append(float(f.prediction_confidence))
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        if not self.err:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Error/Confidence trend…"); return
        w, h = self.width(), self.height()
        n = len(self.err)
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 70), 1)); p.drawLine(10, h - 20, w - 10, h - 20)
        for arr, col in ((self.err, "#e74c3c"), (self.conf, "#2ecc71")):
            p.setPen(QtGui.QPen(_to_qcolor(col), 2))
            for i in range(1, n):
                x0 = 10 + (i - 1) * (w - 20) / max(n - 1, 1)
                x1 = 10 + i * (w - 20) / max(n - 1, 1)
                y0 = h - 20 - float(np.clip(arr[i - 1], 0, 1)) * (h - 40)
                y1 = h - 20 - float(np.clip(arr[i], 0, 1)) * (h - 40)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.setPen(QtGui.QColor(200, 200, 210))
        p.drawText(8, 16, f"err={self.err[-1]:.3f}  conf={self.conf[-1]:.3f}  (last {n})")


class AttentionCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None or not f.attention_indices:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Attention…"); return
        n = len(f.attention_indices)
        bw = (self.width() - 40) / n
        for i in range(n):
            x = 20 + i * bw
            sal = float(np.clip(f.attention_saliences[i], 0, 1)) if i < len(f.attention_saliences) else 0.0
            p.fillRect(int(x), int(self.height() - 30 - sal * (self.height() - 60)),
                       int(bw - 6), int(sal * (self.height() - 60)),
                       QtGui.QColor(155, 89, 182, 200))
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(int(x), self.height() - 12, str(int(f.attention_indices[i])))


class StatusPanel(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.cycle_error: Optional[str] = None

    def set_state(self, f: Optional[ObservabilityFrame], err: Optional[str]) -> None:
        self.frame = f; self.cycle_error = err; self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None and self.cycle_error is None:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Status…"); return
        lines: List[str] = []
        if f is not None:
            rbta_ok = (f.violations_count == 0)
            lines.append(f"Cycle {f.cycle_id}  |  RBTA: {'OK' if rbta_ok else str(f.violations_count)+' violations'}")
            for v in f.rbta_violations[:4]:
                lines.append(f"  {v.get('module','?')}/{v.get('bound_type','?')}: "
                             f"{v.get('measured','?')} > {v.get('allowed','?')}")
            lines.append(f"M3 {f.episode_count}/{f.m3_cap}   M4 {f.fact_count}/{f.m4_cap} (prune→{f.m4_prune_target})")
            lines.append(f"RSS {f.rss_bytes/1e6:.1f} MB   latency {f.latency_ms:.1f} ms")
            ad = f.active_drive_id
            lines.append(f"Active drive: {DRIVE_NAMES.get(ad, ad)}")
            r = f.action_rationale
            if r:
                tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
                lines.append(f"Action: {tag}  ε={r.get('eps', 0):.3f}  K={r.get('k_candidates')}  "
                             f"score={r.get('best_score')}")
        if self.cycle_error:
            p.setPen(QtGui.QColor(231, 76, 60))
            p.drawText(self.rect(), 0x84, f"CYCLE ERROR: {self.cycle_error}")
            return
        p.setPen(QtGui.QColor(220, 220, 230))
        p.setFont(QtGui.QFont("Monospace", 9))
        for i, ln in enumerate(lines):
            p.drawText(10, 20 + i * 16, ln)


# ----- Cognitive Flow tab ----------------------------------------------------

class CognitiveFlowView(_BaseCanvas):
    """Animated directed graph of the pipeline with per-module ms + violations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.heat: Deque[Dict[str, float]] = deque(maxlen=60)
        self.viol_mods: set = set()
        self.pulse: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        self.heat.append(dict(f.module_timings))
        self.viol_mods = {v.get("module", "") for v in f.rbta_violations}
        self.pulse = 12
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Cognitive flow…"); return
        nodes = PIPELINE + SIDE_MODULES
        n = len(PIPELINE)
        # main chain laid out left→right in the upper band
        band_y = h // 3
        side_y = 2 * h // 3
        pos: Dict[str, Any] = {}
        for i, mod in enumerate(PIPELINE):
            pos[mod] = (int((i + 0.5) * w / n), band_y)
        for j, mod in enumerate(SIDE_MODULES):
            pos[mod] = (int((j + 0.5) * w / len(SIDE_MODULES)), side_y)
        # edges: chain
        for i in range(n - 1):
            a, b = pos[PIPELINE[i]], pos[PIPELINE[i + 1]]
            self._edge(p, a, b, anim=self.pulse > 0)
        # side edges attach to nearest main node
        for sm in SIDE_MODULES:
            a = pos[sm]
            # attach to prediction or memory_write for context modules
            anchor = "prediction" if sm in ("gprime_learn", "attn") else "memory_write"
            b = pos.get(anchor, pos[PIPELINE[0]])
            self._edge(p, a, b, dashed=True)
        # nodes
        for mod in nodes:
            x, y = pos[mod]
            ms = float(f.module_timings.get(mod, 0.0))
            viol = mod in self.viol_mods
            col = QtGui.QColor(231, 76, 60) if viol else _to_qcolor(_cost_color(ms))
            r = 26 + (4 if self.pulse > 0 else 0)
            p.setBrush(col); p.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 2))
            p.drawEllipse(int(x - r), int(y - r), int(r * 2), int(r * 2))
            p.setPen(QtGui.QColor(20, 20, 24))
            p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(int(x - 30), int(y - 4), 60, 14, 0x84, PIPELINE_LABEL.get(mod, mod))
            p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(int(x - 24), int(y + 10), 48, 12, 0x84, f"{ms:.1f}ms")
        # bottom strip: module×cycle cost heatmap
        self._heatmap(p)
        if self.pulse > 0:
            self.pulse -= 1

    def _edge(self, p: QtGui.QPainter, a, b, anim: bool = False, dashed: bool = False) -> None:
        pen = QtGui.QPen(QtGui.QColor(241, 196, 15, 220 if anim else 120), 2)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        p.setPen(pen)
        p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

    def _heatmap(self, p: QtGui.QPainter) -> None:
        if not self.heat:
            return
        mods = PIPELINE + SIDE_MODULES
        h = self.height(); strip_h = 28; top = h - strip_h - 4
        cw = (self.width() - 20) / max(len(self.heat), 1)
        for i, mts in enumerate(self.heat):
            for j, mod in enumerate(mods):
                ms = float(mts.get(mod, 0.0))
                a = int(np.clip(ms / 25.0, 0, 1) * 220)
                if a < 8:
                    continue
                p.fillRect(int(10 + i * cw), int(top + j * (strip_h / len(mods))),
                           int(cw), int(strip_h / len(mods)) - 1,
                           QtGui.QColor(231, 76, 60, a))


# ----- Action Selection tab --------------------------------------------------

class CandidateScoreView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.eps_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.ee_hist: Deque[bool] = deque(maxlen=TREND_WINDOW)

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f
        r = f.action_rationale
        if r:
            self.eps_hist.append(float(r.get("eps", 0.0)))
            self.ee_hist.append(bool(r.get("explored", False)))
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Action selection…"); return
        w, h = self.width(), self.height()
        scores = f.candidate_scores
        r = f.action_rationale
        chosen = -1
        if scores and r and not r.get("explored"):
            chosen = int(np.argmax(scores)) if scores else -1
        # top: candidate bars
        top_h = h // 2
        if scores:
            n = len(scores)
            bw = (w - 40) / n
            mx = max(scores) if scores else 1.0
            mx = mx if mx > 1e-6 else 1.0
            for i, s in enumerate(scores):
                x = 20 + i * bw
                bh = int((s / mx) * (top_h - 40))
                col = QtGui.QColor(241, 196, 15) if i == chosen else QtGui.QColor(52, 152, 219, 180)
                p.fillRect(int(x), int(top_h - 20 - bh), int(bw - 6), bh, col)
                p.setPen(QtGui.QColor(200, 200, 210))
                p.drawText(int(x), top_h - 6, str(i))
        p.setPen(QtGui.QColor(200, 200, 210))
        p.drawText(8, 16, f"K={len(scores)}  chosen={chosen}  "
                            f"score={r.get('best_score') if r else None}")
        # middle: continuous action signed bars
        if f.continuous_action is not None:
            ca = f.continuous_action
            mid_y = top_h + 8
            mid_h = (h - top_h - 60) // 2
            p.drawText(8, mid_y + 12, "continuous action vector:")
            n = min(len(ca), 16)
            bw = (w - 40) / n
            for i in range(n):
                x = 20 + i * bw
                v = float(np.clip(ca[i], -1, 1))
                p.fillRect(int(x + bw / 2 - 4), int(mid_y + 30 + mid_h / 2 - abs(v) * mid_h / 2),
                           8, int(abs(v) * mid_h),
                           QtGui.QColor(155, 89, 182) if v >= 0 else QtGui.QColor(231, 76, 60))
            p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1))
            p.drawLine(20, int(mid_y + 30 + mid_h / 2), w - 20, int(mid_y + 30 + mid_h / 2))
        # bottom: ε-greedy decay + explore/exploit dots
        bot_y = h - 50
        if self.eps_hist:
            n = len(self.eps_hist)
            p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
            for i in range(1, n):
                x0 = 10 + (i - 1) * (w - 20) / max(n - 1, 1)
                x1 = 10 + i * (w - 20) / max(n - 1, 1)
                p.drawLine(int(x0), int(bot_y - self.eps_hist[i - 1] * 30),
                           int(x1), int(bot_y - self.eps_hist[i] * 30))
            for i, explored in enumerate(self.ee_hist):
                x = int(10 + i * (w - 20) / max(n - 1, 1))
                p.setBrush(QtGui.QColor(231, 76, 60) if explored else QtGui.QColor(46, 204, 113))
                p.drawEllipse(x - 2, bot_y + 6, 4, 4)
        p.setPen(QtGui.QColor(200, 200, 210))
        p.drawText(8, h - 8, "ε-greedy decay (blue) + explore(red)/exploit(green) dots")


# ----- Phase Space tab -------------------------------------------------------

class TrajectoryView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail: Deque[Any] = deque(maxlen=400)
        self.errmap: Optional[np.ndarray] = None
        self.grid_n: int = 0
        self.is_grid: bool = True

    def set_frame(self, f: ObservabilityFrame) -> None:
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
        w, h = self.width(), self.height()
        if self.is_grid:
            n = self.grid_n or 5
            cell = min(w, h - 30) / n
            ox = (w - cell * n) / 2; oy = 6
            # error map background
            if self.errmap is not None:
                for r in range(n):
                    for c in range(n):
                        a = int(np.clip(self.errmap[r, c], 0, 1) * 200)
                        p.fillRect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell),
                                   QtGui.QColor(231, 76, 60, a))
            # trail
            tl = list(self.trail)
            for i in range(1, len(tl)):
                (r0, c0), (r1, c1) = tl[i - 1], tl[i]
                alpha = int(60 + 195 * i / len(tl))
                p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, alpha), 2))
                p.drawLine(int(ox + c0 * cell + cell / 2), int(oy + r0 * cell + cell / 2),
                           int(ox + c1 * cell + cell / 2), int(oy + r1 * cell + cell / 2))
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(8, h - 8, f"trajectory (trail={len(tl)})  +  G′ spatial |pred−actual| heatmap")
        else:
            p.setPen(QtGui.QColor(160, 160, 160))
            p.drawText(self.rect(), 0x84, "MuJoCo phase-space: see Pareto/dim-error views")


class ParetoView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist: Deque[List[float]] = deque(maxlen=200)

    def set_frame(self, f: ObservabilityFrame) -> None:
        if len(f.drive_levels) >= 6:
            self.hist.append([float(f.drive_levels[0]), float(f.drive_levels[2]),
                              float(f.drive_levels[4])])
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        pts = list(self.hist)
        if len(pts) < 2:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Drives Pareto (D1/D3/D5)…"); return
        for i in range(1, len(pts)):
            x0 = 20 + pts[i - 1][0] * (w - 40); y0 = h - 20 - pts[i - 1][1] * (h - 40)
            x1 = 20 + pts[i][0] * (w - 40); y1 = h - 20 - pts[i][1] * (h - 40)
            alpha = int(40 + 215 * i / len(pts))
            p.setPen(QtGui.QPen(QtGui.QColor(241, 196, 15, alpha), 2))
            p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.setBrush(QtGui.QColor(241, 196, 15))
        p.drawEllipse(int(20 + pts[-1][0] * (w - 40)) - 4, int(h - 20 - pts[-1][1] * (h - 40)) - 4, 8, 8)
        p.setPen(QtGui.QColor(200, 200, 210))
        p.drawText(8, 16, "Drives Pareto scatter (D1 vs D3; D5 = color of trail age)")


# ----- Retention tab ---------------------------------------------------------

class RetentionView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m3: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.m4: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.rss: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.lat: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.m3_cap: int = 0; self.m4_cap: int = 0

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.m3.append(int(f.episode_count)); self.m4.append(int(f.fact_count))
        self.rss.append(float(f.rss_bytes)); self.lat.append(float(f.latency_ms))
        self.m3_cap = int(f.m3_cap); self.m4_cap = int(f.m4_cap)
        self.update()

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        if not self.m3:
            p.setPen(QtGui.QColor(160, 160, 160)); p.drawText(self.rect(), 0x84, "Retention…"); return
        panels = [("M3 episodes", self.m3, self.m3_cap, "#3498db"),
                  ("M4 facts", self.m4, self.m4_cap, "#9b59b6"),
                  ("RSS (MB)", [r / 1e6 for r in self.rss], None, "#e67e22"),
                  ("Latency (ms)", self.lat, None, "#2ecc71")]
        ph = h // 4
        for i, (title, series, cap, col) in enumerate(panels):
            top = i * ph + 4; bot = (i + 1) * ph - 4
            p.setPen(QtGui.QColor(200, 200, 210)); p.drawText(8, top + 12, title)
            n = len(series)
            mx = max(series) if series else 1.0
            mx = mx if mx > 1e-6 else 1.0
            p.setPen(QtGui.QPen(_to_qcolor(col), 2))
            for j in range(1, n):
                x0 = 10 + (j - 1) * (w - 20) / max(n - 1, 1)
                x1 = 10 + j * (w - 20) / max(n - 1, 1)
                y0 = bot - (series[j - 1] / mx) * (bot - top - 16)
                y1 = bot - (series[j] / mx) * (bot - top - 16)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
            if cap and cap > 0:
                cy = bot - (cap / mx) * (bot - top - 16) if mx > 0 else bot
                p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 1, QtCore.Qt.DashLine))
                p.drawLine(10, int(cy), w - 10, int(cy))
                p.setPen(QtGui.QColor(200, 200, 210))


class ViolationTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["cycle", "module", "bound_type", "measured", "allowed"])
        self._seen: int = 0

    def add_frame(self, f: ObservabilityFrame) -> None:
        for v in f.rbta_violations:
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(f.cycle_id)))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(str(v.get("module", ""))))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(v.get("bound_type", ""))))
            self.setItem(row, 3, QtWidgets.QTableWidgetItem(str(v.get("measured", ""))))
            self.setItem(row, 4, QtWidgets.QTableWidgetItem(str(v.get("allowed", ""))))
            self._seen += 1
        # trim to last 200 rows
        while self.rowCount() > 200:
            self.removeRow(0)
        if self._seen > 0 and self.rowCount() > 0:
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
            self.w.pareto.set_frame(f)
            self.w.retention.set_frame(f)
            self.w.viol.add_frame(f)
            # window title progress
            self.w.setWindowTitle(f"PHCA Cognitive Observatory — cycle {f.cycle_id}")
        else:
            self.w.status.set_state(None, cycle_error)
        if cycle_error:
            self.w.status.set_state(f, cycle_error)


class ObservatoryWindow(QtWidgets.QMainWindow):
    def __init__(self, title: str = "PHCA Cognitive Observatory"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1280, 800)
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Overview
        ov = QtWidgets.QWidget(); ov_lay = QtWidgets.QGridLayout(ov)
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
        self.traj = TrajectoryView(); self.pareto = ParetoView()
        ps_lay.addWidget(self.traj); ps_lay.addWidget(self.pareto)
        tabs.addTab(ps, "Phase Space & Trajectory")

        # Retention
        ret = QtWidgets.QWidget(); ret_lay = QtWidgets.QVBoxLayout(ret)
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
    return app
