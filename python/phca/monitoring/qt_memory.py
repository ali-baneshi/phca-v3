"""Memory & Belief panel (extracted from qt_dashboard monolith)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.playback import _Smoother, freeze_sig

# Shared chrome / helpers from the panel module (already loaded when we import).
from phca.monitoring.qt_base import (
    PANEL_BG,
    GRID_COL,
    TEXT_COL,
    DIM_COL,
    _F_AXIS,
    _BaseCanvas,
    _draw_data_contract_banner,
    _window_session_incomplete,
    _drive_color,
    _retention_score,
    _draw_decay_sparkline,
)

class MemoryBeliefView(_BaseCanvas):
    """Belief geography: module entropy chips + G′ σ band (honest; not fake
    per-dim entropy broadcast). M3/M4 ranked cards; |sanitized−raw| inset live."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._m3_sig: str = ""
        self._m4_sig: str = ""
        self._m3_cache: List[dict] = []
        self._m4_cache: List[dict] = []
        self._diff_smooth = _Smoother(0.2)
        self._replay: bool = False
        self._review: bool = False

    def _diff_max(self, f: ObservabilityFrame) -> Optional[float]:
        raw = f.obs_vector
        san = f.sanitized_state
        if raw is None or san is None:
            return None
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        san = np.asarray(san, dtype=np.float32).reshape(-1)
        d = min(len(raw), len(san))
        if d == 0:
            return None
        return float(np.abs(san[:d] - raw[:d]).max())

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        if not histories_done:
            mx = self._diff_max(f)
            if mx is not None:
                self._diff_smooth.value(mx)
        self._dirty = True

    def _memory_live_empty(self, f: ObservabilityFrame) -> bool:
        return not (
            getattr(f, "m3_recent", None) or getattr(f, "m3_top_error", None)
            or getattr(f, "m4_relevant", None) or getattr(f, "m4_top", None))

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self._m3_sig = ""
        self._m4_sig = ""
        self._m3_cache = []
        self._m4_cache = []
        self._diff_smooth.reset()
        for f in frames:
            mx = self._diff_max(f)
            if mx is not None:
                self._diff_smooth.value(mx)
        if frames:
            self.frame = frames[-1]
        self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "Memory & belief…"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review,
            panel_key="memory", multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        # ---- focal: module entropy chips + G′ σ band (no fake per-dim heat) ----
        self._title(p, "Belief geography — module entropies + G′ σ band", x=10, y=14 + y0)
        self._caption(p, "chips = module belief_entropies · strip = G′ posterior σ per dim (dim_names)",
                      x=10, y=26 + y0)
        self._belief_geography(p, f, 10, 32 + y0, w - 20, h // 3)
        # ---- left: M3 ranked bars (frozen via content hash) ----
        col_w = w // 2 - 8
        my = h // 3 + 40 + y0
        self._title(p, "M3 episodic memory — ranked by retention score", x=10, y=my)
        m3 = list(getattr(f, "m3_recent", None) or []) + list(getattr(f, "m3_top_error", None) or [])
        sig = freeze_sig(m3)
        if sig != self._m3_sig:
            self._m3_cache = m3; self._m3_sig = sig
        if self._replay and not m3:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(10, my + 24, "(M3 episodic lists not recorded in JSONL replay)")
        else:
            self._ranked_decay_cards(p, self._m3_cache, f,
                                     label_fn=lambda ep: f"d{ep.get('drive_id','?')} conf={ep.get('confidence','?')}",
                                     score_fn=lambda ep, fr: _retention_score(
                                         max(0, int(fr.cycle_id) - int(ep.get('timestamp', fr.cycle_id))),
                                         max(float(ep.get('confidence', 0.1) or 0.1) * 80.0, 5.0)),
                                     x=10, y=my + 12, w=col_w, h=h - my - 36,
                                     col=QtGui.QColor(52, 152, 219))
        # v8 B6: episodic timeline mark strip (M3 events coloured by salience/confidence)
        self._episodic_timeline(p, self._m3_cache, f, 10, h - 22, col_w, 14)
        # ---- right: M4 retention cards + retention-cap gauge ----
        rx = col_w + 16
        m4 = list(getattr(f, "m4_relevant", None) or []) + list(getattr(f, "m4_top", None) or [])
        m4_has_timestamps = any(
            isinstance(item.get("timestamp"), (int, float)) and item["timestamp"] > 0
            for item in m4 if isinstance(item, dict)
        )
        m4_title = (
            "M4 consolidated facts — ranked by retention score"
            if m4_has_timestamps else
            "M4 consolidated facts — ranked by confidence/support (temporal age unavailable)"
        )
        self._title(p, m4_title, x=rx, y=my)
        sig4 = freeze_sig(m4)
        if sig4 != self._m4_sig:
            self._m4_cache = m4; self._m4_sig = sig4
        if self._replay and not m4:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(rx, my + 24, "(M4 top not in this JSONL · newer sessions record m4_top)")
        else:
            self._ranked_decay_cards(
                p, self._m4_cache, f,
                label_fn=lambda fac: f"{fac.get('fact_type', fac.get('predicate','?'))} {str(fac.get('summary',''))[:18]}",
                score_fn=(
                    (lambda fac, fr: _retention_score(
                        max(0, int(fr.cycle_id) - int(fac.get("timestamp", fr.cycle_id))),
                        max(float(fac.get("confidence", fac.get("frequency", fac.get("support", 0.1))) or 0.1) * 80.0, 5.0),
                    ))
                    if m4_has_timestamps else
                    (lambda fac, fr: float(
                        fac.get("confidence", fac.get("frequency", fac.get("support", 0.0))) or 0.0
                    ))
                ),
                x=rx, y=my + 12, w=w - rx - 10, h=h - my - 40,
                col=QtGui.QColor(155, 89, 182),
                score_label="R" if m4_has_timestamps else "conf",
                draw_decay=m4_has_timestamps,
            )
        # retention-cap gauge (top-right of the M4 column)
        self._cap_gauge(p, rx, my - 14, 120, 16, int(f.fact_count), int(f.m4_cap), "M4 cap")
        # v7: demoted |sanitized−raw| diff as a thin EMA-smoothed inset (bottom strip)
        self._diff_inset(p, f, 10, h - 26, w - 20, 22)

    def _episodic_timeline(self, p, items: list, f: ObservabilityFrame,
                           x: int, y: int, w: int, h: int) -> None:
        """v8 B6: M3 episodic events as a coloured mark strip along cycle time."""
        if not items:
            return
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y - 2, "episodic timeline (mark = M3 event · opacity = confidence)")
        n = len(items); cw = w / max(n, 1)
        cyc = int(getattr(f, "cycle_id", 0) or 0)
        for i, ep in enumerate(items[:40]):
            conf = float(ep.get("confidence", 0.5) or 0.5)
            did = int(ep.get("drive_id", 1) or 1)
            col = _drive_color(did); col.setAlpha(int(80 + 140 * min(1.0, conf)))
            bx = int(x + i * cw)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(bx + 1, y + 2, max(int(cw) - 1, 2), h - 4, col)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + w - 60, y + h - 1, f"now c{cyc}")

    def _belief_geography(self, p, f, x, y, w, h) -> None:
        """Module entropy chips + G′ σ strip — no fake per-dim entropy broadcast."""
        dim_names = list(getattr(f, "dim_names", []) or [])
        be = dict(getattr(f, "belief_entropies", {}) or {})
        unc = (np.asarray(f.gprime_uncertainty, dtype=np.float32).reshape(-1)
               if f.gprime_uncertainty is not None else None)
        # Module chips (non-numeric keys like G'/MDIM/ATTN)
        chips = []
        for k, v in be.items():
            if self._is_per_dim_entropy_key(k, dim_names):
                continue
            try:
                chips.append((str(k), float(v)))
            except (TypeError, ValueError):
                continue
        chip_h = 16
        cx = x
        for name, val in chips[:6]:
            label = f"{name}={val:.2f}"
            tw = max(48, len(label) * 6 + 10)
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRoundedRect(cx, y, tw, chip_h, 3, 3)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(cx + 4, y + 12, label)
            cx += tw + 6
        if not chips and (unc is None or unc.size == 0):
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 14, "(no module entropies / G′ σ yet)")
            return
        strip_y = y + chip_h + 6
        strip_h = max(12, h - chip_h - 22)
        if unc is None or unc.size == 0:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, strip_y + 12, "(G′ σ unavailable)")
            return
        n = int(unc.size)
        umx = float(unc.max()) if unc.size else 1.0
        umx = umx if umx > 1e-6 else 1.0
        bw = w / max(n, 1)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRect(x, strip_y, w, strip_h)
        for i in range(n):
            bx = int(x + i * bw)
            uh = int(float(np.clip(unc[i] / umx, 0, 1)) * strip_h)
            p.fillRect(bx + 1, strip_y + strip_h - uh, max(int(bw) - 2, 1), uh,
                       QtGui.QColor(46, 204, 113, 180))
        step = max(1, n // 16)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        for i in range(0, n, step):
            lbl = dim_names[i] if i < len(dim_names) else f"d{i}"
            p.drawText(int(x + i * bw), strip_y + strip_h + 12, lbl[:6])
        self._legend(p, [("▒ G′ σ", QtGui.QColor(46, 204, 113))],
                     y=y + h - 2, x=x)

    @staticmethod
    def _is_per_dim_entropy_key(k, dim_names: list) -> bool:
        if k in dim_names:
            return True
        if isinstance(k, (int, float)):
            return True
        if isinstance(k, str) and k.lower().startswith("d") and k[1:].replace(".", "", 1).isdigit():
            return True
        try:
            float(k)
            return True
        except (TypeError, ValueError):
            return False

    def _per_dim_entropy(self, be: dict, dim_names: list, f) -> List[float]:
        """Legacy helper kept for tests: only true per-dim keys, else empty.

        Does NOT broadcast module scalars across dims (that lied in the UI).
        """
        if not be:
            return []
        per = {}
        for k, v in be.items():
            if not self._is_per_dim_entropy_key(k, dim_names):
                continue
            try:
                if k in dim_names:
                    per[float(dim_names.index(k))] = float(v)
                elif isinstance(k, str) and k.lower().startswith("d"):
                    per[float(k[1:])] = float(v)
                else:
                    per[float(k)] = float(v)
            except (TypeError, ValueError):
                continue
        if not per:
            return []
        n = int(max(per.keys())) + 1
        return [per.get(float(i), 0.0) for i in range(n)]

    def _ranked_decay_cards(self, p, items: list, f: ObservabilityFrame,
                            label_fn, score_fn, x, y, w, h, col,
                            *, score_label: str = "R", draw_decay: bool = True) -> None:
        """Ranked cards with a decay trace only when a temporal age is recorded."""
        if not items:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x, y + 12, "(empty)"); return
        scored = [(float(score_fn(d, f)), d) for d in items]
        ranked = sorted(scored, key=lambda t: t[0], reverse=True)[:8]
        rh = max(16, min(30, h // max(len(ranked), 1)))
        for i, (rscore, d) in enumerate(ranked):
            ry = y + i * rh
            if ry + rh > y + h:
                break
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.drawRect(x, ry + 4, 6, rh - 10)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            p.drawText(x + 12, ry + rh - 8, label_fn(d)[:28])
            spark_x = x + 12; spark_w = w - 120
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
            p.drawRect(spark_x, ry + 6, spark_w, rh - 14)
            if draw_decay:
                S = max(rscore * 0.5, 2.0) if rscore > 1 else max(1.0 / max(rscore, 0.05), 2.0)
                _draw_decay_sparkline(p, spark_x + 2, ry + 7, spark_w - 4, rh - 16, S, col)
            else:
                p.setPen(QtGui.QPen(col, 2))
                p.drawLine(spark_x + 2, ry + rh // 2, spark_x + spark_w - 2, ry + rh // 2)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            val_lbl = f"{score_label}={rscore:.2f}"
            p.drawText(x + w - 58, ry + rh - 8, val_lbl)

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
