"""Retention & Resources tab — M3/M4/RSS/latency trends, RBTA bounds, violation table."""
from __future__ import annotations
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from phca.monitoring.qt_base import *  # noqa: F401, F403
from phca.monitoring.qt_base import (
    _BaseCanvas,
    ScaleState, _Smoother,
    _draw_data_contract_banner, _window_session_incomplete,
    _elide_line,
    RBTA_TO_FLOW, PIPELINE,
    _retention_score, _draw_measured_sparkline,
    _CAPTION_COL, _FOOTER_COL, _F_CAPTION, _F_AXIS, _F_LABEL, _F_LABEL_B,
)


class RetentionView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m3: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.m4: Deque[int] = deque(maxlen=TREND_WINDOW)
        self.rss: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.lat: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.m3_cap: int = 0; self.m4_cap: int = 0
        self.m3_events: List[int] = []
        self.m4_events: List[int] = []
        self.m3_reasons: Dict[int, str] = {}
        self.m4_reasons: Dict[int, str] = {}
        self.cycle_base: int = 0
        self._panel_scales: Dict[str, ScaleState] = {}
        self._leak_smooth = _Smoother(0.08)
        self.frame: Optional[ObservabilityFrame] = None
        self._replay: bool = False
        self._review: bool = False
        self._mech_frames: List[ObservabilityFrame] = []
        self._prefix_len: int = 0

    def _mechanism_line(self) -> str:
        if not self._mech_frames:
            return ""
        counts = mechanism_histogram(self._mech_frames, window=256)
        pct = mechanism_pct(counts)
        top = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)[:4]
        if not top:
            return ""
        parts = [f"{k} {v}%" for k, v in top]
        line = f"mechanism (last {len(self._mech_frames)}): " + " · ".join(parts)
        if self._review and self.frame is not None:
            line += f" · through cycle {int(self.frame.cycle_id)}"
            if self._prefix_len > 256:
                line += " · window ≤256 (session report uses full session)"
        elif not self._review:
            line += " · window ≤256"
        return line

    def _sync_mechanism_window(self, frames: List[ObservabilityFrame]) -> None:
        self._mech_frames = list(frames[-256:])

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.m3.clear()
        self.m4.clear()
        self.rss.clear()
        self.lat.clear()
        self.m3_events.clear()
        self.m4_events.clear()
        self.m3_reasons.clear()
        self.m4_reasons.clear()
        self.cycle_base = 0
        self._leak_smooth.reset()
        self._panel_scales.clear()
        prev_m3 = None
        prev_m4 = None
        for f in frames:
            if not self.m3:
                self.cycle_base = int(f.cycle_id)
            ep_count = int(getattr(f, "m3_count", 0) or 0) or int(f.episode_count)
            fact_count = int(f.fact_count)
            self.m3.append(ep_count)
            self.m4.append(fact_count)
            self.rss.append(float(f.rss_bytes))
            self.lat.append(float(f.latency_ms))
            self.m3_cap = int(f.m3_cap)
            self.m4_cap = int(f.m4_cap)
            reason = self._prune_reason(f)
            if prev_m3 is not None and ep_count < prev_m3:
                self.m3_events.append(len(self.m3) - 1)
                self.m3_reasons[len(self.m3) - 1] = reason
            if prev_m4 is not None and fact_count < prev_m4:
                self.m4_events.append(len(self.m4) - 1)
                self.m4_reasons[len(self.m4) - 1] = reason
            prev_m3 = ep_count
            prev_m4 = fact_count
        if frames:
            self.frame = frames[-1]
        self._sync_mechanism_window(frames)
        self._dirty = True

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        if histories_done:
            self._dirty = True
            return
        if not self._mech_frames or self._mech_frames[-1] is not f:
            self._mech_frames.append(f)
            if len(self._mech_frames) > 256:
                self._mech_frames = self._mech_frames[-256:]
        if not self.m3:
            self.cycle_base = int(f.cycle_id)
        prev_m3 = self.m3[-1] if self.m3 else None
        prev_m4 = self.m4[-1] if self.m4 else None
        m3_n = int(getattr(f, "m3_count", 0) or 0) or int(f.episode_count)
        self.m3.append(m3_n); self.m4.append(int(f.fact_count))
        self.rss.append(float(f.rss_bytes)); self.lat.append(float(f.latency_ms))
        self.m3_cap = int(f.m3_cap); self.m4_cap = int(f.m4_cap)
        reason = self._prune_reason(f)
        if prev_m3 is not None and m3_n < prev_m3:
            self.m3_events.append(len(self.m3) - 1)
            self.m3_reasons[len(self.m3) - 1] = reason
        if prev_m4 is not None and int(f.fact_count) < prev_m4:
            self.m4_events.append(len(self.m4) - 1)
            self.m4_reasons[len(self.m4) - 1] = reason
        self._dirty = True

    def _prune_reason(self, f: ObservabilityFrame) -> str:
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
        slope = float(np.polyfit(xs, ys, 1)[0])
        return slope

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        if not self.m3:
            self._empty(p, "Retention…"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review,
            panel_key="retention", multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        leak = self._leak_smooth.value(self._leak_rate())
        env_ok, env_seg = self._envelope_status()
        mech = self._mechanism_line()
        row_y = 14 + y0
        if mech:
            p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
            p.drawText(10, row_y, _elide_line(p, mech, w - 20))
            row_y += 14
        self._title(p, f"Retention & resources   RSS leak-rate ≈ {leak:+.1f} B/cyc", y=row_y)
        gauge_h = 36 if h < 220 else 46
        gauge_y = row_y + 18
        self._envelope_gauge(p, 10, gauge_y, w - 20, gauge_h, env_seg, env_ok)
        caption_y = gauge_y + gauge_h + 6
        self._caption(p, "focal: inside-envelope? M3·M4·RSS·lat vs caps · below: RSS (orange) + latency (green) trend",
                      y=caption_y)
        spark_top = caption_y + 10
        footer_h = 14
        min_spark_h = 48
        spark_bot = min(h - 8, spark_top + min_spark_h + footer_h)
        if spark_bot - spark_top >= min_spark_h:
            self._resources_panel(p, spark_top, spark_bot - footer_h)
            self._footer_caption(p, "shared 0..1 normalised scale · stable bounds",
                                 y=spark_bot - 4)
        else:
            p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
            p.drawText(10, spark_top + 12, "sparkline needs more height — widen window")

    def _envelope_status(self) -> Tuple[bool, List[Tuple[str, float, bool]]]:
        segs = []
        m3 = float(self.m3[-1]) if self.m3 else 0.0
        m4 = float(self.m4[-1]) if self.m4 else 0.0
        rss = float(self.rss[-1]) if self.rss else 0.0
        lat = float(self.lat[-1]) if self.lat else 0.0
        c3 = float(self.m3_cap) if self.m3_cap else 1.0
        c4 = float(self.m4_cap) if self.m4_cap else 1.0
        bounds = getattr(self.frame, "rbta_bounds", {}) or {} if self.frame else {}
        mem_b = max((float(b.get("mem", 0.0)) for b in bounds.values() if isinstance(b, dict) and b.get("mem")), default=0.0)
        time_s = max((float(b.get("time", 0.0)) for b in bounds.values() if isinstance(b, dict) and b.get("time")), default=0.0)
        time_b_ms = time_s * 1000.0 if time_s > 0 else 0.0
        rss_b = mem_b if mem_b > 0 else (max(self.rss) * 1.25 if self.rss else 1.0)
        lat_b = time_b_ms if time_b_ms > 0 else (max(self.lat) * 1.25 if self.lat else 1.0)
        for lbl, val, cap in (("M3", m3, c3), ("M4", m4, c4), ("RSS", rss, rss_b), ("lat", lat, lat_b)):
            r = val / cap if cap > 0 else 0.0
            segs.append((lbl, min(r, 1.5), r > 1.0))
        return all(not s[2] for s in segs), segs

    def _envelope_gauge(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int,
                        segs: List[Tuple[str, float, bool]], ok: bool) -> None:
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(TEXT_COL if ok else QtGui.QColor(231, 76, 60)); p.setFont(_F_LABEL_B)
        p.drawText(x + 6, y + 14, "inside envelope?" + ("" if ok else "  ✗ OVER"))
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
            p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
            p.drawText(sx + 2, y + h - 6, f"{lbl} {ratio*100:.0f}%")

    def _resources_panel(self, p: QtGui.QPainter, top: int, bot: int) -> None:
        w = self.width()
        left = 60
        right = min(w - 10, max(left + 80, w - 180))
        self._title(p, "Resources — RSS (B, orange) + latency (ms, green)", y=top + 12)
        rss = list(self.rss); lat = list(self.lat)
        if not rss and not lat:
            return
        if w >= 420:
            p.setPen(_FOOTER_COL); p.setFont(_F_LABEL)
            rss_txt = f"RSS now={rss[-1]/1e6:.1f}MB" if rss else "RSS —"
            lat_txt = f"lat now={lat[-1]:.1f}ms med={float(np.median(lat)):.1f}ms" if lat else "lat —"
            p.drawText(right - 90, top + 12, rss_txt)
            p.drawText(right - 90, top + 24, lat_txt)
        pt0, pt1 = top + 30, bot - 6
        if pt1 <= pt0 + 4:
            return
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
        n_hist = max(len(rss), len(lat), 1)
        for idx in self.m3_events:
            if idx < 0 or idx >= n_hist:
                continue
            xi = int(left + idx * (right - left) / max(n_hist - 1, 1))
            p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
            p.drawLine(xi, pt0, xi, pt1)
        for idx in self.m4_events:
            if idx < 0 or idx >= n_hist:
                continue
            xi = int(left + idx * (right - left) / max(n_hist - 1, 1))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 2))
            p.drawLine(xi, pt0, xi, pt1)
        if self.m3_events or self.m4_events:
            p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
            p.drawText(left, top + 38, "│ prune: M3 blue · M4 purple")


class ViolationTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["cycle", "module", "bound_type", "measured", "allowed"])
        self.setAlternatingRowColors(True)
        self._seen: int = 0
        self._worst: str = ""
        self._last_resize: int = 0
        self._at_bottom: bool = True

    def add_frame(self, f: ObservabilityFrame) -> None:
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
        now = time.monotonic()
        if self._seen > 0 and self.rowCount() > 0 and (now - self._last_resize) > 0.25:
            self.resizeColumnsToContents()
            self._last_resize = now
        if self._seen > 0 and self.rowCount() > 0 and self._at_bottom:
            self.scrollToBottom()

    def rebuild_from_frames(self, frames: List[ObservabilityFrame]) -> None:
        self.setRowCount(0)
        self._seen = 0
        self._worst = ""
        seen_keys: set = set()
        worst_ratio = 0.0
        for f in frames:
            for v in f.rbta_violations:
                mid = str(v.get("module_id") or v.get("module", ""))
                btype = str(v.get("bound_type", ""))
                key = (int(f.cycle_id), mid, btype)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                row = self.rowCount()
                self.insertRow(row)
                self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(f.cycle_id)))
                self.setItem(row, 1, QtWidgets.QTableWidgetItem(mid))
                self.setItem(row, 2, QtWidgets.QTableWidgetItem(btype))
                self.setItem(row, 3, QtWidgets.QTableWidgetItem(
                    f"{v.get('measured',''):.4f}"
                    if isinstance(v.get("measured"), (int, float))
                    else str(v.get("measured", ""))))
                self.setItem(row, 4, QtWidgets.QTableWidgetItem(
                    f"{v.get('allowed',''):.4f}"
                    if isinstance(v.get("allowed"), (int, float))
                    else str(v.get("allowed", ""))))
                meas = float(v.get("measured", 0.0) or 0.0)
                allowed = float(v.get("allowed", 0.0) or 0.0)
                ratio = (meas / allowed) if allowed > 0 else 0.0
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
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    self._worst = mid
                self._seen += 1
        while self.rowCount() > 200:
            self.removeRow(0)
        if self.rowCount() > 0:
            self.resizeColumnsToContents()

    def violations_seen(self) -> int:
        return self._seen

    def summary(self) -> str:
        if self._seen == 0:
            return "0 violations"
        return f"{self._seen} violations · worst {self._worst}"


class RBTABoundsView(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist: Dict[str, Deque[float]] = {}
        self._smooth: Dict[str, _Smoother] = {}
        self._replay: bool = False
        self._review: bool = False

    def _measured(self, f: ObservabilityFrame, flow: str, btype: str) -> float:
        if btype == "time":
            return float((f.module_timings or {}).get(flow, 0.0))
        if btype == "mem":
            return float((getattr(f, "memory_log", {}) or {}).get(flow, 0.0))
        if btype == "energy":
            return float((getattr(f, "energy_log", {}) or {}).get(flow, 0.0))
        return 0.0

    def _append_bounds_sample(self, f: ObservabilityFrame) -> None:
        bounds = getattr(f, "rbta_bounds", None) or {}
        for mid, bound_set in bounds.items():
            if not isinstance(bound_set, dict):
                continue
            flow = RBTA_TO_FLOW.get(str(mid), str(mid))
            for btype in ("time", "mem", "energy"):
                bv = bound_set.get(btype)
                if bv is None or float(bv) <= 0:
                    continue
                meas = self._measured(f, flow, btype)
                key = f"{mid}:{btype}"
                sm = self._smooth.setdefault(key, _Smoother(0.25))
                meas_s = sm.value(meas)
                self._hist.setdefault(key, deque(maxlen=60)).append(meas_s)

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self._hist.clear()
        self._smooth.clear()
        for frame in frames:
            self._append_bounds_sample(frame)
        if frames:
            self.frame = frames[-1]
        self._dirty = True

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        if not histories_done:
            self._append_bounds_sample(f)
        self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        w, h = self.width(), self.height()
        if f is None:
            self._empty(p, "RBTA bounds…"); return
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review,
            panel_key="rbta", multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        bounds = getattr(f, "rbta_bounds", None) or {}
        btypes = [("time", "B_time (ms)", QtGui.QColor(52, 152, 219)),
                  ("mem", "B_mem (B)", QtGui.QColor(155, 89, 182)),
                  ("energy", "B_energy", QtGui.QColor(230, 126, 34))]
        mods = sorted({mid for mid, b in bounds.items() if isinstance(b, dict)
                       and any(b.get(t) for t, _, _ in btypes)})
        if not mods:
            self._empty(p, "RBTA bound envelope (no bounds)"); return
        self._title(p, f"RBTA bound envelope — retention score R=e^{{-t/S}}  ({len(mods)} mods)",
                    y=12 + y0)
        self._caption(p, "sparkline = retention decay · ▮ measured vs │ bound · red = violation",
                      y=26 + y0)
        col_w = w // max(len(btypes), 1)
        top, bot = 40 + y0, h - 8
        min_row = 24
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
                key = f"{mid}:{bt}"
                hist = self._hist.get(key)
                meas_s = float(hist[-1]) if hist else self._measured(f, flow, bt)
                items.append((mid, flow, meas_s, float(bv), key))
            if not items:
                continue
            order = {k: i for i, k in enumerate(PIPELINE)}
            items.sort(key=lambda r: (order.get(r[1], 999), r[0]))
            n = len(items)
            if bot - top < 48 or n * min_row > bot - top:
                p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
                p.drawText(cx, top + 14, "panel too small — widen window")
                continue
            rowh = max(min_row, (bot - top - 8) / max(n, 1))
            bw_max = max(8, col_w - 170)
            if col_w < 120 or bw_max < 8:
                p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
                p.drawText(cx, top + 14, "panel too small — widen window")
                continue
            for i, (mid, flow, meas, b, key) in enumerate(items):
                y = top + int(i * rowh)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(cx, y + 12, f"{mid[:10]:>10}")
                bound_scale = b * 1000.0 if bt == "time" else b
                ratio = meas / bound_scale if bound_scale > 0 else 0.0
                col = QtGui.QColor(231, 76, 60) if ratio > 1.0 else bcol
                S = max(bound_scale / max(meas, 1e-6), 2.0) * 4.0
                spark_x, spark_w, spark_h = cx + 100, bw_max - 8, 10
                p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
                p.drawRect(spark_x, y + 4, spark_w, spark_h)
                hist = self._hist.get(key)
                _draw_measured_sparkline(
                    p, spark_x + 2, y + 5, spark_w - 4, spark_h - 2,
                    hist if hist is not None else deque(), col,
                    fallback_S=max(bound_scale / max(meas, 1e-6), 2.0) * 4.0)
                bar_y = y + 16
                p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG)
                p.drawRect(cx + 100, bar_y, bw_max, 8)
                fw = int(min(ratio, 1.0) * bw_max)
                p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
                p.fillRect(cx + 100, bar_y, fw, 8, col)
                if ratio > 1.0:
                    over = int(min((ratio - 1.0) * bw_max, bw_max * 0.5))
                    p.fillRect(cx + 100 + bw_max, bar_y, over, 8, QtGui.QColor(231, 76, 60, 200))
                p.setPen(QtGui.QPen(ACCENT, 2))
                p.drawLine(cx + 100 + bw_max, bar_y - 1, cx + 100 + bw_max, bar_y + 9)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                r_now = _retention_score(0.0, S)
                over_t = "  OVER" if ratio > 1.0 else ""
                p.drawText(cx + 100 + bw_max + 4, bar_y + 8, f"R={r_now:.2f}{over_t}")
