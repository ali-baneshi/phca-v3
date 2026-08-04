"""Action Selection tab — candidate scores, ranked list, rollout cloud, ε-strip."""

from __future__ import annotations
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui

from phca.monitoring.action_explain import build_explain_chain
from phca.monitoring.qt_base import *  # noqa: F401, F403
from phca.monitoring.qt_base import (
    _BaseCanvas, _RolloutCloudCache,
    _action_layout,
    _draw_data_contract_banner, _window_session_incomplete,
    _action_status_line, _action_chosen_idx, _action_candidate_label, _action_score_margin,
    _draw_belief_rollout_cloud, _draw_score_proxy_cloud, _draw_rollout_score_legend,
    _draw_mechanism_stacked_bar, _draw_action_decision_card,
    _draw_moment_chips, _draw_moment_ticks, _elide_line, _to_qcolor, _map_pt,
    _CAPTION_COL, _F_CAPTION, _F_AXIS, _F_LABEL_B,
    _ACTION_SPARK_GAP, _ACTION_DECISION_H, _ACTION_MECH_H, _ACTION_EXPLAIN_H,
)


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
        self._score_scale = ScaleState(contract=0.05, head=0.10)
        self._rank_sig: tuple = ()
        self._rank_scores: List[float] = []
        self._rank_chosen: int = -1
        self._rank_names: List[str] = []
        self._rank_pareto: set = set()
        self._replay: bool = False
        self._review: bool = False
        self._last_scores_cycle: int = -1
        self._rollout_cache = _RolloutCloudCache()
        self._moment_series: List[Dict[str, Any]] = []
        self._pred_err_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._margin_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._prev_drive_id: Optional[int] = None
        self._prev_best_score: Optional[float] = None
        self._prefix_len: int = 0
        self._mech_frames: List[ObservabilityFrame] = []

    def set_projection(self, proj: BeliefProjection) -> None:
        self.proj = proj

    def _mechanism_line(self) -> str:
        if not self._mech_frames:
            return ""
        counts = mechanism_histogram(self._mech_frames, window=256)
        pct = mechanism_pct(counts)
        top = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)[:4]
        if not top:
            return ""
        parts = [f"{k} {v}%" for k, v in top]
        return f"mechanism (last {len(self._mech_frames)}): " + " · ".join(parts)

    def _sync_mechanism_window(self, frames: List[ObservabilityFrame]) -> None:
        self._mech_frames = list(frames[-256:])

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        self.eps_hist.clear()
        self.ee_hist.clear()
        self.score_hist.clear()
        self._margin_hist.clear()
        self._pred_err_hist.clear()
        self._rollout_cache.clear()
        self._rank_sig = ()
        self._prev_drive_id = None
        self._prev_best_score = None
        self._moment_series = build_moment_series(frames)
        self._sync_mechanism_window(frames)
        for f in frames:
            r = f.action_rationale or {}
            if r:
                self.eps_hist.append(float(r.get("eps", 0.0)))
                self.ee_hist.append(bool(r.get("explored", False)))
                bs = r.get("best_score")
                if isinstance(bs, (int, float)):
                    self.score_hist.append(float(bs))
            self._pred_err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
            sc = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
            m = _action_score_margin(sc)
            if m is not None:
                self._margin_hist.append(m)
        if frames:
            last = frames[-1]
            if last.candidate_scores:
                sc = [float(x) for x in last.candidate_scores]
                self.last_scores = sc
                self.last_chosen = _action_chosen_idx(last, sc)
                self._last_scores_cycle = int(last.cycle_id)
            if last.continuous_action is not None:
                self.last_continuous = last.continuous_action
            from .cognitive_panels import goal_id_from_frame, _best_score
            gid = goal_id_from_frame(last)
            if gid:
                self._prev_drive_id = gid
            self._prev_best_score = _best_score(last)

        self._score_scale.reset()

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.frame = f
        self._replay = replay
        self._review = review
        r = f.action_rationale or {}
        if not histories_done:
            if r:
                self.eps_hist.append(float(r.get("eps", 0.0)))
                self.ee_hist.append(bool(r.get("explored", False)))
                bs = r.get("best_score")
                if isinstance(bs, (int, float)):
                    self.score_hist.append(float(bs))
            sc = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
            m = _action_score_margin(sc)
            if m is not None:
                self._margin_hist.append(m)
            self._moment_series, self._prev_drive_id, self._prev_best_score = (
                append_cognitive_moment(
                    self._moment_series, f, self._pred_err_hist,
                    prev_drive_id=self._prev_drive_id,
                    prev_best_score=self._prev_best_score,
                    maxlen=TREND_WINDOW))
        if f.candidate_scores:
            self.last_scores = [float(x) for x in f.candidate_scores]
            self.last_chosen = _action_chosen_idx(f, self.last_scores)
            self._last_scores_cycle = int(f.cycle_id)
        elif r.get("explored"):
            self.last_scores = []
            self.last_chosen = -1
        if f.continuous_action is not None:
            self.last_continuous = f.continuous_action
        elif getattr(f, "last_action_vector", None) is not None:
            self.last_continuous = f.last_action_vector
        if not histories_done:
            if not self._mech_frames or self._mech_frames[-1] is not f:
                self._mech_frames.append(f)
                if len(self._mech_frames) > 256:
                    self._mech_frames = self._mech_frames[-256:]
        self._dirty = True

    def _action_context_band(self, p: QtGui.QPainter, f: ObservabilityFrame,
                             x: int, y: int, w: int) -> None:
        r = f.action_rationale or {}
        gid = r.get("goal_id") or getattr(f, "active_drive_id", None)
        lvls = list(getattr(f, "drive_levels", []) or [])
        err = float(getattr(f, "prediction_error", 0.0) or 0.0)
        moment = self._moment_series[-1] if self._moment_series else None
        peu_s = ""
        if moment and moment.get("peu_mean") is not None:
            peu_s = f" PEŪ={moment['peu_mean']:.2f}"
        lvl_s = ""
        if gid and lvls and 0 < int(gid) <= len(lvls):
            lvl_s = f" drive_lvl={lvls[int(gid) - 1]:.2f}"
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        fm = p.fontMetrics()
        line1 = f"ctx: goal={gid}{lvl_s} err={err:.2f}{peu_s}"
        p.drawText(x, y, fm.elidedText(line1, QtCore.Qt.ElideRight, w))
        link = flow_action_link_line(f)
        if link:
            p.drawText(x, y + 12, fm.elidedText(link, QtCore.Qt.ElideRight, w))

    def _draw_action_explain_band(
        self,
        p: QtGui.QPainter,
        f: ObservabilityFrame,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        chain = build_explain_chain(f)
        p.setPen(DIM_COL)
        p.setFont(_F_AXIS)
        fm = p.fontMetrics()
        if not chain:
            r = f.action_rationale or {}
            msg = (
                "explain: (legacy rationale — no decision_reason)"
                if r else "explain: —"
            )
            p.drawText(x, y + 10, fm.elidedText(msg, QtCore.Qt.ElideRight, w))
            return
        line_h = max(12, fm.height())
        for i, line in enumerate(chain[:3]):
            p.drawText(x, y + 10 + i * line_h, fm.elidedText(line, QtCore.Qt.ElideRight, w))

    def _rollout_cloud(self, p: QtGui.QPainter, f: ObservabilityFrame,
                       x0: int, top: int, x1: int, cloud_bot: int,
                       footer_y: int, footer_h: int,
                       is_continuous: bool,
                       *, has_scores: bool = False) -> None:
        rollouts = list(getattr(f, "candidate_rollouts", []) or [])
        scores = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
        chosen_idx = _action_chosen_idx(f, scores)
        label = ("MPC candidate cloud (chosen solid · alts faded)" if is_continuous
                 else "per-action predicted-next (chosen solid · alts faded)")
        varexp_fn = getattr(self.proj, "variance_explained", None)
        varexp = varexp_fn() if callable(varexp_fn) else None
        if varexp:
            label += f"  PCA {varexp:.0f}%"
        if chosen_idx >= 0:
            label += f" · chosen=#{chosen_idx}"
        self._title(p, label, y=top + 10, x=x0)
        chart_top = top + 22
        chart_bot = cloud_bot - 2
        px0, py0, px1, py1 = x0 + 4, chart_top, x1 - 4, chart_bot
        p.setPen(QtGui.QPen(GRID_COL, 1))
        p.drawRect(px0, py0, px1 - px0, py1 - py0)
        drawn = False
        if rollouts and self.proj is not None and self.proj.history:
            p.save()
            try:
                p.setClipRect(px0, py0, px1 - px0, py1 - py0)
                drawn = _draw_belief_rollout_cloud(
                    p, f, self.proj, px0, py0, px1, py1, self._rollout_cache,
                    replay=self._replay, is_continuous=is_continuous)
                tau = f.continuous_action if f.continuous_action is not None else self.last_continuous
                if tau is None and getattr(f, "last_action_vector", None) is not None:
                    tau = f.last_action_vector
                if is_continuous and tau is not None:
                    cur_v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
                    anchor = self.proj.project(cur_v) if cur_v is not None else None
                    if anchor is not None:
                        bounds = self.proj.bounds()
                        tcx, tcy = _map_pt(anchor, bounds, px0, py0, px1, py1)
                    else:
                        tcx, tcy = (px0 + px1) // 2, (py0 + py1) // 2
                    tv = np.asarray(tau, dtype=np.float32).reshape(-1)
                    if tv.size >= 2:
                        sc = min(px1 - px0, py1 - py0) * 0.15
                        p.setPen(QtGui.QPen(ACCENT, 2))
                        p.drawLine(tcx, tcy, int(tcx + float(tv[0]) * sc),
                                   int(tcy - float(tv[1]) * sc))
                        p.setPen(DIM_COL); p.setFont(_F_AXIS)
                        p.drawText(tcx + 4, tcy - 4, "τ")
            finally:
                p.restore()
        elif has_scores and self.proj is not None and self.proj.history:
            p.save()
            try:
                p.setClipRect(px0, py0, px1 - px0, py1 - py0)
                drawn = _draw_score_proxy_cloud(
                    p, f, self.proj, px0, py0, px1, py1, scores, chosen_idx)
            finally:
                p.restore()
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(px0 + 4, py0 + 12, "score proxy cloud · rollouts live-only")
        elif not rollouts:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            r = f.action_rationale or {}
            if has_scores:
                p.drawText(px0 + 4, py0 + 14,
                           "rollouts live-only · scores recorded")
                self._draw_score_bars_fallback(
                    p, f, scores, chosen_idx, px0, py0, px1, py1, is_continuous)
            elif bool(r.get("explored")):
                p.drawText(px0 + 4, py0 + 14,
                           "ε-greedy — sampling τ · rollouts not computed")
            else:
                p.drawText(px0 + 4, py0 + 14,
                           "scores=MISSING · (explore/D5 branch)")
        if drawn:
            if getattr(self.proj, "basis_changed", False):
                p.setPen(QtGui.QColor(241, 196, 15)); p.setFont(_F_LABEL_B)
                p.drawText(px0 + 4, py1 - 4, "⟳ PCA re-fit")
        fy = footer_y + 2
        _draw_rollout_score_legend(p, px0, fy + 2, px1 - px0, 14)
        p.setPen(_CAPTION_COL); p.setFont(_F_CAPTION)
        p.drawText(px0, fy + 18,
                   _elide_line(p, "shared PCA plane with Phase Space · window ≤256",
                               px1 - px0))

    def _draw_score_bars_fallback(
        self, p: QtGui.QPainter, f: ObservabilityFrame,
        scores: List[float], chosen_idx: int,
        x0: int, y0: int, x1: int, y1: int, is_continuous: bool,
    ) -> None:
        n = len(scores)
        if n == 0:
            return
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        smax = max(scores)
        smin = min(scores)
        rng = (smax - smin) or 1.0
        bh = max(4, (y1 - y0 - 20) // max(n, 1))
        bw = x1 - x0 - 16
        for rank, idx in enumerate(order[:min(n, 8)]):
            ry = y0 + 18 + rank * bh
            frac = (scores[idx] - smin) / rng
            col = ACCENT if idx == chosen_idx else QtGui.QColor(52, 152, 219, 200)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(x0 + 8, ry, int(bw * frac), bh - 2, col)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x0 + 8 + int(bw * frac) + 4, ry + bh - 4, f"#{idx} {scores[idx]:.2f}")

    def _draw(self, p: QtGui.QPainter) -> None:
        f = self.frame
        if f is None:
            self._empty(p, "Action selection…"); return
        w, h = self.width(), self.height()
        lay = _action_layout(w, h, replay=self._replay)
        r = f.action_rationale or {}
        scores = [float(x) for x in f.candidate_scores]
        chosen = _action_chosen_idx(f, scores)
        is_continuous = bool(r.get("continuous", f.continuous_action is not None))
        goal_id = r.get("goal_id")
        if is_continuous and goal_id is None:
            goal_lbl = "τ∈[-1,1]²"
        else:
            goal_lbl = DRIVE_NAMES.get(goal_id, goal_id) if goal_id is not None else "—"
        explored = bool(r.get("explored"))
        note = r.get("note", "")
        cr_t = float(getattr(f, "cr_temperature", 0.0) or 0.0)
        pareto = set(int(x) for x in (getattr(f, "pareto_front", []) or []))
        names = list(getattr(f, "action_names", []) or [])
        _draw_data_contract_banner(p, self.width(), replay=self._replay,
                                   review=self._review, panel_key="action",
                                   multi_agent=self._multi_agent,
                                   incomplete=_window_session_incomplete(self))
        from phca.monitoring.cognitive_panels import frame_is_geometry_control
        ctrl = "CONTROL=geometry" if frame_is_geometry_control(f) else "CONTROL=scored"
        hdr = (
            f"{ctrl}  goal={goal_lbl}  "
            f"{'EXPLORE' if explored else 'EXPLOIT'}  ε={r.get('eps',0):.3f}  T={cr_t:.2f}"
        )
        bs = r.get("best_score")
        if isinstance(bs, (int, float)):
            hdr += f"  score={bs:.3f}"
        self._title(p, hdr, y=lay["title_y"])
        moment = self._moment_series[-1] if self._moment_series else None
        status = _action_status_line(
            f, scores, chosen, replay=self._replay, review=self._review,
            prefix_len=self._prefix_len, moment=moment)
        kind = "continuous τ" if is_continuous else "discrete"
        status += f" · {kind} · pareto={len(pareto)}/{len(scores)}"
        if note:
            status += f" · {note}"
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(10, lay["status_y"] + 10,
                   _elide_line(p, status, lay["status_w"]))
        chip_x = _draw_moment_chips(p, 10, lay["chips_y"], moment)
        mech = classify_action_mechanism(r)
        if mech != "prediction":
            col = QtGui.QColor(52, 152, 219) if mech == "explore" else DIM_COL
            lbl = mech[:10]
            tw = 8 + len(lbl) * 6
            p.setPen(QtGui.QPen(col, 1))
            p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), CHIP_FILL_ALPHA))
            p.drawRoundedRect(chip_x + 4, lay["chips_y"], tw, 14, 3, 3)
            p.setPen(col); p.setFont(_F_AXIS)
            p.drawText(chip_x + 8, lay["chips_y"] + 11, lbl)
        sx = lay["spark_x0"]
        sy = lay["spark_y0"]
        sw, sh, sg = lay["spark_w"], lay["spark_h"], _ACTION_SPARK_GAP
        self._best_score_spark(p, sx, sy, sw, sh)
        self._margin_spark(p, sx + sw + sg, sy, sw, sh)
        self._pred_err_spark(p, sx + 2 * (sw + sg), sy, sw, sh)
        _draw_action_decision_card(
            p, f, scores, chosen, moment, pareto,
            lay["left_x"], lay["decision_y"], w - lay["left_x"] - 8, _ACTION_DECISION_H)
        self._action_context_band(p, f, lay["left_x"], lay["ctx_y"] + 10, lay["left_w"])
        self._draw_action_explain_band(
            p, f, lay["left_x"], lay["explain_y"] + 4, lay["left_w"], _ACTION_EXPLAIN_H)
        _draw_mechanism_stacked_bar(
            p, self._mech_frames, lay["left_x"], lay["mech_y"],
            w - lay["left_x"] - 8, _ACTION_MECH_H)
        list_x, list_y = lay["left_x"], lay["left_y"]
        list_w, list_h = lay["left_w"], lay["left_h"]
        rx0, right_w = lay["right_x"], lay["right_w"]
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(list_x - 4, list_y - 4, list_w + 8, list_h + lay["tau_slot_h"] + 10, 6, 6)
        if scores:
            sig = freeze_sig((tuple(names), tuple(round(s, 4) for s in scores), chosen))
            if sig != self._rank_sig:
                self._rank_sig = sig
                self._rank_scores = list(scores)
                self._rank_chosen = chosen
                self._rank_names = list(names)
                self._rank_pareto = set(pareto)
            self._ranked_list(p, self._rank_scores, self._rank_chosen, self._rank_pareto,
                             self._rank_names, is_continuous,
                             list_x, list_y, list_w, list_h)
        else:
            tag = "EXPLORE (ε-greedy random τ)" if explored else (note or "D5 ENERGY → STAY (no candidates)")
            p.setPen(QtGui.QColor(231, 76, 60) if not explored else QtGui.QColor(52, 152, 219))
            p.setFont(_F_LABEL_B)
            p.drawText(list_x, list_y + 20, f"● {tag}")
            p.setFont(_F_AXIS); p.setPen(TEXT_COL)
            if explored:
                p.drawText(list_x, list_y + 42,
                           "ε-greedy — no score table this cycle")
            else:
                p.drawText(list_x, list_y + 42, "(candidate scores are not computed for this branch)")
            if (self.last_scores and self._last_scores_cycle == int(f.cycle_id)
                    and not self._replay and not explored):
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                cn = _action_candidate_label(self.last_chosen, names, is_continuous)
                p.drawText(list_x, list_y + 62, f"last chosen: {cn}")
        tau_vec = getattr(f, "continuous_action", None)
        if tau_vec is None:
            tau_vec = self.last_continuous
        if is_continuous and tau_vec is not None:
            self._action_heatmap(p, tau_vec, list(getattr(f, "dim_names", []) or []),
                                 list_x, lay["tau_slot_y"], list_w - 10, lay["tau_slot_h"])
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(rx0 - 4, lay["cloud_top"] - 4, right_w + 8,
                          lay["body_h"] + 8, 6, 6)
        self._rollout_cloud(
            p, f, rx0, lay["cloud_top"], rx0 + right_w, lay["cloud_bot"],
            lay["footer_y"], lay["footer_h"],
            is_continuous, has_scores=bool(scores))
        self._epsilon_strip(p, rx0, lay["epsilon_y"], right_w, lay["epsilon_h"])

    def _ranked_list(self, p: QtGui.QPainter, scores: List[float], chosen: int,
                     pareto: set, names: List[str], is_continuous: bool,
                     x: int, y: int, w: int, h: int) -> None:
        n = len(scores)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        smax = max(scores)
        smin = min(scores) if scores else 0.0
        lo, hi = self._score_scale.update(smin, smax)
        rng = (hi - lo) or 1.0
        rh = max(14, min(34, h // max(n, 1)))
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        n_par = len(pareto)
        p.drawText(x, y - 4, f"rank · name · score (Δ2nd) · Pareto {n_par}/{n}")
        moment = self._moment_series[-1] if self._moment_series else None
        shift = bool(moment and moment.get("decision_shift"))
        if shift:
            p.setPen(_to_qcolor(MOMENT_COLORS["decision_shift"])); p.setFont(_F_LABEL_B)
            p.drawText(x + w - 44, y - 4, "SHIFT")
        for rank, idx in enumerate(order):
            ry = y + 6 + rank * rh
            if ry + rh > y + h:
                break
            s = scores[idx]
            is_chosen = (idx == chosen)
            is_pareto = idx in pareto
            name = _action_candidate_label(idx, names, is_continuous)
            if is_chosen:
                p.setPen(QtGui.QPen(ACCENT, 2)); p.setBrush(QtGui.QColor(241, 196, 15, CHIP_FILL_ALPHA))
                p.drawRoundedRect(x, ry, w, rh - 3, 4, 4)
            elif is_pareto:
                p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182, 160), 1))
                p.setBrush(QtGui.QColor(155, 89, 182, 60))
                p.drawRoundedRect(x, ry, w, rh - 3, 4, 4)
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, ry + rh - 8, f"#{rank + 1}")
            p.setPen(TEXT_COL)
            p.setFont(_F_LABEL_B if is_chosen else _F_AXIS)
            if is_continuous:
                name = f"τ[{idx}]"
            else:
                name = _action_candidate_label(idx, names, is_continuous)
            p.drawText(x + 34, ry + rh - 8, str(name)[:14])
            bar_x = x + 110; bar_w = w - 110 - 78
            frac = max(0.0, min(1.0, (s - lo) / rng))
            p.setPen(QtGui.QPen(GRID_COL, 1)); p.setBrush(PANEL_BG_ALT)
            p.drawRect(bar_x, ry + 4, bar_w, rh - 12)
            col = ACCENT if is_chosen else QtGui.QColor(52, 152, 219, 220)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            p.fillRect(bar_x, ry + 4, int(bar_w * frac), rh - 12, col)
            p.setPen(TEXT_COL); p.setFont(_F_AXIS)
            txt = f"{s:.3f}"
            if rank == 0 and n > 1:
                txt += f"  Δ{(s - scores[order[1]]):+.3f}"
            p.drawText(bar_x + bar_w + 6, ry + rh - 8, txt)
            if is_chosen:
                p.setPen(ACCENT); p.setFont(_F_LABEL_B)
                p.drawText(x + w - 14, ry + rh - 8, "►")
            dim_names = list(getattr(self.frame, "dim_names", []) or []) if self.frame else []
            if is_chosen and is_continuous and idx < len(dim_names) and dim_names[idx]:
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(x + w - 96, ry + rh - 8, f"{str(dim_names[idx])[:10]}")
            elif is_pareto:
                p.setPen(QtGui.QColor(155, 89, 182)); p.setFont(_F_AXIS)
                p.drawEllipse(x + w - 22, ry + 4, 10, 10)

    def _action_heatmap(self, p: QtGui.QPainter, vec: np.ndarray,
                        dim_names: List[str], x: int, y: int, w: int, h: int) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = v.size
        if n == 0:
            return
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y, "chosen action τ (signed, per-dim)")
        gy = y + 14
        chart_h = max(h - 18, 12)
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, gy, w, chart_h, 4, 4)
        bw = (w - 8) / n
        mid = gy + chart_h // 2
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x + 4, mid, x + w - 4, mid)
        for i in range(n):
            val = float(np.clip(v[i], -1, 1))
            bh = int(abs(val) * (chart_h // 2 - 4))
            bx = int(x + 4 + i * bw)
            col = QtGui.QColor(155, 89, 182, 220) if val >= 0 else QtGui.QColor(231, 76, 60, 220)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(col)
            if val >= 0:
                p.fillRect(bx + 1, mid - bh, max(int(bw) - 2, 1), bh, col)
            else:
                p.fillRect(bx + 1, mid, max(int(bw) - 2, 1), bh, col)
        step = max(1, n // 12)
        label_y = min(gy + chart_h - 2, y + h - 4)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        for i in range(0, n, step):
            lbl = dim_names[i] if i < len(dim_names) else str(i)
            p.drawText(int(x + 4 + i * bw), label_y, lbl[:6])

    def _best_score_spark(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
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
        ms = self._moment_series[-n:] if self._moment_series else []
        _draw_moment_ticks(p, x + 4, y + 10, x + w - 4, y + h - 2, ms)

    def _margin_spark(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 10, "Δ2nd margin")
        vals = list(self._margin_hist)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.drawText(x + 4, y + h - 4, "—"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        n = len(vals)
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + 4 + i * (w - 8) / max(n - 1, 1)
            py = (y + h - 3) - (v - lo) / (hi - lo) * (h - 16)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.drawPath(path)

    def _pred_err_spark(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 10, "pred err")
        vals = list(self._pred_err_hist)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.drawText(x + 4, y + h - 4, "—"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        n = len(vals)
        p.setPen(QtGui.QPen(QtGui.QColor(231, 76, 60), 2))
        path = QtGui.QPainterPath()
        for i, v in enumerate(vals):
            px = x + 4 + i * (w - 8) / max(n - 1, 1)
            py = (y + h - 3) - (v - lo) / (hi - lo) * (h - 16)
            (path.moveTo if i == 0 else path.lineTo)(px, py)
        p.drawPath(path)
        ms = self._moment_series[-n:] if self._moment_series else []
        _draw_moment_ticks(p, x + 4, y + 10, x + w - 4, y + h - 2, ms)

    def _epsilon_strip(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int) -> None:
        if not self.eps_hist:
            return
        n = len(self.eps_hist)
        p.setPen(QtGui.QPen(PANEL_BORDER, 1)); p.setBrush(PANEL_BG_ALT)
        p.drawRoundedRect(x, y, w, h, 4, 4)
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x + 4, y + 10, "ε-decay (blue) · explore(red)/exploit(green)")
        base_y = y + h - 8
        chart_top = y + 14
        for gy in range(chart_top, base_y, 6):
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 60), 1))
            p.drawLine(x + 4, gy, x + w - 4, gy)
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219), 2))
        for i in range(1, n):
            x0 = x + (i - 1) * w / max(n - 1, 1)
            x1 = x + i * w / max(n - 1, 1)
            p.drawLine(int(x0), int(base_y - self.eps_hist[i - 1] * (h - 20)),
                       int(x1), int(base_y - self.eps_hist[i] * (h - 20)))
        for i, e in enumerate(self.ee_hist):
            xx = int(x + 4 + i * (w - 8) / max(n - 1, 1))
            c = QtGui.QColor(231, 76, 60) if e else QtGui.QColor(46, 204, 113)
            p.setBrush(c); p.setPen(QtGui.QPen(c.darker(120), 1))
            p.drawEllipse(xx - 3, base_y + 2, 6, 6)
        ms = self._moment_series[-n:] if self._moment_series else []
        _draw_moment_ticks(p, x + 4, chart_top, x + w - 4, base_y, ms)
