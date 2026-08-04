"""Overview tab — one unified agent card, portrait, world, drives, trends, attention, status."""

from __future__ import annotations
from dataclasses import replace
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# Shared base, constants, drawing helpers, and all re-exports.
from phca.monitoring.qt_base import *  # noqa: F401, F403

# Explicit underscore imports needed by overview view classes below.
from phca.monitoring.qt_base import (
    _BaseCanvas, _ChartCanvas,
    _OVERVIEW_HZ, _OVERVIEW_HEADER_H, _OVERVIEW_PHASE_H,
    _OVERVIEW_NARRATIVE_H, _OVERVIEW_EVENT_LOG_H, _OVERVIEW_RIBBON_H,
    _OVERVIEW_MARGIN, _OVERVIEW_EVENT_HOLD, _REACHER_TRAIL_MAX, TREND_WINDOW,
    _Smoother, _AUTOSCALE_FROZEN,
    _F_AXIS, _F_LABEL, _F_LABEL_B,
    _draw_data_contract_banner, _window_session_incomplete,
    _draw_overview_header, _draw_overview_phase_strip, _draw_session_results_panel,
    _draw_overview_narrative, _draw_overview_event_log,
    _draw_agent_glyph, _draw_agent_limbs, _draw_overview_body, _draw_vitals_ribbon,
    _render_reacher_schematic_pixmap, _apply_overview_camera_overlay,
    _normalize_rgb_frame, _prediction_heatmap, _draw_qimage, _draw_tau_bar,
    _vec_pca2, _arrow, _map_pt, _dim_arrow_label,
    _drive_color, _drive_short, _drive_name,
    _overview_moment_flags, _overview_goal_id, _overview_new_events,
    _reacher_kinematics_from_obs,
)


class CameraCaptureTimer(QtCore.QObject):
    """Capture MuJoCo RGB on the main thread outside paintEvent (~4 Hz)."""

    def __init__(self, view: "OverviewAgentView", capture_hz: float = _OVERVIEW_HZ,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent or view)
        self._view = view
        self._tabs: Optional[QtWidgets.QTabWidget] = None
        self._overview_tab_index: int = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000.0 / max(capture_hz, 0.5)))
        self._timer.timeout.connect(self._on_tick)

    def bind_tabs(self, tabs: QtWidgets.QTabWidget, overview_tab_index: int = 0) -> None:
        self._tabs = tabs
        self._overview_tab_index = overview_tab_index

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _on_tick(self) -> None:
        v = self._view
        if v._camera_mode == "schematic":
            return
        if v._camera_provider is None:
            return
        if self._tabs is not None and self._tabs.currentIndex() != self._overview_tab_index:
            return
        before = (
            id(v._camera_pixmap),
            bool(v._camera_stale),
            v._camera_cycle_id,
            int(v._camera_fail_count),
        )
        frame = v._sync_camera_from_provider()
        after = (
            id(v._camera_pixmap),
            bool(v._camera_stale),
            v._camera_cycle_id,
            int(v._camera_fail_count),
        )
        if frame is not None or after != before:
            v.mark_dirty()


class OverviewAgentView(_BaseCanvas):
    """v8.1 Overview: one unified agent card — mind glyph | body camera + vitals ribbon."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # QLabel camera avoids QPainter+WA_OpaquePaintEvent green GPU garbage.
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, False)
        self._camera_label = QtWidgets.QLabel(self)
        self._camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self._camera_label.setStyleSheet("background-color: rgb(12, 12, 16);")
        self._camera_label.setScaledContents(True)
        self.frame: Optional[ObservabilityFrame] = None
        self.cycle_error: Optional[str] = None
        self.proj: Optional["BeliefProjection"] = None
        self.trail: Deque[Any] = deque(maxlen=160)
        self.arena_trail: Deque[Tuple[float, float]] = deque(maxlen=80)
        self._reacher_trail: Deque[Tuple[float, float]] = deque(maxlen=_REACHER_TRAIL_MAX)
        self._ax = ScaleState(contract=0.04, head=0.08)
        self._ay = ScaleState(contract=0.04, head=0.08)
        self._err_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._conf_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._glyph_hist: Deque[Tuple[List[float], float, int]] = deque(maxlen=3)
        self._glyph_sig: tuple = ()
        self._t0 = time.monotonic()
        self._emp_sm = _Smoother(alpha=0.25)
        self._tau_sms: List[_Smoother] = [_Smoother(alpha=0.35), _Smoother(alpha=0.35)]
        self._dist_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._prev_best_score: Optional[float] = None
        self._last_paint_t: float = 0.0
        self._camera_pixmap: Optional[QtGui.QPixmap] = None
        self._camera_numpy: Optional[np.ndarray] = None
        self._camera_stale: bool = False
        self._camera_cycle_id: Optional[int] = None
        self._camera_fail_count: int = 0
        self._camera_provider: Optional[Callable[[], Any]] = None
        self._camera_debug: bool = False
        self._camera_glitch_profile: str = "default"
        self._camera_mode: str = "auto"  # auto | live | schematic
        self._camera_gl_disabled: bool = False
        self._camera_gl_permanent_disabled: bool = False
        self._camera_status: str = "unavailable"  # live|recovering|schematic_fallback|permanent|unavailable
        self._camera_probe_countdown: int = 0
        self._camera_probe_every: int = 18
        self._camera_diag_limit: int = 8
        self._camera_diag_samples: List[Dict[str, Any]] = []
        self._camera_last_reason: str = "not_started"
        self._capture_timer = CameraCaptureTimer(self)
        self._last_active_drive: Optional[int] = None
        self._drive_change: Optional[Tuple[int, int]] = None
        self._prev_explored: Optional[bool] = None
        self._event_lines: List[Tuple[str, int]] = []
        self._moment_series: List[Dict[str, Any]] = []
        self._prev_drive_id: Optional[int] = None
        self._replay: bool = False
        self._review: bool = False
        self._session_results: Optional[List[str]] = None
        self._review_mode: bool = False

    def set_session_results(self, lines: Optional[List[str]]) -> None:
        self._session_results = list(lines) if lines else None
        self._review_mode = bool(lines)
        self._dirty = True

    def set_review_mode(self, on: bool) -> None:
        self._review_mode = bool(on)
        self._dirty = True

    def _current_moment(self, f: ObservabilityFrame) -> Dict[str, Any]:
        if self._moment_series:
            return self._moment_series[-1]
        return _overview_moment_flags(f, self._err_hist,
                                      prev_drive_id=self._prev_drive_id,
                                      prev_best_score=self._prev_best_score)

    def _update_event_log(self, f: ObservabilityFrame,
                          moment: Optional[Dict[str, Any]] = None) -> None:
        """Hold recent overview events for several heartbeats."""
        self._event_lines = [(t, h - 1) for t, h in self._event_lines if h > 1]
        flags = moment if moment is not None else self._current_moment(f)
        flags = _overview_moment_flags(f, self._err_hist, moment=flags)
        explored = bool(flags.get("explored", False))
        explore_entered = explored and not bool(self._prev_explored)
        self._prev_explored = explored
        dc = flags.get("drive_change")
        if isinstance(dc, (list, tuple)) and len(dc) == 2:
            drive_change: Optional[Tuple[int, int]] = (int(dc[0]), int(dc[1]))
        else:
            drive_change = self._drive_change
        for text in _overview_new_events(
                f, flags, drive_change, explore_entered=explore_entered):
            self._event_lines.append((text, _OVERVIEW_EVENT_HOLD))

    def visible_event_lines(self) -> List[str]:
        return [t for t, _ in self._event_lines]

    def _track_drive_change(self, f: ObservabilityFrame) -> None:
        """Record MDIM goal switch for header chip (live frames only)."""
        act = _overview_goal_id(f)
        self._drive_change = None
        if act and self._last_active_drive is not None and act != self._last_active_drive:
            self._drive_change = (self._last_active_drive, act)
        if act:
            self._last_active_drive = act

    def _sync_drive_change_from_moment(self, moment: Dict[str, Any]) -> None:
        dc = moment.get("drive_change")
        if isinstance(dc, (list, tuple)) and len(dc) == 2:
            self._drive_change = (int(dc[0]), int(dc[1]))
        else:
            self._drive_change = None

    def bind_camera_tabs(self, tabs: QtWidgets.QTabWidget,
                         overview_tab_index: int = 0) -> None:
        self._capture_timer.bind_tabs(tabs, overview_tab_index)

    def start_camera_capture(self) -> None:
        if self._camera_mode != "schematic":
            self._capture_timer.start()

    def stop_camera_capture(self) -> None:
        self._capture_timer.stop()

    def set_camera_provider(self, provider: Optional[Callable[[], Any]],
                            *, debug: bool = False,
                            mode: str = "auto",
                            glitch_profile: str = "default") -> None:
        self._camera_provider = provider
        self._camera_debug = bool(debug)
        self._camera_glitch_profile = (glitch_profile or "default").lower()
        self._camera_mode = (mode or "auto").lower()
        if self._camera_mode == "schematic":
            self._camera_gl_disabled = True
            self._camera_status = "schematic_fallback"
            self._capture_timer.stop()
        else:
            self._camera_status = "recovering"
        self._dirty = True
        self._update_camera_label()
        self.update()

    def _fail_threshold(self) -> int:
        return 3

    def _maybe_disable_gl(self) -> None:
        if self._camera_fail_count >= self._fail_threshold():
            if self._camera_fail_count >= 9 and not self._camera_gl_permanent_disabled:
                self._camera_gl_permanent_disabled = True
                self._camera_status = "permanent"
                self._capture_timer.stop()
                if self._camera_debug:
                    import sys
                    print(f"[camera] permanently disabled after {self._camera_fail_count} failures",
                          file=sys.stderr)
                return
            self._camera_gl_disabled = True
            self._camera_status = "schematic_fallback"
            self._camera_probe_countdown = self._camera_probe_every
            self._capture_timer.stop()
            if self._camera_debug:
                import sys
                print(f"[camera] GL disabled; using 2D schematic fallback "
                      f"(reason={self._camera_last_reason})",
                      file=sys.stderr)

    def set_projection(self, proj: "BeliefProjection") -> None:
        self.proj = proj

    def _append_trails_from_frame(self, f: ObservabilityFrame) -> None:
        """Update motion trails and glyph history from one frame."""
        if f.grid is not None and f.agent_pos is not None:
            ap = tuple(f.agent_pos)
            if not self.trail or self.trail[-1] != ap:
                self.trail.append(ap)
        kind = (getattr(f, "env_kind", "") or "").lower()
        sd = int(getattr(f, "state_dim", 0) or 0)
        if kind == "continuous" and 2 <= sd <= 4 and f.obs_vector is not None:
            v = np.asarray(f.obs_vector, dtype=np.float32).reshape(-1)
            if v.size >= 2:
                pt = (float(v[0]), float(v[1]))
                if not self.arena_trail or self.arena_trail[-1] != pt:
                    self.arena_trail.append(pt)
        if self._reacher_obs_ok(f):
            v = f.obs_vector if f.obs_vector is not None else f.sanitized_state
            kin = _reacher_kinematics_from_obs(v)
            if kin is not None:
                pt = (kin["fx"], kin["fy"])
                if not self._reacher_trail or self._reacher_trail[-1] != pt:
                    self._reacher_trail.append(pt)
                self._dist_hist.append(float(kin["dist"]))
        elif f.agent_pos is not None and f.goal_pos is not None:
            ap, gp = f.agent_pos, f.goal_pos
            if len(ap) >= 2 and len(gp) >= 2:
                self._dist_hist.append(
                    float(abs(int(ap[0]) - int(gp[0])) + abs(int(ap[1]) - int(gp[1])))
                )
        levels = list(getattr(f, "drive_levels", []) or [])
        active = int(_overview_goal_id(f) or 0)
        sig = freeze_sig((f.cycle_id, tuple(round(x, 3) for x in levels), active))
        if sig != self._glyph_sig and levels:
            from phca.monitoring.cognitive_panels import frame_display_confidence
            self._glyph_sig = sig
            self._glyph_hist.append(([float(x) for x in levels],
                                     float(frame_display_confidence(f)),
                                     active))

    def rebuild_histories(self, frames: List[ObservabilityFrame]) -> None:
        """Rebuild sparklines/trails from a rolling window (scrub/live sync)."""
        from phca.monitoring.cognitive_panels import frame_display_confidence
        self._err_hist = deque(
            (float(f.prediction_error) for f in frames), maxlen=TREND_WINDOW)
        self._conf_hist = deque(
            (frame_display_confidence(f) for f in frames), maxlen=TREND_WINDOW)
        self.trail.clear()
        self.arena_trail.clear()
        self._reacher_trail.clear()
        self._glyph_hist.clear()
        self._glyph_sig = ()
        self._emp_sm.reset()
        self._dist_hist.clear()
        self._prev_best_score = None
        self._prev_explored = None
        self._event_lines = []
        self._moment_series = build_moment_series(frames)
        self._tau_sms = [_Smoother(alpha=0.35), _Smoother(alpha=0.35)]
        self._ax.reset()
        self._ay.reset()
        for f in frames:
            self._emp_sm.value(float(getattr(f, "empowerment", 0.0) or 0.0))
            self._append_trails_from_frame(f)
        if frames:
            from .cognitive_panels import goal_id_from_frame, _best_score
            gid = goal_id_from_frame(frames[-1])
            if gid:
                self._last_active_drive = gid
                self._prev_drive_id = gid
            self._prev_best_score = _best_score(frames[-1])
        self._drive_change = None

    def set_state(self, f: Optional[ObservabilityFrame],
                  err: Optional[str] = None) -> None:
        self.frame = f
        self.cycle_error = err
        self._dirty = True

    def set_frame(self, f: ObservabilityFrame, *, histories_done: bool = False,
                  replay: bool = False, review: bool = False) -> None:
        self.set_state(f, None)
        self._replay = replay
        if review:
            self._review_mode = True
        if not histories_done:
            from phca.monitoring.cognitive_panels import frame_display_confidence
            self._conf_hist.append(float(frame_display_confidence(f)))
            self._emp_sm.value(float(getattr(f, "empowerment", 0.0) or 0.0))
            self._append_trails_from_frame(f)
            self._moment_series, self._prev_drive_id, self._prev_best_score = (
                append_cognitive_moment(
                    self._moment_series, f, self._err_hist,
                    prev_drive_id=self._prev_drive_id,
                    prev_best_score=self._prev_best_score,
                    maxlen=TREND_WINDOW))
            moment = self._moment_series[-1] if self._moment_series else {}
            self._sync_drive_change_from_moment(moment)
            gid = _overview_goal_id(f)
            if gid:
                self._last_active_drive = gid
            self._update_event_log(f, moment)
        self._update_camera_label()

    def _glyph_display_frame(self, f: ObservabilityFrame) -> ObservabilityFrame:
        """Apply UI-only smoothing for continuous action rendering."""
        ca = getattr(f, "continuous_action", None)
        kind = (getattr(f, "env_kind", "") or "").lower()
        if ca is None or kind != "mujoco_rgb":
            return f
        vec = np.asarray(ca, dtype=np.float32).reshape(-1)
        if vec.size != 2:
            return f
        sm = np.array([
            self._tau_sms[0].value(float(vec[0])),
            self._tau_sms[1].value(float(vec[1])),
        ], dtype=np.float32)
        out = replace(f, continuous_action=sm)
        return out

    def _camera_badge(self, schematic: bool) -> str:
        f = self.frame
        if schematic:
            if self._camera_gl_disabled and self._camera_mode == "live":
                if self._camera_status == "recovering":
                    return "RECOVERING"
                if self._camera_status == "unavailable":
                    return "UNAVAILABLE"
                return "2D fallback"
            return "2D"
        if self._camera_stale:
            return "STALE"
        if (
            f is not None
            and self._camera_cycle_id is not None
            and int(self._camera_cycle_id) != int(f.cycle_id)
        ):
            return "SYNC?"
        return "LIVE"

    def _parse_camera_provider_result(
            self, raw: Any) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if raw is None:
            return None, None
        if isinstance(raw, dict):
            return _normalize_rgb_frame(raw.get("frame")), raw.get("cycle_id")
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return _normalize_rgb_frame(raw[0]), raw[1]
        return _normalize_rgb_frame(raw), None

    def camera_label_rect(self) -> QtCore.QRect:
        """Camera QLabel geometry within the overview body panel."""
        body = self.main_body_rect()
        tau_h = min(36, max(28, body.height() // 6))
        cam_h = max(8, body.height() - tau_h - 4)
        return QtCore.QRect(body.x() + 4, body.y() + 2, body.width() - 8, cam_h)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout_camera_label()

    def _layout_camera_label(self) -> None:
        self._camera_label.setGeometry(self.camera_label_rect())
        self._camera_label.raise_()

    def _update_camera_label(self) -> None:
        """Push camera/schematic pixmap to QLabel (never in paintEvent)."""
        r = self.camera_label_rect()
        w, h = max(r.width(), 2), max(r.height(), 2)
        f = self.frame
        if f is not None and f.grid is not None:
            self._camera_label.clear()
            self._camera_label.hide()
            return
        use_schematic = (
            self._camera_mode == "schematic"
            or self._camera_gl_disabled
            or (f is not None and self._use_reacher_schematic(f))
        )
        if use_schematic and f is not None and self._reacher_obs_ok(f):
            pm = _render_reacher_schematic_pixmap(f, w, h, trail=self._reacher_trail)
            badge = self._camera_badge(schematic=True)
            pm = _apply_overview_camera_overlay(
                pm, f, badge=badge, camera_cycle_id=self._camera_cycle_id,
                stale=self._camera_stale, schematic=True)
            self._camera_label.setPixmap(pm)
            self._camera_label.show()
            return
        if (
            self._camera_pixmap is not None
            and not self._camera_pixmap.isNull()
            and self._camera_numpy is not None
            and not is_glitchy_rgb_frame(self._camera_numpy)
        ):
            scaled, sw, sh = fit_pixmap_to_box(self._camera_pixmap, w, h)
            base = scaled if sw >= 1 and sh >= 1 else self._camera_pixmap
            badge = self._camera_badge(schematic=False)
            pm = _apply_overview_camera_overlay(
                base, f, badge=badge, camera_cycle_id=self._camera_cycle_id,
                stale=self._camera_stale, schematic=False)
            self._camera_label.setPixmap(pm)
            self._camera_label.show()
            return
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtGui.QColor(12, 12, 16))
        p = QtGui.QPainter(pm)
        p.setPen(DIM_COL)
        p.setFont(_F_LABEL)
        p.drawText(pm.rect(), QtCore.Qt.AlignCenter, "Camera unavailable")
        p.end()
        self._camera_label.setPixmap(pm)
        self._camera_label.show()

    def _live_camera_frame(self, *, allow_disabled_probe: bool = False) -> Optional[np.ndarray]:
        if (self._camera_gl_disabled and not allow_disabled_probe) or self._camera_mode == "schematic":
            return None
        if self._camera_provider is None:
            return None
        try:
            raw = self._camera_provider()
        except Exception:
            return None
        arr, cid = self._parse_camera_provider_result(raw)
        self._camera_cycle_id = int(cid) if cid is not None else None
        if arr is None:
            self._camera_last_reason = "provider_none"
            return None
        glitchy = is_glitchy_rgb_frame(arr, profile=self._camera_glitch_profile)
        stats = camera_frame_stats(arr)
        stats["cycle_id"] = self._camera_cycle_id
        stats["reason"] = "glitchy_rgb" if glitchy else "ok"
        if len(self._camera_diag_samples) < self._camera_diag_limit:
            self._camera_diag_samples.append(stats)
        if self._camera_debug:
            import sys
            print(
                f"[camera] shape={stats['shape']} std={stats['std']:.1f} "
                f"mean={stats['mean_rgb']} green_frac={stats['green_frac']:.2f} "
                f"glitchy={glitchy} reason={stats['reason']}",
                file=sys.stderr,
            )
        return arr

    def _save_camera_debug(self, frame: np.ndarray, pm: QtGui.QPixmap,
                           *, reason: str = "ok") -> None:
        try:
            from PIL import Image
            Image.fromarray(frame).save(f"/tmp/phca_obs_numpy_{reason}.png")
        except Exception:
            pass
        try:
            pm.toImage().save(f"/tmp/phca_obs_pixmap_{reason}.png")
        except Exception:
            pass

    def _sync_camera_from_provider(self) -> Optional[np.ndarray]:
        if self._camera_gl_permanent_disabled:
            return self._camera_numpy
        allow_probe = False
        if self._camera_gl_disabled and self._camera_mode == "live":
            if self._camera_probe_countdown > 0:
                self._camera_probe_countdown -= 1
                self._camera_status = "recovering"
                self._update_camera_label()
                return self._camera_numpy
            allow_probe = True
        frame = self._live_camera_frame(allow_disabled_probe=allow_probe)
        if frame is not None and is_glitchy_rgb_frame(frame, profile=self._camera_glitch_profile):
            self._camera_fail_count += 1
            self._camera_last_reason = "glitchy_rgb"
            self._camera_stale = False
            self._camera_pixmap = None
            self._camera_numpy = None
            self._maybe_disable_gl()
            self._camera_status = "recovering" if self._camera_gl_disabled else "unavailable"
            self._update_camera_label()
            return None
        if frame is not None and not is_glitchy_rgb_frame(frame, profile=self._camera_glitch_profile):
            pm = rgb_frame_to_pixmap(frame, already_checked=True)
            if not pm.isNull() and not is_glitchy_pixmap(pm):
                self._camera_fail_count = 0
                self._camera_numpy = frame
                self._camera_pixmap = pm
                self._camera_stale = False
                self._camera_gl_disabled = False
                self._camera_probe_countdown = self._camera_probe_every
                self._camera_status = "live"
                self._camera_last_reason = "ok"
                if self._camera_debug:
                    self._save_camera_debug(frame, pm, reason="ok")
                self._update_camera_label()
                return frame
            self._camera_fail_count += 1
            self._camera_last_reason = "pixmap_invalid"
            self._camera_stale = False
            self._camera_pixmap = None
            self._camera_numpy = None
            self._maybe_disable_gl()
            self._camera_status = "recovering" if self._camera_gl_disabled else "unavailable"
            if self._camera_debug:
                self._save_camera_debug(frame, pm, reason="pixmap_invalid")
            self._update_camera_label()
            return None
        self._camera_fail_count += 1
        self._camera_last_reason = "provider_none"
        self._maybe_disable_gl()
        if self._camera_gl_disabled:
            self._camera_status = "recovering"
            self._camera_probe_countdown = self._camera_probe_every
        if self._camera_pixmap is not None and self._camera_numpy is not None:
            self._camera_stale = True
            if self._camera_status == "unavailable":
                self._camera_status = "recovering"
        else:
            self._camera_status = "recovering" if self._camera_gl_disabled else "unavailable"
            self._camera_stale = False
            self._camera_pixmap = None
            self._camera_numpy = None
        self._update_camera_label()
        return self._camera_numpy

    def _reacher_obs_ok(self, f: ObservabilityFrame) -> bool:
        kind = (getattr(f, "env_kind", "") or "").lower()
        if kind != "mujoco_rgb":
            return False
        v = f.obs_vector if f.obs_vector is not None else f.sanitized_state
        return v is not None and np.asarray(v).size >= 4

    def _use_reacher_schematic(self, f: ObservabilityFrame) -> bool:
        if not self._reacher_obs_ok(f):
            return False
        if self._camera_mode == "schematic" or self._camera_gl_disabled:
            return True
        if self._camera_mode == "live":
            return (
                self._camera_numpy is None
                or self._camera_pixmap is None
                or self._camera_pixmap.isNull()
                or is_glitchy_rgb_frame(
                    self._camera_numpy, profile=self._camera_glitch_profile)
            )
        if self._camera_fail_count < 1:
            return False
        return True

    def repaint_if_dirty(self) -> None:
        if not self._dirty:
            return
        now = time.monotonic()
        if now - self._last_paint_t < 1.0 / _OVERVIEW_HZ:
            return
        self._dirty = False
        self._last_paint_t = now
        self.update()

    def _overview_top_h(self) -> int:
        return (_OVERVIEW_HEADER_H + _OVERVIEW_PHASE_H + _OVERVIEW_NARRATIVE_H
                + _OVERVIEW_EVENT_LOG_H)

    def main_body_rect(self) -> QtCore.QRect:
        """Expose main body rect for layout regression tests."""
        w, h = self.width(), self.height()
        m = _OVERVIEW_MARGIN
        top = m + self._overview_top_h()
        main_h = h - top - _OVERVIEW_RIBBON_H - m * 2
        mind_w = int((w - m * 3) * 0.38)
        return QtCore.QRect(m * 2 + mind_w, top, w - mind_w - m * 3, main_h)

    def _draw(self, p: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        if self.frame is None and self.cycle_error is None:
            self._empty(p, "Overview (collecting…)"); return
        if self.cycle_error and self.frame is None:
            p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(QtGui.QFont("Monospace", 10))
            p.drawText(self.rect(), 0x84, f"CYCLE ERROR:\n{self.cycle_error}")
            return
        f = self.frame
        m = _OVERVIEW_MARGIN
        top_h = self._overview_top_h()
        p.setPen(QtGui.QPen(GRID_COL, 1))
        p.setBrush(QtGui.QColor(22, 22, 28))
        p.drawRoundedRect(2, 2, w - 4, h - 4, 6, 6)
        top = m + top_h
        main_h = h - top - _OVERVIEW_RIBBON_H - m * 2
        mind_w = int((w - m * 3) * 0.38)
        body_w = w - mind_w - m * 3
        header_rect = QtCore.QRect(m, m, w - 2 * m, _OVERVIEW_HEADER_H)
        phase_rect = QtCore.QRect(m, m + _OVERVIEW_HEADER_H, w - 2 * m, _OVERVIEW_PHASE_H)
        narrative_rect = QtCore.QRect(
            m, m + _OVERVIEW_HEADER_H + _OVERVIEW_PHASE_H, w - 2 * m,
            _OVERVIEW_NARRATIVE_H)
        event_rect = QtCore.QRect(
            m, m + _OVERVIEW_HEADER_H + _OVERVIEW_PHASE_H + _OVERVIEW_NARRATIVE_H,
            w - 2 * m, _OVERVIEW_EVENT_LOG_H)
        mind_rect = QtCore.QRect(m, top, mind_w, main_h)
        body_rect = QtCore.QRect(m * 2 + mind_w, top, body_w, main_h)
        ribbon_rect = QtCore.QRect(m, h - _OVERVIEW_RIBBON_H - m, w - 2 * m, _OVERVIEW_RIBBON_H)
        y0 = _draw_data_contract_banner(
            p, w, replay=self._replay, review=self._review_mode, panel_key="overview",
            multi_agent=self._multi_agent,
            incomplete=_window_session_incomplete(self))
        if f is not None:
            flags = self._current_moment(f)
            flags = _overview_moment_flags(f, self._err_hist, moment=flags)
            _draw_overview_header(p, f, QtCore.QRect(
                header_rect.x(), header_rect.y() + y0,
                header_rect.width(), header_rect.height()), self._drive_change, flags,
                session_complete=self._review_mode)
            _draw_overview_phase_strip(
                p, f, QtCore.QRect(
                    phase_rect.x(), phase_rect.y() + y0,
                    phase_rect.width(), phase_rect.height()),
                learn_burst=bool(flags.get("learn_burst")))
            results_rect = QtCore.QRect(
                narrative_rect.x(), narrative_rect.y() + y0,
                narrative_rect.width(),
                narrative_rect.height() + event_rect.height())
            if self._session_results:
                _draw_session_results_panel(p, self._session_results, results_rect)
            else:
                _draw_overview_narrative(p, f, QtCore.QRect(
                    narrative_rect.x(), narrative_rect.y() + y0,
                    narrative_rect.width(), narrative_rect.height()),
                    self._err_hist, self._dist_hist)
                _draw_overview_event_log(p, self.visible_event_lines(), QtCore.QRect(
                    event_rect.x(), event_rect.y() + y0,
                    event_rect.width(), event_rect.height()))
            p.setPen(QtGui.QPen(GRID_COL, 1))
            p.drawLine(mind_rect.right(), mind_rect.y(), mind_rect.right(), mind_rect.bottom())
            cx = mind_rect.x() + mind_rect.width() // 2
            cy = mind_rect.y() + mind_rect.height() // 2 - 16
            R = min(mind_rect.width() * 0.35, mind_rect.height() * 0.40)
            import math as _m
            pulse = 0.5
            if not _AUTOSCALE_FROZEN:
                pulse = 0.5 + 0.5 * _m.sin(2 * _m.pi * (time.monotonic() - self._t0) * 0.8)
            p.save()
            try:
                p.setClipRect(mind_rect)
                gf = self._glyph_display_frame(f)
                _draw_agent_glyph(p, gf, cx, cy, int(R), self._glyph_hist, self._t0,
                                  self._emp_sm._v, active_pulse=pulse,
                                  show_head_label=True)
                p.setPen(DIM_COL); p.setFont(_F_AXIS)
                p.drawText(mind_rect.x() + 4, mind_rect.bottom() - 4,
                           "core=conf · head=MDIM · purple=τ · green arc=D3")
            finally:
                p.restore()
            # Skip body painting when live camera or valid stalled frame is showing
            _cam_showing = (
                self._camera_mode not in ("schematic",)
                and self._camera_pixmap is not None
                and not self._camera_pixmap.isNull()
            )
            if not _cam_showing:
                _draw_overview_body(p, f, body_rect, self.proj, self.trail,
                                    self.arena_trail, self._ax, self._ay)
            _draw_vitals_ribbon(p, f, ribbon_rect, self._err_hist, self._conf_hist)
        if self.cycle_error:
            p.setPen(QtGui.QColor(231, 76, 60)); p.setFont(_F_LABEL_B)
            p.drawText(m + 8, m + _OVERVIEW_HEADER_H - 4, f"⚠ {self.cycle_error[:80]}")


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
    Trend: prediction_error sparkline + ↘/↗ direction arrow (not a health claim).
    """

    GAUGES = ("free-energy", "mutual-info", "empowerment", "cr-temp", "goal-pri")
    # v7: gauge units + human meaning (for the legend/tooltip under each arc).
    GAUGE_META = {
        "free-energy": ("", "prediction error (free energy)"),
        "mutual-info": ("bit", "G′ mutual information"),
        "empowerment": ("", "MDIM empowerment"),
        "cr-temp": ("", "CR cooling temperature"),
        "goal-pri": ("", "active goal priority"),
    }
    _PORTRAIT_HZ = 4.0  # v8 Overview: calmer repaint than global 6 Hz

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self._sm = {k: _Smoother(alpha=0.25) for k in self.GAUGES}
        self._raw: Dict[str, float] = {k: 0.0 for k in self.GAUGES}
        self._fe_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
        self._t0 = time.monotonic()
        # v8 B1: 3-ghost trail of recent drive-halo snapshots (frozen per cycle).
        self._glyph_hist: Deque[Tuple[List[float], float, int]] = deque(maxlen=3)
        self._glyph_sig: tuple = ()
        self._gauge_hist: Dict[str, Deque[float]] = {
            k: deque(maxlen=40) for k in self.GAUGES}
        self._last_paint_t: float = 0.0

    def repaint_if_dirty(self) -> None:
        """v8 Overview: throttle to ~4 Hz (stay dirty between ticks)."""
        if not self._dirty:
            return
        now = time.monotonic()
        if now - self._last_paint_t < 1.0 / self._PORTRAIT_HZ:
            return
        self._dirty = False
        self._last_paint_t = now
        self.update()

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
        self._sm["empowerment"].value(emp); self._raw["empowerment"] = emp
        gp = float(getattr(f, "goal_priority", 0.0) or 0.0)
        self._sm["goal-pri"].value(gp); self._raw["goal-pri"] = gp
        for name in self.GAUGES:
            self._gauge_hist[name].append(self._raw[name])
        # v8 B1: ghost trail — one snapshot per cycle (freeze_sig on levels+active).
        levels = list(getattr(f, "drive_levels", []) or [])
        active = int(getattr(f, "active_drive_id", 0) or 0)
        sig = freeze_sig((f.cycle_id, tuple(round(x, 3) for x in levels), active))
        if sig != self._glyph_sig and levels:
            from phca.monitoring.cognitive_panels import frame_display_confidence
            self._glyph_sig = sig
            self._glyph_hist.append(([float(x) for x in levels],
                                     float(frame_display_confidence(f)),
                                     active))
        self._dirty = True

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
                bound_ms = float(t) * 1000.0
                worst = max(worst, m / bound_ms)
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
        _draw_agent_glyph(p, f, glyph_cx, glyph_cy, R, self._glyph_hist, self._t0,
                          self._sm["empowerment"]._v)
        # gauges grid (5 gauges: 3+2) on the right half + mini sparklines
        gx0 = w // 2
        gw = (w - gx0 - 8) // 3
        gh = (h // 2 - 36) // 2
        for i, name in enumerate(self.GAUGES):
            col = i % 3; row = i // 3
            x = gx0 + col * gw; y = 36 + row * (gh + 8)
            self._draw_gauge(p, name, self._sm[name]._v, x + 4, y + 4, gw - 8, gh - 14)
            self._draw_gauge_spark(p, name, x + 4, y + gh - 8, gw - 8, 10)
        # free-energy convergence trend at the bottom-right
        self._draw_trend(p, gx0, h // 2 + 14, w - gx0 - 8, h // 2 - 28)

    def _draw_glyph(self, p, f, cx, cy, R):
        _draw_agent_glyph(p, f, cx, cy, R, self._glyph_hist, self._t0,
                          self._sm["empowerment"]._v)

    def _draw_limbs(self, p, f, cx, cy, cr, R):
        _draw_agent_limbs(p, f, cx, cy, cr, R)

    def _draw_gauge(self, p, name, v, x, y, w, h):
        import math
        v = max(0.0, min(1.0, float(v)))
        cx = x + w // 2; cy = y + h - 6; R = min(w // 2 - 4, h - 16)
        p.setPen(QtGui.QPen(GRID_COL, 4)); p.setBrush(QtCore.Qt.NoBrush)
        p.drawArc(cx - R, cy - R, R * 2, R * 2, 225 * 16, -270 * 16)
        col = QtGui.QColor(46, 204, 113) if v > 0.66 else (
            QtGui.QColor(241, 196, 15) if v > 0.33 else QtGui.QColor(231, 76, 60))
        p.setPen(QtGui.QPen(col, 4)); p.drawArc(cx - R, cy - R, R * 2, R * 2,
                                               225 * 16, int(-270 * 16 * v))
        ang = math.radians(225 + 270 * v)
        nx = cx + (R - 2) * math.cos(ang); ny = cy - (R - 2) * math.sin(ang)
        p.setPen(QtGui.QPen(QtCore.Qt.white, 2)); p.drawLine(cx, cy, int(nx), int(ny))
        p.setBrush(col); p.setPen(QtCore.Qt.white); p.drawEllipse(cx - 3, cy - 3, 6, 6)
        unit, meaning = self.GAUGE_META.get(name, ("", ""))
        raw = self._raw.get(name, v)
        p.setPen(TEXT_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + 10, name)
        p.setPen(DIM_COL)
        p.drawText(x, y + h - 2, f"{raw:.2f}{unit}")
        p.setPen(DIM_COL); p.setFont(_F_AXIS)
        p.drawText(x, y + 22, meaning[:22])

    def _draw_gauge_spark(self, p, name: str, x: int, y: int, w: int, h: int) -> None:
        """v8 Overview: mini sparkline under each gauge (raw history)."""
        vals = list(self._gauge_hist.get(name, []))
        if len(vals) < 2:
            return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1
        n = len(vals)
        p.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219, 180), 1))
        for i in range(1, n):
            x0 = x + (i - 1) * w / max(n - 1, 1)
            x1 = x + i * w / max(n - 1, 1)
            y0 = y + h - (vals[i - 1] - lo) / (hi - lo) * h
            y1 = y + h - (vals[i] - lo) / (hi - lo) * h
            p.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _draw_trend(self, p, x, y, w, h):
        self._title(p, "prediction-error trend (↘ falling)", x=x, y=y + 12)
        vals = list(self._fe_hist)
        p.setPen(QtGui.QPen(GRID_COL, 1)); p.drawLine(x, y + h - 4, x + w, y + h - 4)
        if len(vals) < 2:
            p.setPen(DIM_COL); p.setFont(_F_AXIS)
            p.drawText(x + 4, y + 30, "collecting…"); return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9: hi = lo + 1
        n = len(vals)
        # Trend arrow only — do not claim free-energy "health".
        q = max(1, n // 4)
        early = float(np.mean(vals[:q])); late = float(np.mean(vals[-q:]))
        arrow = "↘ falling" if late <= early else "↗ rising"
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
        # v8 Overview: full camera + τ bar, no misleading RGB overlays.
        self._overview_mode: bool = False

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
        self._dirty = True

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
        from phca.monitoring.cognitive_panels import frame_display_confidence
        dconf = frame_display_confidence(f)
        env_s = "—"
        if f.agent_pos is not None and f.goal_pos is not None:
            env_s = f"agent={tuple(f.agent_pos)}→goal={tuple(f.goal_pos)}"
        self._title(p, f"GridWorld {n}×{n}  conf={dconf:.2f}  "
                       f"err={f.prediction_error:.2f}  {env_s}")
        self._caption(p, "agent ● + trail · amber ghost = predicted next cell · "
                         "conf=min(epi,qual) · env cell ≠ MDIM drive")

    def _draw_rgb(self, p, f, w, h):
        if self._overview_mode:
            self._draw_rgb_overview(p, f, w, h)
            return
        frame = getattr(f, "env_frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._draw_projection(p, f, w, h); return
        dx, dy, dw, dh = _draw_qimage(p, frame, 10, 28, int(w * 0.62) - 10, h - 40)
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 70), 1))
        p.drawRect(dx - 1, dy - 1, dw + 2, dh + 2)
        from phca.monitoring.cognitive_panels import frame_display_confidence
        dconf = frame_display_confidence(f)
        self._title(p, f"MuJoCo camera  cycle={f.cycle_id}  conf={dconf:.2f}  "
                       f"err={f.prediction_error:.2f}")
        self._caption(p, "live RGB camera frame · env=mujoco_rgb (dim-adaptive; high-dim → see Phase Space)")
        cx, cy = dx + dw // 2, dy + dh // 2
        dim_names = list(getattr(f, "dim_names", []) or [])
        pdims: List[int] = []
        adims: List[int] = []
        pred = f.predicted_state
        if pred is not None:
            pvec = np.asarray(pred, dtype=np.float32).reshape(-1)
            if pvec.size >= 1:
                pvx, pvy, pdims = _vec_pca2(pvec)
                ex = int(np.clip(cx + pvx * 40, dx, dx + dw))
                ey = int(np.clip(cy - pvy * 40, dy, dy + dh))
                _arrow(p, cx, cy, ex, ey, QtGui.QColor(39, 174, 96), size=8)
        act = f.continuous_action if f.continuous_action is not None else f.last_action_vector
        if act is not None:
            avec = np.asarray(act, dtype=np.float32).reshape(-1)
            if avec.size >= 1:
                avx, avy, adims = _vec_pca2(avec)
                ax = int(np.clip(cx + avx * 40, dx, dx + dw))
                ay = int(np.clip(cy - avy * 40, dy, dy + dh))
                _arrow(p, cx, cy, ax, ay, ACCENT, size=8)
        if self.proj is not None:
            inset_x = int(w * 0.62) + 6
            self._draw_projection_inset(p, f, inset_x, 28, w - inset_x - 8, h - 40)
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        plbl = _dim_arrow_label(pdims, dim_names) if pdims else "pred"
        albl = _dim_arrow_label(adims, dim_names) if adims else "act"
        p.drawText(dx, dy + dh + 12, f"green=predicted({plbl})  amber=action({albl}) · PCA-2D when dim>2")

    def _draw_rgb_overview(self, p, f, w, h) -> None:
        """v8 Overview: full-width camera, signed τ bar below — no RGB arrows."""
        frame = getattr(f, "env_frame", None)
        from phca.monitoring.cognitive_panels import frame_display_confidence
        ak = getattr(f, "action_kind", "") or "?"
        dconf = frame_display_confidence(f)
        self._title(p, f"MuJoCo camera  cycle={f.cycle_id}  conf={dconf:.2f}  "
                       f"err={f.prediction_error:.2f}  action={ak}")
        self._caption(p, "live RGB · τ bar below = chosen continuous action (signed per-dim)")
        cam_h = h - 52
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._empty(p, "no camera frame (collecting…)"); return
        dx, dy, dw, dh = _draw_qimage(p, frame, 10, 28, w - 20, cam_h)
        p.setPen(QtGui.QPen(GRID_COL, 1))
        p.drawRect(dx - 1, dy - 1, dw + 2, dh + 2)
        act = f.continuous_action if f.continuous_action is not None else f.last_action_vector
        dim_names = list(getattr(f, "dim_names", []) or [])
        ty = dy + dh + 8
        self._draw_tau_bar(p, act, dim_names, 10, ty, w - 20, h - ty - 6)

    def _draw_tau_bar(self, p, act, dim_names: List[str], x: int, y: int, w: int, h: int) -> None:
        _draw_tau_bar(p, act, dim_names, x, y, w, h)

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
    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.compact = compact

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self._dirty = True

    def _draw_compact(self, p: QtGui.QPainter) -> None:
        """v8 Overview: dot strip — level = dot size (no redundant legend)."""
        f = self.frame
        if f is None or not f.drive_levels:
            self._empty(p, "Drives…"); return
        w, h = self.width(), self.height()
        n = len(f.drive_levels)
        self._title(p, f"Drives  active={_drive_short(f.active_drive_id) if f.active_drive_id else '?'}")
        self._caption(p, "dot size = drive level 0..1 · colour = drive id")
        cy = h // 2 + 4
        span = w - 40
        for i in range(n):
            did = i + 1
            v = float(np.clip(f.drive_levels[i], 0, 1))
            cx = int(20 + (i + 0.5) * span / n)
            rad = int(4 + 10 * v)
            col = _drive_color(did)
            if f.active_drive_id == did:
                p.setPen(QtGui.QPen(ACCENT, 2))
            else:
                p.setPen(QtCore.Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(cx - rad, cy - rad, rad * 2, rad * 2)
            p.setPen(DIM_COL if f.active_drive_id != did else TEXT_COL)
            p.setFont(_F_AXIS)
            p.drawText(cx - 12, cy + rad + 12, f"{v:.2f}")

    def _draw(self, p: QtGui.QPainter) -> None:
        if self.compact:
            self._draw_compact(p); return
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
            col = _drive_color(did)
            p.setBrush(col)
            p.setPen(QtGui.QPen(col.darker(140), 1))
            bh = int(v * (bot - top))
            p.drawRect(int(x + 4), int(bot - bh), int(barw - 10), bh)
            # v8: raw value beside the compressed bar (0..1, unitless level)
            p.setPen(TEXT_COL if active else DIM_COL)
            p.setFont(QtGui.QFont("Sans", 7, QtGui.QFont.Bold if active else 0))
            p.drawText(int(x + 4), int(bot - bh) - 2, f"{v:.2f}")
            # target marker
            ty = bot - t * (bot - top)
            p.setPen(QtGui.QPen(ACCENT, 2))
            p.drawLine(int(x + 2), int(ty), int(x + barw - 6), int(ty))
            # label (abbreviated)
            lbl = _drive_short(did)
            p.setPen(ACCENT if active else TEXT_COL)
            if active:
                p.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            p.drawText(int(x + 4), h - 16, lbl)
            p.setFont(QtGui.QFont("Sans", 8))
            if active:
                p.setPen(TEXT_COL)
                p.drawText(int(x + 4), h - 4, _drive_name(did).split()[-1] if " " in _drive_name(did) else _drive_name(did))
        self._title(p, f"Drives  value vs ◆target  active={_drive_short(f.active_drive_id) if f.active_drive_id else '?'}")
        self._caption(p, f"fixed 0..1 scale · ◆ = target · {n} homeostatic drives · numbers = raw level")
        # color legend
        p.setFont(QtGui.QFont("Sans", 7))
        lx = left + 4
        for did in (1, 3, 5):
            if did > n:
                continue
            p.setPen(_drive_color(did)); p.drawLine(lx, 30, lx + 10, 30)
            p.setPen(DIM_COL); p.drawText(lx + 13, 33, _drive_name(did))
            lx += 92


class TrendCanvas(_ChartCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.err: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.conf: Deque[float] = deque(maxlen=TREND_WINDOW)
        self.add_series("err", "#e74c3c", self.err, log=True, fill=True)
        self.add_series("conf", "#2ecc71", self.conf, log=False, fill=False)

    def push(self, f: ObservabilityFrame) -> None:
        from phca.monitoring.cognitive_panels import frame_display_confidence
        dconf = frame_display_confidence(f)
        self.err.append(float(f.prediction_error))
        self.conf.append(float(dconf))
        self.title = (f"Prediction error (red, log) & confidence (green)  "
                      f"err={f.prediction_error:.2f}  conf={dconf:.3f}")
        self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        super()._draw(p)
        if self.err and self.conf:
            p.setPen(TEXT_COL); p.setFont(_F_LABEL_B)
            p.drawText(self.width() - 130, 14,
                       f"err={self.err[-1]:.2f}  conf={self.conf[-1]:.3f}")


class AttentionCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None

    def set_frame(self, f: ObservabilityFrame) -> None:
        self.frame = f; self._dirty = True

    def _draw(self, p: QtGui.QPainter) -> None:
        from phca.monitoring.cognitive_panels import frame_attention_pairs
        f = self.frame
        pairs = frame_attention_pairs(f) if f is not None else []
        if f is None or not pairs:
            self._empty(p, "no salient chunks"); return
        self._title(p, "Attention focus (attended chunk ids, by salience)")
        self._caption(p, "attended chunks · bar height = salience[chunk_id] · gumbel-τ applied upstream")
        w, h = self.width(), self.height()
        n = len(pairs)
        top, bot = 34, h - 24
        bw = (w - 40) / n
        sal_max = max((s for _, s in pairs), default=1.0) or 1.0
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        for tv in (0.0, 0.5, 1.0):
            y = int(bot - tv * (bot - top))
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 50), 1)); p.drawLine(20, y, w - 20, y)
        for i, (idx, raw) in enumerate(pairs):
            x = 20 + i * bw
            sal = float(np.clip(raw / sal_max, 0, 1))
            bh = int(sal * (bot - top))
            p.setBrush(QtGui.QColor(155, 89, 182, 210))
            p.setPen(QtGui.QPen(QtGui.QColor(155, 89, 182), 1))
            p.drawRect(int(x), int(bot - bh), int(bw - 8), bh)
            p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
            p.drawText(int(x), int(bot - bh) - 2, f"{raw:.2g}")
            p.setPen(TEXT_COL); p.setFont(QtGui.QFont("Sans", 8))
            p.drawText(int(x), h - 8, f"#{int(idx)}")
        p.setPen(DIM_COL); p.setFont(QtGui.QFont("Sans", 7))
        p.drawText(w - 90, 16, f"sal max={sal_max:.2g}")


class StatusPanel(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[ObservabilityFrame] = None
        self.cycle_error: Optional[str] = None

    def set_state(self, f: Optional[ObservabilityFrame], err: Optional[str]) -> None:
        self.frame = f; self.cycle_error = err; self._dirty = True

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
        m3_n = int(getattr(f, "m3_count", 0) or 0) or int(getattr(f, "episode_count", 0) or 0)
        m3e = m3_n / f.m3_cap if f.m3_cap else 0.0
        m4e = f.fact_count / f.m4_cap if f.m4_cap else 0.0
        from phca.monitoring.cognitive_panels import (
            frame_display_confidence, frame_is_geometry_control,
        )
        dconf = frame_display_confidence(f)
        epi = float(getattr(f, "prediction_confidence", 0.0) or 0.0)
        qual = float(getattr(f, "prediction_quality", 1.0) or 1.0)
        ctrl = "CONTROL=geometry" if frame_is_geometry_control(f) else "CONTROL=scored"
        p.drawText(10, y, f"M3 {m3_n}/{f.m3_cap} ({m3e*100:.0f}%)   "
                          f"M4 {f.fact_count}/{f.m4_cap} ({m4e*100:.0f}%)  prune→{f.m4_prune_target}")
        y += 14
        p.drawText(10, y, f"{ctrl}  conf={dconf:.2f} (epi={epi:.2f}·qual={qual:.2f})")
        y += 16
        # v7: proportional-width gauges (fit the panel)
        gw = max(80, (W - 40) // 2)
        x = self._gauge(p, 10, y, "RSS", f.rss_bytes / 1e6, 1024.0, "MB",
                        QtGui.QColor(230, 126, 34), gw=gw)
        self._gauge(p, x, y, "lat", f.latency_ms, 50.0, "ms",
                    QtGui.QColor(46, 204, 113), gw=gw)
        y += 24
        r = f.action_rationale or {}
        gid = r.get("goal_id")
        drive_lbl = DRIVE_NAMES.get(gid, gid) if gid is not None else "—"
        env_lbl = "—"
        if f.goal_pos is not None and len(f.goal_pos) >= 2:
            env_lbl = f"({int(f.goal_pos[0])},{int(f.goal_pos[1])})"
        tag = "EXPLORE" if r.get("explored") else "EXPLOIT"
        note = r.get("note", "")
        score = r.get("best_score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        from phca.monitoring.cognitive_panels import geometry_score_label
        score_key = geometry_score_label(f)
        head = f"env={env_lbl} · drive={drive_lbl} · {tag} · {score_key}={score_s}"
        p.setPen(TEXT_COL)
        p.drawText(10, y, head)
        y += 13
        if note:
            p.setPen(DIM_COL)
            p.setFont(QtGui.QFont("Monospace", 8))
            fm = p.fontMetrics()
            one = fm.elidedText(note, QtCore.Qt.ElideRight, W - 20)
            p.drawText(10, y, one)
