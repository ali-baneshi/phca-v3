"""Observatory window + controller (extracted from qt_dashboard monolith).

Panel views remain in ``qt_dashboard.py``; this module owns transport wiring,
session status, and frame→widget updates. ``qt_dashboard`` re-exports these
symbols for stable import paths.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5 import QtWidgets, QtCore, QtGui

from phca.monitoring.observability import OBSERVABILITY_SCHEMA_VERSION, ObservabilityFrame
from phca.monitoring.belief_projection import BeliefProjection, set_autoscale_frozen
from phca.monitoring.cognitive_panels import (
    OBSERVATORY_TAB_LABELS,
    decimate_frames_for_history,
    moment_tab_badge,
    phase_tab_status_line,
    session_status_text,
)
from phca.monitoring.overview_narrative import (
    _overview_moment_flags,
    _phase_frame_is_grid,
)
from phca.monitoring.qt_transport import TransportBar as _TransportBar

from phca.monitoring.qt_dashboard import (
    OverviewAgentView,
    CognitiveFlowView,
    CandidateScoreView,
    TrajectoryView,
    DriveRadarView,
    _PhasePortraitView,
    _DimSelector,
    RetentionView,
    ViolationTable,
    RBTABoundsView,
    MemoryBeliefView,
    GoalsMotivationView,
    _BaseCanvas,
    _qss,
    _phase_layout,
)


class DashboardController:
    """Mutates persistent widget state from frames (no widget rebuild)."""

    # Views that belong to each tab (index → list of window attribute names).
    # Tab 0 (Overview) has its own set_frame called unconditionally; proj/viol
    # are updated via dedicated methods and are not in this table.
    _TAB_VIEW_ATTRS: Dict[int, Tuple[str, ...]] = {
        1: ("flow",),
        2: ("cand",),
        3: ("traj", "radar", "perdim"),
        4: ("retention", "rbta_bounds"),
        5: ("memory",),
        6: ("goals",),
    }

    def __init__(self, window: "ObservatoryWindow"):
        self.w = window
        # v6 flicker-free: skip the whole widget update when the frame is the
        # same cycle as the last one we rendered (idle/no-new-data → 0 repaints).
        self._last_cycle: int = -1
        self._last_agent_id: int = -1
        self._last_rolling: Optional[List[ObservabilityFrame]] = None
        self._pending_tab_rebuilds: set = set()
        # Lazy-tab dedup: cycle id the last time each tab received set_frame
        # via the live-mode lazy path (prevents double-set on tab switch).
        self._tab_last_cycle: Dict[int, int] = {}

    def _contract_flags(self) -> Tuple[bool, bool]:
        review = bool(getattr(self.w, "_review_mode", False))
        transport = getattr(self.w, "_transport", None)
        clock = getattr(transport, "clock", None) if transport else None
        replay = bool(clock is not None and clock.mode == "replay" and not review)
        return review, replay

    def _repaint_visible_tab(self, *, force_sync: bool = False) -> None:
        tab = self.w._tabs.currentWidget()
        if tab is None:
            return
        for cv in tab.findChildren(_BaseCanvas):
            try:
                if force_sync:
                    cv.mark_dirty()
                    cv.repaint()
                else:
                    cv.repaint_if_dirty()
            except Exception as exc:
                self.w.surface_paint_error(exc)

    def _mark_all_tabs_dirty(self) -> None:
        for i in range(self.w._tabs.count()):
            tab = self.w._tabs.widget(i)
            if tab is None:
                continue
            for cv in tab.findChildren(_BaseCanvas):
                cv.mark_dirty()

    def _activate_review_tab(self, tab_idx: int) -> None:
        f = getattr(self.w, "_last_frame", None)
        if tab_idx == 2 and f is not None:
            try:
                self.w.cand.mark_dirty()
                self.w.cand.repaint()
            except Exception as exc:
                self.w.surface_paint_error(exc)
        if tab_idx == 3 and f is not None:
            self.w._apply_phase_layout(f)
            self.w.dim_selector.refresh()
            review, _ = self._contract_flags()
            plen = getattr(self.w.cand, "_prefix_len", 0)
            self.w.update_phase_tab_status(f, review=review, prefix_len=plen)
        tab = self.w._tabs.widget(tab_idx)
        if tab is None:
            return
        for cv in tab.findChildren(_BaseCanvas):
            try:
                cv.mark_dirty()
                cv.repaint()
            except Exception as exc:
                self.w.surface_paint_error(exc)

    def _set_prefix_len(self, prefix_len: int) -> None:
        for attr in ("traj", "cand", "perdim", "radar", "retention", "goals"):
            view = getattr(self.w, attr, None)
            if view is not None:
                setattr(view, "_prefix_len", int(prefix_len))

    def on_tab_changed(self, tab_idx: int) -> None:
        """Lazy rebuild: finish history rebuild when user switches to a tab."""
        if (not self.w._review_mode
                and tab_idx in self._pending_tab_rebuilds
                and self._last_rolling):
            self._rebuild_tab_histories(self._last_rolling, tab_idx)
            self._pending_tab_rebuilds.discard(tab_idx)
        if self.w._review_mode:
            self._activate_review_tab(tab_idx)
        else:
            # Lazy tab: push the latest frame onto views that were skipped
            # during live-mode streaming so they have current data to paint.
            # Skip if the tab already has the latest cycle (prevents double-set).
            f = getattr(self.w, "_last_frame", None)
            if f is not None and self._last_rolling is None:
                cid = int(getattr(f, "cycle_id", -1))
                if self._tab_last_cycle.get(tab_idx) != cid:
                    review, replay = self._contract_flags()
                    for attr in self._TAB_VIEW_ATTRS.get(tab_idx, ()):
                        view = getattr(self.w, attr, None)
                        if view is not None:
                            view.set_frame(f, histories_done=False, replay=replay, review=review)
                    self._tab_last_cycle[tab_idx] = cid
            self._repaint_visible_tab(force_sync=True)

    def _rebuild_tab_histories(
        self,
        rolling: List[ObservabilityFrame],
        tab_idx: int,
    ) -> None:
        """Rebuild rolling histories for one tab (+ shared overview/proj)."""
        self.w.overview.rebuild_histories(rolling)
        self.w.proj.rebuild_from_frames(rolling)
        if tab_idx == 1:
            self.w.flow.rebuild_histories(rolling)
        elif tab_idx == 2:
            self.w.cand.rebuild_histories(rolling)
        elif tab_idx == 3:
            self.w.traj.rebuild_histories(rolling)
            self.w.radar.rebuild_histories(rolling)
            self.w.perdim.rebuild_histories(rolling)
        elif tab_idx == 4:
            self.w.retention.rebuild_histories(rolling)
            self.w.rbta_bounds.rebuild_histories(rolling)
            self.w.viol.rebuild_from_frames(rolling)
        elif tab_idx == 5:
            self.w.memory.rebuild_histories(rolling)
        elif tab_idx == 6:
            self.w.goals.rebuild_histories(rolling)

    def rebuild_all_histories(self, rolling: List[ObservabilityFrame]) -> None:
        """Rebuild all tab histories (post-run review mode)."""
        if len(rolling) > 2000:
            rolling = decimate_frames_for_history(rolling, 2000)
        self.w.proj.rebuild_from_frames(rolling)
        self.w.overview.rebuild_histories(rolling)
        self.w.flow.rebuild_histories(rolling)
        self.w.cand.rebuild_histories(rolling)
        self.w.traj.rebuild_histories(rolling)
        self.w.radar.rebuild_histories(rolling)
        self.w.perdim.rebuild_histories(rolling)
        self.w.retention.rebuild_histories(rolling)
        self.w.rbta_bounds.rebuild_histories(rolling)
        self.w.viol.rebuild_from_frames(rolling)
        self.w.memory.rebuild_histories(rolling)
        self.w.goals.rebuild_histories(rolling)
        self._pending_tab_rebuilds.clear()
        self._last_rolling = rolling
        if not rolling:
            return
        f = rolling[-1]
        review, replay = self._contract_flags()
        self._set_prefix_len(len(rolling))
        self.w.overview.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.flow.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.cand.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w._last_frame = f
        self.w._apply_phase_layout(f)
        self.w.update_phase_tab_status(f, review=review, prefix_len=len(rolling))
        self.w.traj.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.radar.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.perdim.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.dim_selector.refresh()
        self.w.retention.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.rbta_bounds.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.viol.rebuild_from_frames(rolling)
        self.w._viol_summary.setText(self.w.viol.summary())
        self.w.memory.set_frame(f, histories_done=True, replay=replay, review=review)
        self.w.goals.set_frame(f, histories_done=True, replay=replay, review=review)
        self._last_cycle = int(getattr(f, "cycle_id", -1))
        self._mark_all_tabs_dirty()
        self._repaint_visible_tab(force_sync=True)

    def _update_tab_badges(self, flags: Dict[str, Any]) -> None:
        badge = moment_tab_badge(flags)
        tabs = self.w._tabs
        bases = self.w._tab_base_labels
        for i, base in enumerate(bases):
            label = f"{base} • {badge}" if badge else base
            if tabs.tabText(i) != label:
                tabs.setTabText(i, label)

    def _badge_moment_flags(self, f: ObservabilityFrame) -> Dict[str, Any]:
        """Tab-badge flags aligned with Overview history (scrub/review/live)."""
        ov = self.w.overview
        if ov._moment_series:
            return ov._moment_series[-1]
        return _overview_moment_flags(
            f,
            ov._err_hist,
            prev_drive_id=ov._prev_drive_id,
            prev_best_score=ov._prev_best_score,
        )

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
            aid = int(getattr(f, "agent_id", 0) or 0)
            if cycle_error is None and cid == self._last_cycle and aid == self._last_agent_id and not rolling:
                return
            self._last_cycle = cid
            self._last_agent_id = aid
            if rolling:
                if len(rolling) > 2000:
                    rolling = decimate_frames_for_history(rolling, 2000)
                review, replay = self._contract_flags()
                if self.w._review_mode:
                    self.rebuild_all_histories(rolling)
                    self._update_tab_badges(self._badge_moment_flags(f))
                    title = (f"PHCA Cognitive Observatory — review cycle {f.cycle_id}"
                             if review else f"PHCA Cognitive Observatory — cycle {f.cycle_id}")
                    self.w.setWindowTitle(title)
                    self._repaint_visible_tab(force_sync=True)
                    return
                # OBS-002: scrub/seek rolling update rebuilds all tabs (replay + live).
                self.rebuild_all_histories(rolling)
                self._pending_tab_rebuilds.clear()
                self._last_rolling = rolling
            else:
                self.w.proj.update(f)
                v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
                self.w.proj.push_history(self.w.proj.project(v))
                self._pending_tab_rebuilds.clear()
                self._last_rolling = None
            histories_done = bool(rolling)
            review, replay = self._contract_flags()
            if rolling:
                self._set_prefix_len(len(rolling))
            self._update_tab_badges(self._badge_moment_flags(f))
            self.w._last_frame = f
            # Overview is always updated (shared state like _err_hist, badges).
            self.w.overview.set_frame(f, histories_done=histories_done, replay=replay, review=review)
            # Lazy tab updates: only set_frame on visible tab views in live mode.
            # In rolling mode, rebuild_all_histories already called set_frame on
            # all views, so the per-view calls below are skipped.
            if not rolling:
                visible_idx = self.w._tabs.currentIndex()
                self._tab_last_cycle[visible_idx] = cid
                for attr in self._TAB_VIEW_ATTRS.get(visible_idx, ()):
                    view = getattr(self.w, attr, None)
                    if view is not None:
                        view.set_frame(f, histories_done=False, replay=replay, review=review)
                # Phase-tab-only setup always runs (cheap, keeps phase-status current).
                self.w._apply_phase_layout(f)
                plen = getattr(self.w.cand, "_prefix_len", 0)
                self.w.update_phase_tab_status(f, review=review, prefix_len=plen)
                self.w.dim_selector.refresh()
                self.w.viol.add_frame(f)
                self.w._viol_summary.setText(self.w.viol.summary())
            self.w.setWindowTitle(f"PHCA Cognitive Observatory — cycle {f.cycle_id}")
        else:
            self.w.overview.set_state(None, cycle_error)
        if cycle_error:
            self.w.set_cycle_error(cycle_error)
            self.w.overview.set_state(f, cycle_error)
            self.w.refresh_session_strip()


class RenderPacer(QtCore.QObject):
    """v8 calm-render: a single QTimer that repaints only the *visible* tab's
    canvases, at a calm cadence (default 6 Hz). Each canvas repaints only if it
    is dirty (new data since last paint). Paused/no-new-data → 0 repaints.

    Owned by ``ObservatoryWindow``; started once. Cheap: one findChildren + a
    handful of dirty-flag checks per tick."""

    def __init__(self, window: "ObservatoryWindow", render_hz: float = 6.0,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent or window)
        self._window = window
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000.0 / max(render_hz, 0.5)))
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def set_hz(self, render_hz: float) -> None:
        self._timer.setInterval(int(1000.0 / max(render_hz, 0.5)))

    def _tick(self) -> None:
        tab = self._window._tabs.currentWidget()
        if tab is None:
            return
        for cv in tab.findChildren(_BaseCanvas):
            try:
                cv.repaint_if_dirty()
            except Exception as exc:
                self._window.surface_paint_error(exc)


class _PhaseSpaceTab(QtWidgets.QWidget):
    """Phase Space tab — unified status strip + resize-driven layout."""

    def __init__(self, obs_window: "ObservatoryWindow"):
        super().__init__()
        self._obs_window = obs_window
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._phase_status = QtWidgets.QLabel("")
        self._phase_status.setFixedHeight(22)
        self._phase_status.setStyleSheet(
            "color:#a0a0b0; font-family:monospace; font-size:11px; padding:2px 8px;"
            "background:#1a1a22; border-bottom:1px solid #333;"
        )
        outer.addWidget(self._phase_status)
        grid_w = QtWidgets.QWidget()
        self._grid_lay = QtWidgets.QGridLayout(grid_w)
        self._grid_lay.setContentsMargins(6, 6, 6, 6)
        self._grid_lay.setSpacing(6)
        outer.addWidget(grid_w, 1)

    def update_status(self, text: str) -> None:
        self._phase_status.setText(text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        f = self._obs_window._last_frame
        if f is not None:
            self._obs_window._apply_phase_layout(f)


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
        self._status_strip = QtWidgets.QLabel("")
        self._status_strip.setFixedHeight(22)
        self._status_strip.setStyleSheet(
            "color:#a0a0b0; font-family:monospace; font-size:11px; padding:2px 8px;"
            "background:#1a1a22; border-bottom:1px solid #333;"
        )
        cv.addWidget(self._status_strip)
        cv.addWidget(tabs, 1)
        self.setCentralWidget(central)
        self._tabs = tabs
        self._tab_base_labels = list(OBSERVATORY_TAB_LABELS)
        self._session_ctx: Dict[str, Any] = {}
        self._verify_status = ""
        self._review_mode = False
        self._session_incomplete = False
        self._cycle_error = ""
        self._rbta_safe_ratio = 0.0

        # shared dimension-agnostic projection (World / Phase-Space / Action)
        self.proj = BeliefProjection(window=256)

        # Overview — v8.1: single unified agent card
        ov = QtWidgets.QWidget()
        ov_lay = QtWidgets.QVBoxLayout(ov)
        ov_lay.setContentsMargins(6, 6, 6, 6); ov_lay.setSpacing(0)
        self.overview = OverviewAgentView()
        self.overview.set_projection(self.proj)
        self.overview.setMinimumHeight(480)
        self.overview.bind_camera_tabs(tabs, overview_tab_index=0)
        ov_lay.addWidget(self.overview, 1)
        tabs.addTab(ov, "Overview")

        # Cognitive Flow
        self.flow = CognitiveFlowView()
        tabs.addTab(self.flow, "Cognitive Flow")

        # Action Selection
        self.cand = CandidateScoreView(); self.cand.set_projection(self.proj)
        tabs.addTab(self.cand, "Action Selection")

        # Phase Space & Belief Uncertainty
        ps = _PhaseSpaceTab(self)
        ps_lay = ps._grid_lay
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
        self._ps_tab = ps
        self._ps_lay = ps_lay
        self._last_frame: Optional[ObservabilityFrame] = None

        # Retention & Resources
        ret = QtWidgets.QWidget(); ret_lay = QtWidgets.QVBoxLayout(ret)
        ret_lay.setContentsMargins(6, 6, 6, 6); ret_lay.setSpacing(6)
        self.retention = RetentionView()
        self.rbta_bounds = RBTABoundsView()
        self.rbta_bounds.setMinimumHeight(160)
        self.rbta_bounds.setMaximumHeight(220)
        self.viol = ViolationTable()
        self._viol_summary = QtWidgets.QLabel("0 violations")
        self._viol_summary.setStyleSheet("color:#e74c3c; font-weight:bold; padding:2px 6px;")
        ret_lay.addWidget(self.retention, 3)
        ret_lay.addWidget(self.rbta_bounds, 0)
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
        self._tabs.currentChanged.connect(self.controller.on_tab_changed)
        self._camera_provider: Optional[Callable[[], Any]] = None
        self._camera_debug: bool = False
        # v8: calm-render pacer (repaints the visible tab's dirty canvases at
        # ~6 Hz). Constructed here; started by the launcher via start_render.
        self.render_pacer = RenderPacer(self, render_hz=6.0)
        self._transport: Optional[_TransportBar] = None
        self._multi_agent: bool = False
        self._agent_ids: List[int] = [0]
        self._agent_labels: Dict[int, str] = {0: ""}
        self._selected_agent_id: int = 0
        self._all_frames: List[ObservabilityFrame] = []
        self._all_frames_maxlen: int = 8000
        self._moment_matches: List[Any] = []

    def _moment_nav_enabled(self) -> bool:
        transport = self._transport
        if transport is None or transport.clock is None:
            return False
        clock = transport.clock
        if self._review_mode:
            return True
        if clock.mode == "replay" and (clock.paused or clock.scrubbing):
            return True
        return False

    def _rebuild_moment_matches(self) -> None:
        from phca.monitoring.session_query import (
            current_match_position,
            query_frames,
        )
        transport = self._transport
        if transport is None or transport.clock is None:
            self._moment_matches = []
            return
        frames = list(transport.clock._frames)
        if not frames and self._all_frames:
            frames = self.project_frames_for_agent(self._all_frames)
        q = transport.moment_query_from_ui()
        self._moment_matches = query_frames(frames, q)
        pos = current_match_position(self._moment_matches, transport.clock.cursor_int)
        transport.update_moment_counter(pos, len(self._moment_matches))
        transport.set_moment_nav_enabled(self._moment_nav_enabled())

    def _jump_moment(self, direction: int) -> None:
        from phca.monitoring.session_query import navigate_match
        transport = self._transport
        if transport is None or transport.clock is None or not self._moment_nav_enabled():
            return
        if not self._moment_matches:
            self._rebuild_moment_matches()
        idx = navigate_match(
            self._moment_matches, transport.clock.cursor_int, int(direction))
        if idx is None:
            return
        transport.clock.seek(idx)
        transport._sync_slider()
        set_autoscale_frozen(True)
        self._rebuild_moment_matches()

    def _on_moment_filter_changed(self) -> None:
        self._rebuild_moment_matches()

    def init_multi_agent(self, count: int, labels: List[str]) -> None:
        """Configure multi-agent UI (agent selector + per-agent projection)."""
        self._multi_agent = int(count) > 1
        self._agent_ids = list(range(int(count)))
        self._agent_labels = {i: labels[i] for i in range(int(count))}
        self._selected_agent_id = 0
        for cv in self.findChildren(_BaseCanvas):
            cv._multi_agent = self._multi_agent
        if self._transport is not None:
            self._transport.setup_agent_selector(labels, self.select_agent)

    def append_observability_frames(self, frames: List[ObservabilityFrame]) -> None:
        """Accumulate interleaved frames during live multi-agent runs."""
        if not frames:
            return
        self._all_frames.extend(frames)
        if len(self._all_frames) > self._all_frames_maxlen:
            drop = len(self._all_frames) - self._all_frames_maxlen
            self._all_frames = self._all_frames[drop:]

    def load_multi_agent_frames(
        self,
        frames: List[ObservabilityFrame],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load replay frames and configure agent selector when needed."""
        from phca.monitoring.multi_agent import is_multi_agent_session, session_agent_ids

        self._all_frames = list(frames)
        if is_multi_agent_session(meta, frames=frames):
            self._multi_agent = True
            self._agent_ids = session_agent_ids(frames)
            self._agent_labels = {}
            for f in frames:
                aid = int(getattr(f, "agent_id", 0) or 0)
                lbl = str(getattr(f, "agent_label", "") or "")
                if aid not in self._agent_labels and lbl:
                    self._agent_labels[aid] = lbl
            labels = [
                self._agent_labels.get(aid, f"agent_{aid}") for aid in self._agent_ids
            ]
            self.init_multi_agent(len(self._agent_ids), labels)
        self._apply_agent_projection(rebuild=False)
        self._rebuild_moment_matches()

    def project_frames_for_agent(
        self,
        frames: List[ObservabilityFrame],
    ) -> List[ObservabilityFrame]:
        """Return frames for the currently selected agent."""
        if not self._multi_agent:
            return list(frames)
        from phca.monitoring.multi_agent import frames_for_agent
        return frames_for_agent(frames, self._selected_agent_id)

    def select_agent(self, agent_id: int) -> None:
        """Switch dashboard to another agent's timeline."""
        aid = int(agent_id)
        if aid == self._selected_agent_id:
            return
        self._selected_agent_id = aid
        self.controller._last_cycle = -1
        self.controller._last_agent_id = -1
        if self._transport is not None:
            self._transport.set_agent_index(aid)
        self._apply_agent_projection(rebuild=True)

    def _apply_agent_projection(self, *, rebuild: bool = True) -> None:
        from phca.monitoring.multi_agent import frames_for_agent

        projected = (
            frames_for_agent(self._all_frames, self._selected_agent_id)
            if self._all_frames else []
        )
        transport = self._transport
        if transport is None or transport.clock is None:
            return
        clock = transport.clock
        cursor = min(clock.cursor_int, max(0, len(projected) - 1))
        was_paused = clock.paused
        was_scrubbing = clock.scrubbing
        clock.reload_frames(
            projected,
            preserve_transport=True,
            follow_live=not was_paused and not was_scrubbing,
        )
        if projected and (was_paused or was_scrubbing):
            clock._cursor = float(cursor)
        transport.set_range(len(projected))
        transport._sync_slider()
        if projected:
            clock._emit(force_rebuild=True)
        if rebuild and projected:
            prefix = projected[: clock.cursor_int + 1]
            self.controller.rebuild_all_histories(prefix)
        self._rebuild_moment_matches()

    def set_session_context(self, ctx: Dict[str, Any]) -> None:
        self._session_ctx = dict(ctx or {})

    def set_verify_status(self, status: str) -> None:
        self._verify_status = str(status or "")

    def set_cycle_error(self, error: str) -> None:
        """Surface a cycle-thread abort on the global status strip and banners."""
        self._cycle_error = str(error or "")
        if self._cycle_error:
            self._session_incomplete = True

    def surface_paint_error(self, exc: BaseException) -> None:
        """OBS-D09: paint failures go to PlaybackClock.error (status strip ERR)."""
        msg = f"paint failed: {exc}"
        transport = self._transport
        clock = getattr(transport, "clock", None) if transport is not None else None
        if clock is not None:
            prev = getattr(clock, "error", None)
            clock.error = msg if not prev else f"{prev}; {msg}"
        import sys
        print(f"[observatory] {msg}", file=sys.stderr)
        self.refresh_session_strip()

    def mark_session_incomplete(self, *, recorded: int = 0, requested: int = 0) -> None:
        """Mark review as incomplete when recorded cycles < requested."""
        self._session_incomplete = True
        if requested > 0:
            self._session_ctx["target_cycles"] = int(requested)
        if recorded >= 0:
            self._session_ctx["jsonl_count"] = int(recorded)

    def set_rbta_safe_ratio(self, ratio: float) -> None:
        """CORE-A02: surface sustained RBTA safe-mode fraction on the status strip."""
        try:
            self._rbta_safe_ratio = float(ratio)
        except (TypeError, ValueError):
            self._rbta_safe_ratio = 0.0

    def _rbta_safe_ratio_from_frames(self, frames: List[Any]) -> float:
        if not frames:
            return float(self._rbta_safe_ratio or 0.0)
        safe = 0
        for f in frames:
            r = getattr(f, "action_rationale", None) or {}
            if r.get("rbta_safe_mode") or r.get("decision_reason") == "rbta_safe":
                safe += 1
            elif str(getattr(f, "rbta_action", "") or "").upper() == "TERMINATE":
                safe += 1
        return safe / float(len(frames))

    def update_session_strip(
        self,
        frame: Optional[ObservabilityFrame],
        *,
        jsonl_count: int = 0,
        clock_cursor: int = 0,
    ) -> None:
        ctx = self._session_ctx
        cid = int(getattr(frame, "cycle_id", 0) or 0) if frame is not None else 0
        target = int(ctx.get("target_cycles", 0) or 0)
        live_lag = max(0, cid - int(clock_cursor))
        ek = str(getattr(frame, "env_kind", "") or "") if frame is not None else ""
        agent_count = int(ctx.get("agent_count", 1) or 1)
        agent_id = int(getattr(frame, "agent_id", self._selected_agent_id) or 0) if frame else self._selected_agent_id
        agent_label = str(getattr(frame, "agent_label", "") or "") if frame else self._agent_labels.get(agent_id, "")
        playback_error = ""
        transport = self._transport
        if transport is not None and getattr(transport, "clock", None) is not None:
            playback_error = str(transport.clock.error or "")
        incomplete = bool(self._session_incomplete) or (
            target > 0 and int(jsonl_count) > 0 and int(jsonl_count) < target
        )
        if incomplete:
            self._session_incomplete = True
        safe_ratio = float(self._rbta_safe_ratio or 0.0)
        if transport is not None and getattr(transport, "clock", None) is not None:
            frames = getattr(transport.clock, "_frames", None) or []
            if frames:
                safe_ratio = max(safe_ratio, self._rbta_safe_ratio_from_frames(frames))
        text = session_status_text(
            env=str(ctx.get("env", "") or ""),
            env_kind=ek,
            camera=str(ctx.get("camera", "") or ""),
            cycle_id=cid,
            total=target,
            live_lag=live_lag,
            schema_version=OBSERVABILITY_SCHEMA_VERSION,
            jsonl_count=jsonl_count,
            recording=bool(ctx.get("recording", True)),
            verify_status=self._verify_status,
            agent_id=agent_id,
            agent_count=agent_count,
            agent_label=agent_label,
            playback_error=playback_error,
            cycle_error=self._cycle_error,
            incomplete=incomplete,
            rbta_safe_ratio=safe_ratio,
        )
        if self._status_strip.text() != text:
            self._status_strip.setText(text)

    def refresh_session_strip(self, *, jsonl_count: Optional[int] = None) -> None:
        """Refresh status strip after production timers stop (hb_sync no longer runs)."""
        f = getattr(self, "_last_frame", None)
        transport = self._transport
        cursor = 0
        if transport is not None and getattr(transport, "clock", None) is not None:
            cursor = transport.clock.cursor_int
        jcount = jsonl_count
        if jcount is None:
            jcount = int(self._session_ctx.get("jsonl_count", 0) or 0)
        self.update_session_strip(f, jsonl_count=int(jcount), clock_cursor=cursor)

    def enter_review_mode(self, *, verify_status: str = "",
                          resync_only: bool = False) -> None:
        """Pause production; keep transport + render alive for post-run scrub.

        When ``resync_only=True`` (post-run entry), sync transport range/cursor
        without ``follow_live()`` emit — caller runs ``rebuild_all_histories`` once.
        """
        self._review_mode = True
        if verify_status:
            self.set_verify_status(verify_status)
        self.overview.set_review_mode(True)
        self.setWindowTitle(
            "PHCA Cognitive Observatory — review (close window to exit)")
        transport = self._transport
        if transport is not None and getattr(transport, "clock", None) is not None:
            transport.clock.review_mode = True
            transport.set_range(transport.clock.n)
            if resync_only:
                transport.clock.scrubbing = False
                if transport.clock.n > 0:
                    transport.clock._cursor = float(transport.clock.n - 1)
            else:
                transport.clock.follow_live()
            transport.clock.set_paused(True)
            transport.play_btn.setChecked(True)
            transport.play_btn.setText("Review")
            transport._sync_slider()
        self._rebuild_moment_matches()
        self.overview.mark_dirty()
        for cv in self.findChildren(_BaseCanvas):
            try:
                cv.mark_dirty()
            except Exception:
                pass

    def set_session_results(self, lines: Optional[List[str]]) -> None:
        self.overview.set_session_results(lines)

    def set_camera_provider(self, provider: Optional[Callable[[], Any]],
                            *, debug: bool = False,
                            mode: str = "auto",
                            glitch_profile: str = "default") -> None:
        self._camera_provider = provider
        self._camera_debug = bool(debug)
        self.overview.set_camera_provider(
            provider, debug=debug, mode=mode, glitch_profile=glitch_profile)
        if mode != "schematic":
            self.overview.start_camera_capture()

    def start_camera_capture(self) -> None:
        self.overview.start_camera_capture()

    def stop_camera_capture(self) -> None:
        self.overview.stop_camera_capture()

    def start_render(self, render_hz: Optional[float] = None) -> None:
        """Start the calm-render pacer (call after show())."""
        if render_hz is not None:
            self.render_pacer.set_hz(render_hz)
        self.render_pacer.start()

    def _apply_phase_layout(self, f: ObservabilityFrame) -> None:
        """Drive QGridLayout stretches and visibility from _phase_layout."""
        w = max(self._ps_tab.width(), 1)
        h = max(self._ps_tab.height(), 1)
        is_grid = _phase_frame_is_grid(f)
        lay = _phase_layout(w, h, getattr(f, "env_kind", "") or "", is_grid, tab_w=w)
        show_perdim = lay["show_perdim"]
        show_radar = lay["show_radar"]
        self.perdim.setVisible(show_perdim)
        self.dim_selector.setVisible(show_perdim)
        self.radar.setVisible(show_radar)
        gl = self._ps_lay
        if is_grid:
            gl.setRowStretch(0, 3)
            gl.setRowStretch(1, 3)
            gl.setRowStretch(2, 0)
            gl.setRowStretch(3, 0)
        else:
            gl.setRowStretch(0, 2)
            gl.setRowStretch(1, 2)
            gl.setRowStretch(2, 2)
            gl.setRowStretch(3, 0)
        if show_radar:
            radar_h = max(100, min(140, lay["traj_h"] // 3))
            self.radar.setFixedSize(lay["radar_w"], radar_h)
        if show_perdim:
            self.dim_selector.setFixedHeight(28)

    def update_phase_tab_status(self, f: ObservabilityFrame, *, review: bool = False,
                                prefix_len: int = 0) -> None:
        """Refresh unified Phase Space tab status strip."""
        ps = getattr(self, "_ps_tab", None)
        if ps is None or f is None:
            return
        line = phase_tab_status_line(
            f, self.proj, review=review, prefix_len=prefix_len)
        ps.update_status(line)

    def update_phase_panel_visibility(self, f: ObservabilityFrame) -> None:
        """Hide per-dim portrait on GridWorld; compact drive radar on narrow windows."""
        self._apply_phase_layout(f)

    def install_transport(self, bar: QtWidgets.QWidget) -> None:
        """Dock a transport bar at the top of the window (additive)."""
        self._top_layout.addWidget(bar, 1)
        self._top_frame.show()
        if isinstance(bar, _TransportBar):
            self._transport = bar
            bar.setup_moment_navigation(
                self._on_moment_filter_changed,
                self._jump_moment,
                on_cursor_change=self._rebuild_moment_matches,
            )
            if self._multi_agent:
                labels = [
                    self._agent_labels.get(aid, f"agent_{aid}") for aid in self._agent_ids
                ]
                bar.setup_agent_selector(labels, self.select_agent)

    def keyPressEvent(self, ev: QtCore.QEvent) -> None:
        """v8 keyboard transport: Space=pause, Left/Right=step, Home=seek 0,
        End=Esc=follow-live. Falls through to super when no transport."""
        t = self._transport
        if t is not None:
            k = ev.key()
            if k == QtCore.Qt.Key_Space:
                t.play_btn.toggle(); return
            if k == QtCore.Qt.Key_Right:
                t._on_step(); return
            if k == QtCore.Qt.Key_Left:
                # step the view cursor back one (scrub); cognition single-step
                # is forward-only by design.
                t.clock.seek(max(0, t.clock.cursor_int - 1))
                t._sync_slider(); set_autoscale_frozen(True); return
            if k == QtCore.Qt.Key_Home:
                t.clock.seek(0); t._sync_slider(); set_autoscale_frozen(True); return
            if k in (QtCore.Qt.Key_End, QtCore.Qt.Key_Escape):
                t.clock.follow_live(); t._sync_slider(); set_autoscale_frozen(False); return
            if k == QtCore.Qt.Key_BracketLeft:
                self._jump_moment(-1); return
            if k == QtCore.Qt.Key_BracketRight:
                self._jump_moment(1); return
        super().keyPressEvent(ev)


def make_app() -> "QtWidgets.QApplication":
    """Construct (but do not exec) the QApplication. Caller owns it."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(_qss())
    return app
