"""Qt transport bar for Observatory playback (extracted from qt_dashboard)."""
from __future__ import annotations

import math
from typing import Callable, List, Optional

from PyQt5 import QtCore, QtWidgets

from phca.monitoring.belief_projection import set_autoscale_frozen
from phca.monitoring.cognitive_panels import FLOW_ALL_MODULES, PIPELINE_LABEL
from phca.monitoring.playback import throttle_period


class TransportBar(QtWidgets.QFrame):
    """v8 transport: pause-stops-the-world (default) + continuous slow-mo +
    single-step-cognition + scrub + follow-live.

    Governs a ``PlaybackClock`` (view cursor) and, when a ``CyclePacer`` is
    present (live runs), the real cycle thread."""

    _SMIN, _SMAX = 0.05, 2.0

    def __init__(self, clock, pacer=None, parent=None):
        super().__init__(parent)
        self.clock = clock
        self.pacer = pacer
        self._on_agent_change = None
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        self.agent_combo = QtWidgets.QComboBox()
        self.agent_combo.setMinimumWidth(120)
        self.agent_combo.hide()
        self.agent_combo.currentIndexChanged.connect(self._on_agent_combo)
        lay.addWidget(self.agent_combo, 0)
        self.moment_filter = QtWidgets.QComboBox()
        self.moment_filter.setMinimumWidth(100)
        self.moment_filter.addItems([
            "All", "Spike", "Violation", "Explore", "Decision", "Learn burst", "Near bound",
        ])
        self.moment_module = QtWidgets.QComboBox()
        self.moment_module.setMinimumWidth(72)
        for mod in FLOW_ALL_MODULES:
            self.moment_module.addItem(PIPELINE_LABEL.get(mod, mod), mod)
        self.moment_module.hide()
        self.moment_prev = QtWidgets.QToolButton()
        self.moment_prev.setText("◀ moment")
        self.moment_next = QtWidgets.QToolButton()
        self.moment_next.setText("moment ▶")
        self.moment_lbl = QtWidgets.QLabel("—")
        self.moment_lbl.setMinimumWidth(48)
        self._on_moment_filter = None
        self._on_moment_jump = None
        self._on_cursor_change = None
        lay.addWidget(self.moment_filter, 0)
        lay.addWidget(self.moment_module, 0)
        lay.addWidget(self.moment_prev, 0)
        lay.addWidget(self.moment_next, 0)
        lay.addWidget(self.moment_lbl, 0)
        self.moment_filter.currentIndexChanged.connect(self._on_moment_filter_changed)
        self.moment_module.currentIndexChanged.connect(self._on_moment_filter_changed)
        self.moment_prev.clicked.connect(lambda: self._on_moment_jump_click(-1))
        self.moment_next.clicked.connect(lambda: self._on_moment_jump_click(1))
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setText("⏸ Pause")
        self.play_btn.setCheckable(True)
        self.step_btn = QtWidgets.QToolButton()
        self.step_btn.setText("⏭ Step")
        self.follow_btn = QtWidgets.QToolButton()
        self.follow_btn.setText("⤓ Live")
        self._log_lo = math.log(self._SMIN)
        self._log_hi = math.log(self._SMAX)
        self.speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed.setMinimum(0)
        self.speed.setMaximum(1000)
        self.speed.setValue(self._speed_to_int(1.0))
        self.speed.setMaximumWidth(140)
        self.speed_lbl = QtWidgets.QLabel("1×")
        self.speed_lbl.setMinimumWidth(54)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.label = QtWidgets.QLabel("0 / 0")
        self.label.setMinimumWidth(90)
        self.freeze_view = QtWidgets.QCheckBox("freeze view only")
        self.freeze_view.setToolTip(
            "When ON, Pause/speed affect only the VIEW "
            "(the cognition keeps running). Default OFF: "
            "Pause stops the cognition entirely."
        )
        for w in (
            self.play_btn,
            self.step_btn,
            self.follow_btn,
            self.speed,
            self.speed_lbl,
            self.slider,
            self.label,
        ):
            lay.addWidget(w)
        lay.addWidget(self.freeze_view, 0)
        self.play_btn.toggled.connect(self._on_play)
        self.step_btn.clicked.connect(self._on_step)
        self.follow_btn.clicked.connect(self._on_follow)
        self.speed.valueChanged.connect(self._on_speed)
        self.slider.valueChanged.connect(self._on_seek)
        if pacer is None:
            self.freeze_view.hide()
        else:
            self.freeze_view.toggled.connect(self._on_freeze_view)

    def setup_agent_selector(
        self,
        labels: List[str],
        on_change: Callable[[int], None],
    ) -> None:
        self._on_agent_change = on_change
        self.agent_combo.blockSignals(True)
        self.agent_combo.clear()
        for i, lbl in enumerate(labels):
            text = lbl or f"agent_{i}"
            self.agent_combo.addItem(f"{i}: {text}", i)
        self.agent_combo.blockSignals(False)
        self.agent_combo.show()

    def _on_agent_combo(self, index: int) -> None:
        if self._on_agent_change is None or index < 0:
            return
        aid = self.agent_combo.itemData(index)
        if aid is not None:
            self._on_agent_change(int(aid))

    def set_agent_index(self, agent_id: int) -> None:
        for i in range(self.agent_combo.count()):
            if int(self.agent_combo.itemData(i)) == int(agent_id):
                self.agent_combo.blockSignals(True)
                self.agent_combo.setCurrentIndex(i)
                self.agent_combo.blockSignals(False)
                break

    def setup_moment_navigation(
        self,
        on_filter_change: Callable[[], None],
        on_jump: Callable[[int], None],
        on_cursor_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_moment_filter = on_filter_change
        self._on_moment_jump = on_jump
        self._on_cursor_change = on_cursor_change
        self._sync_moment_nav_enabled()

    def _on_moment_filter_changed(self, *_args) -> None:
        key = self.moment_filter.currentText()
        self.moment_module.setVisible(key == "Near bound")
        if self._on_moment_filter is not None:
            self._on_moment_filter()

    def _on_moment_jump_click(self, direction: int) -> None:
        if self._on_moment_jump is not None:
            self._on_moment_jump(int(direction))

    def moment_query_from_ui(self):
        from phca.monitoring.session_query import MomentQuery

        key = self.moment_filter.currentText()
        q = MomentQuery()
        if key == "Spike":
            q.spike = True
        elif key == "Violation":
            q.violation = True
        elif key == "Explore":
            q.explore = True
        elif key == "Decision":
            q.decision_shift = True
        elif key == "Learn burst":
            q.learn_burst = True
        elif key == "Near bound":
            q.near_bound_module = str(self.moment_module.currentData() or "")
        return q

    def update_moment_counter(self, pos: int, total: int) -> None:
        if total <= 0:
            self.moment_lbl.setText("—")
        else:
            self.moment_lbl.setText(f"{pos}/{total}")

    def set_moment_nav_enabled(self, enabled: bool) -> None:
        for w in (
            self.moment_filter,
            self.moment_module,
            self.moment_prev,
            self.moment_next,
        ):
            w.setEnabled(bool(enabled))
        if not enabled:
            self.moment_lbl.setText("—")
        else:
            self.moment_module.setVisible(
                self.moment_filter.currentText() == "Near bound"
            )

    def _sync_moment_nav_enabled(self) -> None:
        pass

    def _speed_to_int(self, s: float) -> int:
        s = max(self._SMIN, min(self._SMAX, float(s)))
        t = (math.log(s) - self._log_lo) / (self._log_hi - self._log_lo)
        return int(round(t * 1000))

    def _int_to_speed(self, v: int) -> float:
        t = max(0.0, min(1.0, v / 1000.0))
        return math.exp(self._log_lo + t * (self._log_hi - self._log_lo))

    def set_range(self, n: int) -> None:
        self.slider.setMaximum(max(0, n - 1))

    def _cognition_active(self) -> bool:
        return self.pacer is not None and not self.freeze_view.isChecked()

    def _on_play(self, checked: bool) -> None:
        self.play_btn.setText("▶ Play" if checked else "⏸ Pause")
        self.clock.set_paused(checked)
        set_autoscale_frozen(checked)
        if self._cognition_active():
            self.pacer.set_paused(checked)
        elif self.pacer is not None and self.freeze_view.isChecked():
            self.pacer.set_paused(False)

    def _on_step(self) -> None:
        if self._cognition_active() and self.play_btn.isChecked():
            self.pacer.step_once()
        else:
            self.clock.step()
            self._sync_slider()
            set_autoscale_frozen(True)

    def _on_follow(self) -> None:
        self.clock.follow_live()
        self._sync_slider()
        set_autoscale_frozen(False)

    def _on_speed(self, val: int) -> None:
        s = self._int_to_speed(val)
        self.speed_lbl.setText(f"{s:g}×")
        self.clock.set_speed(s)
        if self._cognition_active():
            self.pacer.set_period(throttle_period(s))

    def _on_seek(self, val: int) -> None:
        self.clock.seek(val)
        self.label.setText(f"{val} / {max(0, self.clock.n - 1)}")
        set_autoscale_frozen(True)
        if self._on_cursor_change is not None:
            self._on_cursor_change()

    def _on_freeze_view(self, on: bool) -> None:
        if self.pacer is None:
            return
        if on:
            self.pacer.set_paused(False)
            self.pacer.set_period(0.0)
            set_autoscale_frozen(self.play_btn.isChecked())
        else:
            self.pacer.set_paused(self.play_btn.isChecked())
            self.pacer.set_period(
                throttle_period(self._int_to_speed(self.speed.value()))
            )

    def _sync_slider(self) -> None:
        i = self.clock.cursor_int
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.label.setText(f"{i} / {max(0, self.clock.n - 1)}")


# Legacy private alias used by qt_dashboard and scripts.
_TransportBar = TransportBar
