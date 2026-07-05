"""Observability v6 — playback transport primitives.

Three small, dependency-light classes that decouple the *display* of cognitive
state from the *production* of cycles, and give the user a transport
(pause / variable speed / step / scrub) over both the live view and replays.

- ``CyclePacer``   : throttles/pauses the cognitive cycle THREAD (opt-in).
- ``PlaybackClock``: a heartbeat-driven cursor over a frame buffer that drives
                     ``DashboardController.update`` at a controllable speed.
- ``_Smoother``    : EMA smoother for fast-changing scalars (kills bar jitter).

Pure main-thread Qt for the clock; pure threading for the pacer. Neither touches
core cognitive behaviour — the pacer is a no-op at full speed.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional


class _Smoother:
    """Exponential moving average for one scalar signal.

    Used to glide fast-changing bar/gauge values instead of letting them jump
    per heartbeat. ``alpha`` near 1 = barely smooth; near 0 = very smooth.
    """

    __slots__ = ("alpha", "_v", "_have")

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self._v: float = 0.0
        self._have: bool = False

    def value(self, v: Optional[float]) -> float:
        if v is None:
            return self._v if self._have else 0.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return self._v if self._have else 0.0
        if not self._have:
            self._v = fv
            self._have = True
        else:
            self._v += (fv - self._v) * self.alpha
        return self._v

    def reset(self) -> None:
        self._have = False
        self._v = 0.0


def freeze_sig(values) -> str:
    """Stable content signature for discrete visuals (lists/tables/badges/dim
    labels) so they repaint only when contents actually change — killing the
    per-frame flicker on M3/M4 lists etc. Tolerates dicts/ndarrays/scalars."""
    import hashlib
    import json
    try:
        import numpy as _np
        def _def(o):
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return str(o)
    except Exception:
        def _def(o):
            return str(o)
    try:
        return hashlib.md5(
            json.dumps(values, default=_def, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except Exception:
        return repr(values)


class CyclePacer:
    """Throttle / pause the cognitive cycle thread.

    ``wait()`` is called once per ``cycle.step()``. At full speed
    (``period == 0`` and not paused) it is essentially a no-op (just records the
    timestamp). When ``period > 0`` it sleeps so steps are spaced at >= period.
    When ``paused`` it blocks until unpaused or ``stop()`` is called.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._period: float = 0.0          # seconds between steps; 0 = full speed
        self._paused: bool = False
        self._stop: bool = False
        self._last = time.monotonic()
        # v8: one-shot gate for single-step-cognition (released by step_once()).
        self._step_once: int = 0

    def set_period(self, seconds: float) -> None:
        with self._lock:
            self._period = max(0.0, float(seconds))

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)

    def step_once(self) -> None:
        """v8: release exactly one blocked ``wait()`` (single-step cognition).
        Idempotent: while paused, calling this lets the next ``cycle.step()``
        through, then pauses again."""
        with self._lock:
            self._step_once += 1

    def stop(self) -> None:
        with self._lock:
            self._stop = True

    def wait(self) -> None:
        # Pause: block until unpaused, stopped, or a one-shot step token exists.
        while True:
            with self._lock:
                if self._stop:
                    return
                if self._step_once > 0:
                    self._step_once -= 1
                    break  # allow one step through, then re-loop to re-check pause
                if not self._paused:
                    break
            time.sleep(0.02)
        if self._stop:
            return
        p = self._period
        now = time.monotonic()
        if p > 0.0:
            dt = now - self._last
            if dt < p:
                time.sleep(p - dt)
        self._last = time.monotonic()


# v8: continuous slow-mo range exported for the transport slider.
_SPEED_MIN, _SPEED_MAX = 0.05, 2.0




class PlaybackClock:
    """Heartbeat-driven cursor over a frame buffer.

    Mode ``live``: frames are appended via :meth:`push`; the cursor catches up
    toward the newest frame at ``speed`` per heartbeat (1x = follow live,
    <1 = smooth slow-mo lag, >1 = snap to latest). Scrubbing seeks backward and
    freezes; :meth:`follow_live` re-attaches to the live tail.

    Mode ``replay``: frames are set once via :meth:`set_frames`; the cursor
    advances ``speed`` frames per heartbeat (1x = heartbeat_fps; 2x/4x skip).

    The clock calls ``on_update(frame, rolling, error)`` on each heartbeat when
    not paused/scrubbing. ``on_end`` is called once when a replay reaches the
    end. All Qt-less so it is cheap to drive from a QTimer in the caller.
    """

    def __init__(self, heartbeat_hz: float = 10.0, mode: str = "live",
                 maxlen: Optional[int] = None):
        self.mode = mode
        self._maxlen = int(maxlen) if maxlen is not None and maxlen > 0 else None
        self._frames: List = []
        self._cursor: float = 0.0
        self.speed: float = 1.0
        self.paused: bool = False
        self.scrubbing: bool = False
        self.error: Optional[str] = None
        self._last_t = time.monotonic()
        self._heartbeat_dt = 1.0 / max(float(heartbeat_hz), 0.1)
        self.on_update: Optional[Callable] = None
        self.on_end: Optional[Callable] = None
        self._last_emitted: Optional[int] = None
        self.review_mode: bool = False

    # --- buffer -------------------------------------------------------------
    def push(self, f) -> None:
        """Append a newly produced frame (live mode)."""
        self._frames.append(f)
        if self._maxlen is not None and len(self._frames) > self._maxlen:
            drop = len(self._frames) - self._maxlen
            self._frames = self._frames[drop:]
            self._cursor = max(0.0, self._cursor - drop)

    def set_frames(self, frames: List) -> None:
        """Load a fixed frame list (replay mode)."""
        self._frames = list(frames)
        self._cursor = 0.0
        self.scrubbing = False
        self.paused = False
        self._last_emitted = None

    def reload_frames(
        self,
        frames: List,
        *,
        preserve_transport: bool = False,
        follow_live: bool = False,
    ) -> None:
        """Replace buffer contents without disturbing transport when requested.

        ``preserve_transport=False`` resets transport like :meth:`set_frames`.
        When ``preserve_transport=True``, ``paused``/``scrubbing``/``speed`` are
        kept; the cursor is clamped to the new length. If ``follow_live`` is set
        and the clock is not paused/scrubbing, the cursor snaps to the newest
        frame (live multi-agent poll path).
        """
        if not preserve_transport:
            self.set_frames(frames)
            return
        old_cursor = self.cursor_int if self._frames else 0
        self._frames = list(frames)
        self._last_emitted = None
        if not self._frames:
            self._cursor = 0.0
            return
        if follow_live and not self.paused and not self.scrubbing:
            self._cursor = float(self.n - 1)
        else:
            self._cursor = float(min(old_cursor, self.n - 1))

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def n(self) -> int:
        return len(self._frames)

    @property
    def cursor_int(self) -> int:
        if not self._frames:
            return 0
        return int(max(0, min(self._cursor, self.n - 1)))

    # --- transport controls -------------------------------------------------
    def set_speed(self, s: float) -> None:
        self.speed = float(s)

    def set_paused(self, p: bool) -> None:
        self.paused = bool(p)
        if p:
            self.scrubbing = False
        else:
            # un-pause (play) resumes following from the cursor → exit scrub mode.
            self.scrubbing = False

    def seek(self, i: int) -> None:
        if not self._frames:
            return
        self._cursor = float(max(0, min(i, self.n - 1)))
        self.scrubbing = True
        self._emit(force_rebuild=True)

    def step(self) -> None:
        """Advance one frame (used while paused)."""
        if not self._frames:
            return
        self._cursor = float(min(self._cursor + 1, self.n - 1))
        self._emit()

    def follow_live(self) -> None:
        """Re-attach to the live tail (exit scrub)."""
        self.scrubbing = False
        if self._frames:
            self._cursor = float(self.n - 1)
            self._emit(force_rebuild=True)

    # --- heartbeat ----------------------------------------------------------
    def tick(self) -> None:
        """Advance the cursor by one heartbeat and emit the current frame."""
        if not self._frames:
            return
        if self.paused or self.scrubbing:
            return  # frozen; change-detection in canvases skips repaint anyway
        now = time.monotonic()
        self._last_t = now
        if self.mode == "live":
            newest = self.n - 1
            gap = newest - self._cursor
            self._cursor += self.speed * gap
            if self._cursor > newest:
                self._cursor = float(newest)
        else:  # replay: linear advance of `speed` frames per heartbeat
            self._cursor += max(1.0, self.speed) if self.speed >= 1.0 else self.speed
            if self._cursor >= self.n - 1:
                self._cursor = float(self.n - 1)
                self._emit()
                if self.on_end is not None:
                    try:
                        self.on_end()
                    except Exception:
                        pass
                self.paused = True
                return
        self._emit()

    def _emit(self, *, force_rebuild: bool = False) -> None:
        if self.on_update is None or not self._frames:
            return
        i = self.cursor_int
        f = self._frames[i]
        if force_rebuild:
            rebuild = True
        elif self._last_emitted is None:
            rebuild = True
        elif i == self._last_emitted:
            rebuild = False
        else:
            rebuild = i != self._last_emitted + 1
        if rebuild:
            from phca.monitoring.cognitive_panels import rolling_prefix
            rolling = rolling_prefix(
                self._frames, i, review_mode=self.review_mode)
        else:
            rolling = None
        try:
            self.on_update(f, rolling, self.error)
            self._last_emitted = i
        except Exception as exc:
            import sys
            msg = f"update failed: {exc}"
            self.error = msg if self.error is None else f"{self.error}; {msg}"
            print(f"[observatory] {msg}", file=sys.stderr)


def throttle_period(speed: float) -> float:
    """v8: continuous slow-mo. See module docstring above."""
    s = max(float(speed), 1e-3)
    period = (1.0 / s - 1.0) * 0.2
    return max(0.0, min(period, 4.0))
