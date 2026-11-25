"""Qt-free belief-space projection helpers (PCA / autoscale)."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from phca.monitoring.observability import ObservabilityFrame

_AUTOSCALE_FROZEN: bool = False


def set_autoscale_frozen(frozen: bool) -> None:
    global _AUTOSCALE_FROZEN
    _AUTOSCALE_FROZEN = bool(frozen)


class ScaleState:
    """Stable autoscale bounds with hysteresis + EMA contraction."""

    __slots__ = ("lo", "hi", "_ema_lo", "_ema_hi", "contract", "head", "_have",
                 "_last_rlo", "_last_rhi")

    def __init__(self, contract: float = 0.05, head: float = 0.05):
        self.lo: float = 0.0
        self.hi: float = 1.0
        self._ema_lo: float = 0.0
        self._ema_hi: float = 1.0
        self.contract = contract
        self.head = head
        self._have: bool = False
        self._last_rlo: Optional[float] = None
        self._last_rhi: Optional[float] = None

    def reset(self) -> None:
        self.lo = 0.0
        self.hi = 1.0
        self._ema_lo = 0.0
        self._ema_hi = 1.0
        self._have = False
        self._last_rlo = None
        self._last_rhi = None

    def update(self, rlo: float, rhi: float) -> Tuple[float, float]:
        if _AUTOSCALE_FROZEN:
            return self.lo, self.hi
        rlo_f, rhi_f = float(rlo), float(rhi)
        if (self._have and self._last_rlo is not None and self._last_rhi is not None
                and abs(rlo_f - self._last_rlo) < 1e-12
                and abs(rhi_f - self._last_rhi) < 1e-12):
            return self.lo, self.hi
        self._last_rlo = rlo_f
        self._last_rhi = rhi_f
        if not self._have:
            self._ema_lo = rlo_f
            self._ema_hi = rhi_f
            self.lo = rlo_f
            self.hi = rhi_f
            self._have = True
        else:
            self._ema_lo += (rlo_f - self._ema_lo) * 0.2
            self._ema_hi += (rhi_f - self._ema_hi) * 0.2
            self.lo = min(rlo_f, self.lo + (self._ema_lo - self.lo) * self.contract)
            self.hi = max(rhi_f, self.hi + (self._ema_hi - self.hi) * self.contract)
        span = (self.hi - self.lo) or 1.0
        self.lo -= span * self.head
        self.hi += span * self.head
        if self.hi - self.lo < 1e-9:
            self.hi = self.lo + 1.0
        return self.lo, self.hi


class BeliefProjection:
    """Rolling PCA 2-D projection of the state space."""

    def __init__(self, window: int = 256):
        self.window = window
        self._buf: Deque[np.ndarray] = deque(maxlen=window)
        self._mean: Optional[np.ndarray] = None
        self._comp: Optional[np.ndarray] = None
        self._top_dims: Optional[np.ndarray] = None
        self._fallback: Tuple[int, int] = (0, min(1, 0))
        self._raw2d: bool = False
        self._hist: Deque[Tuple[float, float]] = deque(maxlen=window)
        self._bx = ScaleState(contract=0.04, head=0.08)
        self._by = ScaleState(contract=0.04, head=0.08)
        self._sing: Optional[np.ndarray] = None
        self._prev_comp: Optional[np.ndarray] = None
        self._basis_changed: bool = False
        self._live_skip: int = -1         # C1: only recompute SVD every 5 live frames (starts at -1 so first eligible increment → 0 → triggers recompute)

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
            self._live_skip += 1
            if self._live_skip % 5 == 0:
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
        var = (M - mean).var(axis=0)
        idx = np.argsort(var)[-2:]
        self._fallback = (int(idx[0]), int(idx[1]))
        self._comp = None
        self._top_dims = None
        self._mean = mean
        self._sing = None

    def variance_explained(self) -> Optional[float]:
        if self._sing is None or self._sing.size == 0:
            return None
        tot = float(self._sing.sum())
        if tot <= 0:
            return None
        return float(self._sing[:2].sum() / tot * 100.0)

    @property
    def basis_changed(self) -> bool:
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
        i0, i1 = self._fallback
        if i1 >= v.size:
            i1 = 0
        return (float(v[i0] - self._mean[i0]), float(v[i1] - self._mean[i1]))

    def push_history(self, pt: Optional[Tuple[float, float]]) -> None:
        if pt is not None:
            self._hist.append(pt)

    def rebuild_from_frames(self, frames: List[ObservabilityFrame]) -> None:
        self._buf.clear()
        self._hist.clear()
        self._mean = None
        self._comp = None
        self._top_dims = None
        self._raw2d = False
        self._sing = None
        self._basis_changed = False
        self._bx = ScaleState(contract=0.04, head=0.08)
        self._by = ScaleState(contract=0.04, head=0.08)
        for f in frames:
            v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            if v is None:
                continue
            try:
                v = np.asarray(v, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if v.size == 0 or not np.all(np.isfinite(v)):
                continue
            self._buf.append(v)
        if len(self._buf) >= 4:
            self._recompute()
        elif self._buf:
            M = np.array(list(self._buf), dtype=np.float64)
            self._mean = M.mean(axis=0)
            if M.shape[1] <= 3:
                self._raw2d = True
                self._comp = None
        for f in frames:
            v = f.sanitized_state if f.sanitized_state is not None else f.obs_vector
            self.push_history(self.project(v))

    @property
    def history(self) -> List[Tuple[float, float]]:
        return list(self._hist)

    def bounds(self) -> Tuple[float, float, float, float]:
        pts = self._hist
        if not pts:
            return (-1.0, 1.0, -1.0, 1.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xlo, xhi = self._bx.update(float(min(xs)), float(max(xs)))
        ylo, yhi = self._by.update(float(min(ys)), float(max(ys)))
        return (xlo, xhi, ylo, yhi)

    def uncertainty_ellipse(self, per_dim_std: Optional[np.ndarray]
                            ) -> Optional[Tuple[Tuple[float, float], float]]:
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
        eigval = eigval[order]
        eigvec = eigvec[:, order]
        import math
        axes = (float(np.sqrt(max(eigval[0], 0.0))),
                float(np.sqrt(max(eigval[1], 0.0))))
        angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
        return (axes, angle)
