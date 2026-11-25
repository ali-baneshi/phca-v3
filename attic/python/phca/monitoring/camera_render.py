"""Robust RGB ndarray → Qt image/pixmap conversion.

Uses an owned RGB32 buffer so Qt never wraps a transient numpy view (which can
produce solid magenta/purple placeholders on some builds). PPM is kept as a
fallback when RGB32 packing fails.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui

from .observability import _normalize_rgb_frame

# Legacy tau-bar purple — detect glitched frames, not used for drawing.
_LEGACY_TAU_PURPLE = (155, 89, 182)

# Minimum per-channel variation for a plausible MuJoCo camera frame.
_MIN_CAMERA_STD = 8.0


def camera_frame_stats(frame: Any) -> dict:
    """Compact diagnostics for camera debug logs."""
    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return {
            "shape": None,
            "std": None,
            "mean_rgb": None,
            "green_frac": None,
            "glitch_frac": None,
            "glitchy": True,
        }
    return {
        "shape": tuple(arr.shape),
        "std": float(arr.std()),
        "mean_rgb": arr.mean(axis=(0, 1)).tolist(),
        "green_frac": _green_dominance_frac(arr),
        "glitch_frac": _glitch_fraction_rgb(arr),
        "glitchy": False,
    }


def _glitch_fraction_rgb(arr: np.ndarray, *, tol: int = 35) -> float:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    magenta = (r > 250) & (b > 250) & (g < 10)
    pr, pg, pb = _LEGACY_TAU_PURPLE
    purple = (
        (np.abs(r - pr) < tol)
        & (np.abs(g - pg) < tol)
        & (np.abs(b - pb) < tol)
    )
    total = int(r.size)
    if total == 0:
        return 1.0
    return float((magenta | purple).sum()) / total


def _green_dominance_frac(arr: np.ndarray) -> float:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    green = (g > 100) & (g > r + 25) & (g > b + 25)
    n = max(int(r.size), 1)
    return float(green.sum()) / n


def is_uniform_rgb_frame(frame: Any, *, min_spatial_std: float = _MIN_CAMERA_STD,
                         same_pixel_frac: float = 0.92) -> bool:
    """True when the frame is a solid GL clear colour (green/red/orange slab)."""
    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return True
    flat = arr.reshape(-1, 3)
    if flat.shape[0] == 0:
        return True
    # Solid GL failure: almost every pixel is the exact same RGB triple.
    ref = flat[0]
    if float(np.all(flat == ref, axis=1).mean()) >= same_pixel_frac:
        return True
    # Real camera: mean luminance varies across the image.
    gray = arr.astype(np.float32).mean(axis=2)
    if float(gray.std()) < min_spatial_std:
        return True
    return _is_gl_clear_slab(arr)


def _is_gl_clear_slab(arr: np.ndarray) -> bool:
    """Detect dominant green/red GL clear screens (may have slight noise)."""
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    green = (g > 120) & (g > r + 30) & (g > b + 30)
    red = (r > 120) & (r > g + 30) & (r > b + 30)
    blue = (b > 120) & (b > r + 30) & (b > g + 30)
    n = max(int(r.size), 1)
    if float(green.sum()) / n > 0.70:
        return True
    if float(red.sum()) / n > 0.70:
        return True
    if float(blue.sum()) / n > 0.70:
        return True
    return False


def is_glitchy_rgb_frame(frame: Any, *, min_glitch_frac: float = 0.15,
                         profile: str = "default") -> bool:
    """True when the frame is unusable (purple/magenta placeholder or uniform GL)."""
    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return True
    p = (profile or "default").lower()
    min_spatial_std = 5.0 if p in ("pendulum", "pendulum-v1") else _MIN_CAMERA_STD
    green_dom_thresh = 0.62 if p in ("pendulum", "pendulum-v1") else 0.45
    glitch_frac = _glitch_fraction_rgb(arr)
    if glitch_frac >= min_glitch_frac:
        return True
    if float(arr.std()) < 2.0 and glitch_frac > 0.05:
        return True
    if is_uniform_rgb_frame(arr, min_spatial_std=min_spatial_std):
        return True
    # EGL failure: mostly-green slab that still has channel variance.
    if _green_dominance_frac(arr) > green_dom_thresh:
        mean = arr.mean(axis=(0, 1))
        if float(mean[1]) > float(mean[0]) + 35 and float(mean[1]) > float(mean[2]) + 35:
            return True
    return False


def is_glitchy_pixmap(pixmap: QtGui.QPixmap, *, min_glitch_frac: float = 0.15) -> bool:
    """True when a converted pixmap still looks like a Qt placeholder slab."""
    if pixmap is None or pixmap.isNull():
        return True
    if pixmap_looks_like_green_slab(pixmap):
        return True
    img = pixmap.toImage()
    if img.isNull() or img.width() < 2 or img.height() < 2:
        return True
    w, h = img.width(), img.height()
    step_x = max(1, w // 16)
    step_y = max(1, h // 16)
    pr, pg, pb = _LEGACY_TAU_PURPLE
    tol = 35
    glitch = 0
    total = 0
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            c = img.pixelColor(x, y)
            total += 1
            r, g, b = c.red(), c.green(), c.blue()
            if r > 250 and b > 250 and g < 10:
                glitch += 1
            elif abs(r - pr) < tol and abs(g - pg) < tol and abs(b - pb) < tol:
                glitch += 1
    if total == 0:
        return True
    return (glitch / total) >= min_glitch_frac


def pixmap_looks_like_green_slab(pixmap: QtGui.QPixmap, *, green_frac: float = 0.45) -> bool:
    """Fast sample: True when pixmap pixels are mostly dominant green (EGL clear)."""
    if pixmap is None or pixmap.isNull():
        return False
    img = pixmap.toImage()
    if img.isNull() or img.width() < 2 or img.height() < 2:
        return False
    w, h = img.width(), img.height()
    step_x = max(1, w // 12)
    step_y = max(1, h // 12)
    green = 0
    total = 0
    sum_g = sum_r = sum_b = 0.0
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            c = img.pixelColor(x, y)
            r, g, b = c.red(), c.green(), c.blue()
            total += 1
            sum_r += r
            sum_g += g
            sum_b += b
            if g > 100 and g > r + 25 and g > b + 25:
                green += 1
    if total == 0:
        return False
    if green / total > green_frac:
        return True
    mean_r = sum_r / total
    mean_g = sum_g / total
    mean_b = sum_b / total
    return mean_g > mean_r + 35 and mean_g > mean_b + 35 and green / total > 0.35


def _rgb32_qimage_from_array(arr: np.ndarray) -> QtGui.QImage:
    """Pack uint8 RGB into Format_RGB32 with a Qt-owned buffer."""
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.uint32)
    g = arr[:, :, 1].astype(np.uint32)
    b = arr[:, :, 2].astype(np.uint32)
    packed = (0xFF000000 | (r << 16) | (g << 8) | b).astype(np.uint32)
    packed = np.ascontiguousarray(packed)
    qimg = QtGui.QImage(
        packed.data, w, h, 4 * w, QtGui.QImage.Format_RGB32,
    )
    return qimg.copy()


def _ppm_qimage_from_array(arr: np.ndarray) -> QtGui.QImage:
    h, w, _ = arr.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return QtGui.QImage.fromData(header + arr.tobytes(), "PPM")


def rgb_frame_to_qimage(frame: Any) -> QtGui.QImage:
    """Coerce frame to QImage (PPM first on Linux, then RGB32 + RGB888 fallbacks)."""
    import sys

    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return QtGui.QImage()
    if sys.platform.startswith("linux"):
        try:
            qimg = _ppm_qimage_from_array(arr)
            if not qimg.isNull():
                return qimg
        except Exception:
            pass
    try:
        qimg = _rgb32_qimage_from_array(arr)
        if not qimg.isNull():
            return qimg
    except Exception:
        pass
    qimg = _ppm_qimage_from_array(arr)
    if not qimg.isNull():
        return qimg
    ch = arr.shape[2]
    qimg = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], ch * arr.shape[1],
                        QtGui.QImage.Format_RGB888)
    return qimg.copy()


def rgb_frame_to_pixmap(frame: Any, *, already_checked: bool = False) -> QtGui.QPixmap:
    """Convert a numpy RGB frame to QPixmap (glitch check on numpy + pixmap)."""
    if not already_checked and is_glitchy_rgb_frame(frame):
        return QtGui.QPixmap()
    qimg = rgb_frame_to_qimage(frame)
    if qimg.isNull():
        return QtGui.QPixmap()
    pm = QtGui.QPixmap.fromImage(qimg)
    if pm.isNull() or is_glitchy_pixmap(pm) or pixmap_looks_like_green_slab(pm):
        return QtGui.QPixmap()
    return pm


def fit_pixmap_to_box(
    pixmap: QtGui.QPixmap, max_w: int, max_h: int,
) -> Tuple[QtGui.QPixmap, int, int]:
    if pixmap.isNull() or max_w < 2 or max_h < 2:
        return QtGui.QPixmap(), 0, 0
    scaled = pixmap.scaled(
        max_w, max_h,
        QtCore.Qt.KeepAspectRatio,
        QtCore.Qt.SmoothTransformation,
    )
    return scaled, scaled.width(), scaled.height()
