"""Robust RGB ndarray → Qt image/pixmap conversion.

Uses an owned RGB32 buffer so Qt never wraps a transient numpy view (which can
produce solid magenta/purple placeholders on some builds). PPM is kept as a
fallback when RGB32 packing fails.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui

from .observability import _normalize_rgb_frame

# Legacy tau-bar purple — detect glitched frames, not used for drawing.
_LEGACY_TAU_PURPLE = (155, 89, 182)

# Minimum per-channel variation for a plausible MuJoCo camera frame.
_MIN_CAMERA_STD = 8.0


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


def is_glitchy_rgb_frame(frame: Any, *, min_glitch_frac: float = 0.15) -> bool:
    """True when the frame is unusable (purple/magenta placeholder or uniform GL)."""
    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return True
    glitch_frac = _glitch_fraction_rgb(arr)
    if glitch_frac >= min_glitch_frac:
        return True
    if float(arr.std()) < 2.0 and glitch_frac > 0.05:
        return True
    if is_uniform_rgb_frame(arr):
        return True
    # EGL failure: mostly-green slab that still has channel variance.
    if _green_dominance_frac(arr) > 0.45:
        mean = arr.mean(axis=(0, 1))
        if float(mean[1]) > float(mean[0]) + 35 and float(mean[1]) > float(mean[2]) + 35:
            return True
    return False


def is_glitchy_pixmap(pixmap: QtGui.QPixmap, *, min_glitch_frac: float = 0.15) -> bool:
    """True when a converted pixmap still looks like a Qt placeholder slab."""
    if pixmap is None or pixmap.isNull():
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
    """Coerce frame to QImage via owned RGB32 (PPM + RGB888 fallbacks)."""
    arr = _normalize_rgb_frame(frame)
    if arr is None:
        return QtGui.QImage()
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


def rgb_frame_to_pixmap(frame: Any) -> QtGui.QPixmap:
    """Convert a numpy RGB frame to QPixmap (glitch check on numpy only)."""
    if is_glitchy_rgb_frame(frame):
        return QtGui.QPixmap()
    qimg = rgb_frame_to_qimage(frame)
    if qimg.isNull():
        return QtGui.QPixmap()
    return QtGui.QPixmap.fromImage(qimg)


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
