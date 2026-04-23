"""Histogram equalization: full image (gray) or Y channel in YUV (RGB color)"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils import is_grayscale, squeeze_gray


def _global_histogram_equalize_uint8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.uint8)
    n = g.size
    if n == 0:
        return g.copy()
    hist = np.bincount(g.ravel().astype(np.uint16), minlength=256)
    cdf = np.cumsum(hist, dtype=np.float64)
    n_f = float(cdf[-1])
    if n_f <= 0.0:
        return g.copy()
    l_minus_1 = 255.0
    lut = np.rint(l_minus_1 * cdf / n_f).clip(0, 255).astype(np.uint8)
    return lut[g]


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if is_grayscale(image):
        g = squeeze_gray(image)
        out = _global_histogram_equalize_uint8(g)
        return out if g.ndim == 2 else out[:, :, np.newaxis]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    yuv[:, :, 0] = _global_histogram_equalize_uint8(y)
    bgr2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
