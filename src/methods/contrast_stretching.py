"""Contrast stretching: per-channel (RGB) or global (grayscale) min-max to [0, 255]."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils import EPS, is_grayscale, to_uint8


def contrast_stretch(
    image: np.ndarray,
    per_channel: bool = True,
) -> np.ndarray:
    """Stretch contrast to use full 0--255 range.

    Grayscale: uses min/max of the 2D array. RGB with `per_channel=True` (default)
    stretches each channel independently. Set `per_channel=False` to stretch only
    the luminance (Y) in YCrCb, keeping chroma.
    """
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if is_grayscale(image):
        g = image if image.ndim == 2 else image[:, :, 0]
        lo = float(g.min())
        hi = float(g.max())
        if hi - lo < EPS:
            return image.copy() if image.ndim == 2 else image.copy()
        s = (g.astype(np.float64) - lo) * 255.0 / (hi - lo + EPS)
        out = to_uint8(s)
        return out if image.ndim == 2 else out[:, :, np.newaxis]

    if not per_channel:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)
        lo, hi = float(y.min()), float(y.max())
        if hi - lo < EPS:
            return image.copy()
        ycrcb[:, :, 0] = (y - lo) * 255.0 / (hi - lo + EPS)
        ycrcb = to_uint8(ycrcb)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    f = image.astype(np.float64)
    out = np.empty_like(f)
    for c in range(3):
        ch = f[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        if hi - lo < EPS:
            out[:, :, c] = ch
        else:
            out[:, :, c] = (ch - lo) * 255.0 / (hi - lo + EPS)
    return to_uint8(out)
