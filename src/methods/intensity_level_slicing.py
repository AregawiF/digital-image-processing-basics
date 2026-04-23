"""Intensity level slicing: highlight pixels in [low, high] on a scalar intensity or V channel (RGB)."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils import is_grayscale, squeeze_gray, to_uint8


def intensity_level_slicing(
    image: np.ndarray,
    low: int = 100,
    high: int = 200,
    *,
    preserve_background: bool = False,
    use_value_channel: bool = True,
) -> np.ndarray:
    """Map intensities: pixels inside [low, high] go to 255, outside to 0 (binary slice).

    If `preserve_background` is True, values outside the range keep their original
    intensity (grayscale) or are unchanged in HSV and converted back (RGB, when
    `use_value_channel` is True).
    Grayscale: operates on 2D intensity. RGB: if `use_value_channel` (default),
    slicing applies to V in HSV; otherwise converts to gray for the mask and applies
    to all channels of the result for display (optional pattern).
    """
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    low, high = int(np.clip(low, 0, 255)), int(np.clip(high, 0, 255))
    if low > high:
        low, high = high, low

    if is_grayscale(image):
        g = squeeze_gray(image)
        m = (g >= low) & (g <= high)
        if preserve_background:
            out = g.copy().astype(np.uint8)
            out[m] = 255
            return out if g.ndim == 2 else out[:, :, np.newaxis]
        out = np.where(m, 255, 0).astype(np.uint8)
        return out if g.ndim == 2 else out[:, :, np.newaxis]

    if use_value_channel:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        m = (v >= low) & (v <= high)
        if preserve_background:
            hsv2 = hsv.copy()
            hsv2[:, :, 2] = np.where(m, 255, v)
        else:
            hsv2 = hsv.copy()
            hsv2[:, :, 2] = np.where(m, 255, 0)
        return cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    m = (g >= low) & (g <= high)
    if preserve_background:
        return to_uint8(
            np.dstack(
                [np.where(m, 255, image[:, :, i]) for i in range(3)]
            )
        )
    return to_uint8(np.dstack([np.where(m, 255, 0)] * 3))
