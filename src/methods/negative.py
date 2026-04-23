"""Image negative: s = L - 1 - r (per channel for RGB)."""

from __future__ import annotations

import numpy as np

from src.utils import is_grayscale, to_uint8


def image_negative(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if is_grayscale(image):
        g = image if image.ndim == 2 else image[:, :, 0]
        return to_uint8(255 - g.astype(np.int32))
    return to_uint8(255 - image.astype(np.int32))
