"""Log transform: s = c * log(1 + I) with I in [0, 255], c = 255 / log(1 + 255)."""

from __future__ import annotations

import numpy as np

from src.utils import is_grayscale, squeeze_gray, to_uint8


def logarithmic_transform(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    c = 255.0 / np.log(256.0)  # = 255 / log(1 + 255)
    f = image.astype(np.float64)
    if is_grayscale(image):
        g = squeeze_gray(f)
        s = c * np.log(1.0 + g)
        return to_uint8(s)
    s = c * np.log(1.0 + f)
    return to_uint8(s)
