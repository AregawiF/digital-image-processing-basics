"""Gamma (power-law) transform: s = c * r**gamma with c = 1, r in [0, 1], per channel for RGB."""

from __future__ import annotations

import numpy as np

from src.utils import is_grayscale, to_float01, to_uint8


def gamma_correction(
    image: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    r = to_float01(image)
    if is_grayscale(image):
        g = np.squeeze(r, axis=2) if r.ndim == 3 else r
        s = np.power(g, gamma)
        return to_uint8(s * 255.0)
    s = np.power(r, gamma)
    return to_uint8(s * 255.0)
