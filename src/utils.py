"""Shared helpers for uint8 images (grayscale or RGB, channel-last)."""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def to_float01(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) / 255.0


def squeeze_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image
