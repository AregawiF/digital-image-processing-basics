"""Bit plane extraction and optional reconstruction (grayscale and per-channel color)."""

from __future__ import annotations

import numpy as np

from src.utils import is_grayscale, squeeze_gray, to_uint8


def bit_planes(image: np.ndarray) -> np.ndarray:
    """Return stack of 8 bit planes, shape (H, W, 8) for gray or (H, W, 3, 8) for RGB.

    Each plane is 0 or 255 for visualization.
    """
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if is_grayscale(image):
        g = squeeze_gray(image)
        planes = _planes_8(g)
        return planes
    b = np.empty((image.shape[0], image.shape[1], 3, 8), dtype=np.uint8)
    for c in range(3):
        b[:, :, c, :] = _planes_8(image[:, :, c])
    return b


def _planes_8(g: np.ndarray) -> np.ndarray:
    p = np.empty((*g.shape, 8), dtype=np.uint8)
    for k in range(8):
        p[..., k] = to_uint8(((g.astype(np.uint16) >> k) & 1) * 255)
    return p


def reconstruct_from_planes(
    planes: np.ndarray,
    *,
    bits: list[int] | None = None,
) -> np.ndarray:
    """Reconstruct uint8 from plane stack. `bits` defaults to all 0..7.

    For grayscale: `planes` is (H, W, 8). For RGB: (H, W, 3, 8).
    """
    if bits is None:
        bits = list(range(8))
    if planes.ndim == 3:
        acc = np.zeros(planes.shape[:2], dtype=np.float64)
        for k in bits:
            if 0 <= k < 8:
                acc += (planes[:, :, k] > 127).astype(np.float64) * (2**k)
        return to_uint8(acc)
    if planes.ndim == 4 and planes.shape[2] == 3:
        out = np.empty((*planes.shape[0:2], 3), dtype=np.uint8)
        for c in range(3):
            out[:, :, c] = reconstruct_from_planes(planes[:, :, c, :], bits=bits)
        return out
    raise ValueError("planes must be (H,W,8) or (H,W,3,8)")
