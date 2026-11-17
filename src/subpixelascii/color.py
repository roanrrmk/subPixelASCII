"""sRGB transfer functions."""

from __future__ import annotations

import numpy as np


def srgb_decode(v: np.ndarray) -> np.ndarray:
    """Gamma-encoded sRGB values in [0, 1] -> linear light."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    """Linear light in [0, 1] -> gamma-encoded sRGB values (roughly perceptually uniform)."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)
