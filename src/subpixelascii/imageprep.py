"""Turning an input image into a grid of glyph-sized cells of target ink coverage."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageOps

from .color import srgb_decode, srgb_encode

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_image(source: str | os.PathLike[str] | Image.Image) -> Image.Image:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    return ImageOps.exif_transpose(image)


def linear_luminance(image: Image.Image, background: str = "dark") -> np.ndarray:
    """Relative luminance in linear light, with transparency flattened onto ``background``."""
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        gray = np.asarray(image, dtype=np.float32) / 65535.0
        return srgb_decode(gray)
    if image.has_transparency_data:
        image = image.convert("RGBA")
        fill = (0, 0, 0, 255) if background == "dark" else (255, 255, 255, 255)
        image = Image.alpha_composite(Image.new("RGBA", image.size, fill), image)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return srgb_decode(rgb) @ _LUMA


def grid_rows(image_size: tuple[int, int], columns: int, cell_height: int, cell_width: int) -> int:
    """Number of rows that preserves the image's aspect ratio for ``columns`` cells."""
    width, height = image_size
    return max(1, round(height * columns * cell_width / (width * cell_height)))


def image_to_cells(
    source: str | os.PathLike[str] | Image.Image,
    columns: int,
    rows: int | None,
    cell_height: int,
    cell_width: int,
    *,
    background: str = "dark",
    linear: bool = True,
    autocontrast: bool = True,
    gamma: float = 1.0,
) -> np.ndarray:
    """Resample the image to the glyph grid and return target ink per pixel.

    Returns an array of shape ``(rows, columns, cell_height, cell_width)`` with values in
    ``[0, 1]``. On a ``"dark"`` background ink is light, so ink = brightness; on a ``"light"``
    background ink is dark, so ink = 1 - brightness. With ``linear`` the brightness is linear
    light, which is what the eye averages over a cell's ink and background pixels.
    """
    if background not in ("dark", "light"):
        raise ValueError("background must be 'dark' or 'light'")
    if columns < 1 or (rows is not None and rows < 1):
        raise ValueError("columns and rows must be positive")
    image = load_image(source)
    if rows is None:
        rows = grid_rows(image.size, columns, cell_height, cell_width)

    lum = Image.fromarray(linear_luminance(image, background).astype(np.float32))
    size = (columns * cell_width, rows * cell_height)
    if lum.size != size:
        lum = lum.resize(size, Image.Resampling.LANCZOS)
    value = srgb_encode(np.asarray(lum))

    if autocontrast:
        lo, hi = np.percentile(value, [0.5, 99.5])
        if hi - lo > 1e-3:
            value = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        value = value**gamma
    brightness = srgb_decode(value) if linear else value
    ink = brightness if background == "dark" else 1.0 - brightness
    return ink.reshape(rows, cell_height, columns, cell_width).transpose(0, 2, 1, 3).astype(np.float32)
