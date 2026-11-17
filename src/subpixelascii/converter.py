"""High-level image -> text art conversion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .embedding import PerceptualEmbedding
from .fonts import load_font
from .glyphs import GlyphSet
from .imageprep import image_to_cells
from .matching import nearest, tile_rows


@dataclass(frozen=True)
class TextArt:
    """The result of a conversion: rows of ``(span, glyph_index)`` tokens."""

    glyphs: GlyphSet
    rows: list[list[tuple[int, int]]]
    background: str = "dark"

    @property
    def lines(self) -> list[str]:
        narrow, wide = self.glyphs.narrow_chars, self.glyphs.wide_chars
        return ["".join(narrow[i] if span == 1 else wide[i] for span, i in row) for row in self.rows]

    @property
    def columns(self) -> int:
        return sum(span for span, _ in self.rows[0]) if self.rows else 0

    def __str__(self) -> str:
        return "\n".join(self.lines)

    def to_image(self, scale: int = 1) -> Image.Image:
        """Render the art with the matched glyph bitmaps (what the matcher "saw")."""
        h, w = self.glyphs.cell_height, self.glyphs.cell_width
        canvas = np.zeros((len(self.rows) * h, self.columns * w), dtype=np.float32)
        for r, row in enumerate(self.rows):
            x = 0
            for span, i in row:
                bitmap = self.glyphs.narrow[i] if span == 1 else self.glyphs.wide[i]
                canvas[r * h : (r + 1) * h, x : x + span * w] = bitmap
                x += span * w
        if self.background == "light":
            canvas = 1.0 - canvas
        image = Image.fromarray(np.round(canvas * 255).astype(np.uint8))
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        return image


class Converter:
    """Matches image cells to glyphs. Construct once and reuse for many images."""

    def __init__(
        self,
        glyphs: GlyphSet | None = None,
        *,
        blur: float = 1.5,
        tone_weight: float = 1.0,
        perceptual: bool = False,
    ) -> None:
        self.glyphs = glyphs if glyphs is not None else default_glyphs()
        g = self.glyphs
        self.embedding = PerceptualEmbedding(
            g.cell_height, g.cell_width, blur=blur, tone_weight=tone_weight, perceptual=perceptual
        )
        self._narrow = self.embedding(g.narrow)
        # A wide glyph is compared half by half so its cost is comparable with two narrow glyphs.
        w = g.cell_width
        self._wide = (
            np.concatenate([self.embedding(g.wide[..., :w]), self.embedding(g.wide[..., w:])], axis=1)
            if len(g.wide)
            else None
        )

    def convert(
        self,
        image: str | os.PathLike[str] | Image.Image,
        columns: int = 100,
        rows: int | None = None,
        *,
        background: str = "dark",
        linear: bool = True,
        autocontrast: bool = True,
        gamma: float = 1.0,
        allow_wide: bool = True,
    ) -> TextArt:
        """Convert ``image`` to text art ``columns`` cells wide (and ``rows`` tall).

        ``background`` is the colour the art will be shown on ("dark": light text on a dark
        terminal; "light": dark text on white). See :func:`image_to_cells` for the tone options.
        """
        g = self.glyphs
        cells = image_to_cells(
            image,
            columns,
            rows,
            g.cell_height,
            g.cell_width,
            background=background,
            linear=linear,
            autocontrast=autocontrast,
            gamma=gamma,
        )
        n_rows, n_cols = cells.shape[:2]
        emb = self.embedding(cells)
        narrow_index, narrow_cost = nearest(emb.reshape(n_rows * n_cols, -1), self._narrow)
        narrow_index = narrow_index.reshape(n_rows, n_cols)
        narrow_cost = narrow_cost.reshape(n_rows, n_cols)

        if allow_wide and self._wide is not None and n_cols >= 2:
            pairs = np.concatenate([emb[:, :-1], emb[:, 1:]], axis=2)
            wide_index, wide_cost = nearest(pairs.reshape(n_rows * (n_cols - 1), -1), self._wide)
            tiled = tile_rows(
                narrow_index,
                narrow_cost,
                wide_index.reshape(n_rows, n_cols - 1),
                wide_cost.reshape(n_rows, n_cols - 1),
            )
        else:
            tiled = tile_rows(narrow_index, narrow_cost)
        return TextArt(g, tiled, background)


@lru_cache(maxsize=1)
def default_glyphs() -> GlyphSet:
    """All safe glyphs of GNU Unifont (downloaded and cached on first use)."""
    return GlyphSet.from_fonts([load_font()])


def convert(image: str | os.PathLike[str] | Image.Image, columns: int = 100, **options) -> TextArt:
    """Convert ``image`` with GNU Unifont and default matching settings."""
    return Converter().convert(image, columns, **options)
