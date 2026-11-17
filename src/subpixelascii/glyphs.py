"""The set of candidate glyphs the converter may choose from."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .fonts import FontGlyphs
from .unicode_filter import cell_width, is_safe

CHARSETS: dict[str, tuple[tuple[int, int], ...] | None] = {
    "all": None,
    "ascii": ((0x20, 0x7E),),
    # Box drawing, block elements, geometric shapes and symbols for legacy computing.
    "blocks": ((0x20, 0x20), (0x2500, 0x25FF), (0x1FB00, 0x1FBFF)),
    "braille": ((0x20, 0x20), (0x2800, 0x28FF)),
}


@dataclass(frozen=True)
class GlyphSet:
    """Distinct glyph bitmaps for narrow (one-cell) and wide (two-cell) characters.

    Bitmaps are ``float32`` ink coverage in ``[0, 1]``: ``narrow`` has shape
    ``(n, cell_height, cell_width)`` and ``wide`` has shape ``(m, cell_height, 2 * cell_width)``.
    """

    cell_height: int
    cell_width: int
    narrow_chars: tuple[str, ...]
    narrow: np.ndarray
    wide_chars: tuple[str, ...]
    wide: np.ndarray

    def __len__(self) -> int:
        return len(self.narrow_chars) + len(self.wide_chars)

    @classmethod
    def from_fonts(
        cls,
        fonts: Sequence[FontGlyphs],
        charset: str = "all",
        chars: Iterable[str] | None = None,
        allow_wide: bool = True,
        safe_only: bool = True,
    ) -> GlyphSet:
        """Collect glyphs from ``fonts`` (earlier fonts take precedence).

        ``charset`` names a preset from :data:`CHARSETS`; ``chars``, if given, is an explicit
        collection of allowed characters and overrides ``charset``.
        With ``safe_only`` only characters that render predictably in plain text are kept
        (see :mod:`subpixelascii.unicode_filter`), and a glyph's width must match the number
        of cells a terminal gives the character.
        """
        if not fonts:
            raise ValueError("at least one font is required")
        height, width = fonts[0].cell_height, fonts[0].cell_width
        for font in fonts[1:]:
            if (font.cell_height, font.cell_width) != (height, width):
                raise ValueError(
                    f"font {font.name} has {font.cell_width}x{font.cell_height} cells, "
                    f"expected {width}x{height} like {fonts[0].name}"
                )

        allowed = _charset_predicate(charset, chars)
        seen: set[int] = set()
        collected: dict[int, tuple[list[int], list[np.ndarray]]] = {1: ([], []), 2: ([], [])}
        for font in fonts:
            for span, cps in font.codepoints.items():
                if span not in collected or (span == 2 and not allow_wide):
                    continue
                for i, cp in enumerate(cps.tolist()):
                    if cp in seen or not allowed(cp):
                        continue
                    if safe_only and (not is_safe(cp) or cell_width(cp) != span):
                        continue
                    seen.add(cp)
                    collected[span][0].append(cp)
                    collected[span][1].append(font.bitmaps[span][i])

        narrow_chars, narrow = _dedupe(*collected[1], (height, width))
        wide_chars, wide = _dedupe(*collected[2], (height, 2 * width))
        if not narrow_chars:
            raise ValueError("no usable single-cell glyphs for the requested character set")
        return cls(height, width, narrow_chars, narrow, wide_chars, wide)


def _charset_predicate(charset: str, chars: Iterable[str] | None):
    if chars is not None:
        return frozenset(ord(c) for c in chars).__contains__
    if charset not in CHARSETS:
        raise ValueError(f"unknown charset {charset!r}; choose from {', '.join(CHARSETS)}")
    ranges = CHARSETS[charset]
    if ranges is None:
        return lambda cp: True
    return lambda cp: any(lo <= cp <= hi for lo, hi in ranges)


def _dedupe(
    cps: list[int], bitmaps: list[np.ndarray], shape: tuple[int, int]
) -> tuple[tuple[str, ...], np.ndarray]:
    """Drop glyphs whose bitmap duplicates one with a lower code point."""
    if not cps:
        return (), np.zeros((0, *shape), dtype=np.float32)
    order = np.argsort(cps, kind="stable")
    stacked = np.stack(bitmaps)[order]
    _, first = np.unique(stacked.reshape(len(order), -1), axis=0, return_index=True)
    first.sort()
    chars = tuple(chr(cps[order[i]]) for i in first)
    return chars, stacked[first].astype(np.float32) / 255.0
