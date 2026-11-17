"""Decide which characters can be emitted safely in plain-text art.

A character is "safe" when a typical text renderer will draw it on its own, left to right,
in exactly the number of terminal cells we expect, without combining it with neighbours.
"""

from __future__ import annotations

from bisect import bisect_right

try:  # Prefer an up-to-date Unicode database; the stdlib one may lag behind the font.
    import unicodedata2 as ucd
except ImportError:  # pragma: no cover - exercised only without the optional backport
    import unicodedata as ucd

from ._emoji_data import EMOJI_PRESENTATION_RANGES

UNICODE_VERSION: str = ucd.unidata_version

# Controls, format characters, surrogates, private use, unassigned, combining marks and
# line/paragraph separators either render invisibly, render as placeholder boxes, or attach
# to the previous character.
_EXCLUDED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Mn", "Mc", "Me", "Zl", "Zp"})

# Right-to-left and explicit-embedding bidi classes would reorder the surrounding text.
_EXCLUDED_BIDI = frozenset(
    {"R", "AL", "AN", "NSM", "BN", "B", "S", "LRE", "LRO", "RLE", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
)

# Blocks whose characters join or compose with their neighbours when shaped.
_JOINING_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo (conjoining)
    (0x1800, 0x18AF),  # Mongolian
    (0xA840, 0xA87F),  # Phags-pa
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
)

# Characters that are blank in normal fonts but that Unifont draws as visible placeholders.
_BLANK = frozenset({0x2800, 0x3164, 0xFFA0})  # braille blank, Hangul fillers

_SPACES = frozenset({0x20, 0x3000})


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...], starts: list[int]) -> bool:
    i = bisect_right(starts, cp) - 1
    return i >= 0 and cp <= ranges[i][1]


_EMOJI_STARTS = [lo for lo, _ in EMOJI_PRESENTATION_RANGES]
_JOINING_STARTS = [lo for lo, _ in _JOINING_RANGES]


def cell_width(cp: int) -> int:
    """Number of terminal cells the character occupies (1 or 2)."""
    return 2 if ucd.east_asian_width(chr(cp)) in ("W", "F") else 1


def is_safe(cp: int) -> bool:
    """Whether the code point renders as a standalone, left-to-right, non-emoji glyph."""
    ch = chr(cp)
    category = ucd.category(ch)
    if category in _EXCLUDED_CATEGORIES or cp in _BLANK:
        return False
    if category == "Zs" and cp not in _SPACES:
        return False
    if ucd.bidirectional(ch) in _EXCLUDED_BIDI:
        return False
    if _in_ranges(cp, EMOJI_PRESENTATION_RANGES, _EMOJI_STARTS):
        return False
    return not _in_ranges(cp, _JOINING_RANGES, _JOINING_STARTS)
