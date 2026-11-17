import numpy as np
import pytest

from subpixelascii.fonts import load_hex
from subpixelascii.glyphs import GlyphSet
from subpixelascii.unicode_filter import cell_width, is_safe


@pytest.mark.parametrize(
    ("char", "safe", "width"),
    [
        ("A", True, 1),
        ("░", True, 1),
        ("中", True, 2),
        ("\u3000", True, 2),
        ("\u0301", False, 1),  # combining mark
        ("\u05d0", False, 1),  # right-to-left letter
        ("\u0628", False, 1),  # Arabic (joining, right-to-left)
        ("\u1100", False, 2),  # conjoining Hangul jamo
        ("\U0001f600", False, 2),  # emoji presentation
        ("\ue000", False, 1),  # private use
        ("\u00a0", False, 1),  # no-break space (only plain spaces are kept)
        ("\u2800", False, 1),  # blank braille: Unifont draws placeholder dots
        ("\x07", False, 1),
    ],
)
def test_is_safe_and_cell_width(char, safe, width):
    assert is_safe(ord(char)) is safe
    assert cell_width(ord(char)) == width


def test_load_hex_parses_narrow_and_wide_bitmaps(tiny_hex):
    font = load_hex(tiny_hex)
    assert (font.cell_height, font.cell_width) == (16, 8)
    narrow = dict(zip(font.codepoints[1].tolist(), font.bitmaps[1]))
    wide = dict(zip(font.codepoints[2].tolist(), font.bitmaps[2]))

    assert narrow[0x2588].min() == 255
    assert (narrow[0x2580][:8] == 255).all() and (narrow[0x2580][8:] == 0).all()
    assert (narrow[0x258C][:, :4] == 255).all() and (narrow[0x258C][:, 4:] == 0).all()
    # Row 5 of "A" is 0x24 = 00100100.
    assert (narrow[0x41][5] // 255).tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    # "一" is a single horizontal stroke 0xFFFE on row 7.
    assert wide[0x4E00].shape == (16, 16)
    assert np.flatnonzero(wide[0x4E00].any(axis=1)).tolist() == [7]
    assert (wide[0x4E00][7] // 255).tolist() == [1] * 15 + [0]


def test_glyph_set_filters_unsafe_and_dedupes(tiny_hex):
    glyphs = GlyphSet.from_fonts([load_hex(tiny_hex)])
    # Latin and Cyrillic "A" share a bitmap: the lower code point wins.
    assert glyphs.narrow_chars == (" ", "A", "▀", "█", "▌")
    assert glyphs.wide_chars == ("一",)
    assert glyphs.narrow.dtype == np.float32 and glyphs.narrow.max() == 1.0


def test_glyph_set_charset_options(tiny_hex):
    font = load_hex(tiny_hex)
    assert GlyphSet.from_fonts([font], charset="ascii").narrow_chars == (" ", "A")
    only = GlyphSet.from_fonts([font], chars="█A", allow_wide=False)
    assert only.narrow_chars == ("A", "█") and only.wide_chars == ()
    with pytest.raises(ValueError):
        GlyphSet.from_fonts([font], charset="nope")
    with pytest.raises(ValueError):
        GlyphSet.from_fonts([font], chars="一")  # no single-cell glyph to tile with
