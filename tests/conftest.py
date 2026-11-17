import gzip

import numpy as np
import pytest

from subpixelascii.fonts import FontGlyphs

FULL = "F" * 32
UPPER_HALF = "FF" * 8 + "00" * 8
LEFT_HALF = "F0" * 16
LETTER_A = "0000000018242442427E424242420000"
IDEOGRAPH_ONE = "0" * 28 + "FFFE" + "0" * 32

TINY_HEX = {
    0x0020: "0" * 32,
    0x0041: LETTER_A,
    0x0301: "0" * 30 + "FF",  # combining acute accent: must be filtered out
    0x0410: LETTER_A,  # Cyrillic capital A: same bitmap as Latin "A"
    0x05D0: "0" * 30 + "0F",  # Hebrew alef (right-to-left): must be filtered out
    0x2580: UPPER_HALF,
    0x2588: FULL,
    0x258C: LEFT_HALF,
    0x4E00: IDEOGRAPH_ONE,
    0x1F600: "0" * 62 + "FF",  # emoji: must be filtered out
}


@pytest.fixture
def tiny_hex(tmp_path):
    path = tmp_path / "tiny.hex.gz"
    with gzip.open(path, "wt", encoding="ascii") as f:
        for cp, data in TINY_HEX.items():
            f.write(f"{cp:04X}:{data}\n")
    return path


@pytest.fixture
def random_glyphs():
    """Distinct random glyphs on safe code points (Latin letters and CJK ideographs)."""
    from subpixelascii.glyphs import GlyphSet

    rng = np.random.default_rng(0)
    narrow, wide, height, width = 24, 6, 16, 8
    font = FontGlyphs(
        "random",
        height,
        width,
        {1: np.arange(0x41, 0x41 + narrow), 2: np.arange(0x4E00, 0x4E00 + wide)},
        {
            1: (rng.random((narrow, height, width)) < 0.5).astype(np.uint8) * 255,
            2: (rng.random((wide, height, 2 * width)) < 0.5).astype(np.uint8) * 255,
        },
    )
    return GlyphSet.from_fonts([font])
