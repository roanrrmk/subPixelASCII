"""Convert images to Unicode text art by perceptually matching glyph bitmaps to image pixels."""

from .converter import Converter, TextArt, convert, default_glyphs
from .embedding import PerceptualEmbedding
from .fonts import FontGlyphs, load_font
from .glyphs import CHARSETS, GlyphSet

__all__ = [
    "CHARSETS",
    "Converter",
    "FontGlyphs",
    "GlyphSet",
    "PerceptualEmbedding",
    "TextArt",
    "convert",
    "default_glyphs",
    "load_font",
]
__version__ = "0.1.0"
