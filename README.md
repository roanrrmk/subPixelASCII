# subPixelASCII

Sub-pixel Unicode text art: convert an image to text by matching **every glyph of a font with
near-complete Unicode coverage** against the exact pixels each character cell covers.

Most "ASCII art" converters pick a character by brightness alone. subPixelASCII looks at the
*shape* inside each cell: a cell containing a diagonal edge becomes a glyph with a diagonal
stroke in the same place, and a flat grey area becomes a glyph whose texture averages out to
that grey. Glyphs are never coloured. The output is plain monochrome text.

## How it works

1. **Font.** By default the converter uses [GNU Unifont](https://unifoundry.com/unifont/), the
   font with the widest Unicode coverage (about 127,000 glyphs across the BMP and the
   supplementary planes). Its `.hex` distribution stores each glyph as an exact 8×16
   (single-width) or 16×16 (double-width) bitmap, so the matcher sees precisely the pixels that
   will be drawn. The font is downloaded once, checked against a pinned SHA-256, and cached
   (see [Font licence](#font-licence)).
2. **Safe characters only.** Characters that would not render as a standalone glyph in plain
   text are excluded:
   - combining marks, control, format and private-use characters, and unassigned code points;
   - right-to-left scripts and characters that join with their neighbours (Arabic, Mongolian,
     conjoining Hangul jamo);
   - characters that default to colour emoji.

   A glyph's width must also match the number of terminal cells the character occupies (East
   Asian Width), so lines stay aligned. Identical bitmaps are deduplicated, and the lowest code
   point wins (so you get `A`, not a lookalike). This leaves about 60,000 distinct glyphs: 9,100
   single-width and 51,000 double-width.
3. **Image to cells.** The image is converted to linear-light luminance and resampled so
   that every character cell gets exactly one glyph-sized block of pixels (8×16 for Unifont).
   Contrast is stretched automatically. The target "ink" per pixel is linear brightness, because
   the eye averages ink and background pixels in linear light.
4. **Perceptual embedding.** Each cell and each glyph is mapped to a vector. The squared
   distance between vectors approximates how different the two patches look. The embedding is
   a dense, weighted form of the DCT perceptual hash (pHash):
   - take the orthonormal 2-D DCT of the patch;
   - attenuate each coefficient with a Gaussian contrast-sensitivity weight (equivalent to
     blurring by `--blur` pixels, as the eye does);
   - give extra weight to the DC term, which is the cell's tone (`--tone-weight`);
   - keep only the low-frequency coefficients that matter (43 dimensions by default).
5. **Nearest neighbour.** An exact nearest-neighbour search finds the closest single-width
   glyph for every cell and the closest double-width glyph for every pair of adjacent cells.
   Each half of a double-width glyph is embedded separately, so its cost is directly comparable
   with two single-width glyphs.
6. **Optimal row tiling.** Dynamic programming chooses, per row, the mix of single- and
   double-width glyphs with the lowest total error.

## Installation

Requires Python 3.10+.

```bash
pip install .            # or: pip install -e '.[dev]' for development
```

## Usage

```bash
subpixelascii photo.jpg                     # 100 columns to stdout
subpixelascii photo.jpg -w 160 -o art.txt   # write UTF-8 text to a file
subpixelascii photo.jpg --png art.png       # also render the result with the font bitmaps
subpixelascii photo.jpg --background light  # dark text for white backgrounds / paper
subpixelascii photo.jpg --charset ascii     # restrict to printable ASCII
subpixelascii photo.jpg --charset blocks    # box drawing, blocks, shapes, legacy computing
subpixelascii photo.jpg --chars ' .:-=+*#%@'
subpixelascii photo.jpg --no-wide           # avoid double-width (CJK) characters
python -m subpixelascii --help
```

Main options:

| Option | Meaning |
| --- | --- |
| `-w/--width`, `--height` | Output size in character cells. The height defaults to keeping the aspect ratio. |
| `--background dark\|light` | Where the art will be shown: `dark` = light glyphs on dark (terminals, default); `light` = dark glyphs on white. |
| `--charset all\|ascii\|blocks\|braille`, `--chars` | Which characters may be used. |
| `--no-wide` | Single-width characters only. |
| `--font PATH` | Use another `.hex`/`.hex.gz` bitmap font or a TTF/OTF font (monospaced works best). Repeat for fallbacks. |
| `--blur` | Perceptual blur in glyph pixels (default 1.5). Lower values reproduce exact pixel shapes; higher values reproduce tone. |
| `--tone-weight` | Extra weight on matching each cell's mean brightness (default 1). |
| `--perceptual` | Measure errors on a perceptual lightness scale instead of linear light. |
| `--gamma`, `--no-autocontrast`, `--no-linear` | Tone adjustments. `--gamma` below 1 brightens the result. |
| `--png PATH`, `--scale N` | Save a pixel-exact rendering of the result. |

### Viewing the text

The art was matched against Unifont's bitmaps, so view it in GNU Unifont. Install the
`unifont` and `unifont_upper` OpenType fonts from
[unifoundry.com](https://unifoundry.com/unifont/) and select Unifont in your terminal or
editor. Other fonts will show the right characters but not the shapes they were matched for,
and may be missing many of them. `--png` always produces an exact rendering.

### Python API

```python
from subpixelascii import Converter, GlyphSet, load_font

art = Converter().convert("photo.jpg", columns=120)  # GNU Unifont, default settings
print(art)
art.to_image(scale=2).save("art.png")

# Reuse one converter for many images, with a custom glyph set and matching settings:
glyphs = GlyphSet.from_fonts([load_font()], charset="blocks")
converter = Converter(glyphs, blur=1.0, tone_weight=2.0)
art = converter.convert("photo.jpg", columns=80, background="light")
```

## Font licence

GNU Unifont is not bundled. On first use it is downloaded from the GNU mirrors (falling back
to unifoundry.com) into `~/.cache/subpixelascii`, or `$XDG_CACHE_HOME/subpixelascii`, or
`%LOCALAPPDATA%\subpixelascii` on Windows. Override the location with `SUBPIXELASCII_CACHE`. For
offline use, place `unifont_all-18.0.01.hex.gz` in that directory or pass `--font`. GNU Unifont
is dual-licensed under the SIL Open Font License 1.1 and the GNU GPL v2+ with the font
embedding exception.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The table of emoji-presentation characters (`src/subpixelascii/_emoji_data.py`) is generated
from the Unicode data files by `python tools/generate_emoji_table.py [UNICODE_VERSION]`.
