"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__
from .converter import Converter
from .fonts import load_font
from .glyphs import CHARSETS, GlyphSet
from .imageprep import load_image
from .unicode_filter import is_safe

log = logging.getLogger("subpixelascii")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subpixelascii",
        description=(
            "Convert an image to Unicode text art. Every glyph of GNU Unifont is compared with "
            "the pixels each character cell covers, using a perceptual (DCT, contrast-weighted) "
            "embedding, and the closest glyph is chosen."
        ),
    )
    p.add_argument("image", help="input image (any format Pillow can read)")
    p.add_argument("-w", "--width", type=int, default=100, help="output width in character cells (default: 100)")
    p.add_argument("--height", type=int, help="output height in rows (default: keep the aspect ratio)")
    p.add_argument("-o", "--output", help="write the text to this file (UTF-8) instead of stdout")
    p.add_argument("--png", help="also render the result with the matched font bitmaps to this PNG file")
    p.add_argument("--scale", type=int, default=1, help="integer pixel scale for --png (default: 1)")
    p.add_argument(
        "--background",
        choices=("dark", "light"),
        default="dark",
        help="background the art is shown on: 'dark' = light text on dark (terminals), "
        "'light' = dark text on white (default: dark)",
    )

    chars = p.add_argument_group("characters")
    chars.add_argument("--charset", choices=tuple(CHARSETS), default="all", help="character preset (default: all)")
    chars.add_argument("--chars", help="explicit characters to choose from (overrides --charset)")
    chars.add_argument("--no-wide", action="store_true", help="never use double-width (e.g. CJK) characters")
    chars.add_argument(
        "--font",
        action="append",
        help="font to use instead of GNU Unifont: a .hex/.hex.gz bitmap font or a TTF/OTF file. "
        "Repeat for fallback fonts; all must have the same cell size.",
    )
    chars.add_argument("--font-size", type=int, default=16, help="pixel size for TTF/OTF fonts (default: 16)")

    tuning = p.add_argument_group("matching and tone")
    tuning.add_argument(
        "--blur",
        type=float,
        default=1.5,
        help="perceptual blur in glyph pixels; lower favours exact pixel shapes, higher favours "
        "overall tone (default: 1.5)",
    )
    tuning.add_argument(
        "--tone-weight", type=float, default=1.0, help="extra weight on each cell's mean brightness (default: 1)"
    )
    tuning.add_argument(
        "--perceptual",
        action="store_true",
        help="measure errors on a perceptual lightness scale instead of linear light",
    )
    tuning.add_argument("--gamma", type=float, default=1.0, help="gamma applied to the image; >1 darkens (default: 1)")
    tuning.add_argument("--no-autocontrast", action="store_true", help="do not stretch the image's contrast")
    tuning.add_argument(
        "--no-linear",
        action="store_true",
        help="treat gamma-encoded pixel values as ink coverage (brighter, less faithful tones)",
    )

    p.add_argument("-q", "--quiet", action="store_true", help="no progress messages on stderr")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    if args.width < 1 or (args.height is not None and args.height < 1) or args.scale < 1:
        print("subpixelascii: --width, --height and --scale must be positive", file=sys.stderr)
        return 2

    try:
        image = load_image(args.image)
        start = time.perf_counter()
        fonts = [load_font(path, args.font_size, include=is_safe) for path in args.font or [None]]
        glyphs = GlyphSet.from_fonts(fonts, charset=args.charset, chars=args.chars, allow_wide=not args.no_wide)
        log.info(
            "Loaded %d distinct glyphs (%d single-width, %d double-width) in %.1fs",
            len(glyphs),
            len(glyphs.narrow_chars),
            len(glyphs.wide_chars),
            time.perf_counter() - start,
        )

        start = time.perf_counter()
        converter = Converter(glyphs, blur=args.blur, tone_weight=args.tone_weight, perceptual=args.perceptual)
        art = converter.convert(
            image,
            args.width,
            args.height,
            background=args.background,
            linear=not args.no_linear,
            autocontrast=not args.no_autocontrast,
            gamma=args.gamma,
        )
        log.info("Matched %d x %d cells in %.1fs", art.columns, len(art.rows), time.perf_counter() - start)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"subpixelascii: {exc}", file=sys.stderr)
        return 1

    text = str(art) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure and (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
            reconfigure(encoding="utf-8")
        sys.stdout.write(text)
    if args.png:
        art.to_image(args.scale).save(args.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
