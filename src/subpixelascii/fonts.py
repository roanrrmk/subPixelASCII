"""Font loading.

The default font is GNU Unifont, the font with the widest Unicode coverage available: it has a
glyph for every visible character of the Basic Multilingual Plane and most of the
supplementary planes. Its ``.hex`` distribution stores each glyph as an exact 8x16 or 16x16
bitmap, which is ideal for pixel-accurate matching. Other bitmap ``.hex`` fonts and outline
fonts (TTF/OTF/TTC) are supported too.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import sys
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

UNIFONT_VERSION = "18.0.01"
UNIFONT_FILENAME = f"unifont_all-{UNIFONT_VERSION}.hex.gz"
UNIFONT_SHA256 = "14c96e497466a82e46cf20032ce510e186bdc8658c9158e2290f452cda9bc498"
UNIFONT_URLS = (
    f"https://ftpmirror.gnu.org/unifont/unifont-{UNIFONT_VERSION}/{UNIFONT_FILENAME}",
    f"https://unifoundry.com/pub/unifont/unifont-{UNIFONT_VERSION}/font-builds/{UNIFONT_FILENAME}",
)

HEX_CELL_HEIGHT = 16
HEX_CELL_WIDTH = 8


@dataclass(frozen=True)
class FontGlyphs:
    """Glyph bitmaps of one font, grouped by the number of character cells they span.

    ``bitmaps[span]`` has shape ``(n, cell_height, span * cell_width)`` and holds ink coverage
    as ``uint8`` (0 = background, 255 = fully inked); ``codepoints[span]`` holds the matching
    code points.
    """

    name: str
    cell_height: int
    cell_width: int
    codepoints: dict[int, np.ndarray]
    bitmaps: dict[int, np.ndarray]

    def __len__(self) -> int:
        return sum(len(cps) for cps in self.codepoints.values())


def cache_dir() -> Path:
    """Directory used to cache downloaded fonts (override with ``SUBPIXELASCII_CACHE``)."""
    if env := os.environ.get("SUBPIXELASCII_CACHE"):
        return Path(env)
    if sys.platform == "win32" and (local := os.environ.get("LOCALAPPDATA")):
        return Path(local) / "subpixelascii"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "subpixelascii"


def unifont_path(download: bool = True) -> Path:
    """Path to the cached GNU Unifont ``.hex.gz`` file, downloading it on first use."""
    path = cache_dir() / UNIFONT_FILENAME
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"GNU Unifont is not cached at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in UNIFONT_URLS:
        log.info("Downloading GNU Unifont %s from %s", UNIFONT_VERSION, url)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != UNIFONT_SHA256:
            errors.append(f"{url}: checksum mismatch ({digest})")
            continue
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
        return path
    raise RuntimeError(
        "Could not download GNU Unifont. Download "
        f"{UNIFONT_FILENAME} manually into {path.parent} or pass --font.\n" + "\n".join(errors)
    )


def load_hex(path: str | os.PathLike[str]) -> FontGlyphs:
    """Load a GNU Unifont-style ``.hex`` (optionally gzipped) bitmap font."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    by_length: dict[int, tuple[list[int], list[str]]] = {}
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            cp_hex, sep, data = line.strip().partition(":")
            if not sep or cp_hex.startswith("#"):
                continue
            cps, rows = by_length.setdefault(len(data), ([], []))
            cps.append(int(cp_hex, 16))
            rows.append(data)

    codepoints: dict[int, np.ndarray] = {}
    bitmaps: dict[int, np.ndarray] = {}
    for length, (cps, rows) in by_length.items():
        width = length * 4 // HEX_CELL_HEIGHT
        if length * 4 % HEX_CELL_HEIGHT or width % HEX_CELL_WIDTH or width == 0:
            log.warning("Skipping %d glyphs with unsupported hex length %d", len(cps), length)
            continue
        raw = np.frombuffer(bytes.fromhex("".join(rows)), dtype=np.uint8)
        bits = np.unpackbits(raw.reshape(len(cps), HEX_CELL_HEIGHT, width // 8), axis=2)
        span = width // HEX_CELL_WIDTH
        codepoints[span] = np.asarray(cps, dtype=np.int64)
        bitmaps[span] = bits * np.uint8(255)
    return FontGlyphs(path.name, HEX_CELL_HEIGHT, HEX_CELL_WIDTH, codepoints, bitmaps)


def load_outline_font(
    path: str | os.PathLike[str],
    size: int = 16,
    include: Callable[[int], bool] | None = None,
) -> FontGlyphs:
    """Rasterise every mapped character of a TTF/OTF/TTC font at ``size`` pixels.

    The cell width is the font's most common advance among printable ASCII characters, so
    this works best with monospaced fonts. Glyphs whose advance is not one or two cells wide
    are skipped. ``include`` can be used to skip code points early.
    """
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFont

    path = Path(path)
    with TTFont(path, fontNumber=0, lazy=True) as tt:
        cmap = tt.getBestCmap() or {}
    font = ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)
    ascent, descent = font.getmetrics()
    height = ascent + descent

    ascii_advances = Counter(round(font.getlength(chr(cp))) for cp in range(0x21, 0x7F) if cp in cmap)
    if not ascii_advances:
        raise ValueError(f"{path} has no printable ASCII glyphs to derive a cell width from")
    width = ascii_advances.most_common(1)[0][0]
    tolerance = max(0.5, 0.05 * width)

    groups: dict[int, tuple[list[int], list[np.ndarray]]] = {}
    for cp in sorted(cmap):
        if include is not None and not include(cp):
            continue
        ch = chr(cp)
        advance = font.getlength(ch)
        span = round(advance / width)
        if span not in (1, 2) or abs(advance - span * width) > tolerance:
            continue
        canvas = Image.new("L", (span * width, height), 0)
        ImageDraw.Draw(canvas).text((0, ascent), ch, font=font, fill=255, anchor="ls")
        cps, bitmaps = groups.setdefault(span, ([], []))
        cps.append(cp)
        bitmaps.append(np.asarray(canvas, dtype=np.uint8))

    return FontGlyphs(
        path.name,
        height,
        width,
        {span: np.asarray(cps, dtype=np.int64) for span, (cps, _) in groups.items()},
        {span: np.stack(bms) for span, (_, bms) in groups.items()},
    )


def load_font(
    path: str | os.PathLike[str] | None = None,
    size: int = 16,
    include: Callable[[int], bool] | None = None,
) -> FontGlyphs:
    """Load a ``.hex``/``.hex.gz`` bitmap font or an outline font; default is GNU Unifont."""
    if path is None:
        return load_hex(unifont_path())
    name = Path(path).name.lower()
    if name.endswith((".hex", ".hex.gz")):
        return load_hex(path)
    return load_outline_font(path, size, include)
