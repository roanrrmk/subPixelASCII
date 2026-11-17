import numpy as np
import pytest
from PIL import Image

from subpixelascii import Converter
from subpixelascii.cli import main
from subpixelascii.converter import TextArt
from subpixelascii.fonts import unifont_path
from subpixelascii.unicode_filter import cell_width


@pytest.mark.parametrize("background", ["dark", "light"])
def test_image_drawn_from_glyphs_converts_back_to_the_same_glyphs(random_glyphs, background):
    glyphs = random_glyphs
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(4):
        row, used = [], 0
        while used < 9:
            span = 2 if used < 8 and rng.random() < 0.3 else 1
            pool = len(glyphs.wide_chars) if span == 2 else len(glyphs.narrow_chars)
            row.append((span, int(rng.integers(pool))))
            used += span
        rows.append(row)
    expected = TextArt(glyphs, rows, background)

    art = Converter(glyphs, blur=0.0).convert(expected.to_image(), columns=9, background=background, autocontrast=False)

    assert art.rows == rows
    assert str(art) == str(expected)
    assert np.array_equal(np.asarray(art.to_image()), np.asarray(expected.to_image()))


def test_cli_writes_text_and_png(tiny_hex, tmp_path):
    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (32, 1))
    Image.fromarray(gradient).save(tmp_path / "in.png")
    out_txt, out_png = tmp_path / "out.txt", tmp_path / "out.png"

    code = main([str(tmp_path / "in.png"), "-w", "12", "--font", str(tiny_hex), "-o", str(out_txt),
                 "--png", str(out_png), "--scale", "2", "-q"])

    assert code == 0
    lines = out_txt.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 64x32 image, 12 columns of 1:2 cells
    assert all(sum(cell_width(ord(c)) for c in line) == 12 for line in lines)
    assert lines[0][0] == " " and lines[0][-1] == "█"  # dark to bright, light glyphs on dark
    assert Image.open(out_png).size == (12 * 8 * 2, 3 * 16 * 2)


def test_cli_reports_missing_image(capsys):
    assert main(["does-not-exist.png", "-q"]) == 1
    assert "does-not-exist.png" in capsys.readouterr().err


def _unifont_cached():
    try:
        unifont_path(download=False)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _unifont_cached(), reason="GNU Unifont is not cached")
def test_unifont_output_lines_keep_their_width():
    yy, xx = np.mgrid[0:120, 0:200]
    image = Image.fromarray(((np.sin(xx / 9.0) * np.cos(yy / 7.0) + 1) * 127.5).astype(np.uint8))
    art = Converter().convert(image, columns=50)
    assert any(span == 2 for row in art.rows for span, _ in row)
    for line in art.lines:
        assert sum(cell_width(ord(c)) for c in line) == 50
