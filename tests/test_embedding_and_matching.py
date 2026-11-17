import numpy as np

from subpixelascii import matching
from subpixelascii.embedding import PerceptualEmbedding
from subpixelascii.matching import nearest, tile_rows


def test_unblurred_embedding_distance_equals_pixel_distance():
    rng = np.random.default_rng(1)
    a, b = rng.random((2, 16, 8))
    emb = PerceptualEmbedding(16, 8, blur=0.0, tone_weight=0.0)
    assert emb.dims == 16 * 8
    ea, eb = emb(np.stack([a, b]))
    assert np.isclose(np.sum((ea - eb) ** 2), np.sum((a - b) ** 2), rtol=1e-4)


def test_tone_weight_scales_mean_error_and_blur_suppresses_detail():
    emb = PerceptualEmbedding(16, 8, blur=1.5, tone_weight=3.0)
    flat_a, flat_b = np.full((16, 8), 0.2), np.full((16, 8), 0.5)
    ea, eb = emb(np.stack([flat_a, flat_b]))
    assert np.isclose(np.sum((ea - eb) ** 2), (1 + 3.0) * 128 * 0.3**2, rtol=1e-4)

    checker = (np.indices((16, 8)).sum(axis=0) % 2).astype(float)
    gray = np.full((16, 8), 0.5)
    ec, eg = emb(np.stack([checker, gray]))
    # A fine checkerboard looks like flat 50% grey once blurred.
    assert np.sum((ec - eg) ** 2) < 1e-3 * np.sum((checker - gray) ** 2)


def test_nearest_matches_brute_force_across_blocks(monkeypatch):
    monkeypatch.setattr(matching, "_BLOCK_ELEMENTS", 50)  # force many query blocks
    rng = np.random.default_rng(2)
    queries, refs = rng.random((37, 5)), rng.random((11, 5))
    d = ((queries[:, None] - refs[None]) ** 2).sum(-1)
    index, dist = nearest(queries, refs)
    assert index.tolist() == d.argmin(1).tolist()
    assert np.allclose(dist, d.min(1), atol=1e-5)


def test_tile_rows_uses_wide_glyphs_only_when_cheaper():
    narrow_index = np.array([[0, 1, 2], [3, 4, 5]])
    narrow_cost = np.ones((2, 3))
    wide_index = np.array([[7, 8], [9, 10]])
    wide_cost = np.array([[5.0, 1.5], [2.5, 2.5]])
    rows = tile_rows(narrow_index, narrow_cost, wide_index, wide_cost)
    assert rows[0] == [(1, 0), (2, 8)]
    assert rows[1] == [(1, 3), (1, 4), (1, 5)]
    assert tile_rows(narrow_index, narrow_cost) == [[(1, 0), (1, 1), (1, 2)], [(1, 3), (1, 4), (1, 5)]]
