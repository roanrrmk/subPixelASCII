"""Nearest-neighbour glyph search and optimal tiling of rows with wide glyphs."""

from __future__ import annotations

import numpy as np

_BLOCK_ELEMENTS = 1 << 23  # Bound the (queries x references) distance block to ~32 MB.


def nearest(queries: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest reference (squared Euclidean distance) for every query vector.

    Returns ``(index, squared_distance)`` arrays with one entry per query.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    references = np.ascontiguousarray(references, dtype=np.float32)
    ref_norms = np.einsum("ij,ij->i", references, references)
    index = np.empty(len(queries), dtype=np.int64)
    dist = np.empty(len(queries), dtype=np.float32)
    step = max(1, _BLOCK_ELEMENTS // max(1, len(references)))
    for start in range(0, len(queries), step):
        q = queries[start : start + step]
        # ||q - r||^2 = ||q||^2 + ||r||^2 - 2 q.r ; ||q||^2 does not affect the argmin.
        d = ref_norms[None, :] - 2.0 * (q @ references.T)
        best = np.argmin(d, axis=1)
        index[start : start + len(q)] = best
        dist[start : start + len(q)] = d[np.arange(len(q)), best] + np.einsum("ij,ij->i", q, q)
    return index, np.maximum(dist, 0.0)


def tile_rows(
    narrow_index: np.ndarray,
    narrow_cost: np.ndarray,
    wide_index: np.ndarray | None = None,
    wide_cost: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    """Choose, per row, the cheapest sequence of one-cell and two-cell glyphs.

    ``narrow_*`` have shape ``(rows, columns)``: the best one-cell glyph for every cell.
    ``wide_*`` have shape ``(rows, columns - 1)``: the best two-cell glyph for every pair of
    adjacent cells. Costs are additive, so dynamic programming over each row finds the
    tiling with minimal total cost. Returns, per row, ``(span, glyph_index)`` tokens.
    """
    rows, cols = narrow_index.shape
    if wide_index is None or wide_cost is None or cols < 2:
        return [[(1, int(i)) for i in row] for row in narrow_index]

    best = np.zeros((rows, cols + 1), dtype=np.float64)
    use_wide = np.zeros((rows, cols + 1), dtype=bool)
    best[:, 1] = narrow_cost[:, 0]
    for c in range(2, cols + 1):
        single = best[:, c - 1] + narrow_cost[:, c - 1]
        double = best[:, c - 2] + wide_cost[:, c - 2]
        use_wide[:, c] = double < single
        best[:, c] = np.minimum(single, double)

    tiled = []
    for r in range(rows):
        tokens = []
        c = cols
        while c > 0:
            if use_wide[r, c]:
                tokens.append((2, int(wide_index[r, c - 2])))
                c -= 2
            else:
                tokens.append((1, int(narrow_index[r, c - 1])))
                c -= 1
        tokens.reverse()
        tiled.append(tokens)
    return tiled
