"""Perceptual embedding of glyph-sized pixel patches.

Every image cell (the block of pixels one glyph covers) and every glyph bitmap is mapped to a
vector so that the squared Euclidean distance between two vectors approximates how different
the two patches *look*. Matching is then a nearest-neighbour search. The embedding is a
dense, weighted generalisation of the DCT perceptual hash (pHash):

1. Take the orthonormal 2-D DCT-II of the patch.
2. Attenuate every coefficient with a Gaussian contrast-sensitivity weight. This is the same
   as blurring the patch by ``blur`` pixels: the eye integrates neighbouring pixels, so
   coarse structure matters more than single-pixel detail.
3. Optionally (``perceptual=True``) convert the blurred patch from linear light to a roughly
   perceptually uniform lightness scale (the sRGB curve), so that errors in dark regions
   count as much as errors in bright ones, and take the DCT again.
4. Scale the DC coefficient (the patch's mean, i.e. its tone) by ``sqrt(1 + tone_weight)``.
5. Keep only the low-frequency coefficients whose weight is not negligible.

Because the DCT is orthonormal, with ``blur=0``, ``tone_weight=0`` and ``perceptual=False``
the distance is exactly the sum of squared pixel differences.
"""

from __future__ import annotations

import numpy as np

from .color import srgb_encode


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix ``C`` (``C @ x`` transforms a length-``n`` signal)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


class PerceptualEmbedding:
    """Maps ``(..., cell_height, cell_width)`` patches of ink coverage to ``(..., dims)`` vectors."""

    def __init__(
        self,
        cell_height: int,
        cell_width: int,
        blur: float = 1.5,
        tone_weight: float = 1.0,
        perceptual: bool = False,
        min_weight: float = 0.02,
    ) -> None:
        if blur < 0 or tone_weight < 0:
            raise ValueError("blur and tone_weight must be non-negative")
        self.cell_height = cell_height
        self.cell_width = cell_width
        self.blur = blur
        self.tone_weight = tone_weight
        self.perceptual = perceptual

        # Basis[k] is the k-th 2-D DCT basis image, flattened; frequencies are in cycles/pixel.
        basis = np.kron(dct_matrix(cell_height), dct_matrix(cell_width))
        fy = np.arange(cell_height) / (2.0 * cell_height)
        fx = np.arange(cell_width) / (2.0 * cell_width)
        freq2 = (fy[:, None] ** 2 + fx[None, :] ** 2).ravel()
        lowpass = np.exp(-2.0 * np.pi**2 * blur**2 * freq2)

        keep = np.flatnonzero(lowpass >= min_weight)
        keep = keep[np.argsort(freq2[keep], kind="stable")]
        weights = np.ones(len(keep)) if perceptual else lowpass[keep]
        weights[0] *= np.sqrt(1.0 + tone_weight)  # keep[0] is the DC coefficient

        self._projection = (basis[keep] * weights[:, None]).T.astype(np.float32)
        # Spatial blur operator (applied before the perceptual transfer curve).
        self._blur = (basis.T @ (lowpass[:, None] * basis)).T.astype(np.float32)

    @property
    def dims(self) -> int:
        return self._projection.shape[1]

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches, dtype=np.float32)
        lead = patches.shape[:-2]
        flat = patches.reshape(-1, self.cell_height * self.cell_width)
        if self.perceptual:
            flat = srgb_encode(flat @ self._blur)
        return (flat @ self._projection).reshape(*lead, self.dims)
