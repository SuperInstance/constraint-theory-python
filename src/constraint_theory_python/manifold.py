"""Pythagorean manifold — the core data structure for exact constraint satisfaction.

Implements the snapping operation that maps continuous 2D vectors to the nearest
exact Pythagorean point on the unit circle S^1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .errors import (
    NaNInputError,
    ZeroVectorError,
    InfinityInputError,
    InvalidDimensionError,
    ManifoldEmptyError,
)
from .kdtree import KDTree


@dataclass(frozen=True)
class PythagoreanTriple:
    """A triple (a, b, c) where a^2 + b^2 = c^2 exactly.

    Represents the fundamental geometric constraint. Generated via Euclid's formula:
        a = m^2 - n^2,  b = 2mn,  c = m^2 + n^2
    where m > n > 0, (m - n) is odd, and gcd(m, n) = 1.
    """

    a: int
    b: int
    c: int

    def normalized(self) -> tuple[float, float]:
        """Return the normalized point (a/c, b/c) on the unit circle."""
        return (self.a / self.c, self.b / self.c)


def _gcd(a: int, b: int) -> int:
    """Stein's binary GCD algorithm."""
    if a == 0:
        return b
    if b == 0:
        return a
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
    while (a & 1) == 0:
        a >>= 1
    while b != 0:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            a, b = b, a
        b -= a
    return a << shift


def _generate_triples(density: int) -> list[PythagoreanTriple]:
    """Generate primitive Pythagorean triples using Euclid's formula."""
    triples: list[PythagoreanTriple] = []
    for m in range(2, density + 1):
        for n in range(1, m):
            if (m - n) % 2 == 0:
                continue
            if _gcd(m, n) != 1:
                continue
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            triples.append(PythagoreanTriple(a, b, c))
    return triples


def _triples_to_vectors(triples: list[PythagoreanTriple]) -> list[tuple[float, float]]:
    """Convert triples to normalized unit vectors with quadrant reflections."""
    vectors: list[tuple[float, float]] = []
    # Cardinal directions
    vectors.extend([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
    for t in triples:
        nx, ny = t.normalized()
        vectors.append((nx, ny))
        vectors.append((-nx, ny))
        vectors.append((nx, -ny))
        vectors.append((-nx, -ny))
        vectors.append((ny, nx))
        vectors.append((-ny, nx))
        vectors.append((ny, -nx))
        vectors.append((-ny, -nx))
    return vectors


class PythagoreanManifold:
    """Precomputed set of exact Pythagorean vectors on S^1 with KD-tree index.

    The central entry point for all snapping operations. Maps continuous 2D
    vectors to the nearest exact rational point on the unit circle.

    Parameters
    ----------
    density : int
        Controls the number of precomputed states. Higher density = finer
        resolution. Recommended range: 50-500.

    Examples
    --------
    >>> manifold = PythagoreanManifold(200)
    >>> exact, noise = snap(manifold, (0.577, 0.816))
    >>> exact
    (0.6, 0.8)
    """

    def __init__(self, density: int = 200) -> None:
        if density <= 0:
            raise ValueError("density must be a positive integer")
        self.density = density
        self._triples = _generate_triples(density)
        self._vectors = _triples_to_vectors(self._triples)
        self._tree = KDTree(self._vectors)

    def snap(self, vec: tuple[float, float]) -> tuple[tuple[float, float], float]:
        """Snap a 2D vector to the nearest exact Pythagorean point.

        Parameters
        ----------
        vec : tuple[float, float]
            Input 2D vector (x, y).

        Returns
        -------
        tuple[tuple[float, float], float]
            (snapped_vector, noise_distance). The snapped vector lies exactly
            on the unit circle. Noise is the Euclidean distance between
            input and snapped vector.

        Raises
        ------
        ZeroVectorError
            If the input is (0, 0).
        NaNInputError
            If the input contains NaN.
        InfinityInputError
            If the input contains Infinity.
        """
        x, y = vec
        if len(vec) != 2:
            raise InvalidDimensionError("Expected 2D vector (x, y)")
        if math.isnan(x) or math.isnan(y):
            raise NaNInputError("Input contains NaN values")
        if math.isinf(x) or math.isinf(y):
            raise InfinityInputError("Input contains Infinity values")
        if x == 0.0 and y == 0.0:
            raise ZeroVectorError("Cannot snap the zero vector")

        # Normalize to unit length
        mag = math.sqrt(x * x + y * y)
        nx, ny = x / mag, y / mag

        nearest = self._tree.nearest((nx, ny))
        noise = math.sqrt((nx - nearest[0]) ** 2 + (ny - nearest[1]) ** 2)
        return (nearest, noise)

    def snap_batch(self, vectors: list[tuple[float, float]]) -> list[tuple[tuple[float, float], float]]:
        """Snap multiple vectors in a single call.

        Parameters
        ----------
        vectors : list[tuple[float, float]]
            List of 2D vectors.

        Returns
        -------
        list[tuple[tuple[float, float], float]]
            List of (snapped_vector, noise_distance) pairs.
        """
        return [self.snap(v) for v in vectors]

    @property
    def triple_count(self) -> int:
        """Number of primitive Pythagorean triples generated."""
        return len(self._triples)

    @property
    def state_count(self) -> int:
        """Total number of discrete states (vectors) on the manifold."""
        return len(self._vectors)

    def __repr__(self) -> str:
        return (
            f"PythagoreanManifold(density={self.density}, "
            f"triples={self.triple_count}, states={self.state_count})"
        )


def snap(manifold: PythagoreanManifold, vec: tuple[float, float]) -> tuple[tuple[float, float], float]:
    """Convenience function: snap a vector to the nearest exact Pythagorean point.

    Parameters
    ----------
    manifold : PythagoreanManifold
        The precomputed manifold.
    vec : tuple[float, float]
        Input 2D vector.

    Returns
    -------
    tuple[tuple[float, float], float]
        (snapped_vector, noise_distance).
    """
    return manifold.snap(vec)
