"""Holonomy verification — consistency checking around closed cycles.

Measures accumulated inconsistency when parallel-transporting a vector
around a closed loop. Zero holonomy (identity matrix) means globally
consistent constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HolonomyResult:
    """Result of a holonomy computation.

    Attributes
    ----------
    matrix : list[list[float]]
        3x3 holonomy matrix (rotation matrices product around cycle).
    frobenius_deviation : float
        |Hol(gamma) - I|_F, deviation from identity.
    is_identity : bool
        Whether holonomy is identity (globally consistent).
    angular_deviation : float
        Angular deviation in radians from identity.
    information_content : float
        I = -log2|Hol(gamma)|, infinite for exact identity.
    """

    matrix: list[list[float]]
    frobenius_deviation: float = 0.0
    is_identity: bool = True
    angular_deviation: float = 0.0
    information_content: float = float("inf")


def _rotation_matrix_2d(angle: float) -> list[list[float]]:
    """Create a 2D rotation matrix embedded in 3x3."""
    c, s = math.cos(angle), math.sin(angle)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _frobenius_norm(m: list[list[float]]) -> float:
    """Compute Frobenius norm of a matrix."""
    return math.sqrt(sum(x * x for row in m for x in row))


class HolonomyChecker:
    """Incremental cycle verification for holonomy.

    Provides a step-by-step API for building and verifying cycles.

    Examples
    --------
    >>> checker = HolonomyChecker()
    >>> checker.apply(0.1)
    >>> checker.apply(-0.1)
    >>> result = checker.check_closed()
    >>> result.is_identity
    True
    """

    def __init__(self) -> None:
        self._matrix: list[list[float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self._angles: list[float] = []

    def apply(self, angle: float) -> HolonomyChecker:
        """Apply a rotation to the holonomy accumulator.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.

        Returns
        -------
        HolonomyChecker
            self, for chaining.
        """
        rot = _rotation_matrix_2d(angle)
        self._matrix = _mat_mul(rot, self._matrix)
        self._angles.append(angle)
        return self

    def check_partial(self) -> HolonomyResult:
        """Check current holonomy without requiring a closed cycle."""
        identity = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        diff = [[self._matrix[i][j] - identity[i][j] for j in range(3)] for i in range(3)]
        dev = _frobenius_norm(diff)
        # Angular deviation from trace: trace(R) = 1 + 2*cos(theta)
        trace = sum(self._matrix[i][i] for i in range(3))
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = math.acos(cos_theta)

        det = (
            self._matrix[0][0]
            * (self._matrix[1][1] * self._matrix[2][2] - self._matrix[1][2] * self._matrix[2][1])
            - self._matrix[0][1]
            * (self._matrix[1][0] * self._matrix[2][2] - self._matrix[1][2] * self._matrix[2][0])
            + self._matrix[0][2]
            * (self._matrix[1][0] * self._matrix[2][1] - self._matrix[1][1] * self._matrix[2][0])
        )
        info = -math.log2(abs(det)) if abs(det) > 1e-15 else float("inf")

        return HolonomyResult(
            matrix=[row[:] for row in self._matrix],
            frobenius_deviation=dev,
            is_identity=dev < 1e-6,
            angular_deviation=angle,
            information_content=info,
        )

    def check_closed(self) -> HolonomyResult:
        """Check holonomy for a closed cycle.

        For a closed cycle, sum of angles should be ~0 (mod 2*pi).
        Zero holonomy means globally consistent constraints.
        """
        return self.check_partial()

    def reset(self) -> None:
        """Reset the accumulator to identity."""
        self._matrix = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        self._angles.clear()


def compute_holonomy(angles: list[float]) -> HolonomyResult:
    """Compute holonomy for a sequence of rotation angles.

    Parameters
    ----------
    angles : list[float]
        Rotation angles in radians.

    Returns
    -------
    HolonomyResult
        Holonomy analysis result.
    """
    checker = HolonomyChecker()
    for angle in angles:
        checker.apply(angle)
    return checker.check_closed()
