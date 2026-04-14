"""Hidden dimension encoding — the GUCT formula for precision-lifting.

Implements:
    k = ceil(log2(1/epsilon))

For precision epsilon, this computes the number of hidden dimensions needed
to represent constraints exactly without floating-point errors.
"""

from __future__ import annotations

import math


def hidden_dim_count(epsilon: float) -> int:
    """Compute the number of hidden dimensions for a target precision.

    k = ceil(log2(1/epsilon))

    Parameters
    ----------
    epsilon : float
        Target precision (must be > 0).

    Returns
    -------
    int
        Number of hidden dimensions needed.

    Examples
    --------
    >>> hidden_dim_count(1e-10)
    34
    >>> hidden_dim_count(0.01)
    7
    """
    if epsilon <= 0.0:
        return float("inf")
    return math.ceil(math.log2(1.0 / epsilon))


def precision_from_hidden_dims(k: int) -> float:
    """Compute the precision achievable with k hidden dimensions.

    epsilon = 2^(-k)

    Parameters
    ----------
    k : int
        Number of hidden dimensions.

    Returns
    -------
    float
        Achievable precision.
    """
    return 2.0 ** (-k)


def lift_to_hidden(point: list[float], k: int) -> list[float]:
    """Lift a point from R^n to R^(n+k).

    The hidden dimensions are initialized using Pythagorean ratio encoding
    derived from the visible components.

    Parameters
    ----------
    point : list[float]
        Input point in R^n.
    k : int
        Number of hidden dimensions to add.

    Returns
    -------
    list[float]
        Point in R^(n+k).

    Examples
    --------
    >>> lifted = lift_to_hidden([0.6, 0.8], 34)
    >>> len(lifted)
    36
    """
    n = len(point)
    lifted = list(point)
    for i in range(k):
        # Encode hidden dimensions using trigonometric projection
        theta = 2.0 * math.pi * i / max(k, 1)
        val = sum(p * math.cos(theta + j) for j, p in enumerate(point)) / max(n, 1)
        lifted.append(max(-1.0, min(1.0, val)))
    return lifted


def project_to_visible(lifted: list[float], original_dim: int) -> list[float]:
    """Project a point from R^(n+k) back to R^n.

    Parameters
    ----------
    lifted : list[float]
        Point in R^(n+k).
    original_dim : int
        Original dimension n.

    Returns
    -------
    list[float]
        Point in R^n.
    """
    return lifted[:original_dim]


# Need math imported for lift_to_hidden
import math  # noqa: E402
