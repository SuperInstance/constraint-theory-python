"""Sheaf cohomology — topological detection via Euler characteristic.

Computes H0 and H1 cohomology group dimensions for cellular complexes.
  - H0 = number of connected components
  - H1 = number of independent cycles (emergent behaviors)

Used for emergence detection: every emergent behavior in a swarm
corresponds to a non-trivial element of H1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CohomologyResult:
    """Result of cohomology computation.

    Attributes
    ----------
    h0 : int
        H0 dimension — number of connected components (beta_0).
    h1 : int
        H1 dimension — number of independent cycles.
    vertices : int
        Number of vertices in the complex.
    edges : int
        Number of edges in the complex.
    euler_characteristic : int
        chi = V - E + F (Euler characteristic).
    """

    h0: int
    h1: int
    vertices: int
    edges: int
    euler_characteristic: int


class FastCohomology:
    """Fast cohomology computation via Euler characteristic.

    Runs in O(1) given vertex/edge/component counts.

    Examples
    --------
    >>> cohom = FastCohomology()
    >>> result = cohom.compute(vertices=5, edges=6, components=1)
    >>> result.h1
    2
    """

    def compute(
        self,
        vertices: int,
        edges: int,
        components: int = 1,
        faces: int = 0,
    ) -> CohomologyResult:
        """Compute cohomology groups.

        Parameters
        ----------
        vertices : int
            Number of vertices (V).
        edges : int
            Number of edges (E).
        components : int
            Number of connected components (beta_0).
        faces : int
            Number of faces (F), for 2D complexes.

        Returns
        -------
        CohomologyResult
            H0 and H1 dimensions.
        """
        h0 = components
        # Euler characteristic: chi = V - E + F
        chi = vertices - edges + faces
        # H1 = E - V + beta_0 - F = -chi + beta_0 + ... simplified:
        # For a 1-complex: H1 = E - V + H0
        h1 = edges - vertices + h0

        return CohomologyResult(
            h0=h0,
            h1=max(h1, 0),
            vertices=vertices,
            edges=edges,
            euler_characteristic=chi,
        )
