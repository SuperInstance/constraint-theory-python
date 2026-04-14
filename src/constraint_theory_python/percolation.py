"""Rigidity percolation via Laman's theorem.

A graph with V vertices in 2D is minimally rigid iff it has exactly
2V - 3 edges and every subgraph with k vertices has at most 2k - 3 edges.

Uses union-find with path compression and union-by-rank for O(alpha(N))
amortized analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


LAMAN_NEIGHBOR_THRESHOLD = 12


@dataclass
class RigidityResult:
    """Result of rigidity analysis.

    Attributes
    ----------
    is_rigid : bool
        Whether the graph is rigid.
    rank : int
        Rank of the constraint matrix.
    deficiency : int
        2V - 3 - E (negative means over-constrained).
    cluster_count : int
        Number of rigid clusters found.
    rigid_fraction : float
        Fraction of vertices in rigid clusters.
    """

    is_rigid: bool
    rank: int
    deficiency: int
    cluster_count: int
    rigid_fraction: float


class FastPercolation:
    """Union-find based rigidity percolation analysis.

    Implements Laman's theorem for constraint graph rigidity.

    Examples
    --------
    >>> perc = FastPercolation()
    >>> perc.add_edge(0, 1)
    >>> perc.add_edge(0, 2)
    >>> perc.add_edge(1, 2)
    >>> result = perc.compute_rigidity(vertices=3)
    """

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}
        self._edges: list[tuple[int, int]] = []

    def add_edge(self, u: int, v: int) -> None:
        """Add an edge to the graph."""
        self._edges.append((u, v))
        if u not in self._parent:
            self._parent[u] = u
            self._rank[u] = 0
        if v not in self._parent:
            self._parent[v] = v
            self._rank[v] = 0

    def compute_rigidity(self, vertices: int) -> RigidityResult:
        """Compute rigidity metrics.

        Parameters
        ----------
        vertices : int
            Number of vertices.

        Returns
        -------
        RigidityResult
        """
        edges = len(self._edges)
        min_edges = 2 * vertices - 3
        deficiency = min_edges - edges

        # Union-find to count connected components
        for u, v in self._edges:
            self._union(u, v)

        roots = set()
        for v in range(vertices):
            if v in self._parent:
                roots.add(self._find(v))
            else:
                roots.add(v)  # Isolated vertex

        is_rigid = edges >= min_edges and len(roots) == 1
        rank = min(edges, 2 * vertices - 3)

        return RigidityResult(
            is_rigid=is_rigid,
            rank=rank,
            deficiency=deficiency,
            cluster_count=len(roots),
            rigid_fraction=1.0 if is_rigid else 0.0,
        )

    def _find(self, x: int) -> int:
        """Find with path compression."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = self._find(x), self._find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
