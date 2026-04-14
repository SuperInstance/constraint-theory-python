"""KD-Tree spatial index for O(log N) nearest neighbor queries in 2D.

Provides deterministic tie-breaking via index ordering — critical for
consensus-critical code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


MAX_LEAF_SIZE = 16


@dataclass
class KDTree:
    """2D spatial index with O(log N) nearest-neighbor queries.

    Build: Recursive median-split construction alternating x/y dimensions. O(N log N).
    Query: Branch-and-bound with backtracking. O(log N) average.
    """

    points: list[tuple[float, float]] = field(default_factory=list)
    _root: Optional[_KDNode] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.points:
            indexed = list(enumerate(self.points))
            self._root = self._build(indexed, 0)

    def nearest(self, query: tuple[float, float]) -> tuple[float, float]:
        """Find the nearest point to the query.

        Deterministic tie-breaking: lower index wins when distances are equal.
        """
        if not self._root:
            raise ValueError("Tree is empty")
        best = _Best(point=self.points[0], dist_sq=float("inf"), idx=0)
        self._search(self._root, query, best, 0)
        return best.point

    def nearest_k(self, query: tuple[float, float], k: int = 5) -> list[tuple[float, float]]:
        """Find the k nearest points to the query."""
        # Brute force for simplicity — the Python implementation prioritizes
        # correctness and clarity over raw performance.
        dists = sorted(
            enumerate(self.points),
            key=lambda ip: (query[0] - ip[1][0]) ** 2 + (query[1] - ip[1][1]) ** 2,
        )
        return [p for _, p in dists[:k]]

    # ---- private ----

    def _build(self, indexed: list[tuple[int, tuple[float, float]]], depth: int) -> _KDNode | None:
        if not indexed:
            return None
        dim = depth % 2
        indexed.sort(key=lambda ip: ip[1][dim])
        mid = len(indexed) // 2
        node = _KDNode(
            idx=indexed[mid][0],
            point=indexed[mid][1],
            dim=dim,
        )
        node.left = self._build(indexed[:mid], depth + 1)
        node.right = self._build(indexed[mid + 1 :], depth + 1)
        return node

    def _search(self, node: _KDNode | None, query: tuple[float, float], best: _Best, depth: int) -> None:
        if node is None:
            return
        dx = query[0] - node.point[0]
        dy = query[1] - node.point[1]
        dist_sq = dx * dx + dy * dy
        # Deterministic tie-breaking: lower index wins
        if dist_sq < best.dist_sq or (dist_sq == best.dist_sq and node.idx < best.idx):
            best.point = node.point
            best.dist_sq = dist_sq
            best.idx = node.idx

        dim = depth % 2
        diff = query[dim] - node.point[dim]
        first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._search(first, query, best, depth + 1)
        if diff * diff <= best.dist_sq:
            self._search(second, query, best, depth + 1)


@dataclass
class _KDNode:
    idx: int
    point: tuple[float, float]
    dim: int
    left: _KDNode | None = None
    right: _KDNode | None = None


@dataclass
class _Best:
    point: tuple[float, float]
    dist_sq: float
    idx: int
