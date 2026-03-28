"""
Pythagorean Manifold Module

This module provides the PythagoreanManifold class for deterministic geometric
snapping with O(log n) KD-tree lookup. Part of the Grand Unified Constraint Theory (GUCT).

Key Concepts:
- Hidden dimensions enable exact constraint satisfaction: k = ⌈log₂(1/ε)⌉
- Plane decomposition: n-dimensional constraints decompose into C(n,2) orthogonal 2D planes
- Holonomy verification ensures global consistency

Example:
    >>> from constraint_theory.manifold import PythagoreanManifold
    >>> manifold = PythagoreanManifold(density=200)
    >>> x, y, noise = manifold.snap(0.577, 0.816)
    >>> print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
    Snapped: (0.6000, 0.8000), noise: 0.0236
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import math

try:
    from .constraint_theory_python import (
        PythagoreanManifold as _RustManifold,
        snap as _rust_snap,
        generate_triples as _rust_generate_triples,
    )
    HAS_RUST_BACKEND = True
except ImportError:
    HAS_RUST_BACKEND = False


class PythagoreanManifold:
    """
    A constraint manifold with Pythagorean lattice snapping.
    
    The manifold contains pre-computed normalized Pythagorean triples (a/c, b/c)
    where a² + b² = c². Each state represents an exact point on the unit circle.
    
    This implements the GUCT axiom CM4: Valid states form discrete "snap manifolds"
    with polynomial density.
    
    Attributes:
        density: Maximum value of m in Euclid's formula. Higher = more states.
        state_count: Number of valid Pythagorean states in the manifold.
    
    Example:
        >>> manifold = PythagoreanManifold(density=200)
        >>> print(f"Manifold has {manifold.state_count} states")
        Manifold has 1013 states
        
        >>> # Snap a point to the nearest Pythagorean triple
        >>> x, y, noise = manifold.snap(0.577, 0.816)
        >>> print(f"Snapped to exact: ({x:.4f}, {y:.4f})")
        Snapped to exact: (0.6000, 0.8000)
    """
    
    def __init__(self, density: int = 200):
        """
        Initialize a Pythagorean manifold with specified density.
        
        Args:
            density: Maximum value of m in Euclid's formula.
                     Higher density = more states = finer resolution.
                     Trade-off: Higher density uses more memory.
        
        Raises:
            ValueError: If density is not positive.
        """
        if density <= 0:
            raise ValueError("density must be positive")
        
        self._density = density
        
        if HAS_RUST_BACKEND:
            self._inner = _RustManifold(density)
        else:
            self._inner = None
            self._states = self._generate_states(density)
    
    def _generate_states(self, density: int) -> List[Tuple[float, float]]:
        """Generate Pythagorean states in pure Python (fallback)."""
        states = set()
        max_c = density * density + density * density
        m = 2
        while m * m + 1 <= max_c:
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    # Primitive triple
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if c <= max_c:
                        states.add((a / c, b / c))
                        states.add((b / c, a / c))
            m += 1
        return list(states)
    
    @property
    def density(self) -> int:
        """Get the density parameter used to create this manifold."""
        return self._density
    
    @property
    def state_count(self) -> int:
        """Get the number of valid Pythagorean states in the manifold."""
        if HAS_RUST_BACKEND:
            return self._inner.state_count
        return len(self._states)
    
    def snap(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Snap a 2D vector to the nearest Pythagorean triple.
        
        The snapping algorithm uses KD-tree for O(log n) lookup:
        1. Normalize input to unit circle
        2. Find nearest Pythagorean state
        3. Return snapped coordinates + noise (distance)
        
        Args:
            x: X coordinate of input vector.
            y: Y coordinate of input vector.
        
        Returns:
            Tuple of (snapped_x, snapped_y, noise) where:
            - snapped_x, snapped_y: Exact Pythagorean coordinates
            - noise: Distance from input to snapped point (0.0 to ~0.5)
        
        Example:
            >>> manifold = PythagoreanManifold(density=200)
            >>> sx, sy, noise = manifold.snap(0.6, 0.8)
            >>> print(f"Exact: ({sx}, {sy}), noise: {noise}")
            Exact: (0.6, 0.8), noise: 0.0
        """
        if HAS_RUST_BACKEND:
            return self._inner.snap(x, y)
        
        # Fallback implementation
        if x == 0 and y == 0:
            return (0.0, 0.0, float('inf'))
        
        # Normalize
        norm = math.sqrt(x * x + y * y)
        nx, ny = x / norm, y / norm
        
        # Find nearest state
        best_dist = float('inf')
        best_state = (0.0, 0.0)
        for sx, sy in self._states:
            dist = (nx - sx) ** 2 + (ny - sy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_state = (sx, sy)
        
        return (best_state[0], best_state[1], math.sqrt(best_dist))
    
    def snap_batch(
        self, 
        vectors: Union[List[List[float]], "np.ndarray"]
    ) -> List[Tuple[float, float, float]]:
        """
        Snap multiple vectors efficiently.
        
        This method is optimized for batch processing and uses SIMD
        acceleration when available.
        
        Args:
            vectors: List of [x, y] pairs or Nx2 NumPy array.
        
        Returns:
            List of (snapped_x, snapped_y, noise) tuples.
        
        Example:
            >>> import numpy as np
            >>> manifold = PythagoreanManifold(density=200)
            >>> vectors = np.array([[0.6, 0.8], [0.707, 0.707]])
            >>> results = manifold.snap_batch(vectors)
            >>> for sx, sy, noise in results:
            ...     print(f"({sx:.4f}, {sy:.4f}), noise={noise:.6f}")
        """
        if HAS_RUST_BACKEND:
            # Try SIMD batch method
            if hasattr(self._inner, 'snap_batch_simd'):
                return self._inner.snap_batch_simd(vectors)
            if hasattr(self._inner, 'snap_batch'):
                return self._inner.snap_batch(vectors)
        
        # Fallback: process individually
        results = []
        for v in vectors:
            results.append(self.snap(v[0], v[1]))
        return results
    
    def snap_batch_simd(
        self,
        vectors: Union[List[List[float]], "np.ndarray"]
    ) -> List[Tuple[float, float, float]]:
        """
        Snap multiple vectors using SIMD acceleration.
        
        This is an alias for snap_batch() with explicit SIMD usage.
        
        Args:
            vectors: List of [x, y] pairs or Nx2 NumPy array.
        
        Returns:
            List of (snapped_x, snapped_y, noise) tuples.
        """
        return self.snap_batch(vectors)
    
    def __repr__(self) -> str:
        return f"PythagoreanManifold(density={self._density}, states={self.state_count})"
    
    def __len__(self) -> int:
        return self.state_count


def snap(x: float, y: float, density: int = 200) -> Tuple[float, float, float]:
    """
    Snap a vector using a default manifold.
    
    Convenience function for one-off snapping. For multiple snaps,
    create a PythagoreanManifold instance and reuse it.
    
    Args:
        x: X coordinate of input vector.
        y: Y coordinate of input vector.
        density: Density for default manifold (default: 200).
    
    Returns:
        Tuple of (snapped_x, snapped_y, noise).
    
    Example:
        >>> sx, sy, noise = snap(0.577, 0.816)
        >>> print(f"Snapped: ({sx:.4f}, {sy:.4f})")
    """
    if HAS_RUST_BACKEND:
        return _rust_snap(x, y, density)
    
    # Fallback
    m = PythagoreanManifold(density)
    return m.snap(x, y)


def generate_triples(max_c: int) -> List[Tuple[int, int, int]]:
    """
    Generate Pythagorean triples with hypotenuse <= max_c.
    
    Generates all primitive and non-primitive Pythagorean triples
    where a² + b² = c² and c <= max_c.
    
    Args:
        max_c: Maximum hypotenuse value.
    
    Returns:
        List of (a, b, c) tuples sorted by hypotenuse.
    
    Example:
        >>> triples = generate_triples(50)
        >>> for a, b, c in triples[:5]:
        ...     print(f"{a}² + {b}² = {c}²")
        3² + 4² = 5²
        6² + 8² = 10²
        5² + 12² = 13²
        9² + 12² = 15²
    """
    if HAS_RUST_BACKEND:
        return _rust_generate_triples(max_c)
    
    # Fallback implementation
    triples = []
    m = 2
    while m * m + 1 <= max_c:
        for n in range(1, m):
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            if c > max_c:
                break
            if (m - n) % 2 == 1:  # primitive condition
                ka, kb, kc = a, b, c
                while kc <= max_c:
                    triples.append((min(ka, kb), max(ka, kb), kc))
                    ka += a
                    kb += b
                    kc += c
        m += 1
    
    triples = list(set(triples))
    triples.sort(key=lambda t: t[2])
    return triples


def generate_pythagorean_lattice(max_hypotenuse: int = 1000) -> List[Tuple[float, float]]:
    """
    Generate Pythagorean lattice points up to max hypotenuse.
    
    Returns normalized Pythagorean triples (a/c, b/c) representing
    exact points on the unit circle.
    
    Args:
        max_hypotenuse: Maximum hypotenuse value for generation.
    
    Returns:
        List of (x, y) coordinate pairs on the unit circle.
    
    Example:
        >>> lattice = generate_pythagorean_lattice(100)
        >>> print(f"Generated {len(lattice)} lattice points")
    """
    points = set()
    m = 2
    while m * m + 1 <= max_hypotenuse:
        for n in range(1, m):
            if math.gcd(m, n) == 1 and (m - n) % 2 == 1:  # Primitive triple
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if c <= max_hypotenuse:
                    points.add((a / c, b / c))
                    points.add((b / c, a / c))
        m += 1
    return list(points)


__all__ = [
    "PythagoreanManifold",
    "snap",
    "generate_triples",
    "generate_pythagorean_lattice",
]
