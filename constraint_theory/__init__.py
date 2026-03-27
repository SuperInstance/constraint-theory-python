"""
Constraint Theory - Python Bindings

Deterministic geometric snapping with O(log n) KD-tree lookup.
"""

from .constraint_theory import (
    PythagoreanManifold,
    snap,
    generate_triples,
    __version__,
)

__all__ = [
    "PythagoreanManifold",
    "snap",
    "generate_triples",
    "__version__",
]
