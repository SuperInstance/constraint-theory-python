"""
Constraint Theory - Python Bindings

Deterministic geometric snapping with O(log n) KD-tree lookup.
"""

try:
    from .constraint_theory_python import (
        PythagoreanManifold,
        snap,
        generate_triples,
        __version__,
    )
except ImportError:
    # Fallback for development
    __version__ = "0.1.0"
    
    class PythagoreanManifold:
        def __init__(self, density: int = 200):
            raise ImportError(
                "constraint-theory is not properly installed. "
                "Please install with: pip install constraint-theory"
            )
    
    def snap(x: float, y: float, density: int = 200):
        raise ImportError(
            "constraint-theory is not properly installed. "
            "Please install with: pip install constraint-theory"
        )
    
    def generate_triples(max_c: int):
        raise ImportError(
            "constraint-theory is not properly installed. "
            "Please install with: pip install constraint-theory"
        )

__all__ = [
    "PythagoreanManifold",
    "snap",
    "generate_triples",
    "__version__",
]
