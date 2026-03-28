"""
Constraint Theory - Python Bindings

Deterministic geometric snapping with O(log n) KD-tree lookup.

Schema Alignment (PASS 5):
==========================
This Python API is designed to match the Rust core (constraint-theory-core) exactly.

Mapping to Rust API:
- Python: PythagoreanManifold(density) -> Rust: PythagoreanManifold::new(density)
- Python: manifold.snap(x, y) -> Rust: manifold.snap([x, y])
- Python: manifold.snap_batch(vectors) -> Rust: manifold.snap_batch_simd(vectors)
- Python: manifold.state_count -> Rust: manifold.state_count()

Key differences from Rust core:
- Python uses tuple unpacking: snap() returns (x, y, noise) instead of ([x, y], noise)
- Python's state_count is a property, not a method
- Python's snap_batch wraps snap_batch_simd internally

WASM vs PyO3 differences:
- WASM uses Float32Array for vectors; PyO3 uses Python lists/tuples
- WASM has no GIL; PyO3 releases GIL for long operations
- WASM returns plain arrays; PyO3 returns Python tuples

Example:
    >>> from constraint_theory import PythagoreanManifold
    >>> manifold = PythagoreanManifold(200)
    >>> x, y, noise = manifold.snap(0.577, 0.816)
    >>> print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
    Snapped: (0.6000, 0.8000), noise: 0.023600

For more information, see:
    https://github.com/SuperInstance/constraint-theory-python
    https://github.com/SuperInstance/constraint-theory-core (Rust implementation)
"""

from typing import List, Tuple, Union, Optional, Protocol, runtime_checkable
import sys

__version__ = "0.1.0"

# Version pinning requirements (PASS 8)
# Compatible with constraint-theory-core >= 1.0.0, < 2.0.0
CORE_MIN_VERSION = (1, 0, 0)
CORE_MAX_VERSION = (2, 0, 0)

# Protocol classes for type checking (PASS 6)
@runtime_checkable
class SnapResult(Protocol):
    """Protocol for snap result - a tuple of (x, y, noise)."""
    def __getitem__(self, index: int) -> float: ...
    def __len__(self) -> int: ...

@runtime_checkable
class Vector2D(Protocol):
    """Protocol for 2D vector input - supports indexing."""
    def __getitem__(self, index: int) -> float: ...
    def __len__(self) -> int: ...

@runtime_checkable
class ManifoldProtocol(Protocol):
    """Protocol defining the manifold interface - matches Rust trait."""
    @property
    def state_count(self) -> int: ...
    def snap(self, x: float, y: float) -> Tuple[float, float, float]: ...
    def snap_batch(self, vectors: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]: ...

try:
    from .constraint_theory_python import (
        PythagoreanManifold,
        snap,
        generate_triples,
    )
except ImportError:
    # Fallback for development - provides helpful error messages
    class PythagoreanManifold:
        """
        Pythagorean manifold for deterministic vector snapping.
        
        Schema Alignment (PASS 5):
        ==========================
        This class wraps the Rust PythagoreanManifold from constraint-theory-core.
        
        Rust API Reference:
        - PythagoreanManifold::new(density: usize) -> Self
        - snap(&self, vector: [f32; 2]) -> ([f32; 2], f32)
        - snap_batch_simd(&self, vectors: &[[f32; 2]]) -> Vec<([f32; 2], f32)>
        - state_count(&self) -> usize
        
        Python Adaptation:
        - __init__(density: int) - matches Rust new()
        - snap(x: float, y: float) -> Tuple[float, float, float]
          Returns (x, y, noise) instead of ([x, y], noise) for Pythonic unpacking
        - snap_batch(vectors) -> List[Tuple[float, float, float]]
          Wraps snap_batch_simd internally
        - state_count (property) - matches Rust method but as property
        
        Note: This is a stub. The actual implementation requires the Rust
        extension to be built. Install with:
        
            pip install constraint-theory
        
        Or build from source:
        
            pip install maturin
            maturin develop --release
        """
        
        def __init__(self, density: int = 200):
            raise ImportError(
                "constraint-theory is not properly installed. "
                "The Rust extension module could not be loaded.\n\n"
                "To install:\n"
                "  pip install constraint-theory\n\n"
                "To build from source:\n"
                "  pip install maturin\n"
                "  maturin develop --release\n\n"
                "For help: https://github.com/SuperInstance/constraint-theory-python\n"
                "Rust core: https://github.com/SuperInstance/constraint-theory-core"
            )
        
        @property
        def state_count(self) -> int:
            """Number of valid Pythagorean states in the manifold."""
            raise ImportError("Rust extension not loaded")
        
        @property
        def density(self) -> int:
            """Density parameter used to create this manifold."""
            raise ImportError("Rust extension not loaded")
        
        def snap(self, x: float, y: float) -> Tuple[float, float, float]:
            """Snap a 2D vector to nearest Pythagorean triple."""
            raise ImportError("Rust extension not loaded")
        
        def snap_batch(self, vectors: Union[List[Tuple[float, float]], 'numpy.ndarray']) -> List[Tuple[float, float, float]]:
            """Batch snap multiple vectors (SIMD optimized)."""
            raise ImportError("Rust extension not loaded")
    
    def snap(x: float, y: float, density: int = 200) -> Tuple[float, float, float]:
        """Stub function - see PythagoreanManifold for details."""
        raise ImportError(
            "constraint-theory is not properly installed. "
            "The Rust extension module could not be loaded.\n\n"
            "To install:\n"
            "  pip install constraint-theory\n\n"
            "To build from source:\n"
            "  pip install maturin\n"
            "  maturin develop --release\n\n"
            "For help: https://github.com/SuperInstance/constraint-theory-python"
        )
    
    def generate_triples(max_c: int) -> List[Tuple[int, int, int]]:
        """Stub function - see PythagoreanManifold for details."""
        raise ImportError(
            "constraint-theory is not properly installed. "
            "The Rust extension module could not be loaded.\n\n"
            "To install:\n"
            "  pip install constraint-theory\n\n"
            "To build from source:\n"
            "  pip install maturin\n"
            "  maturin develop --release\n\n"
            "For help: https://github.com/SuperInstance/constraint-theory-python"
        )

# Type aliases for clarity (PASS 6)
VectorLike = Union[List[Tuple[float, float]], 'numpy.ndarray']
"""Type alias for vector input - supports lists or NumPy arrays."""

SnapResultTuple = Tuple[float, float, float]
"""Type alias for snap result: (snapped_x, snapped_y, noise)."""

PythagoreanTripleTuple = Tuple[int, int, int]
"""Type alias for Pythagorean triple: (a, b, c) where a² + b² = c²."""

__all__ = [
    # Core classes
    "PythagoreanManifold",
    # Functions
    "snap",
    "generate_triples",
    # Version
    "__version__",
    # Type aliases (PASS 6)
    "VectorLike",
    "SnapResultTuple",
    "PythagoreanTripleTuple",
    # Protocols (PASS 6)
    "SnapResult",
    "Vector2D",
    "ManifoldProtocol",
]
