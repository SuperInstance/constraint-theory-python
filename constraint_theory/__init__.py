"""
Constraint Theory - Python Bindings

Deterministic geometric snapping with O(log n) KD-tree lookup.

This package provides Python bindings for the Constraint Theory library,
enabling exact constraint satisfaction in Python applications.

Schema Alignment (PASS 5):
==========================
This Python API is designed to match the Rust core (constraint-theory-core) exactly.

Mapping to Rust API:
- Python: PythagoreanManifold(density) -> Rust: PythagoreanManifold::new(density)
- Python: manifold.snap(x, y) -> Rust: manifold.snap([x, y])
- Python: manifold.snap_batch(vectors) -> Rust: manifold.snap_batch_simd(vectors)
- Python: manifold.state_count -> Rust: manifold.state_count()

Key Features:
- PythagoreanManifold: Discrete manifold with exact Pythagorean states
- PythagoreanQuantizer: Unified quantization (TERNARY, POLAR, TURBO, HYBRID)
- ConstraintEnforcedLayer: ML integration for PyTorch/TensorFlow
- Hidden Dimension Encoding: k = ⌈log₂(1/ε)⌉ for exact constraints
- Financial Applications: Exact arithmetic, portfolio optimization, risk constraints

Quick Start:
    >>> from constraint_theory import PythagoreanManifold
    >>> manifold = PythagoreanManifold(density=200)
    >>> x, y, noise = manifold.snap(0.577, 0.816)
    >>> print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
    Snapped: (0.6000, 0.8000), noise: 0.0236

For ML Integration:
    >>> from constraint_theory import ConstraintEnforcedLayer, QuantizationMode
    >>> layer = ConstraintEnforcedLayer(
    ...     input_dim=128,
    ...     output_dim=64,
    ...     constraints=['unit_norm']
    ... )

For Financial Applications:
    >>> from constraint_theory import ExactMoney, PortfolioOptimizer, RiskConstraints
    >>> price = ExactMoney.from_float(100.50)
    >>> print(price.to_string())  # Exact representation, no floating-point drift
"""

from typing import List, Tuple, Union, Optional, Protocol, runtime_checkable
import sys

__version__ = "0.3.0"

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

# Try to import Rust backend for performance
try:
    from .constraint_theory_python import (
        PythagoreanManifold,
        snap,
        generate_triples,
    )
    HAS_RUST_BACKEND = True
except ImportError:
    # Import pure Python fallbacks
    from .manifold import (
        PythagoreanManifold,
        snap,
        generate_triples,
        generate_pythagorean_lattice,
    )
    HAS_RUST_BACKEND = False

# Unified quantizer module
from .quantizer import (
    QuantizationMode,
    QuantizationResult,
    PythagoreanQuantizer,
    auto_select_mode,
    snap_to_pythagorean,
    quantize,
)

# ML integration module
from .ml import (
    ConstraintConfig,
    ConstraintEnforcedLayer,
    HiddenDimensionNetwork,
    GradientSnapper,
)

# Hidden dimension encoding module
from .hidden_dims import (
    HiddenDimConfig,
    compute_hidden_dim_count,
    lift_to_hidden,
    project_visible,
    snap_in_lifted_space,
    generate_nd_lattice,
    encode_with_hidden_dimensions,
    cross_plane_finetune,
    get_orthogonal_planes,
    project_to_plane,
    reconstruct_from_plane,
    constraint_error,
    holographic_accuracy,
)

# Financial applications module
from .financial import (
    RoundingMode,
    ExactMoney,
    RiskConstraints,
    TradingSignal,
    MultiPlaneOptimizer,
    PortfolioOptimizer,
    PriceLevelSnapper,
    calculate_var,
    calculate_sharpe_ratio,
)

# Type aliases for clarity (PASS 6)
VectorLike = Union[List[Tuple[float, float]], 'numpy.ndarray']
"""Type alias for vector input - supports lists or NumPy arrays."""

SnapResultTuple = Tuple[float, float, float]
"""Type alias for snap result: (snapped_x, snapped_y, noise)."""

PythagoreanTripleTuple = Tuple[int, int, int]
"""Type alias for Pythagorean triple: (a, b, c) where a² + b² = c²."""

__all__ = [
    # Version
    "__version__",
    "HAS_RUST_BACKEND",
    
    # Core classes
    "PythagoreanManifold",
    
    # Functions
    "snap",
    "generate_triples",
    "generate_pythagorean_lattice",
    
    # Type aliases (PASS 6)
    "VectorLike",
    "SnapResultTuple",
    "PythagoreanTripleTuple",
    
    # Protocols (PASS 6)
    "SnapResult",
    "Vector2D",
    "ManifoldProtocol",
    
    # Quantization
    "QuantizationMode",
    "QuantizationResult",
    "PythagoreanQuantizer",
    "auto_select_mode",
    "snap_to_pythagorean",
    "quantize",
    
    # ML Integration
    "ConstraintConfig",
    "ConstraintEnforcedLayer",
    "HiddenDimensionNetwork",
    "GradientSnapper",
    
    # Hidden Dimensions
    "HiddenDimConfig",
    "compute_hidden_dim_count",
    "lift_to_hidden",
    "project_visible",
    "snap_in_lifted_space",
    "generate_nd_lattice",
    "encode_with_hidden_dimensions",
    "cross_plane_finetune",
    "get_orthogonal_planes",
    "project_to_plane",
    "reconstruct_from_plane",
    "constraint_error",
    "holographic_accuracy",
    
    # Financial Applications
    "RoundingMode",
    "ExactMoney",
    "RiskConstraints",
    "TradingSignal",
    "MultiPlaneOptimizer",
    "PortfolioOptimizer",
    "PriceLevelSnapper",
    "calculate_var",
    "calculate_sharpe_ratio",
]
