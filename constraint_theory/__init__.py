"""
Constraint Theory - Python Bindings

Deterministic geometric snapping with O(log n) KD-tree lookup.

This package provides Python bindings for the Constraint Theory library,
enabling exact constraint satisfaction in Python applications.

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

__version__ = "0.3.0"

# Core manifold module
from .manifold import (
    PythagoreanManifold,
    snap,
    generate_triples,
    generate_pythagorean_lattice,
)

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

# Try to import Rust backend for performance
try:
    from .constraint_theory_python import (
        PythagoreanManifold as _RustManifold,
        snap as _rust_snap,
        generate_triples as _rust_generate_triples,
    )
    HAS_RUST_BACKEND = True
except ImportError:
    HAS_RUST_BACKEND = False

__all__ = [
    # Version
    "__version__",
    "HAS_RUST_BACKEND",
    
    # Core manifold
    "PythagoreanManifold",
    "snap",
    "generate_triples",
    "generate_pythagorean_lattice",
    
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
