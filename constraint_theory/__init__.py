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

__version__ = "1.0.1"

# Version pinning requirements (PASS 8)
# Compatible with constraint-theory-core >= 1.0.0, < 2.0.0
CORE_MIN_VERSION = (1, 0, 0)
CORE_MAX_VERSION = (2, 0, 0)


# ============================================
# Exception Classes
# ============================================

class ConstraintTheoryError(Exception):
    """
    Base exception for all constraint theory errors.
    
    All exceptions in this library inherit from this class, making it easy
    to catch all constraint theory related errors with a single except clause.
    
    Attributes:
        message: Human-readable error description
        code: Error code for programmatic handling
        details: Additional context about the error
    """
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class InputValidationError(ConstraintTheoryError):
    """
    Raised when input validation fails.
    
    This exception indicates that the input provided to a function
    does not meet the required criteria.
    
    Common causes:
    - NaN or Infinity values in numeric input
    - Zero vector where non-zero is required
    - Invalid dimension or shape
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="INPUT_VALIDATION_ERROR", details=details)


class NaNInputError(InputValidationError):
    """Raised when input contains NaN values."""
    
    def __init__(self, parameter_name: str = "input"):
        super().__init__(
            f"Input contains NaN (Not a Number) values. "
            f"Parameter '{parameter_name}' must contain only valid numbers.",
            details={"parameter": parameter_name, "issue": "nan_value"}
        )


class InfinityInputError(InputValidationError):
    """Raised when input contains Infinity values."""
    
    def __init__(self, parameter_name: str = "input"):
        super().__init__(
            f"Input contains Infinity values. "
            f"Parameter '{parameter_name}' must contain only finite numbers.",
            details={"parameter": parameter_name, "issue": "infinity_value"}
        )


class ZeroVectorError(InputValidationError):
    """Raised when a zero vector is provided where a non-zero vector is required."""
    
    def __init__(self, operation: str = "snap"):
        super().__init__(
            f"Zero vector provided for operation '{operation}'. "
            f"A non-zero vector is required for normalization.",
            details={"operation": operation, "issue": "zero_vector"}
        )


class ManifoldError(ConstraintTheoryError):
    """Base class for manifold-related errors."""
    
    def __init__(self, message: str, code: str = "MANIFOLD_ERROR", details: Optional[dict] = None):
        super().__init__(message, code=code, details=details)


class InvalidDensityError(ManifoldError):
    """Raised when an invalid density parameter is provided."""
    
    def __init__(self, density: int, reason: str = "must be positive"):
        super().__init__(
            f"Invalid density value: {density}. {reason.capitalize()}. "
            f"Recommended range: 50-500 for most applications.",
            code="INVALID_DENSITY",
            details={"density": density, "reason": reason}
        )


class QuantizationError(ConstraintTheoryError):
    """Base class for quantization-related errors."""
    
    def __init__(self, message: str, code: str = "QUANTIZATION_ERROR", details: Optional[dict] = None):
        super().__init__(message, code=code, details=details)


class UnsupportedModeError(QuantizationError):
    """Raised when an unsupported quantization mode is requested."""
    
    def __init__(self, mode: str, supported_modes: List[str]):
        super().__init__(
            f"Unsupported quantization mode: '{mode}'. "
            f"Supported modes: {', '.join(supported_modes)}",
            code="UNSUPPORTED_MODE",
            details={"requested_mode": mode, "supported_modes": supported_modes}
        )


class ConstraintViolationError(ConstraintTheoryError):
    """Raised when a constraint cannot be satisfied during quantization."""
    
    def __init__(self, constraint: str, reason: str):
        super().__init__(
            f"Constraint violation: '{constraint}' cannot be satisfied. {reason}",
            code="CONSTRAINT_VIOLATION",
            details={"constraint": constraint, "reason": reason}
        )


class BufferSizeMismatchError(ConstraintTheoryError):
    """Raised when input and output buffer sizes don't match."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            f"Buffer size mismatch: input has {input_size} elements, "
            f"output has {output_size} elements. Sizes must match.",
            code="BUFFER_SIZE_MISMATCH",
            details={"input_size": input_size, "output_size": output_size}
        )


# ============================================
# Validation Utilities
# ============================================

def validate_vector_2d(x: float, y: float, param_name: str = "vector") -> None:
    """
    Validate a 2D vector for use in snapping operations.
    
    Args:
        x: X coordinate
        y: Y coordinate
        param_name: Parameter name for error messages
        
    Raises:
        NaNInputError: If x or y is NaN
        InfinityInputError: If x or y is Infinity
    """
    import math
    
    if math.isnan(x) or math.isnan(y):
        raise NaNInputError(param_name)
    if math.isinf(x) or math.isinf(y):
        raise InfinityInputError(param_name)


def validate_density(density: int) -> None:
    """
    Validate density parameter for manifold creation.
    
    Args:
        density: Density value to validate
        
    Raises:
        InvalidDensityError: If density is invalid
    """
    if not isinstance(density, int) or density <= 0:
        raise InvalidDensityError(density, "must be a positive integer")


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
    
    # Exceptions
    "ConstraintTheoryError",
    "InputValidationError",
    "NaNInputError",
    "InfinityInputError",
    "ZeroVectorError",
    "ManifoldError",
    "InvalidDensityError",
    "QuantizationError",
    "UnsupportedModeError",
    "ConstraintViolationError",
    "BufferSizeMismatchError",
    
    # Validation utilities
    "validate_vector_2d",
    "validate_density",
    
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
