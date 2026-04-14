"""Core error types for constraint theory operations."""


class CTError(Exception):
    """Base error for all constraint theory operations.

    Mirrors the Rust CTErr enum with 11 variants covering input validation,
    state errors, and numerical errors.
    """

    pass


class InvalidDimensionError(CTError):
    """Expected 2D vector input."""

    pass


class ManifoldEmptyError(CTError):
    """Manifold not initialized — call new() first."""

    pass


class NumericalInstabilityError(CTError):
    """Numerical instability detected — input may contain NaN or Infinity."""

    pass


class ZeroVectorError(CTError):
    """Input vector is zero length — cannot normalize."""

    pass


class NaNInputError(CTError):
    """Input contains NaN values."""

    pass


class InfinityInputError(CTError):
    """Input contains Infinity values."""

    pass


class BufferSizeMismatchError(CTError):
    """Input and output buffers have different lengths."""

    pass


class OverflowError(CTError):
    """Numerical overflow detected — value exceeds float max."""

    pass


class DivisionByZeroError(CTError):
    """Division by zero attempted."""

    pass


class InvalidDensityError(CTError):
    """Density parameter must be a positive integer."""

    pass
