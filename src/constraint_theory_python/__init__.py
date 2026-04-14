"""
Constraint Theory Python — Deterministic manifold snapping via Pythagorean geometry.

This is the Python companion to constraint-theory-core (Rust), implementing the
Grand Unified Constraint Theory (GUCT) for exact constraint satisfaction.
"""

__version__ = "0.1.0"
__all__ = [
    "PythagoreanManifold",
    "PythagoreanTriple",
    "snap",
    "CTError",
    "hidden_dim_count",
    "lift_to_hidden",
    "project_to_visible",
    "Rational",
    "PythagoreanQuantizer",
    "QuantizationMode",
    "QuantizationResult",
    "HolonomyChecker",
    "HolonomyResult",
    "compute_holonomy",
    "FastCohomology",
    "CohomologyResult",
    "RicciFlow",
    "FastPercolation",
    "RigidityResult",
]

from .manifold import PythagoreanManifold, PythagoreanTriple, snap
from .errors import CTError
from .hidden_dimensions import hidden_dim_count, lift_to_hidden, project_to_visible
from .quantizer import (
    Rational,
    PythagoreanQuantizer,
    QuantizationMode,
    QuantizationResult,
)
from .holonomy import HolonomyChecker, HolonomyResult, compute_holonomy
from .cohomology import FastCohomology, CohomologyResult
from .curvature import RicciFlow
from .percolation import FastPercolation, RigidityResult
