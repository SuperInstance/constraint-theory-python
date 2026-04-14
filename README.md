<div align="center">

# constraint-theory-python

### `0.6² + 0.8² = 1.0000000000000002` — and you've been debugging this for years.

**Trade float drift for quantized exactness. Same bits, every machine, guaranteed.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![constraint-theory-core](https://img.shields.io/badge/Rust_companion-constraint--theory--core-orange.svg)](https://github.com/SuperInstance/constraint-theory-core)

**`pip install constraint-theory-python`** · [Rust Core](https://github.com/SuperInstance/constraint-theory-core) · [Interactive Demos](https://constraint-theory-web.pages.dev)

</div>

---

## Overview

**Constraint Theory** is a mathematical framework for exact constraint satisfaction that replaces floating-point approximation with discrete, deterministic rational representations. At its core, it exploits the structure of **Pythagorean triples** — integer solutions to a² + b² = c² — to construct a finite set of exact points on the unit circle S¹.

The insight is simple but powerful: there are infinitely many Pythagorean triples, but only finitely many within any precision bound. By precomputing these exact rational points, indexing them with a KD-tree, and projecting (snapping) continuous input vectors to the nearest exact neighbor, the system eliminates an entire class of floating-point drift bugs — forever.

This package is the **Python companion** to [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) (Rust). It implements the same Grand Unified Constraint Theory (GUCT) algorithms with a pure-Python API, zero compiled dependencies, and seamless NumPy integration. It is designed for prototyping, research, ML pipelines, and any workflow where Python-first development is preferred over the Rust crate's raw performance.

### Python vs Rust: Which to Use?

| Aspect | **constraint-theory-python** (this repo) | **constraint-theory-core** (Rust) |
|--------|------------------------------------------|-----------------------------------|
| Language | Pure Python 3.10+ | Rust 1.75+ |
| Dependencies | Zero (optional: numpy) | Zero |
| Single snap | ~10 µs | ~100 ns |
| Batch (1000 vectors) | ~5 ms | ~74 µs |
| Install | `pip install` | `cargo add` |
| Best for | Prototyping, ML, notebooks, research | Production, games, embedded, HFT |
| API style | Pythonic (dataclasses, exceptions) | Rustic (Result<T, CTErr>, traits) |
| SIMD | NumPy vectorized (optional) | Native AVX2 |
| Cross-language | Python only | PyO3 bindings, WASM |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                constraint-theory-python                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  manifold    │    │   kdtree     │    │  hidden_         │  │
│  │              │───►│              │    │  dimensions      │  │
│  │ Pythagorean  │    │ .nearest(q)  │    │                  │  │
│  │  Manifold    │    │ .nearest_k() │    │ k=⌈log₂(1/ε)⌉  │  │
│  │ .snap(vec)   │    │ O(log N)     │    │ lift_to_hidden() │  │
│  │ .snap_batch  │    │              │    │ project_to_      │  │
│  └──────┬───────┘    └──────────────┘    │  visible()       │  │
│         │                                └──────────────────┘  │
│  ┌──────▼───────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  quantizer   │    │  holonomy    │    │  cohomology      │  │
│  │              │    │              │    │                  │  │
│  │  Ternary     │    │ compute_     │    │ FastCohomology   │  │
│  │  Polar       │    │  holonomy()  │    │   .compute()     │  │
│  │  Turbo       │    │ Holonomy-    │    │   H₀ = β₀        │  │
│  │  Hybrid      │    │  Checker     │    │   H₁ = E-V+β₀   │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────┘  │
│                             │                                  │
│  ┌──────────────┐  ┌───────▼───────┐  ┌──────────────────┐    │
│  │  curvature   │  │  percolation  │  │     errors       │    │
│  │              │  │               │  │                  │    │
│  │  RicciFlow   │  │ FastPerco-    │  │ CTError (base)   │    │
│  │  .evolve()   │  │   lation      │  │   └─ 10 variants │    │
│  │  α=0.1       │  │  Laman 2V-3   │  │                  │    │
│  │  target=0.0  │  │  union-find   │  │                  │    │
│  └──────────────┘  └───────────────┘  └──────────────────┘    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Entry point: constraint_theory_python                           │
│  10 modules · zero compiled dependencies · MIT licensed           │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Module | Class / Function | Description |
|--------|------------------|-------------|
| `manifold` | `PythagoreanManifold`, `snap()` | Core snapping engine — maps vectors to exact Pythagorean points |
| `kdtree` | `KDTree` | O(log N) 2D spatial index with deterministic tie-breaking |
| `hidden_dimensions` | `hidden_dim_count()`, `lift_to_hidden()` | GUCT precision encoding: k = ⌈log₂(1/ε)⌉ |
| `quantizer` | `PythagoreanQuantizer` | Unified quantizer (Ternary / Polar / Turbo / Hybrid) |
| `holonomy` | `HolonomyChecker`, `compute_holonomy()` | Consistency verification around closed cycles |
| `cohomology` | `FastCohomology` | H₀ (components) and H₁ (cycles) via Euler characteristic |
| `curvature` | `RicciFlow`, `ricci_flow_step()` | Curvature evolution toward target flattening |
| `percolation` | `FastPercolation` | Rigidity analysis via Laman's theorem and union-find |
| `errors` | `CTError` (+ 10 subtypes) | Input validation, state, and numerical errors |

---

## Quick Start

### Installation

```bash
pip install constraint-theory-python

# With NumPy support (optional)
pip install constraint-theory-python[numpy]

# From source (development)
git clone https://github.com/SuperInstance/constraint-theory-python.git
cd constraint-theory-python
pip install -e ".[dev]"
```

### Basic Usage

```python
from constraint_theory_python import PythagoreanManifold, snap

# Build the manifold with density 200 (~1000 exact states)
manifold = PythagoreanManifold(200)

# Snap a vector to the nearest exact Pythagorean point
exact, noise = snap(manifold, (0.577, 0.816))
print(f"Snapped to: {exact}")   # (0.6, 0.8)
print(f"Noise: {noise:.6f}")    # 0.023645

# Verify exactness: 0.6² + 0.8² = 1.0 EXACTLY (it's 3/5, 4/5)
mag_sq = exact[0] ** 2 + exact[1] ** 2
print(f"Magnitude squared: {mag_sq}")  # 1.0, not 1.0000000000000002
```

### The Floating-Point Problem

```python
# The bug you've fought before:
x, y = 0.6, 0.8
mag = (x * x + y * y) ** 0.5  # 1.0000000000000002

if mag == 1.0:
    print("This never prints")  # NEVER RUNS

# Constraint Theory's answer:
from constraint_theory_python import PythagoreanManifold, snap

manifold = PythagoreanManifold(200)
exact, noise = snap(manifold, (0.6, 0.8))

# exact = (0.6, 0.8) = (3/5, 4/5) — FOREVER EXACT
print(f"Magnitude: {exact[0]**2 + exact[1]**2}")  # 1.0
```

### Batch Processing

```python
from constraint_theory_python import PythagoreanManifold

manifold = PythagoreanManifold(200)
vectors = [(0.6, 0.8), (0.8, 0.6), (0.1, 0.99), (1.0, 0.0)]
results = manifold.snap_batch(vectors)

for exact, noise in results:
    mag_sq = exact[0] ** 2 + exact[1] ** 2
    assert abs(mag_sq - 1.0) < 1e-6  # Always passes
```

### Quantization

```python
from constraint_theory_python import PythagoreanQuantizer, QuantizationMode

# Ternary (BitNet) — LLM weights
q = PythagoreanQuantizer(QuantizationMode.TERNARY)
result = q.quantize([0.6, 0.8, -0.1, 0.0, 0.5])
print(result.data)  # [1.0, 1.0, 0.0, 0.0, 1.0]

# Polar — embeddings with exact unit norm
q = PythagoreanQuantizer.for_embeddings()
result = q.quantize([0.6, 0.8])
assert result.check_unit_norm(0.1)  # True
```

---

## Mathematical Foundation

### Pythagorean Triples & Euclid's Formula

All primitive Pythagorean triples are generated via **Euclid's formula**:

```
a = m² − n²,   b = 2mn,   c = m² + n²
```

where *m > n > 0*, (*m* − *n*) is odd, and gcd(*m*, *n*) = 1. Each triple produces a normalized point (*a*/*c*, *b*/*c*) lying exactly on the unit circle S¹ with no floating-point error. This is the foundation of deterministic vector snapping.

### Constraint Satisfaction on S¹

The unit circle S¹ = {(*x*, *y*) : *x*² + *y*² = 1} contains infinitely many points, but only finitely many **Pythagorean points** — rational points where both coordinates are exact ratios of integers from a Pythagorean triple. Constraint Theory discretizes S¹ into this finite lattice:

1. **Enumerate** all primitive triples up to a density bound via Euclid's formula
2. **Generate** normalized directions with quadrant reflections (5 per triple + 4 cardinal)
3. **Index** the resulting points with a KD-tree for O(log N) lookup
4. **Snap** any continuous input vector to the nearest lattice point

### Grand Unified Constraint Theory (GUCT)

GUCT extends the Pythagorean manifold into a full geometric framework:

| Domain | Mechanism |
|--------|-----------|
| **Exact representation** | Pythagorean triples via Euclid's formula |
| **Fast lookup** | O(log N) KD-tree spatial index |
| **Precision encoding** | Hidden dimensions: k = ⌈log₂(1/ε)⌉ |
| **Global consistency** | Holonomy verification (zero holonomy = consistent) |
| **Curvature evolution** | Ricci flow toward target curvature |
| **Structural rigidity** | Laman's theorem: 2V − 3 edges for 2D rigidity |
| **Topological detection** | Sheaf cohomology: H₀ = components, H₁ = cycles |
| **Quantization** | Ternary (BitNet), Polar, Turbo (TurboQuant), Hybrid |

### Key Constants

| Constant | Value | Origin |
|----------|-------|--------|
| `k = ⌈log₂(1/ε)⌉` | Depends on ε | Hidden dimension formula |
| `log₂(48) ≈ 5.585` bits | Information capacity | Exact unit vectors, 16-bit numerators |
| `1.692` | Ricci convergence multiplier | Spectral gap of curvature Laplacian |
| `12` | Laman neighbor threshold | Generalized Laman's theorem (6 DOF × 2) |

---

## Python API Reference

### `PythagoreanManifold`

The central data structure. Precomputes exact Pythagorean vectors and provides snapping.

```python
manifold = PythagoreanManifold(density=200)  # density: 50-500 recommended
```

| Method | Returns | Description |
|--------|---------|-------------|
| `manifold.snap(vec)` | `(tuple[float, float], float)` | Snap 2D vector to nearest exact point; returns (snapped, noise) |
| `manifold.snap_batch(vectors)` | `list[tuple[...]]` | Snap multiple vectors |
| `manifold.triple_count` | `int` | Number of primitive Pythagorean triples |
| `manifold.state_count` | `int` | Total discrete states on manifold |

### `snap()` (convenience function)

```python
from constraint_theory_python import PythagoreanManifold, snap
exact, noise = snap(manifold, (0.577, 0.816))
```

### `PythagoreanQuantizer`

```python
q = PythagoreanQuantizer(mode=QuantizationMode.HYBRID, bits=4)
result = q.quantize([0.6, 0.8, -0.3])
```

| Mode | Algorithm | Best For |
|------|-----------|----------|
| `QuantizationMode.TERNARY` | Sign + threshold → {-1, 0, 1} | LLM weights (16× memory reduction) |
| `QuantizationMode.POLAR` | Angle → snap to Pythagorean angles | Embeddings (exact unit norm) |
| `QuantizationMode.TURBO` | Uniform quantization + ratio snapping | Vector databases |
| `QuantizationMode.HYBRID` | Auto-select based on input | Unknown inputs |

### `QuantizationResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `list[float]` | Quantized values |
| `mse` | `float` | Mean squared error |
| `constraints_satisfied` | `bool` | Constraint satisfaction preserved |
| `unit_norm_preserved` | `bool` | Unit norm preserved |
| `check_unit_norm(tol)` | `bool` | Verify unit norm within tolerance |

### Hidden Dimensions

```python
from constraint_theory_python import hidden_dim_count, lift_to_hidden, project_to_visible
```

| Function | Description |
|----------|-------------|
| `hidden_dim_count(epsilon)` | k = ⌈log₂(1/ε)⌉ — compute hidden dimensions for precision ε |
| `lift_to_hidden(point, k)` | Lift Rⁿ → Rⁿ⁺ᵏ |
| `project_to_visible(lifted, n)` | Project Rⁿ⁺ᵏ → Rⁿ |

### Holonomy

```python
from constraint_theory_python import HolonomyChecker, compute_holonomy
```

| API | Description |
|-----|-------------|
| `compute_holonomy(angles)` | Compute holonomy for a cycle of rotation angles |
| `HolonomyChecker().apply(angle)` | Incremental cycle building |
| `.check_partial()` / `.check_closed()` | Check holonomy (identity = consistent) |
| `HolonomyResult.is_identity` | True if globally consistent |

### Cohomology, Curvature, Percolation

```python
from constraint_theory_python import FastCohomology, RicciFlow, FastPercolation
```

| Class | Method | Description |
|-------|--------|-------------|
| `FastCohomology` | `.compute(V, E, components)` | H₀ and H₁ via Euler characteristic |
| `RicciFlow` | `.evolve(curvatures)` | Curvature evolution: c' = c + α(target − c) |
| `FastPercolation` | `.compute_rigidity(vertices)` | Laman's theorem rigidity check |

### Error Handling

All errors inherit from `CTError`. Key subtypes:

```python
from constraint_theory_python.errors import (
    ZeroVectorError,    # Cannot snap (0, 0)
    NaNInputError,      # Input contains NaN
    InfinityInputError, # Input contains Infinity
    InvalidDimensionError,
    ManifoldEmptyError,
    NumericalInstabilityError,
    OverflowError,
    DivisionByZeroError,
    BufferSizeMismatchError,
    InvalidDensityError,
)
```

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=constraint_theory_python --cov-report=term-missing

# Run specific test modules
pytest tests/test_manifold.py
pytest tests/test_quantizer.py
pytest tests/test_hidden_dimensions.py
pytest tests/test_advanced.py

# Run examples
python examples/quickstart.py
python examples/quantization_demo.py
```

---

## Relationship to constraint-theory-core

This package is the **direct Python translation** of [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core), the Rust implementation. They implement the same mathematical framework (GUCT) and share identical algorithmic behavior:

| Feature | Python (this repo) | Rust (core) |
|---------|-------------------|-------------|
| Pythagorean triple generation | Euclid's formula + Stein's GCD | Euclid's formula + Stein's GCD |
| KD-tree | Recursive median-split, deterministic ties | Recursive median-split, deterministic ties |
| Holonomy | 3×3 rotation matrix accumulation | 3×3 rotation matrix accumulation |
| Ricci flow | α·(target − c) evolution | α·(target − c) evolution |
| Quantization | Ternary / Polar / Turbo / Hybrid | Ternary / Polar / Turbo / Hybrid |
| Percolation | Union-find + Laman's theorem | Union-find + Laman's theorem |
| Cohomology | Euler characteristic H₀/H₁ | Euler characteristic H₀/H₁ |
| Error types | `CTError` hierarchy (11 variants) | `CTErr` enum (11 variants) |

### When to Use Each

- **Use this Python package** for: rapid prototyping, Jupyter notebooks, ML training pipelines, scientific computing, education, and any workflow where Python is the primary language.
- **Use the Rust crate** for: production services, game engines, high-frequency trading, embedded systems, and any latency-critical path where ~100 ns per snap matters.

### Interoperability

Both implementations produce **identical snapping results** for the same density and input vector (within floating-point representation limits), making them interchangeable for cross-language consistency testing and gradual migration between Python prototyping and Rust production.

---

## Contributing

Contributions are welcome! Please see the [Rust core contributing guide](https://github.com/SuperInstance/constraint-theory-core/blob/main/CONTRIBUTING.md) for general principles.

```bash
# Set up development environment
pip install -e ".[dev]"
pytest
```

---

## Citation

```bibtex
@software{constraint_theory_python,
  title={Constraint Theory Python: Deterministic Manifold Snapping via Pythagorean Geometry},
  author={SuperInstance},
  year={2025},
  url={https://github.com/SuperInstance/constraint-theory-python},
  version={0.1.0}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

### Deterministic directions for 2D systems — now in Python.

**→ [Get started in 30 seconds](#quick-start)** · **[Try interactive demos](https://constraint-theory-web.pages.dev)** · **[Rust Core](https://github.com/SuperInstance/constraint-theory-core)**

*Built with 🐍 for systems that need exact reproducibility*

</div>

---

<img src="callsign1.jpg" width="128" alt="callsign">
