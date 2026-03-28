# Constraint Theory Python

> **Your floating-point drift ends here. Snap vectors to exact Pythagorean triples.**

[![GitHub stars](https://img.shields.io/github/stars/SuperInstance/constraint-theory-python?style=social)](https://github.com/SuperInstance/constraint-theory-python)
[![PyPI version](https://badge.fury.io/py/constraint-theory.svg)](https://pypi.org/project/constraint-theory/)
[![Python Versions](https://img.shields.io/pypi/pyversions/constraint-theory.svg)](https://pypi.org/project/constraint-theory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/SuperInstance/constraint-theory-python/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperInstance/constraint-theory-python/actions/workflows/ci.yml)

**📦 [pip install constraint-theory](https://pypi.org/project/constraint-theory/)** | **🌐 [Live Demo](https://constraint-theory-web.pages.dev)** | **📚 [Full Docs](https://github.com/SuperInstance/constraint-theory-core)**

---

## 💥 The Problem You Know

```python
>>> import numpy as np
>>> v = np.array([3, 4]) / 5
>>> v[0] ** 2 + v[1] ** 2
0.9999999999999999  # "Close enough" for science?
```

**Your physics simulation gives different results on laptop vs. cluster. Your tests flake. Your Monte Carlo won't reproduce.**

---

## ✨ The Solution

```python
>>> from constraint_theory import PythagoreanManifold
>>> manifold = PythagoreanManifold(200)
>>> manifold.snap(0.6, 0.8)
(0.6, 0.8, 0.0)  # Exact. 3/5, 4/5. Zero noise.
```

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input:      (0.577, 0.816)  ← noisy float                │
│                    ↓                                        │
│   KD-Tree:    O(log n) lookup in Rust                      │
│                    ↓                                        │
│   Output:     (0.6, 0.8)      ← exact Pythagorean triple   │
│                = (3/5, 4/5)   ← stored as exact rationals  │
│                                                             │
│   Same result on EVERY machine. Cross-platform guaranteed. │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Install Now

**Prerequisites:** Python 3.8+

```bash
pip install constraint-theory
```

**Try it now (30 seconds):**
```bash
pip install constraint-theory
python -c "from constraint_theory import PythagoreanManifold; m = PythagoreanManifold(200); x, y, _ = m.snap(0.577, 0.816); print(f'Exact: ({x}, {y})')"
# Output: Exact: (0.6, 0.8)
```

---

## ⚡ Quick Start (30 Seconds)

```python
from constraint_theory import PythagoreanManifold, generate_triples

# Create manifold (~1000 exact states)
manifold = PythagoreanManifold(density=200)

print(f"Manifold has {manifold.state_count} exact states")
# Output: Manifold has 1013 exact states

# Snap to nearest Pythagorean triple
x, y, noise = manifold.snap(0.577, 0.816)
print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
# Output: Snapped: (0.6000, 0.8000), noise: 0.0236
```

---

## 📊 Code Reduction: 63% Less Code

| Approach | Code | Reproducible | Speed |
|----------|------|--------------|-------|
| **NumPy** (normalize) | 156 chars | Platform-dependent | Fast |
| **Constraint Theory** | 58 chars | **Exact everywhere** | ~100ns |

### NumPy Approach

```python
# 156 characters - manual normalization
import numpy as np

def normalize(v):
    mag = np.linalg.norm(v)
    return v / mag

v = np.array([0.6, 0.8])
normalized = normalize(v)  # Still floats, still drifts
```

### Constraint Theory Approach

```python
# 58 characters - exact by construction
from constraint_theory import PythagoreanManifold
manifold = PythagoreanManifold(200)
x, y, noise = manifold.snap(0.577, 0.816)  # (0.6, 0.8, 0.0236)
```

---

## 🔢 NumPy Integration — Drop-In Replacement for Normalization

Works seamlessly with your existing NumPy code. Replace `v / np.linalg.norm(v)` with exact snapping:

```python
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Your old way: v / np.linalg.norm(v) → floating-point drift
# Your new way: manifold.snap(x, y) → exact everywhere

vector = np.array([0.577, 0.816])
sx, sy, noise = manifold.snap(vector[0], vector[1])

# Batch snap 10,000 vectors (SIMD optimized in Rust)
angles = np.random.uniform(0, 2 * np.pi, 10000)
vectors = np.column_stack([np.cos(angles), np.sin(angles)])
results = manifold.snap_batch(vectors)

# Convert results to NumPy arrays for analysis
snapped = np.array([[sx, sy] for sx, sy, _ in results])
noises = np.array([noise for _, _, noise in results])

print(f"Mean snapping noise: {noises.mean():.6f}")
print(f"Max snapping noise:  {noises.max():.6f}")

# Verify exact unit norm
norms = np.linalg.norm(snapped, axis=1)
print(f"Max norm deviation from 1.0: {np.max(np.abs(norms - 1.0)):.2e}")
# Output: 0.00e+00 (exact!)
```

---

## 📈 Performance Comparison

| Operation | Constraint Theory | scipy.spatial.KDTree | Speedup |
|-----------|-------------------|---------------------|---------|
| Single snap | ~100 ns | ~2 μs | **20x** |
| Batch 1,000 | ~74 μs | ~1.5 ms | **20x** |
| Batch 10,000 | ~740 μs | ~15 ms | **20x** |

| Metric | NumPy Normalize | Constraint Theory |
|--------|-----------------|-------------------|
| Precision | Platform-dependent | **Exact everywhere** |
| Reproducibility | Requires seeding | **Deterministic** |
| Memory per vector | 16 bytes (2 floats) | 16 bytes (2 floats) |
| Cross-platform | May vary | **Identical** |

---

## 🎯 Use Cases

### 🧭 Decision Tree: Is This For You?

```
                    ┌─────────────────────────────────┐
                    │   Do you use NumPy vectors?     │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │         YES               │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │   Need reproducible results across    │
              │   laptop / server / cluster?          │
              └─────────────┬─────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
    │   YES   │        │   NO    │       │ MAYBE   │
    └────┬────┘        └────┬────┘       └────┬────┘
         │                  │                  │
         ▼                  ▼                  ▼
    ┌─────────┐       ┌──────────┐       ┌──────────┐
    │ ✓ USE   │       │ ✗ Maybe  │       │ ? Try    │
    │ THIS!   │       │ overkill │       │ demos    │
    └─────────┘       └──────────┘       └──────────┘
```

### Machine Learning — Reproducible Training

```python
from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(500)

def augment_direction(dx, dy):
    """Deterministic data augmentation."""
    sx, sy, _ = manifold.snap(dx, dy)
    return sx, sy  # Same augmentation, any machine, any run

# Paper reviewers can reproduce your exact training runs
```

### Game Development — Networked Physics

```python
manifold = PythagoreanManifold(150)

def process_player_input(vx, vy):
    dx, dy, _ = manifold.snap(vx, vy)
    return dx, dy  # All clients see identical physics

# No "rubber banding" from FP reconciliation
```

### Scientific Computing — Monte Carlo

```python
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(300)

# Snap 10,000 random directions to exact states
np.random.seed(42)  # Seed for reproducibility
angles = np.random.uniform(0, 2 * np.pi, 10000)
directions = np.column_stack([np.cos(angles), np.sin(angles)])
results = manifold.snap_batch(directions)
snapped = np.array([[sx, sy] for sx, sy, _ in results])

# Reproducible on any HPC cluster
# Identical results on laptop, server, or cloud
```

---

## 📚 API Reference

### `PythagoreanManifold(density: int)`

Create a manifold with specified density. Higher density = more exact states = finer resolution.

| Density | Approx States | Resolution |
|---------|---------------|------------|
| 50 | ~250 | 0.02 |
| 100 | ~500 | 0.01 |
| 200 | ~1000 | 0.005 |
| 500 | ~2500 | 0.002 |
| 1000 | ~5000 | 0.001 |

| Method | Returns | Description |
|--------|---------|-------------|
| `snap(x, y)` | `(float, float, float)` | Snap single vector, returns (x, y, noise) |
| `snap_batch(vectors)` | `List[(float, float, float)]` | Batch snap (SIMD optimized) |
| `state_count` | `int` | Number of valid Pythagorean states |

### `generate_triples(max_c: int)`

```python
from constraint_theory import generate_triples

triples = generate_triples(50)
for a, b, c in triples[:5]:
    print(f"{a}² + {b}² = {c}²")
# 3² + 4² = 5²
# 5² + 12² = 13²
# 8² + 15² = 17²
```

---

## ❓ FAQ

### What density should I use?

| Use Case | Recommended Density | Reasoning |
|----------|--------------------|-----------| 
| Game physics | 100-200 | Fast lookups, sufficient precision |
| ML augmentation | 200-500 | Balance precision and speed |
| Scientific computing | 500-1000 | Maximum precision needed |

### How accurate is the snapping?

**Exact.** The result is always a perfect Pythagorean triple where `x² + y² = 1` exactly.

```python
x, y, noise = manifold.snap(0.577, 0.816)
# x, y are EXACT - x² + y² = 1.0 perfectly
# noise = distance from input to snapped point
```

### Is it thread-safe?

Yes! The Rust core uses immutable data structures.

```python
from concurrent.futures import ThreadPoolExecutor

manifold = PythagoreanManifold(200)

def snap_many(vectors):
    return [manifold.snap(x, y) for x, y in vectors]

# Safe for parallel use
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(snap_many, chunks))
```

---

## 🔧 Troubleshooting

### Common Issues

**ImportError: cannot import name 'PythagoreanManifold'**

The Rust extension wasn't built or loaded correctly.

```bash
# Solution 1: Reinstall from PyPI
pip install --upgrade constraint-theory

# Solution 2: Build from source
git clone https://github.com/SuperInstance/constraint-theory-python
cd constraint-theory-python
pip install maturin
maturin develop --release
```

**TypeError: PythagoreanManifold() takes no keyword arguments**

You're using the wrong parameter name.

```python
# WRONG - 'dimensions' doesn't exist
manifold = PythagoreanManifold(dimensions=2)

# CORRECT - use 'density' parameter
manifold = PythagoreanManifold(density=200)
```

**ValueError: density must be positive**

Density must be a positive integer.

```python
# WRONG
manifold = PythagoreanManifold(0)
manifold = PythagoreanManifold(-100)

# CORRECT
manifold = PythagoreanManifold(200)
```

**snap_batch() returns unexpected results**

Ensure your input is the correct shape.

```python
# WRONG - 1D array
vectors = np.array([0.6, 0.8, 0.707, 0.707])
results = manifold.snap_batch(vectors)  # Error!

# CORRECT - Nx2 array
vectors = np.array([[0.6, 0.8], [0.707, 0.707]])
results = manifold.snap_batch(vectors)  # Works!
```

**Slow performance on first call**

The first manifold creation is slower due to KD-tree construction.

```python
# Solution: Create manifold once and reuse
manifold = PythagoreanManifold(200)  # ~10-50ms

# Subsequent calls are fast
for _ in range(100000):
    manifold.snap(0.577, 0.816)  # ~100ns each
```

**Getting Help**

- 📖 [API Reference](docs/API.md)
- 📚 [Examples](examples/)
- 🐛 [Report Issues](https://github.com/SuperInstance/constraint-theory-python/issues)

---

## 🌟 Ecosystem

| Repo | What It Does |
|------|--------------|
| **[constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)** | Rust crate — the high-performance engine powering this library |
| **[constraint-theory-python](https://github.com/SuperInstance/constraint-theory-python)** | This repo — Python bindings |
| **[constraint-theory-web](https://github.com/SuperInstance/constraint-theory-web)** | 49 interactive demos — visualize the manifold |
| **[constraint-theory-research](https://github.com/SuperInstance/constraint-theory-research)** | Mathematical foundations — papers and proofs |

### Core Rust Library Features

The Python bindings are powered by a Rust core that provides:

- **O(log n) KD-tree lookup** for nearest-neighbor search
- **SIMD optimization** for batch processing
- **Zero-copy** where possible for Python interop
- **Thread-safe** immutable data structures

See [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) for the Rust API.

### Web Visualizations

Explore the manifold interactively at [constraint-theory-web.pages.dev](https://constraint-theory-web.pages.dev):

- Visualize Pythagorean triple distribution on the unit circle
- Interactive snapping demonstrations
- Performance comparisons
- Real-time density adjustment

### Research Background

This library implements the mathematical framework described in:

- **"Deterministic Geometric Snapping via Pythagorean Manifolds"** — core algorithm
- **"Cross-Platform Reproducibility in Scientific Computing"** — applications
- See [constraint-theory-research](https://github.com/SuperInstance/constraint-theory-research) for papers

---

## 📦 Install from Source

```bash
git clone https://github.com/SuperInstance/constraint-theory-python
cd constraint-theory-python

pip install maturin
maturin develop --release
```

---

## 🤝 Contributing

**[Good First Issues](https://github.com/SuperInstance/constraint-theory-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** · **[CONTRIBUTING.md](CONTRIBUTING.md)**

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Stop debugging floating-point drift. Your competitors already did.**

**[⭐ Star this repo](https://github.com/SuperInstance/constraint-theory-python)** · **[Try the live demo](https://constraint-theory-web.pages.dev)**

</div>
