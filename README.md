# Constraint Theory - Python Bindings

Python bindings for [Constraint Theory](https://github.com/SuperInstance/constraint-theory-core) — deterministic geometric snapping with O(log n) KD-tree lookup.

[![PyPI version](https://badge.fury.io/py/constraint-theory.svg)](https://pypi.org/project/constraint-theory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Installation

```bash
pip install constraint-theory
```

Or with NumPy support:

```bash
pip install constraint-theory[numpy]
```

---

## Quick Start

```python
from constraint_theory import PythagoreanManifold, generate_triples

# Create a manifold with ~1000 states
manifold = PythagoreanManifold(density=200)

print(f"Manifold has {manifold.state_count} valid states")

# Snap a vector to the nearest Pythagorean triple
snapped_x, snapped_y, noise = manifold.snap(0.6, 0.8)
print(f"Snapped: ({snapped_x:.4f}, {snapped_y:.4f}), noise: {noise:.6f}")
# Output: Snapped: (0.6000, 0.8000), noise: 0.000000
```

### Batch Processing

```python
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Snap multiple vectors at once (SIMD optimized)
vectors = np.array([
    [0.6, 0.8],
    [0.8, 0.6],
    [0.1, 0.99],
    [0.707, 0.707],
])

results = manifold.snap_batch(vectors)

for i, (sx, sy, noise) in enumerate(results):
    print(f"[{i}] ({vectors[i,0]:.3f}, {vectors[i,1]:.3f}) -> ({sx:.4f}, {sy:.4f}), noise={noise:.6f}")
```

### Generate Pythagorean Triples

```python
from constraint_theory import generate_triples

# Get all primitive triples with hypotenuse <= 50
triples = generate_triples(max_c=50)

for a, b, c in triples[:5]:
    print(f"{a}² + {b}² = {c}²  →  {a}² + {b}² = {a*a + b*b} = {c}²")
```

---

## API Reference

### `PythagoreanManifold(density: int)`

Create a manifold of Pythagorean triples.

- `density`: Controls the resolution. Higher values = more states = finer granularity.

#### Methods

| Method | Description |
|--------|-------------|
| `snap(x, y)` | Snap a single vector. Returns `(snapped_x, snapped_y, noise)` |
| `snap_batch(vectors)` | Snap multiple vectors. Accepts list or Nx2 numpy array. |
| `state_count` | Number of valid states in the manifold |

### `generate_triples(max_c: int)`

Generate all primitive Pythagorean triples with hypotenuse ≤ `max_c`.

### `snap(manifold, x, y)`

Convenience function for one-off snaps.

---

## Performance

| Operation | Time (Python) | Time (Rust core) |
|-----------|---------------|------------------|
| Single snap | ~1 μs | ~100 ns |
| Batch 1000 (NumPy) | ~1 ms | ~74 μs |

The Python overhead is minimal — the heavy lifting happens in Rust.

---

## Related Projects

- **[constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)** — The Rust library
- **[constraint-theory-web](https://github.com/SuperInstance/constraint-theory-web)** — Interactive demos
- **[constraint-theory-research](https://github.com/SuperInstance/constraint-theory-research)** — Mathematical foundations

---

## License

MIT
