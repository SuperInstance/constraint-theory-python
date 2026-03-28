# Quick Reference Card

A single-page reference for using Constraint Theory Python bindings.

## Installation

```bash
pip install constraint-theory
```

## Quick Start

```python
from constraint_theory import PythagoreanManifold, snap, generate_triples

# Create manifold
m = PythagoreanManifold(density=200)

# Snap a vector
x, y, noise = m.snap(0.577, 0.816)  # → (0.6, 0.8, 0.0236)

# Batch snap
vectors = [[0.6, 0.8], [0.707, 0.707]]
results = m.snap_batch(vectors)
```

---

## API Reference

### PythagoreanManifold

| Method | Returns | Description |
|--------|---------|-------------|
| `PythagoreanManifold(density)` | instance | Create manifold with density |
| `.snap(x, y)` | `(x, y, noise)` | Snap single vector |
| `.snap_batch(vectors)` | `[(x, y, noise), ...]` | Batch snap (SIMD) |
| `.state_count` | `int` | Number of exact states |
| `.density` | `int` | Density parameter |

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `snap(x, y, density=200)` | `(x, y, noise)` | Quick snap without manifold |
| `generate_triples(max_c)` | `[(a, b, c), ...]` | Generate Pythagorean triples |

### Type Aliases

```python
VectorLike = Union[List[Tuple[float, float]], numpy.ndarray]
SnapResultTuple = Tuple[float, float, float]
PythagoreanTripleTuple = Tuple[int, int, int]
```

---

## Density Guide

| Density | States | Resolution | Use Case |
|---------|--------|------------|----------|
| 50 | ~250 | 0.02 | Prototyping |
| 100 | ~500 | 0.01 | Game physics |
| 200 | ~1000 | 0.005 | General |
| 500 | ~2500 | 0.002 | ML/Science |
| 1000 | ~5000 | 0.001 | High precision |

---

## Common Patterns

### Deterministic Direction

```python
m = PythagoreanManifold(200)

# Normalize and snap
vx, vy = 0.577, 0.816
mag = (vx**2 + vy**2) ** 0.5
dx, dy, noise = m.snap(vx/mag, vy/mag)
# dx, dy are exact unit vector components
```

### Batch Processing

```python
import numpy as np

m = PythagoreanManifold(200)

# Generate 10000 unit vectors
angles = np.random.uniform(0, 2*np.pi, 10000)
vectors = np.column_stack([np.cos(angles), np.sin(angles)])

# Process all at once
results = m.snap_batch(vectors)
```

### NumPy Integration

```python
import numpy as np
from constraint_theory import PythagoreanManifold

m = PythagoreanManifold(200)

# NumPy array input
arr = np.array([[0.6, 0.8], [0.707, 0.707]], dtype=np.float32)
results = m.snap_batch(arr)

# Convert results to NumPy
snapped = np.array([[sx, sy] for sx, sy, _ in results])
noises = np.array([n for _, _, n in results])
```

### Thread-Safe Parallelism

```python
from concurrent.futures import ThreadPoolExecutor

m = PythagoreanManifold(200)

def process_chunk(chunk):
    return m.snap_batch(chunk)

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_chunk, chunks))
```

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Constructor | 1-50 ms | Depends on density |
| Single snap | ~100 ns | O(log n) KD-tree |
| Batch snap | ~30 ns/vector | SIMD optimized |

### Optimization Tips

1. **Reuse manifold instances** - Construction is expensive
2. **Use batch operations** - 2-5x faster than individual snaps
3. **Choose appropriate density** - Higher = slower construction
4. **Process in chunks** - For datasets > 100K vectors

---

## Type Checking

```python
from constraint_theory import PythagoreanManifold, ManifoldProtocol

def process(m: ManifoldProtocol, x: float, y: float) -> float:
    _, _, noise = m.snap(x, y)
    return noise

# Works with mypy
m = PythagoreanManifold(200)
noise = process(m, 0.577, 0.816)
```

---

## Error Handling

```python
from constraint_theory import PythagoreanManifold

try:
    m = PythagoreanManifold(0)  # Invalid
except (ValueError, RuntimeError):
    print("Density must be positive")

m = PythagoreanManifold(200)

try:
    m.snap("not a number", 0.8)  # TypeError
except TypeError:
    print("Coordinates must be numeric")
```

---

## Return Values

### Snap Result

```python
x, y, noise = m.snap(input_x, input_y)

# x, y: snapped coordinates (unit circle)
# noise: distance from input to snapped point

# Verify unit norm
assert abs(x**2 + y**2 - 1.0) < 1e-10  # Always exact
```

### Pythagorean Triple

```python
triples = generate_triples(50)

for a, b, c in triples[:3]:
    print(f"{a}² + {b}² = {c}²")
# 3² + 4² = 5²
# 5² + 12² = 13²
# 8² + 15² = 17²
```

---

## Platform Support

| Platform | Python | Architecture |
|----------|--------|--------------|
| Linux | 3.8-3.12 | x86_64 |
| macOS | 3.8-3.12 | ARM64, x86_64 |
| Windows | 3.8-3.12 | x86_64 |

---

## Ecosystem Links

| Resource | Link |
|----------|------|
| PyPI | https://pypi.org/project/constraint-theory/ |
| GitHub | https://github.com/SuperInstance/constraint-theory-python |
| Rust Core | https://github.com/SuperInstance/constraint-theory-core |
| Web Demo | https://constraint-theory.superinstance.ai |
| Research | https://github.com/SuperInstance/constraint-theory-research |

---

## Cheat Sheet

```python
# One-liner snap
from constraint_theory import snap
x, y, noise = snap(0.577, 0.816, density=200)

# Create reusable manifold
from constraint_theory import PythagoreanManifold
m = PythagoreanManifold(200)

# Properties
m.state_count  # Number of exact states
m.density      # Density parameter

# Single snap
x, y, noise = m.snap(0.577, 0.816)

# Batch snap
results = m.snap_batch([[0.6, 0.8], [0.707, 0.707]])

# Generate triples
from constraint_theory import generate_triples
triples = generate_triples(100)

# Version
from constraint_theory import __version__
print(__version__)
```

---

## Common Mistakes

```python
# ❌ Creating manifold repeatedly
for x, y in vectors:
    m = PythagoreanManifold(200)  # Slow!
    m.snap(x, y)

# ✅ Reuse manifold
m = PythagoreanManifold(200)
for x, y in vectors:
    m.snap(x, y)

# ❌ Individual snaps for batches
results = [m.snap(x, y) for x, y in vectors]

# ✅ Use batch operation
results = m.snap_batch(vectors)

# ❌ Wrong density range
m = PythagoreanManifold(0)  # Error!

# ✅ Valid density
m = PythagoreanManifold(200)
```

---

## Version

```python
import constraint_theory
print(constraint_theory.__version__)  # e.g., "0.1.0"
```

---

**For detailed documentation, see [API Reference](API.md)**
