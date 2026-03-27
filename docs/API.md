# API Reference

Complete API documentation for Constraint Theory Python bindings.

## Table of Contents

- [Installation](#installation)
- [Module Overview](#module-overview)
- [Classes](#classes)
  - [PythagoreanManifold](#pythagoreanmanifold)
- [Functions](#functions)
  - [snap](#snap)
  - [generate_triples](#generate_triples)
- [Types and Constants](#types-and-constants)
- [Error Handling](#error-handling)
- [Thread Safety](#thread-safety)
- [Performance Characteristics](#performance-characteristics)

---

## Installation

```bash
pip install constraint-theory
```

For development installation:

```bash
git clone https://github.com/SuperInstance/constraint-theory-python
cd constraint-theory-python
pip install maturin
maturin develop --release
```

---

## Module Overview

```python
from constraint_theory import (
    PythagoreanManifold,  # Main manifold class
    snap,                 # Convenience snapping function
    generate_triples,     # Pythagorean triple generator
    __version__,          # Package version string
)
```

---

## Classes

### PythagoreanManifold

The main class for creating and using a Pythagorean manifold for deterministic vector snapping.

#### Constructor

```python
PythagoreanManifold(density: int)
```

Creates a new Pythagorean manifold with the specified density parameter.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `density` | `int` | Maximum value of m in Euclid's formula. Controls the resolution and number of exact states. Higher values = more states = finer resolution. |

**Returns:**

| Type | Description |
|------|-------------|
| `PythagoreanManifold` | A new manifold instance with pre-computed valid states |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If density is 0 or negative (via Rust panic) |

**Example:**

```python
from constraint_theory import PythagoreanManifold

# Create a manifold with moderate density
manifold = PythagoreanManifold(density=200)

# Check the number of pre-computed states
print(f"States: {manifold.state_count}")
# Output: States: 1013
```

**Density Guidelines:**

| Density | Approximate States | Resolution | Use Case |
|---------|-------------------|------------|----------|
| 50 | ~250 | 0.02 | Quick prototypes |
| 100 | ~500 | 0.01 | Game physics |
| 200 | ~1000 | 0.005 | General purpose |
| 500 | ~2500 | 0.002 | ML augmentation |
| 1000 | ~5000 | 0.001 | Scientific computing |
| 2000 | ~10000 | 0.0005 | High-precision CAD |

---

#### Properties

##### `state_count`

```python
manifold.state_count -> int
```

Returns the number of valid Pythagorean states in the manifold.

**Type:** `int` (read-only)

**Description:**

Each state represents a unique point on the unit circle that corresponds to a normalized Pythagorean triple `(a/c, b/c)` where `a² + b² = c²`. The count includes points in all quadrants.

**Example:**

```python
manifold = PythagoreanManifold(100)
count = manifold.state_count
print(f"Manifold contains {count} exact states")
```

---

#### Methods

##### `snap()`

```python
manifold.snap(x: float, y: float) -> tuple[float, float, float]
```

Snap a 2D vector to the nearest Pythagorean triple state.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | X coordinate of the input vector |
| `y` | `float` | Y coordinate of the input vector |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[float, float, float]` | A tuple of `(snapped_x, snapped_y, noise)` |

**Return Values:**

| Value | Type | Description |
|-------|------|-------------|
| `snapped_x` | `float` | X coordinate of the snapped point on the unit circle |
| `snapped_y` | `float` | Y coordinate of the snapped point on the unit circle |
| `noise` | `float` | Distance from input to snapped point (snapping error) |

**Notes:**

- Input vectors are NOT normalized internally; the snap distance is computed from the raw input
- The snapped point always satisfies `snapped_x² + snapped_y² = 1.0` exactly
- The noise value indicates how far your input was from an exact Pythagorean state
- Noise of 0.0 means the input was already an exact state

**Example:**

```python
manifold = PythagoreanManifold(200)

# Snap an approximate vector
x, y, noise = manifold.snap(0.577, 0.816)
print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
# Output: Snapped: (0.6000, 0.8000), noise: 0.0236

# Snap an exact Pythagorean triple
x, y, noise = manifold.snap(0.6, 0.8)
print(f"Snapped: ({x:.4f}, {y:.4f}), noise: {noise:.6f}")
# Output: Snapped: (0.6000, 0.8000), noise: 0.0000
```

**Performance:** O(log n) where n is the number of states, typically ~100ns per call.

---

##### `snap_batch()`

```python
manifold.snap_batch(vectors: list[list[float]] | numpy.ndarray) -> list[tuple[float, float, float]]
```

Snap multiple vectors at once with optimized batch processing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `list[list[float]]` or `numpy.ndarray` | Collection of 2D vectors. Either a list of `[x, y]` pairs or an Nx2 NumPy array. |

**Returns:**

| Type | Description |
|------|-------------|
| `list[tuple[float, float, float]]` | List of `(snapped_x, snapped_y, noise)` tuples, one per input vector |

**Example:**

```python
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Using a list of pairs
vectors = [[0.6, 0.8], [0.8, 0.6], [0.1, 0.99]]
results = manifold.snap_batch(vectors)

for i, (sx, sy, noise) in enumerate(results):
    print(f"[{i}] ({vectors[i][0]}, {vectors[i][1]}) -> ({sx:.4f}, {sy:.4f})")

# Using NumPy array
np_vectors = np.array([[0.6, 0.8], [0.707, 0.707]])
np_results = manifold.snap_batch(np_vectors)
```

**Performance:** Approximately 2-5x faster than individual `snap()` calls due to reduced Python-Rust boundary crossings.

**Memory:** For very large datasets, consider processing in chunks to avoid memory pressure.

---

##### `__repr__()` and `__str__()`

```python
repr(manifold) -> str
str(manifold) -> str
```

Returns a string representation of the manifold.

**Returns:**

| Type | Description |
|------|-------------|
| `str` | String like `"PythagoreanManifold(density=200, states=1013)"` |

---

## Functions

### snap

```python
snap(manifold: PythagoreanManifold, x: float, y: float) -> tuple[float, float, float]
```

Convenience function for one-off snapping operations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `manifold` | `PythagoreanManifold` | The manifold to use for snapping |
| `x` | `float` | X coordinate of the input vector |
| `y` | `float` | Y coordinate of the input vector |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[float, float, float]` | Same as `manifold.snap(x, y)` |

**Example:**

```python
from constraint_theory import PythagoreanManifold, snap

manifold = PythagoreanManifold(200)
result = snap(manifold, 0.577, 0.816)
# Equivalent to: result = manifold.snap(0.577, 0.816)
```

**Note:** For multiple snaps, prefer calling `manifold.snap()` directly or using `manifold.snap_batch()` for better performance.

---

### generate_triples

```python
generate_triples(max_c: int) -> list[tuple[int, int, int]]
```

Generate all primitive Pythagorean triples where the hypotenuse `c` is less than or equal to `max_c`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_c` | `int` | Maximum value of the hypotenuse c |

**Returns:**

| Type | Description |
|------|-------------|
| `list[tuple[int, int, int]]` | List of `(a, b, c)` tuples where `a² + b² = c²` |

**Notes:**

- Only **primitive** triples are generated (where gcd(a, b, c) = 1)
- In each tuple, `a < b` (smaller leg first)
- Triples are generated using Euclid's formula with coprime m, n where m > n

**Example:**

```python
from constraint_theory import generate_triples

# Generate triples with hypotenuse <= 50
triples = generate_triples(50)

for a, b, c in triples[:5]:
    print(f"{a}² + {b}² = {c}²")
# Output:
# 3² + 4² = 5²
# 5² + 12² = 13²
# 8² + 15² = 17²
# 7² + 24² = 25²
# 20² + 21² = 29²

print(f"\nTotal triples: {len(triples)}")
```

**Mathematical Background:**

Pythagorean triples are generated using Euclid's formula:

```
a = m² - n²
b = 2mn
c = m² + n²
```

where m > n > 0, gcd(m, n) = 1, and exactly one of m, n is even.

---

## Types and Constants

### `__version__`

```python
from constraint_theory import __version__

print(__version__)  # e.g., "0.1.0"
```

The package version string, following semantic versioning.

---

## Error Handling

### Common Exceptions

| Exception | Cause | Resolution |
|-----------|-------|------------|
| `ImportError` | Package not installed | Run `pip install constraint-theory` |
| `TypeError` | Wrong argument types | Check parameter types in API |
| `ValueError` | Invalid parameter values | Ensure density > 0 |
| `RuntimeError` | Rust panic | Report as bug with reproduction case |

### Type Validation

The Python bindings perform type checking at the Python-Rust boundary:

```python
manifold = PythagoreanManifold(200)

# Valid
manifold.snap(0.5, 0.8)           # Two floats
manifold.snap_batch([[0.5, 0.8]]) # List of pairs

# Invalid - raises TypeError
manifold.snap("0.5", 0.8)         # String instead of float
manifold.snap_batch([0.5, 0.8])   # Single list, not list of lists
```

---

## Thread Safety

### Thread-Safe Operations

The `PythagoreanManifold` class is **fully thread-safe** for read operations:

```python
from concurrent.futures import ThreadPoolExecutor
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def snap_vectors(vectors):
    """Safe for parallel execution."""
    return [manifold.snap(x, y) for x, y in vectors]

# Multiple threads can safely share the manifold
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(snap_vectors, data_chunks))
```

### Implementation Details

- The Rust core uses **immutable data structures** after construction
- All KD-tree lookups are read-only
- No internal state is modified during `snap()` or `snap_batch()` calls
- The GIL is released during expensive computations in Rust

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| `PythagoreanManifold(density)` | O(density² log density) | 1-50ms |
| `snap()` | O(log n) | ~100ns |
| `snap_batch()` | O(m log n) | ~30ns per vector |
| `generate_triples(max_c)` | O(max_c) | ~1μs per 1000 |

Where `n` is the number of states and `m` is the batch size.

### Memory Usage

| Density | States | Memory |
|---------|--------|--------|
| 100 | ~500 | ~40 KB |
| 200 | ~1000 | ~80 KB |
| 500 | ~2500 | ~200 KB |
| 1000 | ~5000 | ~400 KB |
| 2000 | ~10000 | ~800 KB |

### Optimization Tips

1. **Reuse manifold instances**: Construction is expensive; reuse across calls
2. **Use batch operations**: `snap_batch()` is 2-5x faster than individual `snap()` calls
3. **Choose appropriate density**: Higher density = more memory + slower construction
4. **Process in chunks**: For datasets > 100K vectors, use chunked processing
5. **Release GIL**: Long-running batch operations release the GIL for parallelism

### Benchmark Example

```python
import time
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Single snaps
start = time.perf_counter()
for _ in range(10000):
    manifold.snap(0.577, 0.816)
single_time = time.perf_counter() - start

# Batch snapping
vectors = [[0.577, 0.816] for _ in range(10000)]
start = time.perf_counter()
manifold.snap_batch(vectors)
batch_time = time.perf_counter() - start

print(f"Single: {single_time*1000:.2f}ms")
print(f"Batch:  {batch_time*1000:.2f}ms")
print(f"Speedup: {single_time/batch_time:.1f}x")
```

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial release with core snapping functionality |
| 0.2.0 | Added batch processing with NumPy support |
| 0.3.0 | Performance optimizations and expanded documentation |

---

## See Also

- [Migration Guide](MIGRATION.md) - For users coming from other libraries
- [Examples](../examples/) - Practical code examples
- [Constraint Theory Core](https://github.com/SuperInstance/constraint-theory-core) - Rust implementation
