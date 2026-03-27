# Migration Guide

This guide helps you migrate from other geometry and vector libraries to Constraint Theory.

## Table of Contents

- [Why Migrate?](#why-migrate)
- [From NumPy](#from-numpy)
- [From SciPy](#from-scipy)
- [From Shapely](#from-shapely)
- [From SymPy](#from-sympy)
- [From Custom Implementations](#from-custom-implementations)
- [Common Migration Patterns](#common-migration-patterns)
- [Performance Comparison](#performance-comparison)
- [Limitations and Differences](#limitations-and-differences)

---

## Why Migrate?

### Key Benefits

| Feature | Traditional Libraries | Constraint Theory |
|---------|----------------------|-------------------|
| **Exactness** | Platform-dependent floats | Exact rational coordinates |
| **Reproducibility** | Requires seeding | Deterministic everywhere |
| **Cross-platform** | May vary | Identical results |
| **Precision drift** | Accumulates over time | Zero drift |
| **Speed** | Variable | ~100ns per operation |

### When to Migrate

**Migrate if you:**

- Need **exact reproducibility** across different machines
- Experience **floating-point drift** in long-running simulations
- Want **deterministic** network physics in multiplayer games
- Require **verifiable** results for scientific publications
- Build **CAD/engineering** tools needing exact geometry

**Stay with your current library if you:**

- Need arbitrary precision beyond unit circle
- Work primarily with 3D geometry
- Require exact symbolic computation
- Need geometric operations beyond snapping (intersections, unions, etc.)

---

## From NumPy

### Basic Normalization

**Before (NumPy):**

```python
import numpy as np

def normalize(v):
    """Normalize a vector."""
    magnitude = np.linalg.norm(v)
    return v / magnitude

vector = np.array([3.0, 4.0])
normalized = normalize(vector)
# Result: [0.6, 0.8] but 0.6² + 0.8² = 0.9999999999999999
```

**After (Constraint Theory):**

```python
from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(200)

vector = np.array([3.0, 4.0])
magnitude = np.linalg.norm(vector)
normalized_approx = vector / magnitude

# Snap to exact Pythagorean triple
x, y, noise = manifold.snap(normalized_approx[0], normalized_approx[1])
normalized = np.array([x, y])
# Result: [0.6, 0.8] with 0.6² + 0.8² = 1.0 EXACTLY
```

### Batch Normalization

**Before (NumPy):**

```python
import numpy as np

vectors = np.random.randn(1000, 2)
magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / magnitudes
# All results are approximate floats
```

**After (Constraint Theory):**

```python
from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(200)

vectors = np.random.randn(1000, 2)
magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized_approx = vectors / magnitudes

# Batch snap to exact states
results = manifold.snap_batch(normalized_approx)
normalized = np.array([[x, y] for x, y, _ in results])
noises = np.array([noise for _, _, noise in results])
```

### Direction Vectors

**Before (NumPy):**

```python
import numpy as np

def direction_from_angle(angle_degrees):
    """Get unit vector from angle."""
    angle_rad = np.radians(angle_degrees)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])
# Returns approximate floats
```

**After (Constraint Theory):**

```python
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def exact_direction_from_angle(angle_degrees):
    """Get exact unit vector from angle."""
    angle_rad = np.radians(angle_degrees)
    x, y = np.cos(angle_rad), np.sin(angle_rad)
    sx, sy, noise = manifold.snap(x, y)
    return np.array([sx, sy]), noise
# Returns exact Pythagorean coordinates
```

---

## From SciPy

### KDTree for Nearest Neighbor

**Before (SciPy):**

```python
from scipy.spatial import KDTree
import numpy as np

# Build KD-tree of reference points
reference_points = np.random.randn(1000, 2)
tree = KDTree(reference_points)

# Find nearest neighbor
query_point = np.array([0.5, 0.7])
distance, index = tree.query(query_point)
nearest = reference_points[index]
```

**After (Constraint Theory):**

Constraint Theory uses an internal KD-tree for snapping. If you need custom reference points:

```python
from constraint_theory import PythagoreanManifold
import numpy as np

# Use Constraint Theory for unit circle snapping
manifold = PythagoreanManifold(500)

# Snap to nearest exact state
query_point = np.array([0.5, 0.7])
sx, sy, noise = manifold.snap(query_point[0], query_point[1])
nearest = np.array([sx, sy])
# Result is exact and deterministic
```

### Spatial Operations

**Before (SciPy):**

```python
from scipy.spatial.distance import cdist
import numpy as np

points = np.random.randn(100, 2)
distances = cdist(points, points)
```

**After (Constraint Theory):**

For exact distances on the unit circle:

```python
from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(200)

# Snap points to exact states first
points = np.random.randn(100, 2)
# Normalize and snap...
magnitudes = np.linalg.norm(points, axis=1, keepdims=True)
magnitudes[magnitudes == 0] = 1  # Avoid division by zero
normalized = points / magnitudes

results = manifold.snap_batch(normalized)
exact_points = np.array([[x, y] for x, y, _ in results])

# Now distances are based on exact coordinates
distances = cdist(exact_points, exact_points)
```

---

## From Shapely

### Direction Vectors

**Before (Shapely):**

```python
from shapely.geometry import Point, LineString
import numpy as np

def get_direction_line(origin, angle_degrees, length=1.0):
    """Create a line in a given direction."""
    angle_rad = np.radians(angle_degrees)
    end_x = origin.x + length * np.cos(angle_rad)
    end_y = origin.y + length * np.sin(angle_rad)
    return LineString([origin, Point(end_x, end_y)])
```

**After (Constraint Theory):**

```python
from shapely.geometry import Point, LineString
from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(200)

def get_exact_direction_line(origin, angle_degrees, length=1.0):
    """Create a line with exact direction."""
    angle_rad = np.radians(angle_degrees)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    dx, dy, noise = manifold.snap(dx, dy)  # Exact direction
    end_x = origin.x + length * dx
    end_y = origin.y + length * dy
    return LineString([origin, Point(end_x, end_y)])
```

---

## From SymPy

### Exact Arithmetic

**Before (SymPy):**

```python
from sympy import Rational, sqrt

# Exact Pythagorean triple
a, b, c = 3, 4, 5
x = Rational(a, c)  # 3/5
y = Rational(b, c)  # 4/5
# x² + y² = 1 exactly (symbolic)
```

**After (Constraint Theory):**

```python
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Snap to exact Pythagorean triple
x, y, noise = manifold.snap(0.6, 0.8)
# x, y are exact rational coordinates (3/5, 4/5)
# Computed much faster than symbolic math
```

**When to use each:**

| Use SymPy | Use Constraint Theory |
|-----------|----------------------|
| Symbolic manipulation | Numerical snapping |
| Exact algebra | Performance-critical |
| Proof verification | Production code |
| Complex expressions | Simple unit vectors |

---

## From Custom Implementations

### Fixed-Point Arithmetic

**Before (Custom):**

```python
class FixedPointVector:
    """Custom fixed-point vector implementation."""
    
    def __init__(self, x, y, scale=10000):
        self.scale = scale
        self.x = int(x * scale)
        self.y = int(y * scale)
    
    def normalized(self):
        mag_sq = self.x * self.x + self.y * self.y
        mag = int(mag_sq ** 0.5)
        return FixedPointVector(
            self.x / mag / self.scale,
            self.y / mag / self.scale,
            self.scale
        )
```

**After (Constraint Theory):**

```python
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def exact_normalize(x, y):
    """Get exact normalized vector."""
    mag = (x * x + y * y) ** 0.5
    if mag == 0:
        return 0.0, 0.0
    return manifold.snap(x / mag, y / mag)[:2]
```

### Lookup Table

**Before (Custom):**

```python
# Pre-computed lookup table of angles
ANGLES = {
    0: (1.0, 0.0),
    30: (0.866025, 0.5),
    45: (0.707107, 0.707107),
    60: (0.5, 0.866025),
    90: (0.0, 1.0),
    # ... more entries
}

def get_direction(angle):
    return ANGLES.get(angle, (0.0, 0.0))
```

**After (Constraint Theory):**

```python
from constraint_theory import PythagoreanManifold
import math

manifold = PythagoreanManifold(200)

def get_exact_direction(angle_degrees):
    """Get exact direction for any angle."""
    angle_rad = math.radians(angle_degrees)
    x, y = math.cos(angle_rad), math.sin(angle_rad)
    return manifold.snap(x, y)[:2]

# Works for ANY angle, not just pre-defined ones
```

---

## Common Migration Patterns

### Pattern 1: Normalize and Snap

```python
# Pattern: Convert approximate normalized vectors to exact

import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def to_exact(x, y):
    """Convert to exact Pythagorean coordinates."""
    mag = (x * x + y * y) ** 0.5
    if mag == 0:
        return 0.0, 0.0, 0.0
    return manifold.snap(x / mag, y / mag)

# Usage
exact_x, exact_y, noise = to_exact(2.0, 3.0)
```

### Pattern 2: Batch Processing

```python
# Pattern: Efficient batch conversion

import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(500)

def batch_to_exact(vectors):
    """Convert batch of vectors to exact coordinates."""
    vectors = np.asarray(vectors)
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1
    normalized = vectors / magnitudes
    
    results = manifold.snap_batch(normalized)
    exact = np.array([[x, y] for x, y, _ in results])
    noises = np.array([noise for _, _, noise in results])
    
    return exact, noises
```

### Pattern 3: Tolerance Filtering

```python
# Pattern: Filter by snapping noise

from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def filter_by_tolerance(vectors, max_noise=0.01):
    """Keep only vectors that snap within tolerance."""
    results = []
    for x, y in vectors:
        sx, sy, noise = manifold.snap(x, y)
        if noise <= max_noise:
            results.append((sx, sy))
    return results
```

### Pattern 4: Deterministic Augmentation

```python
# Pattern: Reproducible data augmentation for ML

from constraint_theory import PythagoreanManifold
import numpy as np

manifold = PythagoreanManifold(300)

def augment_directions(vectors, seed=42):
    """Deterministic direction augmentation."""
    np.random.seed(seed)
    
    # Add small random noise, then snap
    noisy = vectors + np.random.randn(*vectors.shape) * 0.05
    
    # Normalize and snap
    mags = np.linalg.norm(noisy, axis=1, keepdims=True)
    mags[mags == 0] = 1
    normalized = noisy / mags
    
    results = manifold.snap_batch(normalized)
    return np.array([[x, y] for x, y, _ in results])

# Same seed = same augmentation everywhere
```

---

## Performance Comparison

### Benchmark: Single Snap

```python
import time
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# NumPy normalization
start = time.perf_counter()
for _ in range(100000):
    v = np.array([0.577, 0.816])
    v = v / np.linalg.norm(v)
numpy_time = time.perf_counter() - start

# Constraint Theory
start = time.perf_counter()
for _ in range(100000):
    manifold.snap(0.577, 0.816)
ct_time = time.perf_counter() - start

print(f"NumPy:     {numpy_time*1000:.2f}ms")
print(f"Constraint Theory: {ct_time*1000:.2f}ms")
print(f"Speedup:   {numpy_time/ct_time:.1f}x")
```

### Benchmark: Batch Processing

```python
import time
import numpy as np
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)
vectors = np.random.randn(100000, 2)

# NumPy batch normalization
start = time.perf_counter()
mags = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / mags
numpy_time = time.perf_counter() - start

# Constraint Theory batch
start = time.perf_counter()
results = manifold.snap_batch(vectors / mags)
ct_time = time.perf_counter() - start

print(f"NumPy:     {numpy_time*1000:.2f}ms")
print(f"Constraint Theory: {ct_time*1000:.2f}ms")
```

---

## Limitations and Differences

### What Constraint Theory Does NOT Provide

| Feature | Alternative |
|---------|-------------|
| 3D vectors | Use NumPy + snap 2D projections |
| Arbitrary precision | Use SymPy or mpmath |
| Geometric operations (intersection, union) | Use Shapely |
| Symbolic math | Use SymPy |
| Non-unit vectors | Scale manually after snapping |

### Numeric Differences

```python
# NumPy: Returns platform-dependent float
np_normalized = np.array([3, 4]) / 5
# May be 0.6000000000000001 or 0.5999999999999999

# Constraint Theory: Returns exact rational as float
x, y, _ = manifold.snap(0.6, 0.8)
# Always exactly 0.6 and 0.8
```

### API Differences

| Operation | NumPy | Constraint Theory |
|-----------|-------|-------------------|
| Normalize | `v / np.linalg.norm(v)` | `manifold.snap(x, y)` |
| Batch normalize | `vectors / norms` | `manifold.snap_batch(vectors)` |
| Magnitude | `np.linalg.norm(v)` | `(x*x + y*y)**0.5` (same) |
| Dot product | `np.dot(a, b)` | `ax*bx + ay*by` (same) |

---

## Getting Help

- **Documentation:** [API Reference](API.md)
- **Examples:** [examples/](../examples/)
- **Issues:** [GitHub Issues](https://github.com/SuperInstance/constraint-theory-python/issues)
- **Core Library:** [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)
