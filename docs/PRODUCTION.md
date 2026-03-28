# Production Readiness Guide

This guide covers production deployment considerations for Constraint Theory Python bindings.

## Table of Contents

- [Security Considerations](#security-considerations)
- [GIL Handling](#gil-handling)
- [Memory Management](#memory-management)
- [Debugging FFI Issues](#debugging-ffi-issues)
- [Performance Optimization](#performance-optimization)
- [Error Handling](#error-handling)

---

## Security Considerations

### PyO3 Security Model

The Python bindings use PyO3, which provides memory-safe FFI between Python and Rust:

| Aspect | Security Property |
|--------|-------------------|
| **Memory Safety** | Rust's ownership system prevents use-after-free, buffer overflows |
| **Thread Safety** | `Send` and `Sync` traits ensure thread-safe data access |
| **Panic Handling** | Rust panics are caught and converted to Python exceptions |
| **Type Safety** | PyO3 validates types at the FFI boundary |

### Security Best Practices

1. **Input Validation**
   ```python
   # The bindings validate types automatically
   manifold = PythagoreanManifold(200)  # Valid
   
   # Invalid types raise TypeError
   manifold.snap("not a float", 0.8)  # TypeError
   manifold.snap(None, 0.8)           # TypeError
   ```

2. **Bounds Checking**
   ```python
   # Density must be positive
   PythagoreanManifold(-100)  # Raises exception
   
   # Very large densities are allowed but may be slow
   PythagoreanManifold(1000000)  # Valid but slow construction
   ```

3. **No Arbitrary Code Execution**
   - The library performs only geometric calculations
   - No file I/O, network access, or system calls
   - No dynamic code evaluation

### Attack Surface

| Component | Risk Level | Mitigation |
|-----------|------------|------------|
| `snap()` | Low | Pure computation, no side effects |
| `snap_batch()` | Low | SIMD computation, GIL released |
| `generate_triples()` | Low | Integer math only |
| Constructor | Low | Memory allocation bounded by density |

### Denial of Service Considerations

```python
# Large densities can cause slow construction
# Consider validating density in application code
MAX_REASONABLE_DENSITY = 10000

def create_manifold(density: int) -> PythagoreanManifold:
    if density > MAX_REASONABLE_DENSITY:
        raise ValueError(f"Density {density} exceeds maximum {MAX_REASONABLE_DENSITY}")
    return PythagoreanManifold(density)
```

---

## GIL Handling

### Global Interpreter Lock (GIL) Behavior

The Python GIL prevents multiple native threads from executing Python bytecode simultaneously. This affects how the bindings operate:

| Operation | GIL Status | Notes |
|-----------|------------|-------|
| `PythagoreanManifold(n)` | **Held** | Construction is fast (~1-50ms) |
| `snap()` | **Held** | Very fast (~100ns), no release needed |
| `snap_batch()` | **Released** | GIL released for batch computation |
| `generate_triples()` | **Held** | Fast integer computation |

### Thread Safety Guarantees

```python
from concurrent.futures import ThreadPoolExecutor
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

def process_batch(vectors):
    # Safe: manifold is immutable after construction
    # GIL is released during snap_batch
    return manifold.snap_batch(vectors)

# This works efficiently because GIL is released
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_batch, data_chunks))
```

### GIL Release Details

The `snap_batch_simd` method explicitly releases the GIL:

```rust
// From src/lib.rs
pub fn snap_batch_simd(&self, py: Python<'_>, vectors: &PyList) -> PyResult<...> {
    // GIL released here
    py.allow_threads(|| {
        let results = self.inner.snap_batch_simd(&input);
        // Pure Rust computation, no Python API calls
        Ok(results.into_iter().map(|(s, n)| (s[0], s[1], n)).collect())
    })
}
```

### When GIL Release Matters

| Batch Size | GIL Released? | Benefit |
|------------|---------------|---------|
| < 100 | Yes | Minimal - overhead dominates |
| 100-1000 | Yes | Moderate - parallel speedup |
| > 1000 | Yes | Significant - other threads can run |

---

## Memory Management

### Memory Layout

```
Python Process
├── Python Objects
│   └── PyManifold (wrapper)
│       └── PythagoreanManifold (Rust struct)
│           ├── valid_states: Vec<[f32; 2]>  (~80KB for density=200)
│           └── kdtree: KDTree                (~160KB for density=200)
└── NumPy Arrays (if used)
    └── Input/Output buffers
```

### Memory Usage by Density

| Density | States | Rust Memory | Python Overhead |
|---------|--------|-------------|-----------------|
| 50 | ~250 | ~20 KB | ~8 bytes (pointer) |
| 100 | ~500 | ~40 KB | ~8 bytes |
| 200 | ~1000 | ~80 KB | ~8 bytes |
| 500 | ~2500 | ~200 KB | ~8 bytes |
| 1000 | ~5000 | ~400 KB | ~8 bytes |
| 5000 | ~25000 | ~2 MB | ~8 bytes |

### Memory Ownership

```python
# Python owns the PyManifold wrapper
manifold = PythagoreanManifold(200)

# Rust owns the internal data
# When manifold is garbage collected, Rust drops the data
del manifold  # Rust memory freed immediately

# Cloning creates a full copy
manifold1 = PythagoreanManifold(200)
manifold2 = manifold1  # Reference, not copy
```

### Zero-Copy Considerations

| Operation | Copy? | Notes |
|-----------|-------|-------|
| `snap(x, y)` | No | Scalar values passed directly |
| `snap_batch(list)` | Yes | Python list converted to Rust Vec |
| `snap_batch(numpy)` | Yes | NumPy buffer copied to Rust Vec |

For maximum performance with NumPy, consider memory-mapped files or shared memory for large datasets.

### Batch Processing Memory

```python
# For very large datasets, process in chunks
def process_large_dataset(vectors, chunk_size=10000):
    manifold = PythagoreanManifold(200)
    results = []
    
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i+chunk_size]
        results.extend(manifold.snap_batch(chunk))
    
    return results
```

---

## Debugging FFI Issues

### Common FFI Problems

1. **ImportError on Module Load**
   ```
   ImportError: cannot import name 'PythagoreanManifold'
   ```
   
   **Causes:**
   - Rust extension not compiled
   - Architecture mismatch (x86_64 vs ARM64)
   - Python version mismatch
   
   **Solutions:**
   ```bash
   # Rebuild the extension
   pip install maturin
   maturin develop --release
   
   # Check architecture
   python -c "import platform; print(platform.machine())"
   
   # Check Python version
   python --version
   ```

2. **Segfault or Crash**
   ```
   Segmentation fault (core dumped)
   ```
   
   **Causes:**
   - Bug in Rust code (should be reported)
   - Memory corruption (rare with safe Rust)
   
   **Debugging:**
   ```bash
   # Enable Rust backtrace
   export RUST_BACKTRACE=1
   python your_script.py
   
   # Run with gdb
   gdb python
   (gdb) run your_script.py
   ```

3. **Type Errors at FFI Boundary**
   ```
   TypeError: argument 'x': must be real number, not str
   ```
   
   **Solution:** Ensure correct types
   ```python
   # Wrong
   manifold.snap("0.6", 0.8)
   
   # Correct
   manifold.snap(0.6, 0.8)
   manifold.snap(float("0.6"), 0.8)
   ```

### Debug Logging

Enable Python-level logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check module load
from constraint_theory import PythagoreanManifold
logging.debug(f"Loaded PythagoreanManifold: {PythagoreanManifold}")
```

Enable Rust-level logging:

```bash
# Set RUST_LOG environment variable
export RUST_LOG=debug
python your_script.py
```

### Valgrind Memory Analysis

For memory leak detection:

```bash
# Install valgrind
sudo apt-get install valgrind

# Run Python under valgrind
valgrind --leak-check=full python your_script.py
```

### Common Debugging Patterns

```python
def debug_snap(manifold, x, y):
    """Debug wrapper with validation."""
    import sys
    
    # Check inputs
    print(f"Input: x={x!r}, y={y!r}", file=sys.stderr)
    print(f"Types: x={type(x)}, y={type(y)}", file=sys.stderr)
    
    # Validate
    if not isinstance(x, (int, float)):
        raise TypeError(f"x must be numeric, got {type(x)}")
    if not isinstance(y, (int, float)):
        raise TypeError(f"y must be numeric, got {type(y)}")
    
    # Call
    result = manifold.snap(float(x), float(y))
    
    print(f"Output: {result}", file=sys.stderr)
    return result
```

---

## Performance Optimization

### Best Practices

1. **Reuse Manifold Instances**
   ```python
   # BAD: Creating manifold repeatedly
   for vec in vectors:
       m = PythagoreanManifold(200)  # Slow!
       m.snap(*vec)
   
   # GOOD: Reuse manifold
   m = PythagoreanManifold(200)
   for vec in vectors:
       m.snap(*vec)
   ```

2. **Use Batch Operations**
   ```python
   # BAD: Individual snaps
   results = [m.snap(x, y) for x, y in vectors]
   
   # GOOD: Batch operation
   results = m.snap_batch(vectors)
   ```

3. **Choose Appropriate Density**
   ```python
   # Game physics: lower density for speed
   game_manifold = PythagoreanManifold(100)
   
   # Scientific computing: higher density for precision
   science_manifold = PythagoreanManifold(500)
   ```

4. **Thread Pool for Large Datasets**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   manifold = PythagoreanManifold(200)
   
   def process_chunk(chunk):
       return manifold.snap_batch(chunk)
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_chunk, chunks))
   ```

### Performance Profiling

```python
import time
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

# Profile single snaps
start = time.perf_counter()
for _ in range(100000):
    manifold.snap(0.577, 0.816)
single_time = time.perf_counter() - start

print(f"Single snap: {single_time/100000*1e6:.1f} μs")

# Profile batch snaps
vectors = [[0.577, 0.816] for _ in range(100000)]
start = time.perf_counter()
results = manifold.snap_batch(vectors)
batch_time = time.perf_counter() - start

print(f"Batch snap: {batch_time/100000*1e6:.1f} μs per vector")
print(f"Speedup: {single_time/batch_time:.1f}x")
```

---

## Error Handling

### Exception Hierarchy

```
Python Exception
├── ImportError          # Module load failure
├── TypeError            # Wrong argument types
├── ValueError           # Invalid values
└── RuntimeError         # Rust panic converted
```

### Handling Errors Gracefully

```python
from constraint_theory import PythagoreanManifold
import logging

def safe_snap(manifold, x, y):
    """Snap with error handling."""
    try:
        return manifold.snap(x, y)
    except TypeError as e:
        logging.warning(f"Invalid input types: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

# Context manager for manifold creation
def create_manifold_safely(density):
    """Create manifold with validation."""
    if not isinstance(density, int):
        raise TypeError("density must be an integer")
    if density <= 0:
        raise ValueError("density must be positive")
    if density > 100000:
        raise ValueError("density exceeds reasonable maximum")
    
    return PythagoreanManifold(density)
```

---

## Version Compatibility

| Python Version | Supported | Notes |
|----------------|-----------|-------|
| 3.8 | Yes | Minimum supported version |
| 3.9 | Yes | |
| 3.10 | Yes | |
| 3.11 | Yes | Recommended for performance |
| 3.12 | Yes | |
| 3.13 | Yes | |

| Platform | Supported | Notes |
|----------|-----------|-------|
| Linux x86_64 | Yes | Primary platform |
| macOS ARM64 | Yes | Apple Silicon |
| macOS x86_64 | Yes | Intel Mac |
| Windows x86_64 | Yes | MSVC toolchain |

---

## Monitoring and Observability

### Metrics to Track

```python
import time
from dataclasses import dataclass

@dataclass
class ManifoldMetrics:
    density: int
    state_count: int
    creation_time_ms: float
    avg_snap_time_us: float

def benchmark_manifold(density: int, iterations: int = 10000) -> ManifoldMetrics:
    start = time.perf_counter()
    m = PythagoreanManifold(density)
    creation_time = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    for _ in range(iterations):
        m.snap(0.577, 0.816)
    snap_time = (time.perf_counter() - start) / iterations * 1e6
    
    return ManifoldMetrics(
        density=density,
        state_count=m.state_count,
        creation_time_ms=creation_time,
        avg_snap_time_us=snap_time
    )
```

---

## See Also

- [API Reference](API.md)
- [Migration Guide](MIGRATION.md)
- [Security Policy](../SECURITY.md)
- [Rust Core Documentation](https://github.com/SuperInstance/constraint-theory-core)
