# Rust FFI Documentation

This document covers the Foreign Function Interface (FFI) between Python and the Rust core library.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [PyO3 Binding Patterns](#pyo3-binding-patterns)
- [Memory Management](#memory-management)
- [Type Conversions](#type-conversions)
- [Debugging FFI Issues](#debugging-ffi-issues)
- [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### Component Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application                            │
│  from constraint_theory import PythagoreanManifold               │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Python C API
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PyO3 Layer                                    │
│  src/lib.rs - Python bindings definitions                        │
│  - #[pyclass] wrapper structs                                    │
│  - #[pymethods] implementations                                  │
│  - #[pyfunction] free functions                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Pure Rust API
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 constraint-theory-core                           │
│  Rust crate with Pythagorean manifold implementation             │
│  - PythagoreanManifold struct                                    │
│  - KD-tree for O(log n) lookups                                  │
│  - SIMD batch processing                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Build Process

```
Python Package (pip install)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  maturin (build backend)                           │
│  - Compiles Rust code with PyO3                    │
│  - Creates CPython extension module (.so/.pyd)     │
│  - Packages as wheel                               │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Native Extension                                  │
│  - Linux: constraint_theory_python.cpython-311-   │
│           x86_64-linux-gnu.so                      │
│  - macOS: constraint_theory_python.cpython-311-   │
│           darwin.so                                │
│  - Windows: constraint_theory_python.cp311-       │
│             win_amd64.pyd                          │
└───────────────────────────────────────────────────┘
```

---

## PyO3 Binding Patterns

### Class Wrapping Pattern

The standard pattern for wrapping Rust structs:

```rust
use pyo3::prelude::*;

/// Wrapper struct that holds the Rust type
#[pyclass(name = "PythagoreanManifold")]
pub struct PyManifold {
    inner: PythagoreanManifold,  // The actual Rust type
    density: usize,               // Additional cached data
}

#[pymethods]
impl PyManifold {
    /// Constructor - called from Python as PythagoreanManifold(density)
    #[new]
    pub fn new(density: usize) -> Self {
        PyManifold {
            inner: PythagoreanManifold::new(density),
            density,
        }
    }

    /// Property getter - accessed as manifold.state_count
    #[getter]
    pub fn state_count(&self) -> usize {
        self.inner.state_count()
    }

    /// Method - called as manifold.snap(x, y)
    pub fn snap(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let (snapped, noise) = self.inner.snap([x, y]);
        (snapped[0], snapped[1], noise)
    }
}
```

### GIL Release Pattern

For long-running operations, release the GIL to allow Python threads to run:

```rust
use pyo3::types::PyList;

pub fn snap_batch_simd(&self, py: Python<'_>, vectors: &PyList) -> PyResult<Vec<(f32, f32, f32)>> {
    // Convert Python types while holding GIL
    let input: Vec<[f32; 2]> = vectors
        .iter()
        .map(|item| {
            let t: (f32, f32) = item.extract()?;
            Ok([t.0, t.1])
        })
        .collect::<PyResult<_>>()?;
    
    // Release GIL for computation
    py.allow_threads(|| {
        let results = self.inner.snap_batch_simd(&input);
        Ok(results.into_iter().map(|(s, n)| (s[0], s[1], n)).collect())
    })
}
```

### Error Handling Pattern

Convert Rust errors to Python exceptions:

```rust
use pyo3::exceptions::{PyValueError, PyRuntimeError};

fn validate_and_snap(&self, x: f32, y: f32) -> PyResult<(f32, f32, f32)> {
    // Validate inputs
    if !x.is_finite() || !y.is_finite() {
        return Err(PyValueError::new_err("Coordinates must be finite numbers"));
    }
    
    // Perform operation
    match self.inner.snap_checked([x, y]) {
        Ok((snapped, noise)) => Ok((snapped[0], snapped[1], noise)),
        Err(e) => Err(PyRuntimeError::new_err(format!("Internal error: {}", e))),
    }
}
```

### Module Registration Pattern

Register classes and functions in the module:

```rust
#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyManifold>()?;
    
    // Register functions
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    
    // Add module-level constants
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}
```

---

## Memory Management

### Ownership Model

```
Python Object Lifecycle
───────────────────────

1. Creation (Python)
   manifold = PythagoreanManifold(200)
   
   ┌─────────────────┐
   │  Python Object  │  ← Python's gc manages this
   │  (PyManifold)   │
   │  ┌───────────┐  │
   │  │ inner:    │──┼──┐
   │  └───────────┘  │  │
   └─────────────────┘  │
                        ▼
          ┌────────────────────────┐
          │  Rust PythagoreanManifold │  ← Rust ownership
          │  - valid_states: Vec    │
          │  - kdtree: KDTree       │
          └────────────────────────┘

2. Usage (Python)
   result = manifold.snap(x, y)
   - Borrow happens internally
   - No cloning required

3. Deletion (Python)
   del manifold
   # OR when refcount reaches 0
   
   ┌─────────────────┐
   │  Python calls   │
   │  __dealloc__    │
   └────────┬────────┘
            │
            ▼
   ┌────────────────────────┐
   │  Rust drop() called    │
   │  - Frees valid_states  │
   │  - Frees kdtree        │
   └────────────────────────┘
```

### Memory Layout

| Component | Location | Lifetime | Size |
|-----------|----------|----------|------|
| PyManifold wrapper | Python heap | Python GC | ~16 bytes |
| PythagoreanManifold | Rust heap | Owned by PyManifold | ~240KB (density=200) |
| valid_states Vec | Rust heap | Owned by PythagoreanManifold | ~80KB |
| KDTree | Rust heap | Owned by PythagoreanManifold | ~160KB |

### Memory Safety Guarantees

1. **No use-after-free**: Rust's ownership system prevents accessing freed memory
2. **No double-free**: `drop` is called exactly once
3. **Thread-safe**: All access is through immutable references after construction
4. **No buffer overflows**: Rust bounds-checking is enforced

### Copy Semantics

```python
# Python reference semantics
m1 = PythagoreanManifold(200)
m2 = m1  # m2 is the same object (reference)

# To create a copy, construct a new manifold
m3 = PythagoreanManifold(m1.density)  # New instance with same density
```

### Memory Profiling Example

```python
import sys
import tracemalloc
from constraint_theory import PythagoreanManifold

# Track memory usage
tracemalloc.start()

# Create manifold
m = PythagoreanManifold(200)
current, peak = tracemalloc.get_traced_memory()
print(f"Manifold memory: {current / 1024:.1f} KB")

# Expected output: ~240-320 KB depending on density

# Delete manifold
del m
current, _ = tracemalloc.get_traced_memory()
print(f"After deletion: {current / 1024:.1f} KB")

# Expected output: ~0 KB (all Rust memory freed)
```

---

## Type Conversions

### Primitive Types

| Rust Type | Python Type | Conversion | Notes |
|-----------|-------------|------------|-------|
| `i32` | `int` | Automatic | Exact |
| `usize` | `int` | Automatic | Platform-dependent in Rust |
| `f32` | `float` | Automatic | Python uses f64 internally |
| `f64` | `float` | Automatic | Exact |
| `bool` | `bool` | Automatic | Exact |

### Collection Types

| Rust Type | Python Type | Conversion | Overhead |
|-----------|-------------|------------|----------|
| `[f32; 2]` | `Tuple[float, float]` | Copy | Minimal |
| `Vec<f32>` | `List[float]` | Copy | O(n) |
| `Vec<[f32; 2]>` | `List[Tuple[float, float]]` | Copy | O(n) |
| `&[f32]` | N/A | Must copy | - |

### Custom Conversion Example

```rust
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

impl PyManifold {
    /// Convert NumPy-style input to Rust Vec
    fn vectors_to_rust<'py>(&self, py: Python<'py>, input: &PyAny) -> PyResult<Vec<[f32; 2]>> {
        // Check if it's a list
        if let Ok(list) = input.downcast::<PyList>() {
            list.iter()
                .map(|item| {
                    if let Ok(tup) = item.downcast::<PyTuple>() {
                        let x: f32 = tup.get_item(0)?.extract()?;
                        let y: f32 = tup.get_item(1)?.extract()?;
                        Ok([x, y])
                    } else if let Ok(lst) = item.downcast::<PyList>() {
                        let x: f32 = lst.get_item(0)?.extract()?;
                        let y: f32 = lst.get_item(1)?.extract()?;
                        Ok([x, y])
                    } else {
                        Err(PyTypeError::new_err("Expected tuple or list"))
                    }
                })
                .collect()
        } else {
            // Could add NumPy array handling here
            Err(PyTypeError::new_err("Expected list of vectors"))
        }
    }
}
```

---

## Debugging FFI Issues

### Enable Debug Logging

```bash
# Enable Rust backtrace
export RUST_BACKTRACE=1

# Enable debug logging
export RUST_LOG=debug

# Run Python script
python your_script.py
```

### Common Issues and Solutions

#### 1. ImportError: cannot import name 'PythagoreanManifold'

**Causes:**
- Extension not built
- Architecture mismatch
- Python version mismatch

**Solution:**
```bash
# Rebuild
pip install maturin
maturin develop --release

# Verify build
ls -la target/release/libconstraint_theory_python.*

# Check Python can find it
python -c "import constraint_theory; print(constraint_theory.__file__)"
```

#### 2. Segmentation Fault

**Causes:**
- Bug in Rust code
- Memory corruption (rare with safe Rust)
- Undefined behavior in unsafe code

**Solution:**
```bash
# Run with gdb
gdb python
(gdb) run your_script.py

# Or with address sanitizer
export RUSTFLAGS="-Zsanitizer=address"
cargo build --target x86_64-unknown-linux-gnu
```

#### 3. Memory Leaks

**Detection:**
```bash
# Use valgrind
valgrind --leak-check=full --show-leak-kinds=all python your_script.py

# Or Python's tracemalloc
python -c "
import tracemalloc
tracemalloc.start()
# Your code here
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
"
```

#### 4. Type Conversion Errors

```python
import traceback
from constraint_theory import PythagoreanManifold

m = PythagoreanManifold(200)

try:
    m.snap("not a float", 0.8)
except TypeError as e:
    traceback.print_exc()
    # Output shows exact type mismatch
```

### Debug Build

For development, use debug builds for better error messages:

```bash
# Build in debug mode
maturin develop

# Features:
# - Bounds checking enabled
# - Debug assertions
# - Better panic messages
# - No optimization
```

### Panic Handling

Rust panics are converted to Python exceptions:

```rust
// In Rust
pub fn new(density: usize) -> Self {
    assert!(density > 0, "density must be positive");
    // ...
}

// Python sees:
// RuntimeError: assertion failed: density must be positive
```

---

## Performance Optimization

### Minimize FFI Crossings

```python
# BAD: Many FFI crossings
for x, y in vectors:
    result = manifold.snap(x, y)  # FFI crossing each time

# GOOD: Single FFI crossing
results = manifold.snap_batch(vectors)  # One FFI crossing
```

### Use Appropriate Types

```python
# BAD: Python creates intermediate objects
def process(vectors):
    return [manifold.snap(float(x), float(y)) for x, y in vectors]

# GOOD: Direct pass-through
def process(vectors):
    return manifold.snap_batch(vectors)
```

### Release GIL for Parallelism

```python
from concurrent.futures import ThreadPoolExecutor

# GIL is released during snap_batch, allowing true parallelism
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(
        lambda chunk: manifold.snap_batch(chunk),
        chunks
    ))
```

### Memory Pool for Batch Operations

```python
import numpy as np
from constraint_theory import PythagoreanManifold

class BatchProcessor:
    def __init__(self, density: int, chunk_size: int = 10000):
        self.manifold = PythagoreanManifold(density)
        self.chunk_size = chunk_size
        # Pre-allocate output buffer
        self.output_buffer = np.zeros((chunk_size, 3), dtype=np.float32)
    
    def process(self, vectors: np.ndarray) -> np.ndarray:
        results = []
        for i in range(0, len(vectors), self.chunk_size):
            chunk = vectors[i:i + self.chunk_size]
            batch_results = self.manifold.snap_batch(chunk)
            results.extend(batch_results)
        return np.array(results)
```

---

## See Also

- [PyO3 Documentation](https://pyo3.rs/)
- [Production Guide](PRODUCTION.md)
- [API Reference](API.md)
- [Rust Core Library](https://github.com/SuperInstance/constraint-theory-core)
