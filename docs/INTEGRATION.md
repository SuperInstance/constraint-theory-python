# Integration Testing Guide

This document covers integration testing between Python bindings and the Rust core.

## Table of Contents

- [Compatibility Tests](#compatibility-tests)
- [Known FFI Limitations](#known-ffi-limitations)
- [CI/CD Pipeline](#cicd-pipeline)
- [Version Pinning](#version-pinning)

---

## Compatibility Tests

### Cross-Language Consistency

The Python bindings must produce identical results to the Rust core for the same inputs. These tests verify the FFI boundary.

```python
"""
Compatibility tests for Python bindings vs Rust core.

Run with: pytest tests/test_compatibility.py -v
"""

import pytest
import math
from constraint_theory import PythagoreanManifold, generate_triples


class TestRustCoreCompatibility:
    """Verify Python bindings match Rust core behavior."""
    
    def test_exact_triple_zero_noise(self):
        """Exact Pythagorean triples should have zero noise."""
        manifold = PythagoreanManifold(200)
        
        # Known Pythagorean triples
        test_cases = [
            (3, 4, 5),      # Classic
            (5, 12, 13),    # Another classic
            (8, 15, 17),    # Another
            (7, 24, 25),    # Another
        ]
        
        for a, b, c in test_cases:
            x, y, noise = manifold.snap(a/c, b/c)
            
            # Should snap to exact coordinates
            assert abs(x - a/c) < 0.001, f"Failed for ({a}, {b}, {c})"
            assert abs(y - b/c) < 0.001, f"Failed for ({a}, {b}, {c})"
            assert noise < 0.001, f"Failed for ({a}, {b}, {c})"
    
    def test_state_count_consistency(self):
        """State count should match expected formula."""
        # Rust uses Euclid's formula to generate states
        # Count should be approximately 5 * density for moderate densities
        
        for density in [50, 100, 200]:
            manifold = PythagoreanManifold(density)
            count = manifold.state_count
            
            # State count should be deterministic
            manifold2 = PythagoreanManifold(density)
            assert manifold2.state_count == count
    
    def test_batch_vs_single_consistency(self):
        """Batch results must match individual snap results."""
        manifold = PythagoreanManifold(200)
        
        # Generate test vectors
        vectors = [
            (0.6, 0.8),
            (0.707, 0.707),
            (0.1, 0.995),
            (-0.5, 0.866),
            (0.999, 0.001),
        ]
        
        # Get batch results
        batch_results = manifold.snap_batch(vectors)
        
        # Compare with individual results
        for i, (x, y) in enumerate(vectors):
            single_result = manifold.snap(x, y)
            batch_result = batch_results[i]
            
            assert abs(single_result[0] - batch_result[0]) < 1e-6
            assert abs(single_result[1] - batch_result[1]) < 1e-6
            assert abs(single_result[2] - batch_result[2]) < 1e-6
    
    def test_determinism(self):
        """Same inputs must produce same outputs across calls."""
        manifold = PythagoreanManifold(200)
        
        test_input = (0.577, 0.816)
        results = [manifold.snap(*test_input) for _ in range(100)]
        
        first = results[0]
        for r in results[1:]:
            assert r == first, "Results should be deterministic"
    
    def test_cross_quadrant_consistency(self):
        """Snapping should work correctly in all quadrants."""
        manifold = PythagoreanManifold(200)
        
        # Test in all quadrants
        test_cases = [
            (0.6, 0.8),     # Q1
            (-0.6, 0.8),    # Q2
            (-0.6, -0.8),   # Q3
            (0.6, -0.8),    # Q4
        ]
        
        for x, y in test_cases:
            sx, sy, noise = manifold.snap(x, y)
            
            # Verify quadrant
            assert (sx > 0) == (x > 0), f"X sign mismatch for ({x}, {y})"
            assert (sy > 0) == (y > 0), f"Y sign mismatch for ({x}, {y})"
            
            # Verify unit vector
            mag = math.sqrt(sx*sx + sy*sy)
            assert abs(mag - 1.0) < 1e-6


class TestTypeSystemCompatibility:
    """Verify type system matches Rust types."""
    
    def test_density_type_validation(self):
        """Density must be a positive integer."""
        # Valid densities
        PythagoreanManifold(1)
        PythagoreanManifold(100)
        PythagoreanManifold(1000)
        
        # Invalid densities should raise
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            PythagoreanManifold(0)
        
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            PythagoreanManifold(-1)
    
    def test_snap_parameter_types(self):
        """Snap parameters must be numeric."""
        manifold = PythagoreanManifold(200)
        
        # Valid types
        manifold.snap(0.6, 0.8)           # floats
        manifold.snap(3, 4)               # ints
        manifold.snap(3.0, 4)             # mixed
        
        # Invalid types
        with pytest.raises(TypeError):
            manifold.snap("0.6", 0.8)
        
        with pytest.raises(TypeError):
            manifold.snap(0.6, None)
    
    def test_return_types(self):
        """Return types should match expected Python types."""
        manifold = PythagoreanManifold(200)
        
        # snap returns tuple of floats
        result = manifold.snap(0.6, 0.8)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
        
        # state_count returns int
        assert isinstance(manifold.state_count, int)
        
        # generate_triples returns list of tuples
        triples = generate_triples(50)
        assert isinstance(triples, list)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in triples)


class TestNumericalPrecision:
    """Verify numerical precision matches Rust f32 behavior."""
    
    def test_unit_vector_exactness(self):
        """Snapped vectors should be exactly on unit circle."""
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            sx, sy, _ = manifold.snap(x, y)
            mag_sq = sx*sx + sy*sy
            
            # Should be exactly 1.0 within float precision
            assert abs(mag_sq - 1.0) < 1e-6
    
    def test_noise_range(self):
        """Noise should be in valid range [0, 2]."""
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            
            _, _, noise = manifold.snap(x, y)
            
            assert 0.0 <= noise <= 2.0


class TestBatchProcessing:
    """Test batch processing edge cases."""
    
    def test_empty_batch(self):
        """Empty batch should return empty list."""
        manifold = PythagoreanManifold(200)
        results = manifold.snap_batch([])
        assert results == []
    
    def test_single_element_batch(self):
        """Single element batch should work."""
        manifold = PythagoreanManifold(200)
        results = manifold.snap_batch([[0.6, 0.8]])
        
        assert len(results) == 1
        assert abs(results[0][0] - 0.6) < 0.01
        assert abs(results[0][1] - 0.8) < 0.01
    
    def test_large_batch(self):
        """Large batches should be handled correctly."""
        manifold = PythagoreanManifold(200)
        
        vectors = [[0.5, 0.8] for _ in range(10000)]
        results = manifold.snap_batch(vectors)
        
        assert len(results) == 10000
        for sx, sy, noise in results:
            assert -1.0 <= sx <= 1.0
            assert -1.0 <= sy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Known FFI Limitations

### Type System Differences

| Rust Type | Python Type | Limitation |
|-----------|-------------|------------|
| `f32` | `float` | Python uses 64-bit floats; precision may differ slightly |
| `usize` | `int` | Platform-dependent size in Rust, always 64-bit in Python 3 |
| `[T; N]` | `tuple` | Rust arrays have fixed size; Python tuples are dynamic |
| `&[T]` | `list` | Rust slices are borrowed; Python lists are copied |

### Performance Limitations

| Operation | Limitation | Mitigation |
|-----------|------------|------------|
| Small batches | FFI overhead dominates | Use individual `snap()` for <10 vectors |
| NumPy arrays | Copy required at FFI boundary | Accept overhead, benefit from SIMD |
| String conversion | Not supported | Pre-convert all inputs to float |

### Memory Limitations

| Scenario | Limitation | Mitigation |
|----------|------------|------------|
| Very high density | Memory grows as O(density²) | Limit density to reasonable values |
| Large batches | All vectors in memory at once | Process in chunks |

### Threading Limitations

| Scenario | Limitation | Mitigation |
|----------|------------|------------|
| Multiprocessing | Manifold must be recreated in each process | Use threading instead |
| Async/await | No native async support | Use `run_in_executor` |

```python
# Example: Using with asyncio
import asyncio
from constraint_theory import PythagoreanManifold

manifold = PythagoreanManifold(200)

async def async_snap(x, y):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, manifold.snap, x, y)

async def main():
    result = await async_snap(0.577, 0.816)
    print(result)

asyncio.run(main())
```

### Platform-Specific Limitations

| Platform | Limitation |
|----------|------------|
| Windows | Requires MSVC build tools |
| macOS ARM | Native ARM builds only (no Rosetta for Python extensions) |
| Linux musl | May require static linking |

---

## CI/CD Pipeline

### GitHub Actions Configuration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Rust
      uses: dtolnay/rust-action@stable
    
    - name: Install maturin
      run: pip install maturin
    
    - name: Build extension
      run: maturin develop --release
    
    - name: Install test dependencies
      run: pip install pytest pytest-cov numpy
    
    - name: Run tests
      run: pytest tests/ -v --cov=constraint_theory
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install linters
      run: pip install black isort mypy
    
    - name: Check formatting
      run: |
        black --check constraint_theory/ tests/
        isort --check constraint_theory/ tests/
    
    - name: Type check
      run: mypy constraint_theory/
```

### Build Matrix

| Platform | Python Versions | Rust Version |
|----------|-----------------|--------------|
| Ubuntu 22.04 | 3.8-3.12 | stable |
| macOS 13 (Intel) | 3.8-3.12 | stable |
| macOS 14 (ARM) | 3.8-3.12 | stable |
| Windows Server 2022 | 3.8-3.12 | stable |

### Release Process

```yaml
# .github/workflows/release.yml
name: Release

on:
  release:
    types: [published]

jobs:
  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build wheels
      uses: PyO3/maturin-action@v1
      with:
        command: build
        args: --release --out dist
    
    - name: Upload wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: dist/

  publish:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
    - name: Download wheels
      uses: actions/download-artifact@v3
      with:
        name: wheels
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

---

## Version Pinning

### Version Compatibility Matrix

| Python Bindings | Rust Core | Notes |
|-----------------|-----------|-------|
| 0.1.x | >= 1.0.0, < 2.0.0 | Initial release |

### Dependency Specification

```toml
# Cargo.toml
[dependencies]
# Exact version pin for stability
constraint-theory-core = "1.0.1"

# Or range for flexibility
# constraint-theory-core = ">=1.0.0,<2.0.0"
```

### Semantic Versioning

The Python bindings follow semantic versioning:

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Version Checking

```python
from constraint_theory import __version__, CORE_MIN_VERSION, CORE_MAX_VERSION

def check_version_compatibility():
    """Verify version compatibility at runtime."""
    parts = __version__.split('.')
    major, minor = int(parts[0]), int(parts[1])
    
    # Check if compatible with your application
    if major < 0 or (major == 0 and minor < 1):
        raise RuntimeError(f"Incompatible version: {__version__}")
    
    return True
```

### Lock File

For reproducible deployments, use a lock file:

```
# requirements.lock
constraint-theory==0.1.0
numpy==1.24.3
```

---

## Integration Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run compatibility tests only
pytest tests/test_compatibility.py -v

# Run with coverage
pytest tests/ --cov=constraint_theory --cov-report=html

# Run benchmarks
pytest tests/ --benchmark-only

# Run slow tests
pytest tests/ --runslow

# Run with specific Python version
python3.11 -m pytest tests/
```

---

## See Also

- [API Reference](API.md)
- [Production Guide](PRODUCTION.md)
- [Security Policy](../SECURITY.md)
