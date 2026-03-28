"""
Compatibility tests for Python bindings vs Rust core.

These tests verify the FFI boundary and ensure the Python bindings
produce identical results to the Rust core implementation.

Run with: pytest tests/test_compatibility.py -v
"""

import pytest
import math
from typing import List, Tuple


class TestRustCoreCompatibility:
    """Verify Python bindings match Rust core behavior."""
    
    @pytest.fixture
    def manifold(self):
        """Create a standard test manifold."""
        from constraint_theory import PythagoreanManifold
        return PythagoreanManifold(200)
    
    def test_exact_triple_zero_noise(self, manifold):
        """Exact Pythagorean triples should have zero noise.
        
        Rust core test: test_snap_exact_triple in manifold.rs
        """
        # Known Pythagorean triples
        test_cases = [
            (3, 4, 5),      # Classic
            (5, 12, 13),    # Another classic
            (8, 15, 17),    # Another
            (7, 24, 25),    # Another
            (20, 21, 29),   # Another
        ]
        
        for a, b, c in test_cases:
            x, y, noise = manifold.snap(a/c, b/c)
            
            # Should snap to exact coordinates
            assert abs(x - a/c) < 0.001, f"Failed for ({a}, {b}, {c})"
            assert abs(y - b/c) < 0.001, f"Failed for ({a}, {b}, {c})"
            assert noise < 0.001, f"Failed for ({a}, {b}, {c})"
    
    def test_state_count_deterministic(self):
        """State count should be deterministic for same density.
        
        Rust core: state_count() returns valid_states.len()
        """
        from constraint_theory import PythagoreanManifold
        
        for density in [50, 100, 200]:
            manifold1 = PythagoreanManifold(density)
            manifold2 = PythagoreanManifold(density)
            
            assert manifold1.state_count == manifold2.state_count
            assert manifold1.state_count > 0
    
    def test_state_count_increases_with_density(self):
        """Higher density should produce more states.
        
        Rust core: more m values generate more triples
        """
        from constraint_theory import PythagoreanManifold
        
        m50 = PythagoreanManifold(50)
        m100 = PythagoreanManifold(100)
        m200 = PythagoreanManifold(200)
        
        assert m100.state_count > m50.state_count
        assert m200.state_count > m100.state_count
    
    def test_batch_vs_single_consistency(self, manifold):
        """Batch results must match individual snap results.
        
        Rust core: snap_batch_simd should produce identical results to snap
        """
        # Generate test vectors
        vectors = [
            (0.6, 0.8),
            (0.707, 0.707),
            (0.1, 0.995),
            (-0.5, 0.866),
            (0.999, 0.001),
            (0.577, 0.816),
        ]
        
        # Get batch results
        batch_results = manifold.snap_batch(vectors)
        
        # Compare with individual results
        for i, (x, y) in enumerate(vectors):
            single_result = manifold.snap(x, y)
            batch_result = batch_results[i]
            
            assert abs(single_result[0] - batch_result[0]) < 1e-6, \
                f"X mismatch at {i}: single={single_result[0]} batch={batch_result[0]}"
            assert abs(single_result[1] - batch_result[1]) < 1e-6, \
                f"Y mismatch at {i}: single={single_result[1]} batch={batch_result[1]}"
            assert abs(single_result[2] - batch_result[2]) < 1e-6, \
                f"Noise mismatch at {i}: single={single_result[2]} batch={batch_result[2]}"
    
    def test_determinism(self, manifold):
        """Same inputs must produce same outputs across calls.
        
        Rust core: all operations are deterministic given same input
        """
        test_input = (0.577, 0.816)
        results = [manifold.snap(*test_input) for _ in range(100)]
        
        first = results[0]
        for r in results[1:]:
            assert r == first, "Results should be deterministic"
    
    def test_cross_quadrant_consistency(self, manifold):
        """Snapping should work correctly in all quadrants.
        
        Rust core: generates states for all quadrants
        """
        # Test in all quadrants
        test_cases = [
            (0.6, 0.8),     # Q1
            (-0.6, 0.8),    # Q2
            (-0.6, -0.8),   # Q3
            (0.6, -0.8),    # Q4
        ]
        
        for x, y in test_cases:
            sx, sy, noise = manifold.snap(x, y)
            
            # Verify quadrant preserved
            assert (sx > 0) == (x > 0), f"X sign mismatch for ({x}, {y})"
            assert (sy > 0) == (y > 0), f"Y sign mismatch for ({x}, {y})"
            
            # Verify unit vector
            mag = math.sqrt(sx*sx + sy*sy)
            assert abs(mag - 1.0) < 1e-6, f"Not unit vector for ({x}, {y})"
    
    def test_cardinal_directions(self, manifold):
        """Cardinal directions should snap exactly.
        
        Rust core: [1,0], [0,1], [-1,0], [0,-1] are added explicitly
        """
        cardinals = [
            (1.0, 0.0),    # East
            (0.0, 1.0),    # North
            (-1.0, 0.0),   # West
            (0.0, -1.0),   # South
        ]
        
        for x, y in cardinals:
            sx, sy, noise = manifold.snap(x, y)
            
            assert abs(sx - x) < 0.001, f"Failed for cardinal ({x}, {y})"
            assert abs(sy - y) < 0.001, f"Failed for cardinal ({x}, {y})"
            assert noise < 0.001, f"Non-zero noise for cardinal ({x}, {y})"


class TestTypeSystemCompatibility:
    """Verify type system matches Rust types (PASS 6)."""
    
    def test_density_type_validation(self):
        """Density must be a positive integer.
        
        Rust core: density: usize
        """
        from constraint_theory import PythagoreanManifold
        
        # Valid densities
        PythagoreanManifold(1)
        PythagoreanManifold(100)
        PythagoreanManifold(1000)
        
        # Invalid densities should raise
        with pytest.raises((TypeError, ValueError, RuntimeError, BaseException)):
            PythagoreanManifold(0)
        
        with pytest.raises((TypeError, ValueError, RuntimeError, BaseException)):
            PythagoreanManifold(-1)
    
    def test_snap_parameter_types(self):
        """Snap parameters must be numeric.
        
        Rust core: snap(vector: [f32; 2])
        """
        from constraint_theory import PythagoreanManifold
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
        """Return types should match expected Python types.
        
        Rust core: ([f32; 2], f32) -> Python (float, float, float)
        """
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        # snap returns tuple of floats
        result = manifold.snap(0.6, 0.8)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
        
        # state_count returns int
        assert isinstance(manifold.state_count, int)
        
        # density returns int
        assert isinstance(manifold.density, int)
    
    def test_generate_triples_types(self):
        """generate_triples return types.
        
        Rust core: Vec<(i32, i32, i32)>
        """
        from constraint_theory import generate_triples
        
        triples = generate_triples(50)
        
        assert isinstance(triples, list)
        for t in triples:
            assert isinstance(t, tuple)
            assert len(t) == 3
            assert all(isinstance(x, int) for x in t)


class TestNumericalPrecision:
    """Verify numerical precision matches Rust f32 behavior."""
    
    def test_unit_vector_exactness(self):
        """Snapped vectors should be exactly on unit circle.
        
        Rust core: a/c, b/c where a² + b² = c²
        """
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            sx, sy, _ = manifold.snap(x, y)
            mag_sq = sx*sx + sy*sy
            
            # Should be exactly 1.0 within float precision
            assert abs(mag_sq - 1.0) < 1e-6, f"Not unit: ({sx}, {sy}), mag²={mag_sq}"
    
    def test_noise_range(self):
        """Noise should be in valid range [0, 2].
        
        Rust core: noise = 1.0 - resonance, where resonance is dot product
        """
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            
            _, _, noise = manifold.snap(x, y)
            
            # noise = 1 - dot_product, dot_product ∈ [-1, 1]
            # So noise ∈ [0, 2]
            assert 0.0 <= noise <= 2.0, f"Noise out of range: {noise}"
    
    def test_f32_precision(self):
        """Verify f32 precision limits are respected.
        
        Rust core uses f32 (32-bit floats), Python uses f64.
        Results should match within f32 precision.
        """
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        # Known exact values
        x, y, noise = manifold.snap(3/5, 4/5)
        
        # Check that we get the exact rational representation
        # 3/5 = 0.6 exactly representable in f32
        assert abs(x - 0.6) < 1e-7
        assert abs(y - 0.8) < 1e-7


class TestBatchProcessing:
    """Test batch processing consistency with Rust SIMD."""
    
    def test_empty_batch(self):
        """Empty batch should return empty list."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        results = manifold.snap_batch([])
        assert results == []
    
    def test_single_element_batch(self):
        """Single element batch should match snap()."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        single = manifold.snap(0.6, 0.8)
        batch = manifold.snap_batch([[0.6, 0.8]])[0]
        
        assert abs(single[0] - batch[0]) < 1e-6
        assert abs(single[1] - batch[1]) < 1e-6
        assert abs(single[2] - batch[2]) < 1e-6
    
    def test_large_batch_consistency(self):
        """Large batches should be consistent with individual snaps."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        vectors = [[random.uniform(-1, 1), random.uniform(-1, 1)] 
                   for _ in range(1000)]
        
        batch_results = manifold.snap_batch(vectors)
        
        for i, vec in enumerate(vectors):
            single = manifold.snap(vec[0], vec[1])
            batch = batch_results[i]
            
            assert abs(single[0] - batch[0]) < 1e-6, f"Mismatch at {i}"
            assert abs(single[1] - batch[1]) < 1e-6, f"Mismatch at {i}"
            assert abs(single[2] - batch[2]) < 1e-6, f"Mismatch at {i}"


class TestFFIBoundary:
    """Test FFI boundary edge cases."""
    
    def test_nan_handling(self):
        """NaN handling at FFI boundary."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        # NaN should be handled gracefully (behavior is implementation-defined)
        # The test just ensures no segfault
        try:
            result = manifold.snap(float('nan'), 0.8)
            # If it succeeds, result should be valid
            assert len(result) == 3
        except (ValueError, RuntimeError):
            pass  # Acceptable to reject NaN
    
    def test_inf_handling(self):
        """Infinity handling at FFI boundary."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        # Infinity should be handled gracefully
        try:
            result = manifold.snap(float('inf'), 0.8)
            assert len(result) == 3
        except (ValueError, RuntimeError):
            pass  # Acceptable to reject infinity
    
    def test_very_large_values(self):
        """Very large values should be normalized."""
        from constraint_theory import PythagoreanManifold
        manifold = PythagoreanManifold(200)
        
        # Large values should snap to unit circle
        x, y, noise = manifold.snap(1e10, 1e10)
        
        mag = math.sqrt(x*x + y*y)
        assert abs(mag - 1.0) < 1e-6


class TestVersionCompatibility:
    """Test version-related functionality."""
    
    def test_version_available(self):
        """Version should be available."""
        from constraint_theory import __version__
        
        assert __version__ is not None
        assert isinstance(__version__, str)
    
    def test_version_format(self):
        """Version should follow semver."""
        from constraint_theory import __version__
        
        parts = __version__.split('.')
        assert len(parts) >= 2
        
        # Should be numeric
        int(parts[0])  # Major
        int(parts[1])  # Minor
    
    def test_core_version_bounds(self):
        """Core version bounds should be defined."""
        from constraint_theory import CORE_MIN_VERSION, CORE_MAX_VERSION
        
        assert CORE_MIN_VERSION <= CORE_MAX_VERSION
        assert isinstance(CORE_MIN_VERSION, tuple)
        assert isinstance(CORE_MAX_VERSION, tuple)


class TestProtocolConformance:
    """Test that types conform to defined protocols (PASS 6)."""
    
    def test_manifold_protocol(self):
        """Manifold should conform to ManifoldProtocol."""
        from constraint_theory import PythagoreanManifold, ManifoldProtocol
        
        manifold = PythagoreanManifold(200)
        
        # Check protocol conformance
        assert isinstance(manifold, ManifoldProtocol)
        
        # Check required attributes
        assert hasattr(manifold, 'state_count')
        assert hasattr(manifold, 'snap')
        assert hasattr(manifold, 'snap_batch')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
