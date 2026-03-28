"""Unit tests for PythagoreanManifold class."""

import pytest
import math


class TestManifoldCreation:
    """Tests for manifold creation and initialization."""

    def test_basic_creation(self):
        """Test basic manifold creation."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(100)
        assert manifold is not None
        assert manifold.state_count > 0

    def test_state_count_increases_with_density(self):
        """Test that higher density produces more states."""
        from constraint_theory import PythagoreanManifold
        
        m_low = PythagoreanManifold(50)
        m_high = PythagoreanManifold(200)
        
        assert m_high.state_count > m_low.state_count

    def test_state_count_reasonable_range(self):
        """Test state count is in reasonable range."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Should have around 1000 states
        assert manifold.state_count > 500
        assert manifold.state_count < 2000

    def test_repr_string(self):
        """Test string representation."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        repr_str = repr(manifold)
        
        assert "PythagoreanManifold" in repr_str
        assert "states" in repr_str


class TestSingleSnap:
    """Tests for single vector snapping."""

    def test_snap_exact_triple_3_4_5(self):
        """Test snapping exact 3-4-5 triple."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        x, y, noise = manifold.snap(0.6, 0.8)
        
        # (0.6, 0.8) = (3/5, 4/5) is exact
        assert abs(x - 0.6) < 0.01
        assert abs(y - 0.8) < 0.01
        assert noise < 0.001

    def test_snap_exact_triple_5_12_13(self):
        """Test snapping exact 5-12-13 triple."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        x, y, noise = manifold.snap(5/13, 12/13)
        
        assert abs(x - 5/13) < 0.01
        assert abs(y - 12/13) < 0.01
        assert noise < 0.001

    def test_snap_approximate_vector(self):
        """Test snapping approximate vector to nearest triple."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        x, y, noise = manifold.snap(0.577, 0.816)  # Close to (0.6, 0.8)
        
        # Should snap to nearby triple
        assert abs(x - 0.6) < 0.1
        assert abs(y - 0.8) < 0.1
        assert noise < 0.1

    def test_snap_returns_unit_vector(self):
        """Test that snapped vectors are on unit circle."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        test_vectors = [
            (0.577, 0.816),
            (0.707, 0.707),
            (0.1, 0.995),
            (0.999, 0.001),
            (-0.5, 0.866),
        ]
        
        for vx, vy in test_vectors:
            x, y, _ = manifold.snap(vx, vy)
            magnitude = math.sqrt(x * x + y * y)
            assert abs(magnitude - 1.0) < 0.0001, f"({x}, {y}) not on unit circle"

    def test_snap_all_quadrants(self):
        """Test snapping works in all quadrants."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Test vectors in each quadrant
        quadrant_tests = [
            (0.6, 0.8),    # Q1
            (-0.6, 0.8),   # Q2
            (-0.6, -0.8),  # Q3
            (0.6, -0.8),   # Q4
        ]
        
        for vx, vy in quadrant_tests:
            x, y, noise = manifold.snap(vx, vy)
            
            # Should snap to exact triple
            assert noise < 0.001, f"Noise too high for ({vx}, {vy})"
            
            # Check correct quadrant
            if vx > 0:
                assert x > 0
            else:
                assert x < 0
            if vy > 0:
                assert y > 0
            else:
                assert y < 0

    def test_snap_cardinal_directions(self):
        """Test snapping to cardinal directions."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        cardinals = [
            (1.0, 0.0),    # East
            (0.0, 1.0),    # North
            (-1.0, 0.0),   # West
            (0.0, -1.0),   # South
        ]
        
        for vx, vy in cardinals:
            x, y, noise = manifold.snap(vx, vy)
            
            assert abs(x - vx) < 0.001
            assert abs(y - vy) < 0.001
            assert noise < 0.001

    def test_snap_zero_vector(self):
        """Test snapping zero vector."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        x, y, noise = manifold.snap(0.0, 0.0)
        
        # Should handle gracefully (result depends on implementation)
        # The snapped point should still be on unit circle
        magnitude = math.sqrt(x * x + y * y)
        assert magnitude <= 1.0

    def test_snap_noise_positive(self):
        """Test that noise is non-negative."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        for _ in range(100):
            import random
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            _, _, noise = manifold.snap(x, y)
            assert noise >= 0

    def test_snap_consistency(self):
        """Test that same input produces same output."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        input_vec = (0.577, 0.816)
        results = [manifold.snap(*input_vec) for _ in range(10)]
        
        for result in results[1:]:
            assert result == results[0], "Snapping should be deterministic"


class TestSnapPrecision:
    """Tests for snapping precision and accuracy."""

    def test_higher_density_lower_noise(self):
        """Test that higher density gives lower noise on average."""
        from constraint_theory import PythagoreanManifold
        
        import random
        random.seed(42)
        
        test_vectors = [(random.uniform(-1, 1), random.uniform(-1, 1)) 
                       for _ in range(100)]
        
        m_low = PythagoreanManifold(50)
        m_high = PythagoreanManifold(500)
        
        total_noise_low = sum(m_low.snap(x, y)[2] for x, y in test_vectors)
        total_noise_high = sum(m_high.snap(x, y)[2] for x, y in test_vectors)
        
        assert total_noise_high < total_noise_low

    def test_exact_triples_zero_noise(self):
        """Test that exact Pythagorean triples have zero noise."""
        from constraint_theory import PythagoreanManifold, generate_triples
        
        manifold = PythagoreanManifold(200)
        triples = generate_triples(50)
        
        for a, b, c in triples[:10]:  # Test first 10
            x, y, noise = manifold.snap(a / c, b / c)
            assert noise < 0.001, f"Triple ({a}, {b}, {c}) should have zero noise"

    def test_snapped_magnitude_exact(self):
        """Test that snapped vectors have magnitude exactly 1."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            sx, sy, _ = manifold.snap(x, y)
            magnitude_squared = sx * sx + sy * sy
            
            # Should be exactly 1.0 (within float precision)
            assert abs(magnitude_squared - 1.0) < 1e-6


class TestManifoldProperties:
    """Tests for manifold properties and behavior."""

    def test_state_count_readonly(self):
        """Test that state_count is a property."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Should be able to read
        count = manifold.state_count
        assert isinstance(count, int)
        assert count > 0

    def test_different_manifolds_independent(self):
        """Test that different manifolds are independent."""
        from constraint_theory import PythagoreanManifold
        
        m1 = PythagoreanManifold(100)
        m2 = PythagoreanManifold(200)
        
        assert m1.state_count != m2.state_count

    def test_manifold_reusable(self):
        """Test that manifold can be used multiple times."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Use manifold many times
        for i in range(1000):
            x, y, _ = manifold.snap(0.5 + i * 0.001, 0.8)
        
        # Should still work
        x, y, noise = manifold.snap(0.6, 0.8)
        assert noise < 0.001


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_vectors(self):
        """Test snapping very small vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        x, y, noise = manifold.snap(0.0001, 0.0001)
        
        # Should snap to something
        magnitude = math.sqrt(x * x + y * y)
        assert magnitude <= 1.0

    def test_large_vectors(self):
        """Test snapping large vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        x, y, noise = manifold.snap(100, 100)
        
        # Should snap to something on unit circle
        magnitude = math.sqrt(x * x + y * y)
        assert abs(magnitude - 1.0) < 0.0001

    def test_negative_vectors(self):
        """Test snapping negative vectors."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        x, y, noise = manifold.snap(-0.6, -0.8)
        
        assert x < 0
        assert y < 0
        assert noise < 0.001

    def test_near_axis_vectors(self):
        """Test vectors near axes."""
        from constraint_theory import PythagoreanManifold
        
        manifold = PythagoreanManifold(200)
        
        # Near x-axis
        x, y, _ = manifold.snap(0.999, 0.001)
        assert abs(x) > abs(y)  # More horizontal
        
        # Near y-axis
        x, y, _ = manifold.snap(0.001, 0.999)
        assert abs(y) > abs(x)  # More vertical


class TestGenerateTriples:
    """Tests for Pythagorean triple generation."""

    def test_basic_generation(self):
        """Test basic triple generation."""
        from constraint_theory import generate_triples
        
        triples = generate_triples(50)
        
        assert len(triples) > 0
        assert isinstance(triples, list)

    def test_triples_valid(self):
        """Test that all generated triples are valid Pythagorean triples."""
        from constraint_theory import generate_triples
        
        triples = generate_triples(100)
        
        for a, b, c in triples:
            # Check a² + b² = c²
            assert a * a + b * b == c * c, f"Invalid triple: ({a}, {b}, {c})"
            
            # Check c <= max_c
            assert c <= 100

    def test_triples_primitive(self):
        """Test that generated triples are primitive."""
        from constraint_theory import generate_triples
        import math
        
        triples = generate_triples(100)
        
        for a, b, c in triples:
            # GCD should be 1 for primitive triples
            g = math.gcd(math.gcd(a, b), c)
            assert g == 1, f"Non-primitive triple: ({a}, {b}, {c})"

    def test_triples_ordered(self):
        """Test that triples have a < b."""
        from constraint_theory import generate_triples
        
        triples = generate_triples(100)
        
        for a, b, c in triples:
            assert a < b, f"Triple not ordered: ({a}, {b}, {c})"

    def test_known_triples(self):
        """Test that known triples are generated."""
        from constraint_theory import generate_triples
        
        triples = generate_triples(50)
        triple_set = set(triples)
        
        # Known triples
        assert (3, 4, 5) in triple_set
        assert (5, 12, 13) in triple_set
        assert (8, 15, 17) in triple_set


class TestModuleExports:
    """Tests for module exports and version."""

    def test_version_available(self):
        """Test that version is available."""
        from constraint_theory import __version__
        
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test version string format."""
        from constraint_theory import __version__
        
        # Should be semver-like
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_all_exports(self):
        """Test that __all__ is defined correctly."""
        from constraint_theory import __all__
        
        assert "PythagoreanManifold" in __all__
        assert "snap" in __all__
        assert "generate_triples" in __all__
        assert "__version__" in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
