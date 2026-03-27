"""Tests for Constraint Theory Python bindings."""

import pytest


def test_manifold_creation():
    """Test that manifold can be created with various densities."""
    from constraint_theory import PythagoreanManifold
    
    m = PythagoreanManifold(200)
    assert m.state_count > 0
    assert m.state_count > 500  # Should have at least 500 states


def test_snap_exact_triple():
    """Test snapping an exact Pythagorean triple."""
    from constraint_theory import PythagoreanManifold
    
    m = PythagoreanManifold(200)
    x, y, noise = m.snap(0.6, 0.8)
    
    # (0.6, 0.8) = (3/5, 4/5) is an exact Pythagorean triple
    assert abs(x - 0.6) < 0.01
    assert abs(y - 0.8) < 0.01
    assert noise < 0.001


def test_snap_approximate():
    """Test snapping an approximate vector."""
    from constraint_theory import PythagoreanManifold
    
    m = PythagoreanManifold(200)
    x, y, noise = m.snap(0.599, 0.801)
    
    # Should snap close to (0.6, 0.8)
    assert abs(x - 0.6) < 0.1
    assert abs(y - 0.8) < 0.1
    assert noise < 0.1


def test_snap_batch():
    """Test batch snapping."""
    from constraint_theory import PythagoreanManifold
    
    m = PythagoreanManifold(200)
    vectors = [[0.6, 0.8], [0.8, 0.6], [0.1, 0.99]]
    results = m.snap_batch(vectors)
    
    assert len(results) == 3
    
    for snapped_x, snapped_y, noise in results:
        assert -1.0 <= snapped_x <= 1.0
        assert -1.0 <= snapped_y <= 1.0
        assert 0.0 <= noise <= 2.0


def test_snap_batch_numpy():
    """Test batch snapping with NumPy array."""
    np = pytest.importorskip("numpy")
    from constraint_theory import PythagoreanManifold
    
    m = PythagoreanManifold(200)
    vectors = np.array([[0.6, 0.8], [0.8, 0.6]])
    results = m.snap_batch(vectors)
    
    assert len(results) == 2


def test_generate_triples():
    """Test Pythagorean triple generation."""
    from constraint_theory import generate_triples
    
    triples = generate_triples(50)
    
    assert len(triples) > 0
    
    for a, b, c in triples:
        assert a * a + b * b == c * c
        assert c <= 50


def test_version():
    """Test that version is available."""
    from constraint_theory import __version__
    
    assert __version__ is not None
    assert len(__version__.split(".")) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
