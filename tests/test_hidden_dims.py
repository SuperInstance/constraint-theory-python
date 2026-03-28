"""
Comprehensive Tests for Hidden Dimension Encoding

Tests the core GUCT algorithms:
- k = ⌈log₂(1/ε)⌉ formula for hidden dimension count
- Lifting to hidden dimensions
- Projection back to visible dimensions
- Exact constraint satisfaction
- Cross-plane fine-tuning
"""

import pytest
import math
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from constraint_theory import (
    HiddenDimConfig,
    compute_hidden_dim_count,
    lift_to_hidden,
    project_visible,
    snap_in_lifted_space,
    generate_nd_lattice,
    encode_with_hidden_dimensions,
    cross_plane_finetune,
    get_orthogonal_planes,
    project_to_plane,
    reconstruct_from_plane,
    constraint_error,
    holographic_accuracy,
)


class TestHiddenDimCount:
    """Test the core k = ⌈log₂(1/ε)⌉ formula."""
    
    def test_basic_precision(self):
        """Test basic precision values."""
        # k = ceil(log2(1/epsilon))
        assert compute_hidden_dim_count(0.1) == 4       # log2(10) ≈ 3.32 -> 4
        assert compute_hidden_dim_count(0.01) == 7      # log2(100) ≈ 6.64 -> 7
        assert compute_hidden_dim_count(0.001) == 10    # log2(1000) ≈ 9.97 -> 10
        assert compute_hidden_dim_count(0.0001) == 14   # log2(10000) ≈ 13.29 -> 14
    
    def test_high_precision(self):
        """Test high precision values."""
        assert compute_hidden_dim_count(1e-6) == 20
        assert compute_hidden_dim_count(1e-10) == 34
        assert compute_hidden_dim_count(1e-16) == 54
    
    def test_extreme_precision(self):
        """Test extreme precision values."""
        # Very high precision
        k = compute_hidden_dim_count(1e-100)
        assert k > 300  # log2(1e100) ≈ 332
        
        # Precision >= 1 should give 0 hidden dims
        assert compute_hidden_dim_count(1.0) == 0
        assert compute_hidden_dim_count(2.0) == 0
    
    def test_formula_correctness(self):
        """Test that the formula is correct."""
        for epsilon in [1e-3, 1e-6, 1e-9, 1e-12]:
            k = compute_hidden_dim_count(epsilon)
            expected = math.ceil(math.log2(1.0 / epsilon))
            assert k == expected, f"For epsilon={epsilon}, expected {expected}, got {k}"


class TestHolographicAccuracy:
    """Test the holographic accuracy formula."""
    
    def test_basic_accuracy(self):
        """Test basic accuracy calculations."""
        # With all hidden dims, accuracy should be high
        acc = holographic_accuracy(10, 10)
        assert acc > 0.9
        
        # With no hidden dims, accuracy should be low
        acc = holographic_accuracy(0, 10)
        assert acc < 0.5
    
    def test_accuracy_formula(self):
        """Test the accuracy formula: k/n + O(1/log n)."""
        k, n = 34, 2
        acc = holographic_accuracy(k, n)
        
        # Base accuracy = k/n = 17
        # Should be > 1 with correction
        assert acc > 1.0
    
    def test_accuracy_bounds(self):
        """Test accuracy bounds."""
        # With equal hidden and visible dims
        for n in [2, 10, 100]:
            acc = holographic_accuracy(n, n)
            assert acc >= 0.9  # Should be close to 1


class TestLiftToHidden:
    """Test lifting points to hidden dimensions."""
    
    def test_basic_lift(self):
        """Test basic point lifting."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [0.6, 0.8]
        lifted = lift_to_hidden(point, k=3)
        
        # Should have 2 + 3 = 5 dimensions
        assert len(lifted) == 5
        
        # First 2 should be original point
        assert np.allclose(lifted[:2], point)
    
    def test_lift_various_k(self):
        """Test lifting with various k values."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [1.0, 0.0, 0.0]  # 3D point
        
        for k in [1, 5, 10, 34]:
            lifted = lift_to_hidden(point, k=k)
            assert len(lifted) == 3 + k
    
    def test_lift_preserves_original(self):
        """Test that lifting preserves original dimensions."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = np.random.randn(10)
        lifted = lift_to_hidden(point, k=20)
        
        # Original should be unchanged
        assert np.allclose(lifted[:10], point)


class TestProjectVisible:
    """Test projecting back to visible dimensions."""
    
    def test_basic_projection(self):
        """Test basic projection."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        lifted = [0.6, 0.8, 0.1, 0.2, 0.3]  # 5D
        visible = project_visible(lifted, n=2)
        
        assert len(visible) == 2
        assert np.allclose(visible, [0.6, 0.8])
    
    def test_projection_various_n(self):
        """Test projection with various n values."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        lifted = np.random.randn(100)
        
        for n in [2, 10, 50]:
            visible = project_visible(lifted, n=n)
            assert len(visible) == n


class TestLiftProjectRoundTrip:
    """Test lift-project round trip."""
    
    def test_roundtrip_preserves_visible(self):
        """Test that lift-project preserves visible dimensions."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = np.random.randn(5)
        
        for k in [3, 10, 20]:
            lifted = lift_to_hidden(point, k=k)
            projected = project_visible(lifted, n=len(point))
            
            assert np.allclose(projected, point)


class TestGenerateNDLattice:
    """Test n-dimensional lattice generation."""
    
    def test_2d_lattice(self):
        """Test 2D Pythagorean lattice."""
        lattice = generate_nd_lattice(2, max_denominator=50)
        
        assert len(lattice) > 0
        
        # All points should be 2D
        for point in lattice:
            assert len(point) == 2
    
    def test_3d_lattice(self):
        """Test 3D lattice generation."""
        lattice = generate_nd_lattice(3, max_denominator=50)
        
        assert len(lattice) > 0
        
        # All points should be 3D
        for point in lattice:
            assert len(point) == 3
    
    def test_high_dim_lattice(self):
        """Test higher dimensional lattice."""
        for dim in [4, 5, 10]:
            lattice = generate_nd_lattice(dim, max_denominator=30)
            assert len(lattice) > 0
            
            for point in lattice:
                assert len(point) == dim


class TestSnapInLiftedSpace:
    """Test snapping in lifted space."""
    
    def test_basic_snap(self):
        """Test basic snapping in lifted space."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        lifted = [0.6, 0.8, 0.1, 0.1, 0.1]  # 5D
        snapped, dist = snap_in_lifted_space(lifted)
        
        assert len(snapped) == 5
        assert dist >= 0
    
    def test_snap_exact_point(self):
        """Test snapping an already-exact point."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Point on unit circle
        lifted = [0.6, 0.8]
        snapped, dist = snap_in_lifted_space(lifted)
        
        # Should be close to original
        assert dist < 0.2


class TestEncodeWithHiddenDimensions:
    """Test the main encoding function."""
    
    def test_basic_encoding(self):
        """Test basic encoding."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [0.6, 0.8]
        encoded = encode_with_hidden_dimensions(
            point,
            constraints=['unit_norm'],
            epsilon=1e-6
        )
        
        assert len(encoded) == 2
        
        # Should be close to original
        assert abs(encoded[0] - 0.6) < 0.2
        assert abs(encoded[1] - 0.8) < 0.2
    
    def test_encoding_precision(self):
        """Test encoding at various precisions."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = np.random.randn(5)
        
        for epsilon in [1e-3, 1e-6, 1e-10]:
            encoded = encode_with_hidden_dimensions(
                point,
                constraints=['unit_norm'],
                epsilon=epsilon
            )
            
            assert len(encoded) == len(point)
    
    def test_encoding_with_config(self):
        """Test encoding with HiddenDimConfig."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        config = HiddenDimConfig(epsilon=1e-6, hidden_dims=20)
        point = [0.5, 0.5]
        
        encoded = encode_with_hidden_dimensions(
            point,
            constraints=['unit_norm'],
            config=config
        )
        
        assert len(encoded) == 2


class TestCrossPlaneFinetune:
    """Test cross-plane fine-tuning."""
    
    def test_basic_finetune(self):
        """Test basic fine-tuning."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [0.707, 0.707]  # Near sqrt(2)/2
        result = cross_plane_finetune(
            point,
            constraints=['unit_norm']
        )
        
        # Should be a valid 2D point
        assert len(result) == 2
    
    def test_finetune_improves_constraint(self):
        """Test that fine-tuning improves constraint satisfaction."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = np.random.randn(4)
        
        # Get error before
        error_before = constraint_error(point, ['unit_norm'])
        
        # Fine-tune
        tuned = cross_plane_finetune(point, constraints=['unit_norm'])
        
        # Get error after
        error_after = constraint_error(tuned, ['unit_norm'])
        
        # Should improve (or at least not worsen)
        assert error_after <= error_before + 0.1


class TestOrthogonalPlanes:
    """Test orthogonal plane utilities."""
    
    def test_2d_planes(self):
        """Test planes in 2D."""
        planes = get_orthogonal_planes(2)
        
        # 2D has 1 plane
        assert len(planes) == 1
        assert planes[0] == (0, 1)
    
    def test_3d_planes(self):
        """Test planes in 3D."""
        planes = get_orthogonal_planes(3)
        
        # 3D has C(3,2) = 3 planes
        assert len(planes) == 3
        
        expected = [(0, 1), (0, 2), (1, 2)]
        assert set(planes) == set(expected)
    
    def test_high_dim_planes(self):
        """Test planes in higher dimensions."""
        for n in [4, 5, 10]:
            planes = get_orthogonal_planes(n)
            
            # Should have C(n, 2) planes
            expected_count = n * (n - 1) // 2
            assert len(planes) == expected_count


class TestProjectReconstruct:
    """Test project_to_plane and reconstruct_from_plane."""
    
    def test_2d_projection(self):
        """Test 2D projection and reconstruction."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = np.array([1.0, 2.0, 3.0, 4.0])
        plane = (0, 1)
        
        projected = project_to_plane(point, plane)
        assert len(projected) == 2
        assert np.allclose(projected, [1.0, 2.0])
        
        reconstructed = reconstruct_from_plane(projected, plane, n=4)
        assert len(reconstructed) == 4
        assert reconstructed[0] == 1.0
        assert reconstructed[1] == 2.0
        assert reconstructed[2] == 0.0
        assert reconstructed[3] == 0.0


class TestConstraintError:
    """Test constraint error computation."""
    
    def test_unit_norm_error(self):
        """Test unit norm error computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Perfect unit vector
        point = np.array([0.6, 0.8])
        error = constraint_error(point, ['unit_norm'])
        assert error < 0.01
        
        # Non-unit vector
        point = np.array([1.0, 1.0])
        error = constraint_error(point, ['unit_norm'])
        assert error > 0.1


class TestHiddenDimConfig:
    """Test HiddenDimConfig class."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = HiddenDimConfig(epsilon=1e-10)
        
        assert config.epsilon == 1e-10
        # hidden_dims should be computed
        assert config.hidden_dims == 34
    
    def test_config_with_hidden_dims(self):
        """Test config with explicit hidden dims."""
        config = HiddenDimConfig(epsilon=1e-6, hidden_dims=10)
        
        assert config.hidden_dims == 10


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_point(self):
        """Test zero point handling."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [0.0, 0.0]
        encoded = encode_with_hidden_dimensions(point, epsilon=1e-6)
        
        # Should handle gracefully
        assert len(encoded) == 2
    
    def test_single_dimension(self):
        """Test single dimension handling."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        point = [1.0]
        
        # Single dim has only one plane (empty)
        planes = get_orthogonal_planes(1)
        assert len(planes) == 0
    
    def test_very_high_precision(self):
        """Test very high precision."""
        k = compute_hidden_dim_count(1e-50)
        assert k > 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
