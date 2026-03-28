"""
Comprehensive Tests for PythagoreanQuantizer

Tests all quantization modes and constraint preservation:
- TERNARY: BitNet-style {-1, 0, 1} quantization
- POLAR: Exact unit norm preservation
- TURBO: Near-optimal distortion
- HYBRID: Auto-mode selection
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
    PythagoreanQuantizer,
    QuantizationMode,
    QuantizationResult,
    auto_select_mode,
    snap_to_pythagorean,
    quantize,
)


class TestQuantizationMode:
    """Test QuantizationMode enum."""
    
    def test_mode_values(self):
        """Test that all modes are defined."""
        assert QuantizationMode.TERNARY is not None
        assert QuantizationMode.POLAR is not None
        assert QuantizationMode.TURBO is not None
        assert QuantizationMode.HYBRID is not None
    
    def test_mode_comparison(self):
        """Test mode equality."""
        assert QuantizationMode.POLAR == QuantizationMode.POLAR
        assert QuantizationMode.TERNARY != QuantizationMode.POLAR


class TestPythagoreanQuantizer:
    """Test PythagoreanQuantizer class."""
    
    def test_default_initialization(self):
        """Test default quantizer creation."""
        q = PythagoreanQuantizer()
        assert q is not None
        assert q.mode == QuantizationMode.HYBRID
        assert q.bits == 4
    
    def test_custom_initialization(self):
        """Test custom quantizer parameters."""
        q = PythagoreanQuantizer(mode=QuantizationMode.POLAR, bits=8)
        assert q.mode == QuantizationMode.POLAR
        assert q.bits == 8
    
    def test_factory_methods(self):
        """Test factory method creation."""
        q_llm = PythagoreanQuantizer.for_llm()
        assert q_llm.mode == QuantizationMode.TERNARY
        
        q_emb = PythagoreanQuantizer.for_embeddings()
        assert q_emb.mode == QuantizationMode.POLAR
        
        q_vdb = PythagoreanQuantizer.for_vector_db()
        assert q_vdb.mode == QuantizationMode.TURBO
        
        q_hyb = PythagoreanQuantizer.hybrid()
        assert q_hyb.mode == QuantizationMode.HYBRID


class TestTernaryQuantization:
    """Test TERNARY mode quantization."""
    
    def test_ternary_basic(self):
        """Test basic ternary quantization."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_llm()
        data = [-0.8, -0.1, 0.1, 0.9]
        result = q.quantize(data)
        
        # All values should be -1, 0, or 1
        for val in result.data:
            assert val in [-1.0, 0.0, 1.0], f"Expected ternary value, got {val}"
    
    def test_ternary_sparsity(self):
        """Test that ternary mode creates sparse output."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_llm()
        
        # Data with many small values
        data = [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9]
        result = q.quantize(data)
        
        # Count zeros
        zeros = sum(1 for v in result.data if v == 0.0)
        assert zeros > 0, "Ternary mode should produce zeros for small values"
    
    def test_ternary_compression_ratio(self):
        """Test compression ratio calculation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_llm()
        data = np.random.randn(100)
        result = q.quantize(data)
        
        # Ternary should have high compression ratio
        assert result.compression_ratio > 10, "Expected high compression for ternary"


class TestPolarQuantization:
    """Test POLAR mode quantization."""
    
    def test_polar_unit_norm_2d(self):
        """Test unit norm preservation in 2D."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_embeddings()
        
        # 2D unit vector
        data = np.array([[0.6, 0.8]])
        result = q.quantize(data)
        
        # Check unit norm
        norm = np.linalg.norm(result.data)
        assert abs(norm - 1.0) < 0.1, f"Expected unit norm, got {norm}"
    
    def test_polar_unit_norm_high_dim(self):
        """Test unit norm preservation in higher dimensions."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_embeddings()
        
        # Random unit vector in 128D
        vec = np.random.randn(128)
        vec = vec / np.linalg.norm(vec)
        
        result = q.quantize(vec)
        
        # Check unit norm
        norm = np.linalg.norm(result.data)
        assert abs(norm - 1.0) < 0.1, f"Expected unit norm, got {norm}"
    
    def test_polar_batch_unit_norm(self):
        """Test unit norm preservation for batch."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_embeddings()
        
        # Batch of unit vectors
        vectors = np.random.randn(10, 64)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        result = q.quantize(vectors)
        
        # Check all unit norms
        norms = np.linalg.norm(result.data, axis=1)
        for i, norm in enumerate(norms):
            assert abs(norm - 1.0) < 0.1, f"Row {i}: expected unit norm, got {norm}"
    
    def test_polar_exact_pythagorean_angles(self):
        """Test that polar mode snaps to Pythagorean angles."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_embeddings()
        
        # 3-4-5 triangle
        data = [0.6, 0.8]
        result = q.quantize(data)
        
        # Should be close to 3-4-5 ratio
        assert abs(result.data[0] - 0.6) < 0.15 or abs(result.data[1] - 0.8) < 0.15


class TestTurboQuantization:
    """Test TURBO mode quantization."""
    
    def test_turbo_basic(self):
        """Test basic turbo quantization."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_vector_db()
        data = np.random.randn(100)
        
        result = q.quantize(data)
        
        assert len(result.data) == len(data)
        assert result.mse >= 0
    
    def test_turbo_distortion_bounds(self):
        """Test that distortion is bounded."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_vector_db()
        
        # Uniform data
        data = np.random.uniform(-1, 1, 1000)
        result = q.quantize(data)
        
        # MSE should be reasonable for 4-bit quantization
        # Theoretical: ~0.02 for 4-bit uniform
        assert result.mse < 0.2, f"MSE too high: {result.mse}"
    
    def test_turbo_deterministic(self):
        """Test that turbo quantization is deterministic."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.for_vector_db()
        data = np.random.randn(50)
        
        result1 = q.quantize(data)
        result2 = q.quantize(data)
        
        assert np.allclose(result1.data, result2.data)


class TestHybridMode:
    """Test HYBRID mode auto-selection."""
    
    def test_hybrid_selects_polar_for_unit_norm(self):
        """Test that hybrid selects POLAR for unit norm data."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.hybrid()
        
        # Unit norm vector
        vec = np.random.randn(64)
        vec = vec / np.linalg.norm(vec)
        
        result = q.quantize(vec)
        
        # Should preserve unit norm (indicates POLAR was selected)
        norm = np.linalg.norm(result.data)
        assert abs(norm - 1.0) < 0.2, f"Expected unit norm (POLAR mode), got {norm}"
    
    def test_hybrid_selects_ternary_for_sparse(self):
        """Test that hybrid selects TERNARY for sparse data."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer.hybrid()
        
        # Very sparse data
        data = np.zeros(100)
        data[::10] = np.random.randn(10)  # 10% non-zero
        
        result = q.quantize(data)
        
        # Should have many zeros (indicates TERNARY was selected)
        zeros = np.sum(result.data == 0)
        assert zeros > 50, f"Expected sparse output (TERNARY mode), got {zeros} zeros"


class TestAutoSelectMode:
    """Test auto_select_mode function."""
    
    def test_auto_select_unit_norm(self):
        """Test mode selection for unit norm data."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Unit norm vectors
        vectors = np.random.randn(10, 64)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        mode = auto_select_mode(vectors)
        assert mode == QuantizationMode.POLAR
    
    def test_auto_select_sparse(self):
        """Test mode selection for sparse data."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Sparse data
        data = np.random.randn(10, 64) * 0.01
        
        mode = auto_select_mode(data)
        # May vary based on sparsity check
        assert mode in [QuantizationMode.TERNARY, QuantizationMode.TURBO]
    
    def test_auto_select_embeddings(self):
        """Test mode selection for embedding-like data."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Embedding-like data (unit norm, moderate dimensions)
        embeddings = np.random.randn(100, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        mode = auto_select_mode(embeddings)
        assert mode == QuantizationMode.POLAR


class TestSnapToPythagorean:
    """Test snap_to_pythagorean function."""
    
    def test_snap_3_5(self):
        """Test snapping to 3/5 ratio."""
        result = snap_to_pythagorean(0.6)
        assert abs(result - 0.6) < 0.1, f"Expected ~0.6, got {result}"
    
    def test_snap_4_5(self):
        """Test snapping to 4/5 ratio."""
        result = snap_to_pythagorean(0.8)
        assert abs(result - 0.8) < 0.1, f"Expected ~0.8, got {result}"
    
    def test_snap_near_value(self):
        """Test snapping values near Pythagorean ratios."""
        result = snap_to_pythagorean(0.61)  # Close to 3/5
        assert abs(result - 0.6) < 0.15, f"Expected ~0.6, got {result}"


class TestConvenienceFunction:
    """Test convenience quantize function."""
    
    def test_convenience_basic(self):
        """Test basic convenience quantization."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        data = np.random.randn(100)
        result = quantize(data)
        
        assert result is not None
        assert len(result.data) == 100
    
    def test_convenience_with_mode(self):
        """Test convenience quantization with explicit mode."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        data = np.random.randn(100)
        result = quantize(data, mode=QuantizationMode.TURBO)
        
        assert result.mode == QuantizationMode.TURBO


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test empty input handling."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer()
        result = q.quantize([])
        
        assert len(result.data) == 0
    
    def test_single_element(self):
        """Test single element input."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer()
        result = q.quantize([1.0])
        
        assert len(result.data) == 1
    
    def test_all_zeros(self):
        """Test all zeros input."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer()
        result = q.quantize(np.zeros(100))
        
        assert len(result.data) == 100
    
    def test_extreme_values(self):
        """Test extreme value handling."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        q = PythagoreanQuantizer()
        data = np.array([1e10, -1e10, 1e-10, -1e-10])
        
        result = q.quantize(data)
        
        assert len(result.data) == 4
        assert np.isfinite(result.data).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
