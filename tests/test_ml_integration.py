"""
Comprehensive Tests for ML Integration

Tests the ML integration components:
- ConstraintEnforcedLayer: PyTorch/TensorFlow/NumPy layers with constraints
- HiddenDimensionNetwork: Networks using hidden dimension encoding
- GradientSnapper: Deterministic gradient augmentation
"""

import pytest
import math
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from constraint_theory import (
    ConstraintConfig,
    ConstraintEnforcedLayer,
    HiddenDimensionNetwork,
    GradientSnapper,
    QuantizationMode,
)


class TestConstraintConfig:
    """Test ConstraintConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConstraintConfig(constraint_type='unit_norm')
        
        assert config.constraint_type == 'unit_norm'
        assert config.tolerance == 1e-6
        assert config.enforcement_mode == 'hard'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ConstraintConfig(
            constraint_type='orthogonal',
            tolerance=1e-8,
            enforcement_mode='soft',
            schedule='annealing'
        )
        
        assert config.constraint_type == 'orthogonal'
        assert config.tolerance == 1e-8
        assert config.enforcement_mode == 'soft'
        assert config.schedule == 'annealing'


class TestConstraintEnforcedLayer:
    """Test ConstraintEnforcedLayer class."""
    
    def test_numpy_initialization(self):
        """Test NumPy initialization."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=128,
            output_dim=64,
            constraints=['unit_norm'],
            framework='numpy'
        )
        
        assert layer.input_dim == 128
        assert layer.output_dim == 64
        assert 'unit_norm' in layer.constraints
    
    def test_numpy_forward(self):
        """Test NumPy forward pass."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=['unit_norm'],
            framework='numpy'
        )
        
        x = np.random.randn(8, 10)
        y = layer(x)
        
        assert y.shape == (8, 5)
    
    def test_numpy_unit_norm_enforcement(self):
        """Test unit norm enforcement in NumPy."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=['unit_norm'],
            framework='numpy'
        )
        
        x = np.random.randn(8, 10)
        y = layer(x)
        
        # Check that output rows are approximately unit norm
        norms = np.linalg.norm(y, axis=1)
        for i, norm in enumerate(norms):
            assert abs(norm - 1.0) < 0.1, f"Row {i} has norm {norm}"
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_pytorch_initialization(self):
        """Test PyTorch initialization."""
        layer = ConstraintEnforcedLayer(
            input_dim=128,
            output_dim=64,
            constraints=['unit_norm'],
            framework='pytorch'
        )
        
        assert layer._framework == 'pytorch'
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_pytorch_forward(self):
        """Test PyTorch forward pass."""
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=['unit_norm'],
            framework='pytorch'
        )
        
        x = torch.randn(8, 10)
        y = layer(x)
        
        assert y.shape == (8, 5)
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_pytorch_unit_norm_enforcement(self):
        """Test unit norm enforcement in PyTorch."""
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=2,  # 2D for Pythagorean snapping
            constraints=['unit_norm'],
            framework='pytorch'
        )
        
        x = torch.randn(8, 10)
        y = layer(x)
        
        # Check output norms
        norms = torch.norm(y, dim=1)
        for i, norm in enumerate(norms):
            # Allow some tolerance for Pythagorean snapping
            assert abs(norm.item() - 1.0) < 0.2, f"Row {i} has norm {norm}"
    
    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_tensorflow_initialization(self):
        """Test TensorFlow initialization."""
        layer = ConstraintEnforcedLayer(
            input_dim=128,
            output_dim=64,
            constraints=['unit_norm'],
            framework='tensorflow'
        )
        
        assert layer._framework == 'tensorflow'
    
    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_tensorflow_forward(self):
        """Test TensorFlow forward pass."""
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=['unit_norm'],
            framework='tensorflow'
        )
        
        x = tf.random.normal((8, 10))
        y = layer(x)
        
        assert y.shape == (8, 5)
    
    def test_auto_framework_detection(self):
        """Test automatic framework detection."""
        if HAS_TORCH:
            layer = ConstraintEnforcedLayer(
                input_dim=10,
                output_dim=5,
                framework='auto'
            )
            assert layer._framework == 'pytorch'
        elif HAS_TENSORFLOW:
            layer = ConstraintEnforcedLayer(
                input_dim=10,
                output_dim=5,
                framework='auto'
            )
            assert layer._framework == 'tensorflow'
        elif HAS_NUMPY:
            layer = ConstraintEnforcedLayer(
                input_dim=10,
                output_dim=5,
                framework='auto'
            )
            assert layer._framework == 'numpy'
    
    def test_parameters(self):
        """Test parameter access."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            framework='numpy'
        )
        
        params = layer.parameters()
        assert params is not None
    
    def test_repr(self):
        """Test string representation."""
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=['unit_norm']
        )
        
        repr_str = repr(layer)
        assert 'ConstraintEnforcedLayer' in repr_str
        assert 'input_dim=10' in repr_str
        assert 'output_dim=5' in repr_str


class TestHiddenDimensionNetwork:
    """Test HiddenDimensionNetwork class."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=128,
            epsilon=1e-6,
            hidden_layers=[256, 256]
        )
        
        assert net.visible_dims == 128
        assert net.epsilon == 1e-6
        assert net.hidden_dims == 20  # log2(1e6) ≈ 20
    
    def test_hidden_dim_calculation(self):
        """Test hidden dimension calculation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        for epsilon, expected_k in [(1e-6, 20), (1e-10, 34), (1e-16, 54)]:
            net = HiddenDimensionNetwork(visible_dims=10, epsilon=epsilon)
            assert net.hidden_dims == expected_k
    
    def test_lift(self):
        """Test lifting operation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=10,
            epsilon=1e-6  # k = 20
        )
        
        x = np.random.randn(5, 10)
        lifted = net.lift(x)
        
        # Should have visible + hidden dimensions
        assert lifted.shape == (5, 10 + net.hidden_dims)
    
    def test_project(self):
        """Test projection operation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(visible_dims=10, epsilon=1e-6)
        
        lifted = np.random.randn(5, 10 + net.hidden_dims)
        projected = net.project(lifted)
        
        assert projected.shape == (5, 10)
    
    def test_forward(self):
        """Test forward pass."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=10,
            epsilon=1e-6,
            hidden_layers=[20, 20]
        )
        
        x = np.random.randn(5, 10)
        y = net.forward(x)
        
        assert y.shape == (5, 10)
    
    def test_roundtrip(self):
        """Test lift-process-project roundtrip."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=10,
            epsilon=1e-6,
            hidden_layers=[]
        )
        
        x = np.random.randn(5, 10)
        lifted = net.lift(x)
        projected = net.project(lifted)
        
        # First 10 dimensions should be preserved
        assert np.allclose(projected, x)
    
    def test_repr(self):
        """Test string representation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=128,
            epsilon=1e-10
        )
        
        repr_str = repr(net)
        assert 'HiddenDimensionNetwork' in repr_str
        assert 'visible_dims=128' in repr_str


class TestGradientSnapper:
    """Test GradientSnapper class."""
    
    def test_snapper_initialization(self):
        """Test snapper initialization."""
        snapper = GradientSnapper(density=200, preserve_magnitude=True)
        
        assert snapper.density == 200
        assert snapper.preserve_magnitude
    
    def test_single_snap(self):
        """Test single gradient snap."""
        snapper = GradientSnapper(density=200)
        
        dx, dy = 0.577, 0.816  # Near 3-4-5 ratio
        sx, sy, noise = snapper.snap(dx, dy)
        
        # Should snap to Pythagorean ratio
        assert abs(sx) <= 1.0
        assert abs(sy) <= 1.0
        assert noise >= 0
    
    def test_batch_snap(self):
        """Test batch gradient snap."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        snapper = GradientSnapper(density=200)
        
        gradients = np.random.randn(100, 2)
        snapped = snapper.snap_batch(gradients)
        
        assert snapped.shape == gradients.shape
    
    def test_preserve_magnitude(self):
        """Test magnitude preservation."""
        snapper = GradientSnapper(density=200, preserve_magnitude=True)
        
        dx, dy = 0.577, 0.816
        original_mag = math.sqrt(dx * dx + dy * dy)
        
        sx, sy, _ = snapper.snap(dx, dy)
        snapped_mag = math.sqrt(sx * sx + sy * sy)
        
        # Magnitudes should be similar
        assert abs(snapped_mag - original_mag) < 0.5
    
    def test_no_preserve_magnitude(self):
        """Test without magnitude preservation."""
        snapper = GradientSnapper(density=200, preserve_magnitude=False)
        
        dx, dy = 2.0, 3.0  # Large magnitude
        sx, sy, _ = snapper.snap(dx, dy)
        
        # Snapped should be unit or near-unit
        snapped_mag = math.sqrt(sx * sx + sy * sy)
        assert snapped_mag < 2.0
    
    def test_repr(self):
        """Test string representation."""
        snapper = GradientSnapper(density=200, preserve_magnitude=True)
        
        repr_str = repr(snapper)
        assert 'GradientSnapper' in repr_str
        assert 'density=200' in repr_str


class TestIntegration:
    """Integration tests for ML components."""
    
    def test_layer_with_network(self):
        """Test layer integrated with network."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Create network
        net = HiddenDimensionNetwork(
            visible_dims=10,
            epsilon=1e-6,
            hidden_layers=[20]
        )
        
        # Create layer
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=10,
            constraints=['unit_norm'],
            framework='numpy'
        )
        
        # Forward pass through both
        x = np.random.randn(5, 10)
        y = net.forward(x)
        z = layer(y)
        
        assert z.shape == (5, 10)
    
    def test_deterministic_training_step(self):
        """Test deterministic training step."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        # Setup
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=2,
            constraints=['unit_norm'],
            framework='numpy'
        )
        snapper = GradientSnapper(density=200)
        
        # Simulate training step
        x = np.random.randn(8, 10)
        y = layer(x)
        
        # Compute gradients (simplified)
        grad = np.random.randn(2)
        snapped_grad = snapper.snap_batch(grad.reshape(1, -1))
        
        assert snapped_grad.shape == (1, 2)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_constraints(self):
        """Test layer with no constraints."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            constraints=[],
            framework='numpy'
        )
        
        x = np.random.randn(8, 10)
        y = layer(x)
        
        assert y.shape == (8, 5)
    
    def test_zero_input(self):
        """Test zero input handling."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            framework='numpy'
        )
        
        x = np.zeros((8, 10))
        y = layer(x)
        
        assert y.shape == (8, 5)
    
    def test_single_sample(self):
        """Test single sample input."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        layer = ConstraintEnforcedLayer(
            input_dim=10,
            output_dim=5,
            framework='numpy'
        )
        
        x = np.random.randn(1, 10)
        y = layer(x)
        
        assert y.shape == (1, 5)
    
    def test_very_small_network(self):
        """Test very small network."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required")
        
        net = HiddenDimensionNetwork(
            visible_dims=2,
            epsilon=1e-3,
            hidden_layers=[]
        )
        
        x = np.random.randn(5, 2)
        y = net.forward(x)
        
        assert y.shape == (5, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
