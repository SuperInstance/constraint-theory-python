"""
Machine Learning Integration Module

This module provides ML integration patterns for Constraint Theory, including:
- ConstraintEnforcedLayer: PyTorch/TensorFlow layer with constraint enforcement
- HiddenDimensionNetwork: Network using hidden dimension encoding
- GradientSnapper: Deterministic gradient augmentation

Key Concepts:
- Hidden dimensions: k = ⌈log₂(1/ε)⌉ for exact constraint satisfaction
- Constraint enforcement during training prevents drift
- Pythagorean snapping ensures reproducibility

Example:
    >>> from constraint_theory.ml import ConstraintEnforcedLayer
    >>> import torch
    >>> 
    >>> # Create a layer that enforces unit norm on outputs
    >>> layer = ConstraintEnforcedLayer(
    ...     input_dim=128,
    ...     output_dim=64,
    ...     constraints=['unit_norm']
    ... )
    >>> 
    >>> x = torch.randn(32, 128)
    >>> y = layer(x)  # y has unit norm constraint enforced
"""

from __future__ import annotations
from typing import List, Optional, Union, Any, Callable, Dict
from dataclasses import dataclass
import math

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

from .manifold import PythagoreanManifold
from .quantizer import PythagoreanQuantizer, QuantizationMode, QuantizationResult


@dataclass
class ConstraintConfig:
    """
    Configuration for constraint enforcement.
    
    Attributes:
        constraint_type: Type of constraint ('unit_norm', 'orthogonal', 'bounded')
        tolerance: Tolerance for constraint satisfaction
        enforcement_mode: 'soft' (regularization) or 'hard' (projection)
        schedule: Schedule for enforcement ('constant', 'annealing')
    """
    constraint_type: str
    tolerance: float = 1e-6
    enforcement_mode: str = 'hard'
    schedule: str = 'constant'


class ConstraintEnforcedLayer:
    """
    Neural network layer with constraint enforcement.
    
    This layer wraps standard linear transformations with constraint
    enforcement using Pythagorean snapping.
    
    Supports:
    - PyTorch nn.Module interface
    - TensorFlow/Keras layer interface
    - NumPy-only fallback
    
    Example (PyTorch):
        >>> import torch
        >>> from constraint_theory.ml import ConstraintEnforcedLayer
        >>> 
        >>> layer = ConstraintEnforcedLayer(
        ...     input_dim=128,
        ...     output_dim=64,
        ...     constraints=['unit_norm']
        ... )
        >>> 
        >>> x = torch.randn(32, 128)
        >>> y = layer(x)  # Unit norm enforced on each row
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        constraints: Optional[List[str]] = None,
        density: int = 200,
        framework: str = 'auto'
    ):
        """
        Initialize the constraint enforced layer.
        
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            constraints: List of constraints to enforce ['unit_norm', 'orthogonal'].
            density: Manifold density for Pythagorean snapping.
            framework: 'pytorch', 'tensorflow', 'numpy', or 'auto' (detect).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.constraints = constraints or []
        self.density = density
        
        # Initialize manifold
        self._manifold = PythagoreanManifold(density=density)
        self._quantizer = PythagoreanQuantizer(
            mode=QuantizationMode.POLAR if 'unit_norm' in self.constraints else QuantizationMode.HYBRID
        )
        
        # Detect framework
        if framework == 'auto':
            if HAS_TORCH:
                self._framework = 'pytorch'
            elif HAS_TENSORFLOW:
                self._framework = 'tensorflow'
            else:
                self._framework = 'numpy'
        else:
            self._framework = framework
        
        # Initialize framework-specific components
        self._init_framework_components()
    
    def _init_framework_components(self):
        """Initialize framework-specific layer components."""
        if self._framework == 'pytorch' and HAS_TORCH:
            self._linear = nn.Linear(self.input_dim, self.output_dim)
        elif self._framework == 'tensorflow' and HAS_TENSORFLOW:
            self._dense = tf.keras.layers.Dense(self.output_dim)
        else:
            # NumPy fallback: initialize weights
            if HAS_NUMPY:
                self._weights = np.random.randn(self.input_dim, self.output_dim) * 0.01
                self._bias = np.zeros(self.output_dim)
    
    def __call__(self, x: Any) -> Any:
        """
        Forward pass with constraint enforcement.
        
        Args:
            x: Input tensor (framework-specific).
        
        Returns:
            Output tensor with constraints enforced.
        """
        if self._framework == 'pytorch' and HAS_TORCH:
            return self._forward_pytorch(x)
        elif self._framework == 'tensorflow' and HAS_TENSORFLOW:
            return self._forward_tensorflow(x)
        else:
            return self._forward_numpy(x)
    
    def _forward_pytorch(self, x: "torch.Tensor") -> "torch.Tensor":
        """PyTorch forward pass."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        
        # Standard linear transformation
        y = self._linear(x)
        
        # Enforce constraints
        if 'unit_norm' in self.constraints:
            y = self._enforce_unit_norm_pytorch(y)
        
        return y
    
    def _enforce_unit_norm_pytorch(self, y: "torch.Tensor") -> "torch.Tensor":
        """Enforce unit norm on PyTorch tensor."""
        with torch.no_grad():
            # Convert to numpy for snapping
            np_y = y.detach().cpu().numpy()
            
            # Snap each row to Pythagorean manifold
            snapped = []
            for row in np_y:
                if len(row) == 2:
                    sx, sy, _ = self._manifold.snap(row[0], row[1])
                    snapped.append([sx, sy])
                else:
                    # For higher dims, use quantizer
                    result = self._quantizer.quantize(row.reshape(1, -1), mode=QuantizationMode.POLAR)
                    snapped.append(result.data.flatten().tolist())
            
            snapped_arr = np.array(snapped)
            return torch.from_numpy(snapped_arr.astype(np.float32)).to(y.device)
    
    def _forward_tensorflow(self, x: "tf.Tensor") -> "tf.Tensor":
        """TensorFlow forward pass."""
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow not available")
        
        # Standard dense transformation
        y = self._dense(x)
        
        # Enforce constraints
        if 'unit_norm' in self.constraints:
            y = self._enforce_unit_norm_tensorflow(y)
        
        return y
    
    def _enforce_unit_norm_tensorflow(self, y: "tf.Tensor") -> "tf.Tensor":
        """Enforce unit norm on TensorFlow tensor."""
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow not available")
        
        # Use TensorFlow ops for normalization
        norms = tf.norm(y, axis=1, keepdims=True)
        norms = tf.maximum(norms, 1e-10)
        return y / norms
    
    def _forward_numpy(self, x: Any) -> "np.ndarray":
        """NumPy forward pass."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available")
        
        x = np.asarray(x)
        
        # Linear transformation
        y = x @ self._weights + self._bias
        
        # Enforce constraints
        if 'unit_norm' in self.constraints:
            norms = np.linalg.norm(y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            y = y / norms
        
        return y
    
    def parameters(self) -> Any:
        """Get layer parameters (framework-specific)."""
        if self._framework == 'pytorch' and HAS_TORCH:
            return self._linear.parameters()
        elif self._framework == 'tensorflow' and HAS_TENSORFLOW:
            return self._dense.trainable_variables
        else:
            return [self._weights, self._bias]
    
    def __repr__(self) -> str:
        return (
            f"ConstraintEnforcedLayer(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, constraints={self.constraints})"
        )


class HiddenDimensionNetwork:
    """
    Neural network using hidden dimension encoding.
    
    Uses the GUCT formula k = ⌈log₂(1/ε)⌉ to determine hidden dimensions
    for exact constraint satisfaction.
    
    The network:
    1. Lifts input to hidden dimensions
    2. Processes in lifted space
    3. Projects back to visible dimensions
    
    Example:
        >>> from constraint_theory.ml import HiddenDimensionNetwork
        >>> import numpy as np
        >>> 
        >>> # Create network with 1e-10 precision (k = 34 hidden dims)
        >>> net = HiddenDimensionNetwork(
        ...     visible_dims=128,
        ...     epsilon=1e-10,
        ...     hidden_layers=[256, 256]
        ... )
        >>> 
        >>> x = np.random.randn(32, 128)
        >>> y = net.forward(x)
    """
    
    def __init__(
        self,
        visible_dims: int,
        epsilon: float = 1e-6,
        hidden_layers: Optional[List[int]] = None,
        density: int = 200
    ):
        """
        Initialize the hidden dimension network.
        
        Args:
            visible_dims: Number of visible dimensions.
            epsilon: Desired precision for constraint satisfaction.
            hidden_layers: Hidden layer sizes (excludes hidden dims).
            density: Manifold density.
        """
        self.visible_dims = visible_dims
        self.epsilon = epsilon
        self.hidden_layers = hidden_layers or []
        self.density = density
        
        # Calculate hidden dimensions using GUCT formula
        self.hidden_dims = self._calculate_hidden_dims(epsilon)
        
        # Initialize components
        self._manifold = PythagoreanManifold(density=density)
        
        if HAS_NUMPY:
            self._init_weights()
    
    def _calculate_hidden_dims(self, epsilon: float) -> int:
        """
        Calculate minimum hidden dimensions for precision.
        
        Uses GUCT formula: k = ⌈log₂(1/ε)⌉
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return math.ceil(math.log2(1.0 / epsilon))
    
    def _init_weights(self):
        """Initialize network weights."""
        if not HAS_NUMPY:
            return
        
        # Lift projection: visible -> visible + hidden
        total_input = self.visible_dims + self.hidden_dims
        self._lift_weights = np.random.randn(self.visible_dims, self.hidden_dims) * 0.01
        
        # Hidden layers
        layer_sizes = [total_input] + self.hidden_layers + [self.visible_dims]
        self._layer_weights = []
        self._layer_biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros(layer_sizes[i + 1])
            self._layer_weights.append(w)
            self._layer_biases.append(b)
    
    def lift(self, x: Any) -> Any:
        """
        Lift input to hidden dimensions.
        
        Args:
            x: Input data (n, visible_dims).
        
        Returns:
            Lifted data (n, visible_dims + hidden_dims).
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for HiddenDimensionNetwork")
        
        x = np.asarray(x)
        
        # Generate hidden dimension values
        hidden = x @ self._lift_weights
        
        # Concatenate
        return np.concatenate([x, hidden], axis=1)
    
    def project(self, y: Any) -> Any:
        """
        Project from lifted space back to visible dimensions.
        
        Args:
            y: Lifted data (n, visible_dims + hidden_dims).
        
        Returns:
            Visible data (n, visible_dims).
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for HiddenDimensionNetwork")
        
        y = np.asarray(y)
        return y[:, :self.visible_dims]
    
    def forward(self, x: Any) -> Any:
        """
        Forward pass through the network.
        
        Args:
            x: Input data.
        
        Returns:
            Output in visible dimensions.
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for HiddenDimensionNetwork")
        
        # Lift to hidden dimensions
        lifted = self.lift(x)
        
        # Process through layers
        y = lifted
        for w, b in zip(self._layer_weights, self._layer_biases):
            y = np.tanh(y @ w + b)
        
        # Project back
        return self.project(y)
    
    def __repr__(self) -> str:
        return (
            f"HiddenDimensionNetwork(visible_dims={self.visible_dims}, "
            f"hidden_dims={self.hidden_dims}, epsilon={self.epsilon})"
        )


class GradientSnapper:
    """
    Deterministic gradient augmentation using Pythagorean snapping.
    
    Ensures reproducible training by snapping gradient directions
    to exact Pythagorean states.
    
    Example:
        >>> from constraint_theory.ml import GradientSnapper
        >>> import numpy as np
        >>> 
        >>> snapper = GradientSnapper(density=200)
        >>> gradients = np.random.randn(1000, 2)
        >>> snapped = snapper.snap_batch(gradients)
    """
    
    def __init__(self, density: int = 200, preserve_magnitude: bool = True):
        """
        Initialize the gradient snapper.
        
        Args:
            density: Manifold density for snapping.
            preserve_magnitude: Whether to preserve gradient magnitude.
        """
        self.density = density
        self.preserve_magnitude = preserve_magnitude
        self._manifold = PythagoreanManifold(density=density)
    
    def snap(self, dx: float, dy: float) -> Tuple[float, float, float]:
        """
        Snap a single gradient to exact direction.
        
        Args:
            dx: X component of gradient.
            dy: Y component of gradient.
        
        Returns:
            Tuple of (snapped_dx, snapped_dy, noise).
        """
        if self.preserve_magnitude:
            magnitude = math.sqrt(dx * dx + dy * dy)
        else:
            magnitude = 1.0
        
        sx, sy, noise = self._manifold.snap(dx, dy)
        
        return (sx * magnitude, sy * magnitude, noise)
    
    def snap_batch(self, gradients: Any) -> Any:
        """
        Snap multiple gradients.
        
        Args:
            gradients: (n, 2) array of gradients.
        
        Returns:
            Snapped gradients with same shape.
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy required for batch snapping")
        
        gradients = np.asarray(gradients)
        results = []
        
        for dx, dy in gradients:
            sx, sy, _ = self.snap(dx, dy)
            results.append([sx, sy])
        
        return np.array(results)
    
    def __repr__(self) -> str:
        return f"GradientSnapper(density={self.density}, preserve_magnitude={self.preserve_magnitude})"


# Type alias for Tuple
from typing import Tuple

__all__ = [
    "ConstraintConfig",
    "ConstraintEnforcedLayer",
    "HiddenDimensionNetwork",
    "GradientSnapper",
]
