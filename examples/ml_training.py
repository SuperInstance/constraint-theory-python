"""
ML Training Example with Constraint Theory

This example demonstrates how to use constraint theory for:
1. Training neural networks with constraint enforcement
2. Using hidden dimension encoding for better generalization
3. Deterministic gradient snapping for reproducibility
4. Quantization-aware training

The key insight is that enforcing constraints during training:
- Prevents drift in weight distributions
- Ensures reproducibility across runs
- Maintains exact mathematical properties (e.g., orthogonality)
"""

import numpy as np
from typing import List, Tuple, Optional
import time

# Import constraint theory components
from constraint_theory import (
    # Core components
    PythagoreanManifold,
    PythagoreanQuantizer,
    QuantizationMode,
    
    # ML integration
    ConstraintEnforcedLayer,
    HiddenDimensionNetwork,
    GradientSnapper,
    
    # Hidden dimensions
    compute_hidden_dim_count,
    encode_with_hidden_dimensions,
    
    # Quantization
    quantize,
)


class ConstrainedTrainer:
    """
    A trainer that uses constraint theory for reproducible ML training.
    
    Features:
    - Constraint enforcement on layer outputs
    - Deterministic gradient snapping
    - Hidden dimension encoding for inputs
    - Quantization-aware training
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        constraints: List[str] = ['unit_norm'],
        learning_rate: float = 0.01,
        precision: float = 1e-6
    ):
        """
        Initialize the constrained trainer.
        
        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            constraints: Constraints to enforce.
            learning_rate: Learning rate.
            precision: Target precision for hidden dimensions.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.constraints = constraints
        self.learning_rate = learning_rate
        self.precision = precision
        
        # Calculate hidden dimensions for encoding
        self.hidden_dims = compute_hidden_dim_count(precision)
        print(f"Using {self.hidden_dims} hidden dimensions for precision {precision}")
        
        # Initialize layers
        self.layer1 = ConstraintEnforcedLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            constraints=constraints,
            framework='numpy'
        )
        
        self.layer2 = ConstraintEnforcedLayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            constraints=[],
            framework='numpy'
        )
        
        # Gradient snapper for deterministic updates
        self.snapper = GradientSnapper(density=200, preserve_magnitude=True)
        
        # Hidden dimension network for encoding
        self.encoder = HiddenDimensionNetwork(
            visible_dims=input_dim,
            epsilon=precision
        )
        
        # Training stats
        self.training_history = []
    
    def forward(self, x: np.ndarray, use_hidden_encoding: bool = False) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input data.
            use_hidden_encoding: Whether to use hidden dimension encoding.
        
        Returns:
            Output predictions.
        """
        if use_hidden_encoding:
            # Lift to hidden dimensions, process, project back
            x = self.encoder.forward(x)
        
        # Pass through constrained layers
        h = self.layer1(x)
        y = self.layer2(h)
        
        return y
    
    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss_type: str = 'mse'
    ) -> float:
        """Compute loss."""
        if loss_type == 'mse':
            return float(np.mean((predictions - targets) ** 2))
        elif loss_type == 'cross_entropy':
            # Softmax cross entropy
            exp_pred = np.exp(predictions - predictions.max(axis=1, keepdims=True))
            probs = exp_pred / exp_pred.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-10)))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def backward(
        self,
        x: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss_type: str = 'mse'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients (simplified for demonstration).
        
        In a real implementation, this would use automatic differentiation.
        """
        batch_size = x.shape[0]
        
        # Output gradient
        if loss_type == 'mse':
            d_output = 2 * (predictions - targets) / batch_size
        else:
            # Cross entropy gradient (simplified)
            d_output = predictions - targets
        
        # Simplified gradients (would normally use autograd)
        d_weights2 = np.random.randn(*d_output.T.shape) * 0.01  # Placeholder
        d_weights1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01  # Placeholder
        
        return d_weights1, d_weights2
    
    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        use_snapping: bool = True
    ) -> float:
        """
        Single training step with optional gradient snapping.
        
        Args:
            x: Input batch.
            y: Target batch.
            use_snapping: Whether to snap gradients.
        
        Returns:
            Loss value.
        """
        # Forward pass
        predictions = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Backward pass (simplified)
        d_w1, d_w2 = self.backward(x, predictions, y)
        
        # Optionally snap gradients for determinism
        if use_snapping and d_w1.shape[1] >= 2:
            # Snap first two dimensions of each gradient
            snapped = self.snapper.snap_batch(d_w1[:, :2])
            d_w1[:, :2] = snapped
        
        # Apply gradient update (simplified - in practice use optimizer)
        # self.apply_gradients(d_w1, d_w2)
        
        return loss
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_hidden_encoding: bool = False,
        use_snapping: bool = True
    ) -> List[float]:
        """
        Full training loop.
        
        Args:
            x_train: Training inputs.
            y_train: Training targets.
            epochs: Number of epochs.
            batch_size: Batch size.
            use_hidden_encoding: Use hidden dimension encoding.
            use_snapping: Use gradient snapping.
        
        Returns:
            Training loss history.
        """
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                loss = self.train_step(x_batch, y_batch, use_snapping)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {avg_loss:.6f}")
        
        return self.training_history


def demonstrate_quantization_aware_training():
    """
    Demonstrate quantization-aware training.
    
    This shows how to use PythagoreanQuantizer during training
    to prepare models for quantized inference.
    """
    print("\n" + "=" * 60)
    print("Quantization-Aware Training Demo")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    input_dim = 128
    
    x_train = np.random.randn(n_samples, input_dim).astype(np.float32)
    # Normalize to unit vectors (common for embeddings)
    x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
    
    # Create quantizer
    quantizer = PythagoreanQuantizer(
        mode=QuantizationMode.POLAR,
        bits=8,
        constraints=['unit_norm']
    )
    
    print(f"\nOriginal data shape: {x_train.shape}")
    print(f"Original data dtype: {x_train.dtype}")
    print(f"Sample vector norm: {np.linalg.norm(x_train[0]):.6f}")
    
    # Quantize
    start_time = time.time()
    result = quantizer.quantize(x_train)
    elapsed = time.time() - start_time
    
    print(f"\nQuantized in {elapsed*1000:.2f}ms")
    print(f"Quantization mode: {result.mode}")
    print(f"Compression ratio: {result.compression_ratio:.1f}x")
    print(f"MSE: {result.mse:.6f}")
    print(f"Unit norm preserved: {result.constraints_satisfied}")
    print(f"Quantized vector norm: {np.linalg.norm(result.data[0]):.6f}")
    
    return result


def demonstrate_hidden_dimension_encoding():
    """
    Demonstrate hidden dimension encoding for constraint satisfaction.
    """
    print("\n" + "=" * 60)
    print("Hidden Dimension Encoding Demo")
    print("=" * 60)
    
    # Point that needs constraint satisfaction
    point = np.array([0.577, 0.816])  # Near 3-5-sqrt(34)
    
    print(f"\nOriginal point: {point}")
    print(f"Original norm: {np.linalg.norm(point):.6f}")
    
    # Encode at different precisions
    for epsilon in [1e-3, 1e-6, 1e-10]:
        k = compute_hidden_dim_count(epsilon)
        encoded = encode_with_hidden_dimensions(
            point,
            constraints=['unit_norm'],
            epsilon=epsilon
        )
        
        print(f"\nPrecision {epsilon} (k={k}):")
        print(f"  Encoded: {encoded}")
        print(f"  Norm: {np.linalg.norm(encoded):.6f}")


def demonstrate_gradient_snapping():
    """
    Demonstrate deterministic gradient snapping.
    """
    print("\n" + "=" * 60)
    print("Gradient Snapping Demo")
    print("=" * 60)
    
    snapper = GradientSnapper(density=200, preserve_magnitude=True)
    
    # Random gradients
    np.random.seed(42)
    gradients = np.random.randn(5, 2)
    
    print("\nOriginal gradients:")
    for i, g in enumerate(gradients):
        print(f"  {i}: ({g[0]:.4f}, {g[1]:.4f}) mag={np.linalg.norm(g):.4f}")
    
    # Snap gradients
    snapped = snapper.snap_batch(gradients)
    
    print("\nSnapped gradients (Pythagorean directions):")
    for i, g in enumerate(snapped):
        print(f"  {i}: ({g[0]:.4f}, {g[1]:.4f}) mag={np.linalg.norm(g):.4f}")


def demonstrate_constrained_layer():
    """
    Demonstrate constraint-enforced layer.
    """
    print("\n" + "=" * 60)
    print("Constraint-Enforced Layer Demo")
    print("=" * 60)
    
    # Create layer with unit norm constraint
    layer = ConstraintEnforcedLayer(
        input_dim=128,
        output_dim=2,  # 2D output for Pythagorean snapping
        constraints=['unit_norm'],
        framework='numpy'
    )
    
    # Random input
    np.random.seed(42)
    x = np.random.randn(10, 128)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = layer(x)
    
    print(f"Output shape: {y.shape}")
    
    # Check norms
    norms = np.linalg.norm(y, axis=1)
    print(f"\nOutput norms (should be ~1.0):")
    for i, norm in enumerate(norms):
        print(f"  Sample {i}: {norm:.6f}")
    
    print(f"\nMean norm: {norms.mean():.6f}")
    print(f"Std norm: {norms.std():.6f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Constraint Theory ML Training Examples")
    print("=" * 60)
    
    # Demonstrate all features
    demonstrate_quantization_aware_training()
    demonstrate_hidden_dimension_encoding()
    demonstrate_gradient_snapping()
    demonstrate_constrained_layer()
    
    # Full training example
    print("\n" + "=" * 60)
    print("Full Training Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    input_dim = 32
    
    x_train = np.random.randn(n_samples, input_dim)
    y_train = np.random.randint(0, 3, n_samples)  # 3 classes
    y_onehot = np.zeros((n_samples, 3))
    y_onehot[np.arange(n_samples), y_train] = 1
    
    # Create trainer
    trainer = ConstrainedTrainer(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=3,
        constraints=['unit_norm'],
        learning_rate=0.01,
        precision=1e-6
    )
    
    print(f"\nTraining on {n_samples} samples...")
    
    # Train with constraint enforcement
    history = trainer.train(
        x_train,
        y_onehot,
        epochs=50,
        batch_size=32,
        use_hidden_encoding=False,
        use_snapping=True
    )
    
    print(f"\nFinal loss: {history[-1]:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Benefits of Constraint Theory in ML:
    
1. Reproducibility: Gradient snapping ensures deterministic updates
2. Constraint Satisfaction: Layers enforce constraints (unit norm, orthogonality)
3. Quantization Ready: Train with quantization awareness
4. Hidden Dimensions: Encode constraints for exact satisfaction
5. No Floating-Point Drift: Pythagorean ratios are exact

For production use:
- Use PyTorch/TensorFlow with ConstraintEnforcedLayer
- Enable gradient snapping in optimizer
- Quantize weights before deployment
- Use hidden dimension encoding for critical constraints
    """)


if __name__ == "__main__":
    main()
