#!/usr/bin/env python3
"""
Advanced NumPy Integration Example for Constraint Theory.

This example demonstrates:
1. PythagoreanQuantizer with mode selection
2. Hidden dimension encoding
3. ML integration patterns
4. Batch processing with constraint preservation

Run with: python examples/advanced_numpy_integration.py
"""

import numpy as np
from typing import List, Tuple


def example_quantizer_modes():
    """Demonstrate different quantization modes."""
    print("=" * 70)
    print("1. PythagoreanQuantizer - Mode Selection")
    print("=" * 70)
    
    from constraint_theory import (
        PythagoreanQuantizer,
        QuantizationMode,
        auto_select_mode,
    )
    
    # Generate test data for different scenarios
    np.random.seed(42)
    
    # Scenario 1: Unit norm embeddings (should select POLAR)
    print("\n--- Scenario 1: Unit Norm Embeddings ---")
    embeddings = np.random.randn(100, 128)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    auto_mode = auto_select_mode(embeddings)
    print(f"Auto-selected mode: {auto_mode.name}")
    
    quantizer = PythagoreanQuantizer(mode=QuantizationMode.POLAR)
    result = quantizer.quantize(embeddings)
    
    print(f"  Compression ratio: {result.compression_ratio:.1f}x")
    print(f"  Distortion (MSE): {result.distortion:.6f}")
    print(f"  Constraints satisfied: {result.constraints_satisfied}")
    
    # Verify unit norm preservation
    norms = np.linalg.norm(result.data, axis=1)
    print(f"  Mean norm after quantization: {norms.mean():.10f}")
    print(f"  Norm variance: {norms.var():.2e}")
    
    # Scenario 2: Sparse weight matrix (should select TERNARY)
    print("\n--- Scenario 2: Sparse Weight Matrix ---")
    weights = np.random.randn(100, 256) * 0.1
    weights[np.abs(weights) < 0.05] = 0  # Add sparsity
    
    auto_mode = auto_select_mode(weights)
    print(f"Auto-selected mode: {auto_mode.name}")
    
    quantizer = PythagoreanQuantizer(mode=QuantizationMode.TERNARY)
    result = quantizer.quantize(weights)
    
    print(f"  Compression ratio: {result.compression_ratio:.1f}x")
    print(f"  Distortion (MSE): {result.distortion:.6f}")
    print(f"  Sparsity: {result.metadata.get('sparsity', 0):.2%}")
    
    # Scenario 3: General embeddings (should select TURBO)
    print("\n--- Scenario 3: General Embeddings ---")
    general_vectors = np.random.randn(100, 768)  # Typical embedding dim
    
    auto_mode = auto_select_mode(general_vectors)
    print(f"Auto-selected mode: {auto_mode.name}")
    
    quantizer = PythagoreanQuantizer(mode=QuantizationMode.TURBO, bits=4)
    result = quantizer.quantize(general_vectors)
    
    print(f"  Compression ratio: {result.compression_ratio:.1f}x")
    print(f"  Distortion (MSE): {result.distortion:.6f}")


def example_hidden_dimensions():
    """Demonstrate hidden dimension encoding."""
    print("\n" + "=" * 70)
    print("2. Hidden Dimension Encoding")
    print("=" * 70)
    
    from constraint_theory import (
        compute_hidden_dim_count,
        encode_with_hidden_dimensions,
        lift_to_hidden,
        project_visible,
        holographic_accuracy,
    )
    
    # Compute hidden dimensions for different precisions
    print("\n--- Hidden Dimensions vs Precision ---")
    epsilons = [1e-6, 1e-10, 1e-16]
    
    for eps in epsilons:
        k = compute_hidden_dim_count(eps)
        accuracy = holographic_accuracy(k, 2)
        print(f"  ε = {eps:.0e}: k = {k} hidden dims, accuracy = {accuracy:.1f}")
    
    # Encode a point with hidden dimensions
    print("\n--- Encoding a Point ---")
    point = np.array([0.6, 0.8])
    
    # Lift to hidden dimensions
    k = compute_hidden_dim_count(1e-10)
    lifted = lift_to_hidden(point, k)
    print(f"  Original point: {point}")
    print(f"  Lifted to {len(lifted)}D: first 5 coords = {lifted[:5]}")
    
    # Project back
    visible = project_visible(lifted, 2)
    print(f"  Projected back: {visible}")
    
    # Full encoding with constraint satisfaction
    print("\n--- Full Encoding with Constraints ---")
    noisy_point = np.array([0.577, 0.816])
    
    encoded = encode_with_hidden_dimensions(
        noisy_point,
        constraints=['unit_norm'],
        epsilon=1e-10
    )
    
    original_norm = np.linalg.norm(noisy_point)
    encoded_norm = np.linalg.norm(encoded)
    
    print(f"  Input: {noisy_point}, norm = {original_norm:.6f}")
    print(f"  Encoded: {encoded}, norm = {encoded_norm:.10f}")


def example_ml_integration():
    """Demonstrate ML integration patterns."""
    print("\n" + "=" * 70)
    print("3. ML Integration Patterns")
    print("=" * 70)
    
    from constraint_theory import (
        ConstraintEnforcedLayer,
        GradientSnapper,
    )
    
    # Example 1: Constraint Enforced Layer (NumPy backend)
    print("\n--- ConstraintEnforcedLayer (NumPy backend) ---")
    layer = ConstraintEnforcedLayer(
        input_dim=64,
        output_dim=32,
        constraints=['unit_norm'],
        framework='numpy'
    )
    
    # Forward pass
    batch = np.random.randn(16, 64)
    output = layer(batch)
    
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Verify unit norm
    norms = np.linalg.norm(output, axis=1)
    print(f"  Mean output norm: {norms.mean():.10f}")
    print(f"  Norm variance: {norms.var():.2e}")
    
    # Example 2: Gradient Snapper for reproducible training
    print("\n--- GradientSnapper for Reproducible Training ---")
    snapper = GradientSnapper(density=200, preserve_magnitude=True)
    
    # Simulate gradients
    gradients = np.random.randn(100, 2)
    
    snapped = snapper.snap_batch(gradients)
    
    print(f"  Original gradients sample: {gradients[:3]}")
    print(f"  Snapped gradients sample: {snapped[:3]}")
    
    # Verify determinism
    snapped_again = snapper.snap_batch(gradients)
    print(f"  Deterministic: {np.allclose(snapped, snapped_again)}")


def example_batch_processing():
    """Demonstrate efficient batch processing."""
    print("\n" + "=" * 70)
    print("4. Efficient Batch Processing")
    print("=" * 70)
    
    from constraint_theory import PythagoreanManifold, PythagoreanQuantizer
    
    import time
    
    # Large-scale batch processing
    print("\n--- Large-Scale Batch Snapping ---")
    manifold = PythagoreanManifold(density=500)
    
    n_vectors = 100000
    angles = np.random.uniform(0, 2 * np.pi, n_vectors)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    
    print(f"  Processing {n_vectors:,} vectors...")
    
    start = time.time()
    results = manifold.snap_batch(vectors)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Throughput: {n_vectors/elapsed:,.0f} vectors/sec")
    
    # Analyze noise distribution
    noises = np.array([noise for _, _, noise in results])
    print(f"\n  Noise distribution:")
    print(f"    Mean: {noises.mean():.6f}")
    print(f"    Std:  {noises.std():.6f}")
    print(f"    Max:  {noises.max():.6f}")
    
    # Quantizer batch processing
    print("\n--- Quantizer Batch Processing ---")
    quantizer = PythagoreanQuantizer(mode='POLAR')
    
    # High-dimensional embeddings
    embeddings = np.random.randn(10000, 256)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    start = time.time()
    result = quantizer.quantize(embeddings)
    elapsed = time.time() - start
    
    print(f"  Processed {embeddings.shape[0]:,} vectors of {embeddings.shape[1]}D")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Constraints satisfied: {result.constraints_satisfied}")


def example_cross_plane_optimization():
    """Demonstrate cross-plane fine-tuning."""
    print("\n" + "=" * 70)
    print("5. Cross-Plane Fine-Tuning")
    print("=" * 70)
    
    from constraint_theory import (
        cross_plane_finetune,
        get_orthogonal_planes,
        constraint_error,
    )
    
    # Start with an arbitrary point
    point = np.array([0.707, 0.707])  # Close to sqrt(2)/2
    
    print(f"\n  Original point: {point}")
    print(f"  Original norm: {np.linalg.norm(point):.6f}")
    
    # Fine-tune using cross-plane optimization
    optimized = cross_plane_finetune(
        point,
        constraints=['unit_norm'],
        max_iterations=20
    )
    
    print(f"\n  Optimized point: {optimized}")
    print(f"  Optimized norm: {np.linalg.norm(optimized):.10f}")
    
    # Show orthogonal planes used
    planes = get_orthogonal_planes(len(point))
    print(f"\n  Planes explored: {planes}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Constraint Theory - Advanced NumPy Integration Examples")
    print("=" * 70)
    
    example_quantizer_modes()
    example_hidden_dimensions()
    example_ml_integration()
    example_batch_processing()
    example_cross_plane_optimization()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Takeaways:
1. Use PythagoreanQuantizer with auto_select_mode for best results
2. Hidden dimensions (k = ⌈log₂(1/ε)⌉) enable exact constraint satisfaction
3. ConstraintEnforcedLayer integrates seamlessly with ML workflows
4. Batch processing is highly optimized with SIMD acceleration
5. Cross-plane optimization can improve precision with less compute

Next Steps:
- Try the quantizer with your own data
- Integrate ConstraintEnforcedLayer into your model
- Explore hidden dimension encoding for high-precision applications
    """)
    
    print("Done!")


if __name__ == "__main__":
    main()
