#!/usr/bin/env python3
"""
NumPy integration example for Constraint Theory Python bindings.

This example demonstrates how to use the library with NumPy arrays
for efficient batch processing and array operations.
"""

import numpy as np
from constraint_theory import PythagoreanManifold, generate_triples


def main():
    print("=" * 60)
    print("Constraint Theory - NumPy Integration Example")
    print("=" * 60)
    
    # =====================================================================
    # 1. Basic NumPy Integration
    # =====================================================================
    print("\n1. Basic NumPy Integration")
    print("-" * 40)
    
    manifold = PythagoreanManifold(density=200)
    print(f"   Created manifold with {manifold.state_count} states")
    
    # Create a NumPy array of vectors
    vectors = np.array([
        [0.6, 0.8],
        [0.8, 0.6],
        [0.707, 0.707],
        [0.1, 0.995],
        [0.999, 0.05],
    ], dtype=np.float32)
    
    print(f"\n   Input array shape: {vectors.shape}")
    print(f"   Input array dtype: {vectors.dtype}")
    print("\n   Input vectors:")
    print(vectors)
    
    # Snap batch accepts NumPy arrays directly
    results = manifold.snap_batch(vectors)
    
    print("\n   Snapped results:")
    for i, (sx, sy, noise) in enumerate(results):
        print(f"   [{i}] ({vectors[i, 0]:.3f}, {vectors[i, 1]:.3f}) "
              f"-> ({sx:.4f}, {sy:.4f}), noise={noise:.6f}")
    
    # =====================================================================
    # 2. Processing Large Arrays
    # =====================================================================
    print("\n2. Processing Large Arrays")
    print("-" * 40)
    
    # Generate random unit vectors
    np.random.seed(42)
    n_vectors = 10000
    
    # Generate random angles and convert to unit vectors
    angles = np.random.uniform(0, 2 * np.pi, n_vectors)
    large_vectors = np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ]).astype(np.float32)
    
    print(f"   Generated {n_vectors:,} random unit vectors")
    print(f"   Array shape: {large_vectors.shape}")
    
    import time
    start = time.time()
    results = manifold.snap_batch(large_vectors)
    elapsed = time.time() - start
    
    print(f"   Processed in {elapsed*1000:.2f} ms")
    print(f"   Throughput: {n_vectors/elapsed:,.0f} vectors/second")
    
    # Convert results to NumPy array for analysis
    snapped = np.array([[sx, sy, noise] for sx, sy, noise in results])
    
    print(f"\n   Noise statistics:")
    print(f"     Mean:   {snapped[:, 2].mean():.6f}")
    print(f"     Std:    {snapped[:, 2].std():.6f}")
    print(f"     Min:    {snapped[:, 2].min():.6f}")
    print(f"     Max:    {snapped[:, 2].max():.6f}")
    print(f"     Median: {np.median(snapped[:, 2]):.6f}")
    
    # =====================================================================
    # 3. Visualization with Matplotlib (if available)
    # =====================================================================
    print("\n3. Visualization Analysis")
    print("-" * 40)
    
    # Analyze the distribution of snapped vectors
    exact_matches = np.sum(snapped[:, 2] < 0.001)
    close_matches = np.sum((snapped[:, 2] >= 0.001) & (snapped[:, 2] < 0.05))
    approximate = np.sum(snapped[:, 2] >= 0.05)
    
    print(f"   Exact matches (noise < 0.001):    {exact_matches:>5} ({exact_matches/n_vectors*100:.1f}%)")
    print(f"   Close matches (0.001-0.05):       {close_matches:>5} ({close_matches/n_vectors*100:.1f}%)")
    print(f"   Approximate (noise >= 0.05):      {approximate:>5} ({approximate/n_vectors*100:.1f}%)")
    
    # =====================================================================
    # 4. Working with Different Manifold Densities
    # =====================================================================
    print("\n4. Comparing Manifold Densities")
    print("-" * 40)
    
    densities = [50, 100, 200, 500]
    test_vectors = np.array([
        [0.6, 0.8],
        [0.707, 0.707],
        [0.5, 0.866],  # 30 degrees
    ], dtype=np.float32)
    
    print(f"\n   Testing {len(test_vectors)} vectors with different densities:\n")
    print(f"   {'Density':>8} {'States':>10} {'Avg Noise':>12} {'Time (ms)':>12}")
    print(f"   {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
    
    for density in densities:
        m = PythagoreanManifold(density=density)
        
        start = time.time()
        results = m.snap_batch(test_vectors)
        elapsed = time.time() - start
        
        avg_noise = np.mean([noise for _, _, noise in results])
        
        print(f"   {density:>8} {m.state_count:>10} {avg_noise:>12.6f} {elapsed*1000:>12.2f}")
    
    # =====================================================================
    # 5. Practical Application: Vector Normalization Analysis
    # =====================================================================
    print("\n5. Practical Application: Vector Normalization Analysis")
    print("-" * 40)
    
    # Generate Pythagorean triples and create test vectors
    triples = generate_triples(100)
    print(f"   Using {len(triples)} Pythagorean triples for analysis")
    
    # Create normalized vectors from triples and add small perturbations
    test_data = []
    for a, b, c in triples[:100]:
        # Exact vector
        test_data.append([a/c, b/c])
        # Perturbed vectors
        for eps in [0.01, 0.05, 0.1]:
            test_data.append([a/c + eps * np.random.randn(), 
                             b/c + eps * np.random.randn()])
    
    test_array = np.array(test_data, dtype=np.float32)
    # Renormalize to unit vectors
    norms = np.linalg.norm(test_array, axis=1, keepdims=True)
    test_array = test_array / norms
    
    results = manifold.snap_batch(test_array)
    noise_values = np.array([noise for _, _, noise in results])
    
    # Group by perturbation level
    n_exact = len(triples[:100])
    exact_noise = noise_values[:n_exact]
    
    print(f"\n   Exact Pythagorean vectors:")
    print(f"     Mean noise: {exact_noise.mean():.6f}")
    print(f"     All exact matches: {np.all(exact_noise < 0.001)}")
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
NumPy Integration Tips:
1. Use float32 arrays for best performance
2. snap_batch() accepts Nx2 NumPy arrays directly
3. Higher density = better resolution but slower initialization
4. Batch processing is much faster than individual snaps
5. Convert results to NumPy arrays for further analysis
    """)
    
    print("Done!")


if __name__ == "__main__":
    main()
