#!/usr/bin/env python3
"""
Quick start example for Constraint Theory Python bindings.
"""

from constraint_theory import PythagoreanManifold, generate_triples, snap


def main():
    print("=" * 60)
    print("Constraint Theory - Python Quick Start")
    print("=" * 60)
    
    # Create a manifold
    print("\n1. Creating Pythagorean Manifold...")
    manifold = PythagoreanManifold(density=200)
    print(f"   Manifold has {manifold.state_count} valid states")
    
    # Snap some vectors
    print("\n2. Snapping vectors to Pythagorean triples...")
    test_vectors = [
        (0.6, 0.8),    # Exact: 3-4-5 triangle
        (0.8, 0.6),    # Same, swapped
        (0.707, 0.707), # ~45 degrees
        (0.1, 0.995),  # Near vertical
    ]
    
    for x, y in test_vectors:
        sx, sy, noise = manifold.snap(x, y)
        print(f"   ({x:.3f}, {y:.3f}) -> ({sx:.4f}, {sy:.4f}), noise={noise:.6f}")
    
    # Batch processing
    print("\n3. Batch processing (SIMD optimized)...")
    vectors = [[0.6, 0.8], [0.8, 0.6], [0.1, 0.99], [0.707, 0.707]]
    results = manifold.snap_batch(vectors)
    
    for i, (sx, sy, noise) in enumerate(results):
        print(f"   [{i}] ({vectors[i][0]:.3f}, {vectors[i][1]:.3f}) -> ({sx:.4f}, {sy:.4f})")
    
    # Generate triples
    print("\n4. Generating Pythagorean triples (c <= 30)...")
    triples = generate_triples(30)
    for a, b, c in triples[:5]:
        print(f"   {a}² + {b}² = {c}²  (→ {a}² + {b}² = {a*a + b*b})")
    print(f"   ... and {len(triples) - 5} more")
    
    print("\n" + "=" * 60)
    print("Done! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
