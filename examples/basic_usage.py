#!/usr/bin/env python3
"""
Basic usage example for Constraint Theory Python bindings.

This example demonstrates the core functionality of the library:
- Creating a Pythagorean manifold
- Snapping individual vectors
- Understanding the noise/resonance metric
"""

from constraint_theory import PythagoreanManifold, snap, generate_triples


def main():
    print("=" * 60)
    print("Constraint Theory - Basic Usage Example")
    print("=" * 60)
    
    # =====================================================================
    # 1. Creating a Pythagorean Manifold
    # =====================================================================
    print("\n1. Creating a Pythagorean Manifold")
    print("-" * 40)
    
    # The density parameter controls the resolution of the manifold.
    # Higher density = more valid states = finer snapping resolution.
    # Trade-off: Higher density uses more memory and slower initialization.
    manifold = PythagoreanManifold(density=200)
    
    print(f"   Created manifold with density=200")
    print(f"   Total valid states: {manifold.state_count}")
    print(f"   Note: Each state represents a normalized Pythagorean triple")
    
    # =====================================================================
    # 2. Snapping Individual Vectors
    # =====================================================================
    print("\n2. Snapping Individual Vectors")
    print("-" * 40)
    
    # Test vectors with different characteristics
    test_cases = [
        ("Exact Pythagorean (3-4-5)", (0.6, 0.8)),      # 3/5, 4/5 - should snap exactly
        ("Exact Pythagorean (5-12-13)", (0.384615, 0.923077)),  # 5/13, 12/13
        ("Near 45 degrees", (0.707, 0.707)),            # Close to sqrt(2)/2
        ("Near vertical", (0.1, 0.995)),                # Almost straight up
        ("Near horizontal", (0.999, 0.05)),             # Almost horizontal
        ("Random direction", (0.543, 0.839)),           # Random angle
    ]
    
    for description, (x, y) in test_cases:
        snapped_x, snapped_y, noise = manifold.snap(x, y)
        resonance = 1 - noise
        
        print(f"\n   {description}:")
        print(f"     Input:    ({x:.6f}, {y:.6f})")
        print(f"     Snapped:  ({snapped_x:.6f}, {snapped_y:.6f})")
        print(f"     Noise:    {noise:.6f}")
        print(f"     Resonance: {resonance:.6f} ({resonance*100:.1f}%)")
        
        if noise < 0.001:
            print(f"     Status:   EXACT MATCH!")
        elif noise < 0.05:
            print(f"     Status:   Very close match")
        else:
            print(f"     Status:   Approximate match")
    
    # =====================================================================
    # 3. Using the Convenience snap() Function
    # =====================================================================
    print("\n3. Using the Convenience snap() Function")
    print("-" * 40)
    
    # The snap() function is a convenience wrapper
    # For multiple snaps, create a manifold and use its methods directly
    x, y = 0.6, 0.8
    sx, sy, noise = snap(manifold, x, y)
    print(f"   snap(manifold, {x}, {y}) = ({sx:.4f}, {sy:.4f}, noise={noise:.6f})")
    
    # =====================================================================
    # 4. Generating Pythagorean Triples
    # =====================================================================
    print("\n4. Generating Pythagorean Triples")
    print("-" * 40)
    
    max_hypotenuse = 50
    triples = generate_triples(max_hypotenuse)
    
    print(f"   Generated {len(triples)} primitive triples with c <= {max_hypotenuse}:")
    print()
    print(f"   {'a':>5} {'b':>5} {'c':>5} {'a² + b² = c²':<20}")
    print(f"   {'-'*5} {'-'*5} {'-'*5} {'-'*20}")
    
    for a, b, c in triples[:10]:
        print(f"   {a:>5} {b:>5} {c:>5} {a*a:>5} + {b*b:>5} = {c*c:>5}")
    
    if len(triples) > 10:
        print(f"   ... and {len(triples) - 10} more")
    
    # =====================================================================
    # 5. Understanding the Manifold
    # =====================================================================
    print("\n5. Understanding the Manifold")
    print("-" * 40)
    
    print("   The Pythagorean manifold contains all primitive Pythagorean triples")
    print("   normalized to unit vectors. Each valid state represents:")
    print()
    print("     (a/c, b/c) where a² + b² = c²")
    print()
    print("   Properties:")
    print("   - Deterministic: same input always gives same output")
    print("   - Discrete: only snaps to exact Pythagorean triples")
    print("   - Resonance measures how close the input is to a valid state")
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Takeaways:
1. Create a manifold with appropriate density for your use case
2. Use snap() for single vectors, snap_batch() for multiple vectors
3. Noise < 0.05 indicates a very close match
4. Generate triples with generate_triples() for reference
    """)
    
    print("Done!")


if __name__ == "__main__":
    main()
