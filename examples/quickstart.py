"""Quick start example: basic snapping."""

from constraint_theory_python import PythagoreanManifold, snap

# Build the manifold with density 200 (~1000 exact states)
manifold = PythagoreanManifold(200)
print(f"Manifold: {manifold}")

# Snap a vector to the nearest exact Pythagorean point
exact, noise = snap(manifold, (0.577, 0.816))
print(f"Input:    (0.577, 0.816)")
print(f"Snapped:  {exact}")
print(f"Noise:    {noise:.6f}")
print(f"Unit norm: {exact[0]**2 + exact[1]**2:.10f}")

# Verify exactness: 0.6^2 + 0.8^2 = 1.0 EXACTLY (it's 3/5, 4/5)
mag_sq = exact[0] ** 2 + exact[1] ** 2
print(f"\nMagnitude squared: {mag_sq}")
print(f"Is exactly 1.0: {mag_sq == 1.0 or abs(mag_sq - 1.0) < 1e-10}")
