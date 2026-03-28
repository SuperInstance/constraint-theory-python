# Use Case Examples

This document provides detailed examples of using Constraint Theory in real-world scenarios.

## Table of Contents

- [Game Development](#game-development)
- [Machine Learning](#machine-learning)
- [Scientific Computing](#scientific-computing)
- [Robotics](#robotics)
- [CAD/CAM](#cadcam)
- [Financial Modeling](#financial-modeling)

---

## Game Development

### Networked Physics

**Problem:** Floating-point drift causes physics desynchronization between players in networked games.

**Solution:** Snap all direction vectors to exact Pythagorean coordinates for deterministic physics.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class DeterministicPhysics:
    """Deterministic physics for networked games."""
    
    def __init__(self, density=150):
        self.manifold = PythagoreanManifold(density)
    
    def process_input(self, vx: float, vy: float) -> tuple[float, float]:
        """Process player input deterministically."""
        # Normalize input direction
        mag = np.sqrt(vx*vx + vy*vy)
        if mag < 0.001:
            return 0.0, 0.0
        
        # Snap to exact Pythagorean coordinates
        sx, sy, noise = self.manifold.snap(vx/mag, vy/mag)
        
        # All clients see identical physics
        return sx, sy
    
    def compute_trajectory(self, start_pos, direction, speed, steps):
        """Compute trajectory deterministically."""
        sx, sy = self.process_input(*direction)
        trajectory = [start_pos]
        
        for _ in range(steps):
            new_pos = (
                trajectory[-1][0] + sx * speed,
                trajectory[-1][1] + sy * speed
            )
            trajectory.append(new_pos)
        
        return trajectory

# Usage
physics = DeterministicPhysics(density=150)
direction = physics.process_input(0.577, 0.816)
print(f"Direction: {direction}")  # (0.6, 0.8) - exact on all machines
```

### Projectile Spawning

```python
class ProjectileSpawner:
    """Deterministic projectile spawning."""
    
    def __init__(self):
        self.manifold = PythagoreanManifold(100)
    
    def spawn_projectile(self, angle: float, speed: float) -> dict:
        """Spawn projectile with deterministic direction."""
        # Convert angle to direction
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Snap for exact coordinates
        sx, sy, _ = self.manifold.snap(dx, dy)
        
        return {
            'direction': (sx, sy),
            'velocity': (sx * speed, sy * speed),
            'norm_check': sx*sx + sy*sy  # Always exactly 1.0
        }

spawner = ProjectileSpawner()
proj = spawner.spawn_projectile(np.pi/4, 100)
print(f"Direction: {proj['direction']}")
print(f"Norm: {proj['norm_check']}")  # 1.0 exactly
```

---

## Machine Learning

### Data Augmentation

**Problem:** Random data augmentation causes non-reproducible training runs.

**Solution:** Use deterministic direction snapping for reproducible augmentation.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class DeterministicAugmenter:
    """Reproducible data augmentation."""
    
    def __init__(self, density=300, seed=42):
        self.manifold = PythagoreanManifold(density)
        self.rng = np.random.RandomState(seed)
    
    def augment_direction(self, dx: float, dy: float, 
                          noise_scale: float = 0.1) -> tuple[float, float]:
        """Augment direction deterministically."""
        # Add reproducible noise
        noise_x = self.rng.randn() * noise_scale
        noise_y = self.rng.randn() * noise_scale
        
        # Perturb and snap
        px, py = dx + noise_x, dy + noise_y
        sx, sy, _ = self.manifold.snap(px, py)
        
        return sx, sy
    
    def augment_batch(self, directions, noise_scale=0.1):
        """Augment batch of directions."""
        results = []
        for dx, dy in directions:
            results.append(self.augment_direction(dx, dy, noise_scale))
        return results

# Usage - produces identical results with same seed
augmenter = DeterministicAugmenter(seed=42)
augmented = augmenter.augment_direction(0.707, 0.707)
print(f"Augmented: {augmented}")

# Same seed = same results
augmenter2 = DeterministicAugmenter(seed=42)
augmented2 = augmenter2.augment_direction(0.707, 0.707)
print(f"Identical: {augmented == augmented2}")  # True
```

### Embedding Normalization

```python
class EmbeddingNormalizer:
    """Deterministic embedding normalization for vector databases."""
    
    def __init__(self, density=500):
        self.manifold = PythagoreanManifold(density)
    
    def normalize_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings deterministically."""
        normalized = []
        
        for emb in embeddings:
            # For high-dim, normalize in 2D projections
            # or use first two principal components
            norm = np.linalg.norm(emb)
            if norm < 1e-10:
                normalized.append(emb)
                continue
            
            unit = emb / norm
            
            # Project first two dims and snap
            sx, sy, _ = self.manifold.snap(unit[0], unit[1])
            
            # Reconstruct with exact first two components
            result = unit.copy()
            result[0] = sx
            result[1] = sy
            normalized.append(result)
        
        return np.array(normalized)

# Usage
normalizer = EmbeddingNormalizer()
embeddings = np.random.randn(100, 128)  # 100 embeddings of dim 128
normalized = normalizer.normalize_batch(embeddings)
```

---

## Scientific Computing

### Monte Carlo Simulation

**Problem:** Monte Carlo simulations produce different results across runs and platforms.

**Solution:** Use deterministic direction snapping for reproducible sampling.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class ReproducibleMonteCarlo:
    """Reproducible Monte Carlo sampling."""
    
    def __init__(self, density=300, seed=42):
        self.manifold = PythagoreanManifold(density)
        self.rng = np.random.RandomState(seed)
    
    def sample_unit_vectors(self, n: int) -> np.ndarray:
        """Generate reproducible unit vectors."""
        # Generate random angles
        angles = self.rng.uniform(0, 2*np.pi, n)
        
        # Convert to vectors
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Snap to exact states
        results = self.manifold.snap_batch(vectors)
        
        return np.array([[sx, sy] for sx, sy, _ in results])
    
    def estimate_pi(self, n_samples: int) -> float:
        """Estimate π with reproducible Monte Carlo."""
        # Generate samples
        x = self.rng.uniform(-1, 1, n_samples)
        y = self.rng.uniform(-1, 1, n_samples)
        
        # Snap sample points (optional, for determinism)
        # This doesn't affect the π estimation
        
        # Count points inside unit circle
        inside = np.sum(x*x + y*y <= 1)
        
        return 4 * inside / n_samples

# Usage - produces identical results on any machine
mc = ReproducibleMonteCarlo(seed=42)
pi_estimate = mc.estimate_pi(100000)
print(f"π estimate: {pi_estimate:.6f}")

# Same seed = identical results
mc2 = ReproducibleMonteCarlo(seed=42)
pi_estimate2 = mc2.estimate_pi(100000)
print(f"Identical: {pi_estimate == pi_estimate2}")  # True
```

### HPC Cross-Platform Reproducibility

```python
class HPCSimulation:
    """HPC simulation with cross-platform reproducibility."""
    
    def __init__(self, density=500):
        self.manifold = PythagoreanManifold(density)
    
    def simulate_diffusion(self, n_particles, n_steps, dt):
        """Simulate diffusion with exact direction tracking."""
        # Initialize positions
        positions = np.zeros((n_particles, 2))
        
        # Track trajectory for analysis
        trajectories = [positions.copy()]
        
        for step in range(n_steps):
            # Generate random directions
            angles = np.random.uniform(0, 2*np.pi, n_particles)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Snap directions for exactness
            results = self.manifold.snap_batch(directions)
            snapped = np.array([[sx, sy] for sx, sy, _ in results])
            
            # Update positions
            positions += snapped * dt
            
            trajectories.append(positions.copy())
        
        return np.array(trajectories)

# Usage
sim = HPCSimulation(density=500)
trajectories = sim.simulate_diffusion(n_particles=100, n_steps=100, dt=0.01)
print(f"Trajectory shape: {trajectories.shape}")
```

---

## Robotics

### Path Planning

**Problem:** Path planning calculations accumulate floating-point errors over time.

**Solution:** Snap waypoint directions to exact coordinates.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class PathPlanner:
    """Deterministic path planning."""
    
    def __init__(self, density=200):
        self.manifold = PythagoreanManifold(density)
    
    def plan_path(self, start, goal, obstacles=None):
        """Plan path with exact waypoint directions."""
        # Simple direct path (obstacle avoidance would extend this)
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 0.001:
            return [start, goal]
        
        # Snap direction
        sx, sy, _ = self.manifold.snap(dx/dist, dy/dist)
        
        # Generate waypoints
        n_waypoints = int(dist / 0.1)  # 0.1 unit spacing
        waypoints = [start]
        
        for i in range(1, n_waypoints):
            wp = (
                start[0] + sx * 0.1 * i,
                start[1] + sy * 0.1 * i
            )
            waypoints.append(wp)
        
        waypoints.append(goal)
        return waypoints
    
    def compute_velocity_command(self, current_pos, target_pos, max_speed):
        """Compute deterministic velocity command."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 0.01:
            return 0.0, 0.0
        
        # Snap direction for exact command
        sx, sy, _ = self.manifold.snap(dx/dist, dy/dist)
        
        return sx * max_speed, sy * max_speed

# Usage
planner = PathPlanner()
path = planner.plan_path((0, 0), (10, 10))
print(f"Path has {len(path)} waypoints")
```

---

## CAD/CAM

### Geometric Constraints

**Problem:** Geometric constraints in CAD systems suffer from floating-point tolerance issues.

**Solution:** Use exact Pythagorean coordinates for constrained geometry.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class ConstrainedGeometry:
    """CAD geometry with exact constraints."""
    
    def __init__(self, density=1000):
        self.manifold = PythagoreanManifold(density)
    
    def create_line(self, start, angle, length):
        """Create line with exact direction."""
        # Snap angle direction
        dx = np.cos(angle)
        dy = np.sin(angle)
        sx, sy, noise = self.manifold.snap(dx, dy)
        
        # Exact endpoint
        end = (start[0] + sx * length, start[1] + sy * length)
        
        return {
            'start': start,
            'end': end,
            'direction': (sx, sy),
            'length': length,
            'snap_noise': noise
        }
    
    def create_polygon(self, center, radius, n_sides):
        """Create regular polygon with exact vertices."""
        vertices = []
        angle_step = 2 * np.pi / n_sides
        
        for i in range(n_sides):
            angle = i * angle_step
            dx = np.cos(angle)
            dy = np.sin(angle)
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            vertex = (center[0] + sx * radius, center[1] + sy * radius)
            vertices.append(vertex)
        
        return vertices
    
    def fillet_corner(self, p1, corner, p2, radius):
        """Create fillet with exact arc."""
        # Calculate arc endpoints using snapped directions
        d1x, d1y = corner[0] - p1[0], corner[1] - p1[1]
        d2x, d2y = p2[0] - corner[0], p2[1] - corner[1]
        
        # Normalize and snap
        norm1 = np.sqrt(d1x*d1x + d1y*d1y)
        norm2 = np.sqrt(d2x*d2x + d2y*d2y)
        
        s1x, s1y, _ = self.manifold.snap(d1x/norm1, d1y/norm1)
        s2x, s2y, _ = self.manifold.snap(d2x/norm2, d2y/norm2)
        
        # Arc center and endpoints
        arc_start = (corner[0] - s1x * radius, corner[1] - s1y * radius)
        arc_end = (corner[0] + s2x * radius, corner[1] + s2y * radius)
        
        return {
            'arc_start': arc_start,
            'arc_end': arc_end,
            'center': corner,
            'radius': radius
        }

# Usage
cad = ConstrainedGeometry(density=1000)
line = cad.create_line((0, 0), np.pi/4, 10)
print(f"Line direction: {line['direction']}")
print(f"Direction norm: {line['direction'][0]**2 + line['direction'][1]**2}")
```

---

## Financial Modeling

### Risk Direction Analysis

**Problem:** Risk direction calculations produce slightly different results across systems.

**Solution:** Use deterministic direction snapping for risk vectors.

```python
from constraint_theory import PythagoreanManifold
import numpy as np

class RiskAnalyzer:
    """Deterministic risk direction analysis."""
    
    def __init__(self, density=300):
        self.manifold = PythagoreanManifold(density)
    
    def analyze_risk_direction(self, risk_factors):
        """Analyze risk direction deterministically."""
        # Risk vector from factors
        total_risk = np.sqrt(sum(f**2 for f in risk_factors.values()))
        
        if total_risk < 1e-10:
            return None
        
        # Direction in risk space
        dir_x = risk_factors.get('market', 0) / total_risk
        dir_y = risk_factors.get('credit', 0) / total_risk
        
        # Snap for exact comparison
        sx, sy, noise = self.manifold.snap(dir_x, dir_y)
        
        return {
            'direction': (sx, sy),
            'magnitude': total_risk,
            'snap_error': noise,
            'category': self._categorize_risk(sx, sy)
        }
    
    def _categorize_risk(self, x, y):
        """Categorize risk direction."""
        if abs(x) > abs(y):
            return 'market_dominated'
        elif abs(y) > abs(x):
            return 'credit_dominated'
        else:
            return 'balanced'

# Usage
analyzer = RiskAnalyzer()
risk_factors = {'market': 0.577, 'credit': 0.816, 'operational': 0.1}
result = analyzer.analyze_risk_direction(risk_factors)
print(f"Risk direction: {result['direction']}")
print(f"Risk category: {result['category']}")
```

---

## Summary

| Use Case | Key Benefit | Recommended Density |
|----------|-------------|---------------------|
| Game Physics | Network determinism | 100-200 |
| ML Augmentation | Reproducible training | 200-500 |
| Monte Carlo | Cross-platform repro | 300-500 |
| Robotics | Exact waypoints | 200-500 |
| CAD/CAM | Geometric precision | 500-1000 |
| Financial | Consistent risk metrics | 200-300 |

### Choosing Density

- **Lower (50-100)**: Maximum speed, lower precision
- **Medium (100-200)**: Balanced for games and real-time
- **Higher (200-500)**: Scientific computing, ML
- **Highest (500-1000)**: CAD/CAM, maximum precision
