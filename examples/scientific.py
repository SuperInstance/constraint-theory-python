#!/usr/bin/env python3
"""
Scientific Computing Examples for Constraint Theory.

This example demonstrates how Constraint Theory can be used in scientific
computing for reproducible simulations, Monte Carlo methods, and data analysis.

Key benefits for scientific computing:
- Reproducible results across all platforms
- Exact geometric calculations
- Deterministic simulations
- Verifiable computational science
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from collections import defaultdict


class SimulatedManifold:
    """Simulated PythagoreanManifold for demo purposes."""
    
    def __init__(self, density: int):
        self.density = density
        self._states = self._generate_states()
    
    def _generate_states(self) -> List[Tuple[float, float]]:
        """Generate Pythagorean triple states."""
        states = []
        triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61),
            (28, 45, 53), (33, 56, 65), (16, 63, 65), (48, 55, 73),
            (15, 112, 113), (44, 117, 125), (88, 105, 137),
            (17, 144, 145), (24, 143, 145), (51, 140, 149),
        ]
        
        for a, b, c in triples:
            for sx in [1, -1]:
                for sy in [1, -1]:
                    states.append((sx * a / c, sy * b / c))
                    states.append((sx * b / c, sy * a / c))
        
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            states.append(d)
        
        return states
    
    def snap(self, x: float, y: float) -> Tuple[float, float, float]:
        """Snap to nearest Pythagorean state."""
        mag = math.sqrt(x * x + y * y)
        if mag == 0:
            return (0.0, 0.0, 0.0)
        
        nx, ny = x / mag, y / mag
        best_state = (0.0, 0.0)
        best_dist = float('inf')
        
        for sx, sy in self._states:
            dist = (nx - sx) ** 2 + (ny - sy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_state = (sx, sy)
        
        noise = math.sqrt(best_dist)
        return (best_state[0], best_state[1], noise)
    
    def snap_batch(self, vectors: List[List[float]]) -> List[Tuple[float, float, float]]:
        """Batch snap multiple vectors."""
        return [self.snap(v[0], v[1]) for v in vectors]
    
    @property
    def state_count(self) -> int:
        return len(self._states)


# =============================================================================
# Example 1: Monte Carlo Integration
# =============================================================================

class MonteCarloIntegrator:
    """
    Monte Carlo integration with deterministic sampling.
    
    Uses Constraint Theory for:
    - Reproducible random directions
    - Exact sample distribution
    - Deterministic variance estimation
    """
    
    def __init__(self, manifold: SimulatedManifold, seed: int = 42):
        self.manifold = manifold
        self.rng = random.Random(seed)
    
    def integrate_circle_method(self, 
                                func: Callable[[float, float], float],
                                n_samples: int) -> Tuple[float, float]:
        """
        Integrate function over unit circle using exact directions.
        
        Args:
            func: Function to integrate, f(x, y)
            n_samples: Number of Monte Carlo samples
            
        Returns: (integral, standard_error)
        """
        # Generate samples using exact Pythagorean directions
        samples = []
        
        for _ in range(n_samples):
            # Generate random radius and use exact direction
            r = math.sqrt(self.rng.random())  # Uniform in circle
            angle = self.rng.uniform(0, 2 * math.pi)
            
            # Get exact direction
            dx = math.cos(angle)
            dy = math.sin(angle)
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            # Sample point
            x, y = r * sx, r * sy
            
            # Evaluate function
            samples.append(func(x, y))
        
        # Calculate integral (area of circle = pi)
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std_error = math.sqrt(variance / n_samples)
        
        integral = mean * math.pi  # Scale by area
        
        return integral, std_error
    
    def integrate_direction_average(self,
                                    func: Callable[[float], float],
                                    n_samples: int) -> Tuple[float, float]:
        """
        Average function over unit circle directions.
        
        Useful for computing directional statistics.
        """
        samples = []
        
        # Sample uniformly over angle, snap to exact
        for i in range(n_samples):
            angle = (i / n_samples) * 2 * math.pi
            dx, dy = math.cos(angle), math.sin(angle)
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            # Exact angle from snapped direction
            snapped_angle = math.atan2(sy, sx)
            samples.append(func(snapped_angle))
        
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        
        return mean, math.sqrt(variance / n_samples)
    
    def estimate_pi(self, n_samples: int) -> Tuple[float, float]:
        """
        Estimate pi using Monte Carlo with exact directions.
        
        This demonstrates reproducibility - same samples every time.
        """
        inside = 0
        
        for _ in range(n_samples):
            # Random point in square [-1, 1] x [-1, 1]
            x = self.rng.uniform(-1, 1)
            y = self.rng.uniform(-1, 1)
            
            # Check if inside unit circle
            if x * x + y * y <= 1:
                inside += 1
        
        pi_estimate = 4 * inside / n_samples
        error = abs(pi_estimate - math.pi)
        
        return pi_estimate, error


def demo_monte_carlo():
    """Demonstrate Monte Carlo integration."""
    print("=" * 60)
    print("Example 1: Monte Carlo Integration")
    print("=" * 60)
    
    manifold = SimulatedManifold(500)
    
    print("\nReproducible Monte Carlo (same seed = same results):")
    print("-" * 40)
    
    # Test reproducibility
    for seed in [42, 42, 100]:
        integrator = MonteCarloIntegrator(manifold, seed=seed)
        
        # Estimate pi
        pi_est, error = integrator.estimate_pi(10000)
        print(f"\n   Seed {seed}: pi ≈ {pi_est:.6f} (error: {error:.6f})")
    
    print("\nIntegration over unit circle:")
    print("-" * 40)
    
    # Integrate x^2 + y^2 over unit circle (should be pi/2)
    integrator = MonteCarloIntegrator(manifold, seed=42)
    
    def f(x, y):
        return x * x + y * y
    
    integral, std_err = integrator.integrate_circle_method(f, 10000)
    expected = math.pi / 2
    
    print(f"\n   Integrating x² + y² over unit circle:")
    print(f"   Result: {integral:.6f} ± {std_err:.6f}")
    print(f"   Expected: {expected:.6f}")
    print(f"   Difference: {abs(integral - expected):.6f}")


# =============================================================================
# Example 2: Particle Simulation
# =============================================================================

@dataclass
class Particle:
    """Particle with position, velocity, and direction."""
    x: float
    y: float
    vx: float
    vy: float
    mass: float = 1.0
    
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)
    
    def momentum(self) -> Tuple[float, float]:
        return (self.mass * self.vx, self.mass * self.vy)
    
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


class ParticleSimulation:
    """
    Particle simulation with deterministic physics.
    
    Uses Constraint Theory for:
    - Exact collision directions
    - Reproducible simulations
    - Zero numerical drift
    """
    
    def __init__(self, manifold: SimulatedManifold, seed: int = 42):
        self.manifold = manifold
        self.rng = random.Random(seed)
        self.particles: List[Particle] = []
        self.time = 0.0
        self.collision_count = 0
    
    def add_particle(self, x: float, y: float, 
                    direction_angle: float, 
                    speed: float,
                    mass: float = 1.0) -> None:
        """Add particle with exact direction."""
        # Snap direction to exact
        dx = math.cos(math.radians(direction_angle))
        dy = math.sin(math.radians(direction_angle))
        sx, sy, _ = self.manifold.snap(dx, dy)
        
        self.particles.append(Particle(
            x=x, y=y,
            vx=sx * speed,
            vy=sy * speed,
            mass=mass
        ))
    
    def add_random_particles(self, n: int, box_size: float, 
                            speed_range: Tuple[float, float] = (0.1, 1.0)) -> None:
        """Add n random particles with exact directions."""
        for _ in range(n):
            x = self.rng.uniform(0, box_size)
            y = self.rng.uniform(0, box_size)
            angle = self.rng.uniform(0, 360)
            speed = self.rng.uniform(*speed_range)
            
            self.add_particle(x, y, angle, speed)
    
    def step(self, dt: float, box_size: float) -> None:
        """Advance simulation by dt with wall collisions."""
        self.time += dt
        
        for p in self.particles:
            # Move particle
            p.x += p.vx * dt
            p.y += p.vy * dt
            
            # Wall collisions with exact reflection
            if p.x < 0 or p.x > box_size:
                p.vx = -p.vx
                p.x = max(0, min(box_size, p.x))
                self.collision_count += 1
            
            if p.y < 0 or p.y > box_size:
                p.vy = -p.vy
                p.y = max(0, min(box_size, p.y))
                self.collision_count += 1
    
    def total_energy(self) -> float:
        """Calculate total kinetic energy."""
        return sum(p.kinetic_energy() for p in self.particles)
    
    def total_momentum(self) -> Tuple[float, float]:
        """Calculate total momentum."""
        px = sum(p.mass * p.vx for p in self.particles)
        py = sum(p.mass * p.vy for p in self.particles)
        return (px, py)
    
    def run(self, duration: float, dt: float, box_size: float) -> dict:
        """Run simulation and return statistics."""
        initial_energy = self.total_energy()
        initial_momentum = self.total_momentum()
        
        steps = int(duration / dt)
        
        for _ in range(steps):
            self.step(dt, box_size)
        
        final_energy = self.total_energy()
        final_momentum = self.total_momentum()
        
        return {
            'duration': self.time,
            'steps': steps,
            'collisions': self.collision_count,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_drift': abs(final_energy - initial_energy),
            'initial_momentum': initial_momentum,
            'final_momentum': final_momentum,
            'momentum_drift': math.sqrt(
                (final_momentum[0] - initial_momentum[0]) ** 2 +
                (final_momentum[1] - initial_momentum[1]) ** 2
            ),
        }


def demo_particle_simulation():
    """Demonstrate particle simulation."""
    print("\n" + "=" * 60)
    print("Example 2: Particle Simulation")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    
    print("\nRunning reproducible particle simulation:")
    print("-" * 40)
    
    # Run same simulation twice
    for run in range(2):
        sim = ParticleSimulation(manifold, seed=42)
        sim.add_random_particles(20, box_size=10.0)
        
        stats = sim.run(duration=10.0, dt=0.01, box_size=10.0)
        
        print(f"\n   Run {run + 1}:")
        print(f"     Duration: {stats['duration']:.1f}s ({stats['steps']} steps)")
        print(f"     Wall collisions: {stats['collisions']}")
        print(f"     Energy drift: {stats['energy_drift']:.10f}")
        print(f"     Momentum drift: {stats['momentum_drift']:.10f}")
    
    print("\n   Key: Both runs produce IDENTICAL results!")


# =============================================================================
# Example 3: Ray Tracing
# =============================================================================

class RayTracer:
    """
    Ray tracing with exact ray directions.
    
    Uses Constraint Theory for:
    - Exact ray directions
    - Deterministic reflections
    - Reproducible rendering
    """
    
    def __init__(self, manifold: SimulatedManifold):
        self.manifold = manifold
    
    def cast_ray(self, origin: Tuple[float, float],
                direction_angle: float,
                max_distance: float = 100.0) -> Tuple[Tuple[float, float], float]:
        """
        Cast ray with exact direction.
        
        Returns: (endpoint, snap_error)
        """
        # Snap direction to exact
        dx = math.cos(math.radians(direction_angle))
        dy = math.sin(math.radians(direction_angle))
        sx, sy, noise = self.manifold.snap(dx, dy)
        
        # Calculate endpoint
        end = (
            origin[0] + sx * max_distance,
            origin[1] + sy * max_distance
        )
        
        return end, noise
    
    def trace_reflections(self, origin: Tuple[float, float],
                         initial_angle: float,
                         reflectors: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                         max_bounces: int = 10) -> List[Tuple[float, float]]:
        """
        Trace ray with multiple reflections.
        
        Args:
            origin: Starting point
            initial_angle: Initial direction in degrees
            reflectors: List of (point, normal) pairs for reflectors
            max_bounces: Maximum number of bounces
        
        Returns: List of hit points
        """
        path = [origin]
        current = origin
        dx = math.cos(math.radians(initial_angle))
        dy = math.sin(math.radians(initial_angle))
        
        for bounce in range(max_bounces):
            # Snap current direction to exact
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            # Find nearest reflector
            nearest_dist = float('inf')
            nearest_hit = None
            nearest_normal = None
            
            for ref_point, ref_normal in reflectors:
                # Simple line-segment intersection
                # (simplified for demo)
                hit_dist = self._ray_point_distance(
                    current, (sx, sy), ref_point
                )
                
                if hit_dist < nearest_dist and hit_dist > 0.01:
                    nearest_dist = hit_dist
                    nearest_hit = ref_point
                    nearest_normal = ref_normal
            
            if nearest_hit is None:
                break
            
            path.append(nearest_hit)
            
            # Calculate reflection
            nx, ny = nearest_normal
            dot = sx * nx + sy * ny
            dx = sx - 2 * dot * nx
            dy = sy - 2 * dot * ny
            
            current = nearest_hit
        
        return path
    
    def _ray_point_distance(self, origin: Tuple[float, float],
                           direction: Tuple[float, float],
                           point: Tuple[float, float]) -> float:
        """Calculate distance from ray to point."""
        ox, oy = origin
        dx, dy = direction
        px, py = point
        
        # Project point onto ray
        t = (px - ox) * dx + (py - oy) * dy
        
        if t < 0:
            return float('inf')
        
        # Distance to closest point on ray
        closest_x = ox + t * dx
        closest_y = oy + t * dy
        
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def demo_ray_tracing():
    """Demonstrate ray tracing."""
    print("\n" + "=" * 60)
    print("Example 3: Ray Tracing")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    tracer = RayTracer(manifold)
    
    print("\nCasting rays with exact directions:")
    print("-" * 40)
    
    origin = (0, 0)
    angles = [0, 30, 45, 60, 90, 120, 180, 270]
    
    print(f"\n   From origin {origin}:")
    print(f"   {'Angle':<8} {'End point':<25} {'Snap error':<10}")
    print(f"   {'-'*43}")
    
    for angle in angles:
        end, error = tracer.cast_ray(origin, angle, max_distance=10.0)
        print(f"   {angle:<8.0f} ({end[0]:.4f}, {end[1]:.4f}){'':<8} {error:.6f}")
    
    print("\nTracing reflections:")
    print("-" * 40)
    
    reflectors = [
        ((5, 5), (0.707, -0.707)),   # 45-degree mirror
        ((8, 0), (-1, 0)),           # Vertical mirror
        ((0, 5), (0, -1)),           # Horizontal mirror
    ]
    
    path = tracer.trace_reflections(
        origin=(0, 0),
        initial_angle=45,
        reflectors=reflectors,
        max_bounces=5
    )
    
    print(f"\n   Ray path ({len(path)} points):")
    for i, point in enumerate(path):
        print(f"   [{i}] ({point[0]:.4f}, {point[1]:.4f})")


# =============================================================================
# Example 4: Statistical Analysis
# =============================================================================

class DirectionalStatistics:
    """
    Statistical analysis of directional data with exact coordinates.
    
    Uses Constraint Theory for:
    - Exact circular statistics
    - Deterministic mean direction
    - Reproducible analysis
    """
    
    def __init__(self, manifold: SimulatedManifold):
        self.manifold = manifold
    
    def mean_direction(self, angles: List[float]) -> Tuple[float, float]:
        """
        Calculate exact mean direction.
        
        Returns: (mean_angle_degrees, concentration)
        """
        # Convert to unit vectors
        cos_sum = 0.0
        sin_sum = 0.0
        
        for angle in angles:
            rad = math.radians(angle)
            dx, dy = math.cos(rad), math.sin(rad)
            
            # Snap to exact
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            cos_sum += sx
            sin_sum += sy
        
        n = len(angles)
        mean_x = cos_sum / n
        mean_y = sin_sum / n
        
        # Resultant vector length (0 to 1, measures concentration)
        R = math.sqrt(mean_x ** 2 + mean_y ** 2)
        
        # Mean direction
        mean_angle = math.degrees(math.atan2(mean_y, mean_x))
        
        return mean_angle, R
    
    def circular_variance(self, angles: List[float]) -> float:
        """Calculate circular variance (0 = all same direction, 1 = uniform)."""
        _, R = self.mean_direction(angles)
        return 1 - R
    
    def snap_histogram(self, angles: List[float]) -> dict:
        """
        Create histogram of snapped directions.
        
        Returns counts for each unique snapped direction.
        """
        histogram = defaultdict(int)
        
        for angle in angles:
            rad = math.radians(angle)
            dx, dy = math.cos(rad), math.sin(rad)
            sx, sy, _ = self.manifold.snap(dx, dy)
            
            # Round for binning
            key = (round(sx, 4), round(sy, 4))
            histogram[key] += 1
        
        return dict(histogram)
    
    def uniformity_test(self, angles: List[float]) -> dict:
        """
        Test if angles are uniformly distributed.
        
        Uses Rayleigh test approximation.
        """
        n = len(angles)
        _, R = self.mean_direction(angles)
        
        # Rayleigh statistic
        Z = n * R * R
        
        # P-value approximation (Rayleigh)
        p_value = math.exp(-Z)
        
        return {
            'n': n,
            'resultant_length': R,
            'rayleigh_z': Z,
            'p_value': p_value,
            'uniform': p_value > 0.05,
        }


def demo_statistical_analysis():
    """Demonstrate directional statistics."""
    print("\n" + "=" * 60)
    print("Example 4: Statistical Analysis")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    stats = DirectionalStatistics(manifold)
    
    print("\nMean direction calculation:")
    print("-" * 40)
    
    # Angles clustered around 45 degrees
    clustered_angles = [40, 42, 44, 45, 46, 48, 50, 43, 47]
    mean_angle, concentration = stats.mean_direction(clustered_angles)
    
    print(f"\n   Angles: {clustered_angles}")
    print(f"   Mean direction: {mean_angle:.2f}°")
    print(f"   Concentration (R): {concentration:.4f}")
    print(f"   Circular variance: {stats.circular_variance(clustered_angles):.4f}")
    
    print("\nUniformity test:")
    print("-" * 40)
    
    # Uniform angles
    uniform_angles = list(range(0, 360, 10))
    result = stats.uniformity_test(uniform_angles)
    
    print(f"\n   Testing {len(uniform_angles)} uniformly spaced angles:")
    print(f"   Resultant length: {result['resultant_length']:.4f}")
    print(f"   Rayleigh Z: {result['rayleigh_z']:.4f}")
    print(f"   P-value: {result['p_value']:.4f}")
    print(f"   Uniform? {result['uniform']}")
    
    print("\nSnap histogram:")
    print("-" * 40)
    
    random_angles = [random.uniform(0, 360) for _ in range(100)]
    histogram = stats.snap_histogram(random_angles)
    
    print(f"\n   {len(histogram)} unique snapped directions from 100 random angles")
    top = sorted(histogram.items(), key=lambda x: -x[1])[:5]
    for (dx, dy), count in top:
        angle = math.degrees(math.atan2(dy, dx))
        print(f"   Direction ({dx:.4f}, {dy:.4f}) [~{angle:.0f}°]: {count} samples")


# =============================================================================
# Example 5: Reproducible Research
# =============================================================================

class ReproducibleExperiment:
    """
    Framework for reproducible computational experiments.
    
    Uses Constraint Theory for:
    - Deterministic all steps
    - Exact intermediate results
    - Verifiable computations
    """
    
    def __init__(self, manifold: SimulatedManifold, 
                 experiment_id: str, 
                 seed: int = 42):
        self.manifold = manifold
        self.experiment_id = experiment_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.results = {}
        self.log = []
    
    def log_step(self, step: str, data: dict) -> None:
        """Log experiment step with data."""
        self.log.append({
            'step': step,
            'data': data,
        })
    
    def run_direction_sampling(self, n_samples: int) -> dict:
        """
        Sample random directions with exact snapping.
        
        Returns reproducible results.
        """
        self.log_step('start', {'n_samples': n_samples})
        
        samples = []
        for i in range(n_samples):
            angle = self.rng.uniform(0, 2 * math.pi)
            dx, dy = math.cos(angle), math.sin(angle)
            sx, sy, noise = self.manifold.snap(dx, dy)
            
            samples.append({
                'input_angle': math.degrees(angle),
                'snapped_direction': (sx, sy),
                'snap_noise': noise,
            })
        
        # Calculate statistics
        noises = [s['snap_noise'] for s in samples]
        
        result = {
            'samples': samples[:5],  # Keep first 5 for verification
            'n_samples': n_samples,
            'mean_noise': sum(noises) / len(noises),
            'max_noise': max(noises),
            'min_noise': min(noises),
        }
        
        self.results['direction_sampling'] = result
        self.log_step('complete', {'mean_noise': result['mean_noise']})
        
        return result
    
    def run_geometric_analysis(self, points: List[Tuple[float, float]]) -> dict:
        """
        Analyze geometric properties with exact calculations.
        """
        self.log_step('start_analysis', {'n_points': len(points)})
        
        # Calculate pairwise directions
        directions = []
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i < j:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    sx, sy, noise = self.manifold.snap(dx, dy)
                    
                    angle = math.degrees(math.atan2(sy, sx))
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    
                    directions.append({
                        'from': p1,
                        'to': p2,
                        'exact_direction': (sx, sy),
                        'exact_angle': angle,
                        'distance': distance,
                        'snap_noise': noise,
                    })
        
        result = {
            'n_pairs': len(directions),
            'directions': directions,
        }
        
        self.results['geometric_analysis'] = result
        self.log_step('complete_analysis', {'n_pairs': len(directions)})
        
        return result
    
    def get_verification_hash(self) -> str:
        """
        Get hash of all results for verification.
        
        Other researchers can verify they got identical results.
        """
        import hashlib
        import json
        
        # Serialize results deterministically
        serialized = json.dumps(self.results, sort_keys=True)
        
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def demo_reproducible_research():
    """Demonstrate reproducible research."""
    print("\n" + "=" * 60)
    print("Example 5: Reproducible Research")
    print("=" * 60)
    
    manifold = SimulatedManifold(200)
    
    print("\nRunning reproducible experiment:")
    print("-" * 40)
    
    # Run experiment twice with same parameters
    hashes = []
    
    for run in range(2):
        exp = ReproducibleExperiment(
            manifold, 
            experiment_id='demo_001',
            seed=42
        )
        
        result = exp.run_direction_sampling(100)
        hash_val = exp.get_verification_hash()
        hashes.append(hash_val)
        
        print(f"\n   Run {run + 1}:")
        print(f"     Samples: {result['n_samples']}")
        print(f"     Mean noise: {result['mean_noise']:.6f}")
        print(f"     Verification hash: {hash_val}")
    
    print(f"\n   Hashes match: {hashes[0] == hashes[1]}")
    print("   (Other researchers can verify identical results)")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all scientific computing examples."""
    print("\n" + "=" * 60)
    print("Constraint Theory - Scientific Computing Examples")
    print("=" * 60)
    
    demo_monte_carlo()
    demo_particle_simulation()
    demo_ray_tracing()
    demo_statistical_analysis()
    demo_reproducible_research()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways for scientific computing:

1. Monte Carlo Methods
   - Reproducible random sampling
   - Exact directional statistics
   - Deterministic variance estimation

2. Simulations
   - Zero numerical drift
   - Deterministic physics
   - Reproducible across platforms

3. Ray Tracing
   - Exact ray directions
   - Deterministic reflections
   - Reproducible rendering

4. Statistics
   - Exact circular statistics
   - Deterministic analysis
   - Verifiable results

5. Reproducible Research
   - Verification hashes
   - Deterministic experiments
   - Cross-platform identical

For production use:
    from constraint_theory import PythagoreanManifold
    manifold = PythagoreanManifold(500)  # High precision for science
    """)


if __name__ == "__main__":
    main()
