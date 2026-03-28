# Jupyter Integration Guide

This guide covers using Constraint Theory in Jupyter notebooks for interactive analysis and visualization.

## Table of Contents

- [Installation](#installation)
- [Quick Start Notebook](#quick-start-notebook)
- [Visualization Examples](#visualization-examples)
- [Interactive Widgets](#interactive-widgets)
- [Data Export for Web](#data-export-for-web)
- [Best Practices](#best-practices)

---

## Installation

### Basic Installation

```bash
pip install constraint-theory jupyter
```

### Full Installation with Visualization

```bash
pip install constraint-theory[numpy] jupyter matplotlib numpy
```

### JupyterLab (Recommended)

```bash
pip install constraint-theory jupyterlab matplotlib numpy ipywidgets
jupyter lab
```

---

## Quick Start Notebook

Create a new Jupyter notebook and paste the following cells:

### Cell 1: Setup

```python
# Imports
import numpy as np
import matplotlib.pyplot as plt
from constraint_theory import PythagoreanManifold, generate_triples

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

print("Constraint Theory - Jupyter Integration")
print("=" * 40)
```

### Cell 2: Basic Usage

```python
# Create manifold
manifold = PythagoreanManifold(density=200)
print(f"Manifold created with {manifold.state_count} states")

# Snap a vector
x, y, noise = manifold.snap(0.577, 0.816)
print(f"\nInput:    (0.577, 0.816)")
print(f"Snapped:  ({x:.4f}, {y:.4f})")
print(f"Noise:    {noise:.6f}")

# Verify unit norm
print(f"\nNorm: {np.sqrt(x**2 + y**2):.10f} (should be 1.0)")
```

### Cell 3: Visualization

```python
# Visualize the manifold
fig, ax = plt.subplots(figsize=(10, 10))

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3, label='Unit Circle')

# Get Pythagorean triples
triples = generate_triples(200)
print(f"Generated {len(triples)} triples with c <= 200")

# Plot Pythagorean points
for a, b, c in triples[:50]:  # First 50 triples
    ax.scatter(a/c, b/c, c='green', s=20, alpha=0.7)
    ax.scatter(-a/c, b/c, c='green', s=20, alpha=0.7)
    ax.scatter(a/c, -b/c, c='green', s=20, alpha=0.7)
    ax.scatter(-a/c, -b/c, c='green', s=20, alpha=0.7)

# Plot a snap example
input_vec = np.array([0.577, 0.816])
snapped = np.array([x, y])

ax.scatter(*input_vec, c='red', s=100, marker='x', label='Input', zorder=5)
ax.scatter(*snapped, c='blue', s=100, marker='*', label='Snapped', zorder=5)
ax.plot([input_vec[0], snapped[0]], [input_vec[1], snapped[1]], 
        'r--', alpha=0.5, label=f'Snap Distance: {noise:.4f}')

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.legend()
ax.set_title('Pythagorean Manifold Visualization')

plt.tight_layout()
plt.show()
```

---

## Visualization Examples

### 1. Noise Distribution Analysis

```python
def analyze_noise_distribution(density=200, n_samples=10000, seed=42):
    """Analyze the distribution of snapping noise."""
    np.random.seed(seed)
    
    # Create manifold
    manifold = PythagoreanManifold(density)
    
    # Generate random unit vectors
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Snap all vectors
    results = manifold.snap_batch(vectors)
    noises = np.array([noise for _, _, noise in results])
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(noises, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(noises.mean(), color='red', linestyle='--', 
                    label=f'Mean: {noises.mean():.4f}')
    axes[0].axvline(np.median(noises), color='green', linestyle='--',
                    label=f'Median: {np.median(noises):.4f}')
    axes[0].set_xlabel('Snap Noise')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Noise Distribution (density={density})')
    axes[0].legend()
    
    # CDF
    sorted_noises = np.sort(noises)
    cdf = np.arange(1, len(sorted_noises) + 1) / len(sorted_noises)
    axes[1].plot(sorted_noises, cdf)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(0.9, color='green', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Snap Noise')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Noise CDF')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Statistics for density={density}:")
    print(f"  Mean:   {noises.mean():.6f}")
    print(f"  Std:    {noises.std():.6f}")
    print(f"  Min:    {noises.min():.6f}")
    print(f"  Max:    {noises.max():.6f}")
    print(f"  P50:    {np.percentile(noises, 50):.6f}")
    print(f"  P90:    {np.percentile(noises, 90):.6f}")
    print(f"  P99:    {np.percentile(noises, 99):.6f}")
    
    return noises

# Run analysis
noises = analyze_noise_distribution(density=200)
```

### 2. Density Comparison

```python
def compare_densities(densities=[50, 100, 200, 500], n_samples=5000):
    """Compare noise distributions across different densities."""
    np.random.seed(42)
    
    # Generate test vectors once
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Collect results
    all_noises = {}
    state_counts = {}
    
    for density in densities:
        manifold = PythagoreanManifold(density)
        state_counts[density] = manifold.state_count
        
        results = manifold.snap_batch(vectors)
        all_noises[density] = np.array([noise for _, _, noise in results])
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot([all_noises[d] for d in densities], 
                    labels=[f'd={d}\n({state_counts[d]} states)' for d in densities])
    axes[0].set_ylabel('Snap Noise')
    axes[0].set_title('Noise by Density')
    
    # Mean noise vs density
    means = [all_noises[d].mean() for d in densities]
    axes[1].plot(densities, means, 'o-', markersize=10)
    axes[1].set_xlabel('Density')
    axes[1].set_ylabel('Mean Noise')
    axes[1].set_title('Mean Noise vs Density')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\nDensity Comparison:")
    print("-" * 70)
    print(f"{'Density':>8} {'States':>10} {'Mean':>10} {'Std':>10} {'P90':>10} {'Max':>10}")
    print("-" * 70)
    for d in densities:
        print(f"{d:>8} {state_counts[d]:>10} {all_noises[d].mean():>10.6f} "
              f"{all_noises[d].std():>10.6f} {np.percentile(all_noises[d], 90):>10.6f} "
              f"{all_noises[d].max():>10.6f}")

compare_densities()
```

### 3. Interactive Snap Demonstration

```python
from ipywidgets import interact, FloatSlider, IntSlider

def interactive_snap(x=0.577, y=0.816, density=200):
    """Interactive visualization of snapping."""
    manifold = PythagoreanManifold(density)
    sx, sy, noise = manifold.snap(x, y)
    
    # Normalize input for display
    mag = np.sqrt(x**2 + y**2)
    if mag > 0:
        nx, ny = x/mag, y/mag
    else:
        nx, ny = x, y
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3)
    
    # Input point
    ax.scatter(x, y, c='red', s=150, marker='x', label=f'Input: ({x:.3f}, {y:.3f})', zorder=5)
    
    # Snapped point
    ax.scatter(sx, sy, c='green', s=200, marker='*', 
               label=f'Snapped: ({sx:.4f}, {sy:.4f})', zorder=5)
    
    # Snap line
    ax.plot([x, sx], [y, sy], 'r--', alpha=0.7, linewidth=2)
    
    # Annotation
    ax.annotate(f'noise = {noise:.6f}', xy=(sx, sy), xytext=(sx+0.1, sy+0.1),
                fontsize=12, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Interactive Snap (density={density}, states={manifold.state_count})')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input:    ({x:.6f}, {y:.6f})")
    print(f"Snapped:  ({sx:.6f}, {sy:.6f})")
    print(f"Noise:    {noise:.6f}")
    print(f"Norm:     {np.sqrt(sx**2 + sy**2):.10f}")

# Create interactive widget
interact(interactive_snap,
         x=FloatSlider(min=-2, max=2, step=0.01, value=0.577, description='X:'),
         y=FloatSlider(min=-2, max=2, step=0.01, value=0.816, description='Y:'),
         density=IntSlider(min=50, max=1000, step=50, value=200, description='Density:'));
```

---

## Interactive Widgets

### Performance Benchmark Widget

```python
import time
from ipywidgets import HBox, VBox, Output, Button, IntText, Label

class PerformanceBenchmark:
    def __init__(self):
        self.output = Output()
        self.n_vectors = IntText(value=10000, description='Vectors:')
        self.density = IntText(value=200, description='Density:')
        self.run_button = Button(description='Run Benchmark')
        self.run_button.on_click(self.run_benchmark)
        
        display(VBox([
            HBox([self.n_vectors, self.density, self.run_button]),
            self.output
        ]))
    
    def run_benchmark(self, _):
        with self.output:
            self.output.clear_output()
            
            n = self.n_vectors.value
            density = self.density.value
            
            print(f"Benchmarking with {n:,} vectors, density={density}")
            print("-" * 50)
            
            # Create manifold
            start = time.time()
            manifold = PythagoreanManifold(density)
            create_time = (time.time() - start) * 1000
            print(f"Manifold creation: {create_time:.2f} ms")
            print(f"State count: {manifold.state_count}")
            
            # Generate vectors
            np.random.seed(42)
            angles = np.random.uniform(0, 2*np.pi, n)
            vectors = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Single snap benchmark
            start = time.time()
            for _ in range(min(n, 1000)):
                manifold.snap(0.577, 0.816)
            single_time = (time.time() - start) / min(n, 1000) * 1e6
            print(f"Single snap: {single_time:.1f} ns")
            
            # Batch benchmark
            start = time.time()
            results = manifold.snap_batch(vectors)
            batch_time = (time.time() - start) * 1e6 / n
            print(f"Batch snap: {batch_time:.1f} ns per vector")
            print(f"Speedup: {single_time / batch_time:.1f}x")
            
            # Throughput
            throughput = n / (batch_time * n / 1e6)
            print(f"Throughput: {throughput:,.0f} vectors/second")

benchmark = PerformanceBenchmark()
```

---

## Data Export for Web

### Export Results as JSON

```python
import json

def export_snap_results(vectors, manifold, filename='snap_results.json'):
    """Export snap results for web visualization."""
    results = manifold.snap_batch(vectors)
    
    data = {
        'metadata': {
            'density': manifold.density,
            'state_count': manifold.state_count,
            'n_vectors': len(vectors)
        },
        'vectors': [
            {
                'input': [float(v[0]), float(v[1])],
                'snapped': [float(r[0]), float(r[1])],
                'noise': float(r[2])
            }
            for v, r in zip(vectors, results)
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(vectors)} results to {filename}")
    return data

# Example usage
manifold = PythagoreanManifold(200)
angles = np.linspace(0, 2*np.pi, 100)
vectors = np.column_stack([np.cos(angles), np.sin(angles)])
export_snap_results(vectors, manifold);
```

### Export Manifold Points for D3.js

```python
def export_manifold_for_d3(density=200, filename='manifold.json'):
    """Export manifold points for D3.js visualization."""
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    # Create points in all quadrants
    points = []
    for a, b, c in triples:
        for sx, sy in [(a/c, b/c), (-a/c, b/c), (a/c, -b/c), (-a/c, -b/c)]:
            points.append({
                'x': float(sx),
                'y': float(sy),
                'triple': [int(a), int(b), int(c)]
            })
    
    data = {
        'density': density,
        'state_count': manifold.state_count,
        'points': points
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(points)} manifold points to {filename}")

export_manifold_for_d3();
```

### Export for WebGL/Three.js

```python
def export_manifold_for_webgl(density=200, filename='manifold_buffer.json'):
    """Export manifold as interleaved buffer for WebGL."""
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    # Create interleaved buffer: x, y, r, g, b (position + color)
    vertices = []
    for a, b, c in triples:
        for sx, sy in [(a/c, b/c), (-a/c, b/c), (a/c, -b/c), (-a/c, -b/c)]:
            # Color based on angle
            angle = np.arctan2(sy, sx)
            r = (np.sin(angle) + 1) / 2
            g = (np.cos(angle) + 1) / 2
            b = 0.5
            vertices.extend([float(sx), float(sy), r, g, b])
    
    data = {
        'format': 'interleaved',
        'stride': 5,  # x, y, r, g, b
        'vertexCount': len(vertices) // 5,
        'data': vertices
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Exported {len(vertices) // 5} vertices to {filename}")

export_manifold_for_webgl();
```

---

## Best Practices

### 1. Reuse Manifold Instances

```python
# BAD: Create manifold repeatedly
def process_vectors_bad(vectors_list):
    results = []
    for vectors in vectors_list:
        manifold = PythagoreanManifold(200)  # Slow!
        results.append(manifold.snap_batch(vectors))
    return results

# GOOD: Reuse manifold
def process_vectors_good(vectors_list):
    manifold = PythagoreanManifold(200)  # Create once
    results = []
    for vectors in vectors_list:
        results.append(manifold.snap_batch(vectors))
    return results
```

### 2. Use Batch Operations

```python
# BAD: Individual snaps
results = [manifold.snap(x, y) for x, y in vectors]

# GOOD: Batch operation
results = manifold.snap_batch(vectors)
```

### 3. Handle Large Datasets in Chunks

```python
def process_large_dataset(vectors, chunk_size=50000):
    """Process large datasets without memory issues."""
    manifold = PythagoreanManifold(200)
    results = []
    
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i + chunk_size]
        chunk_results = manifold.snap_batch(chunk)
        results.extend(chunk_results)
    
    return results
```

### 4. Profile Your Notebooks

```python
# Use Jupyter magic commands
%load_ext line_profiler

def profile_snap(n=10000):
    manifold = PythagoreanManifold(200)
    vectors = np.random.randn(n, 2)
    return manifold.snap_batch(vectors)

%lprun -f profile_snap profile_snap()
```

---

## See Also

- [NumPy Integration Example](../examples/numpy_integration.py)
- [Web Demos](https://constraint-theory.superinstance.ai)
- [API Reference](API.md)
