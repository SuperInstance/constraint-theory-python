#!/usr/bin/env python3
"""
Visualization example for Constraint Theory Python bindings.

This example demonstrates various visualization techniques using matplotlib
for understanding and presenting the constraint theory manifold.

Run with: python examples/visualization.py

Requirements:
    pip install constraint-theory matplotlib numpy
"""

import numpy as np
from constraint_theory import PythagoreanManifold, generate_triples

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PathCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def plot_manifold_2d(density=200, figsize=(10, 10), save_path=None):
    """
    Visualize the 2D Pythagorean manifold.
    
    Parameters
    ----------
    density : int
        Density parameter for the manifold
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create manifold
    manifold = PythagoreanManifold(density)
    
    # Draw unit circle
    unit_circle = Circle((0, 0), 1, fill=False, color='blue', 
                          alpha=0.3, linestyle='--', linewidth=2)
    ax.add_patch(unit_circle)
    
    # Get and plot Pythagorean points
    triples = generate_triples(density)
    
    # Plot in all quadrants
    for a, b, c in triples:
        for sx, sy in [(a/c, b/c), (-a/c, b/c), (a/c, -b/c), (-a/c, -b/c)]:
            ax.scatter(sx, sy, c='green', s=15, alpha=0.6)
    
    # Add axis lines
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Labels and title
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Pythagorean Manifold (density={density}, {manifold.state_count} states)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_snap_visualization(x, y, density=200, figsize=(8, 8), save_path=None):
    """
    Visualize a single snap operation.
    
    Parameters
    ----------
    x, y : float
        Input coordinates
    density : int
        Manifold density
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create manifold and snap
    manifold = PythagoreanManifold(density)
    sx, sy, noise = manifold.snap(x, y)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', alpha=0.3, linewidth=2)
    
    # Plot some manifold points for context
    triples = generate_triples(min(density, 100))
    for a, b, c in triples:
        ax.scatter(a/c, b/c, c='lightgray', s=10, alpha=0.3)
        ax.scatter(-a/c, b/c, c='lightgray', s=10, alpha=0.3)
        ax.scatter(a/c, -b/c, c='lightgray', s=10, alpha=0.3)
        ax.scatter(-a/c, -b/c, c='lightgray', s=10, alpha=0.3)
    
    # Plot input point
    ax.scatter(x, y, c='red', s=150, marker='x', linewidths=3, 
               label=f'Input: ({x:.4f}, {y:.4f})', zorder=10)
    
    # Plot snapped point
    ax.scatter(sx, sy, c='green', s=200, marker='*', 
               label=f'Snapped: ({sx:.4f}, {sy:.4f})', zorder=10)
    
    # Draw snap line
    ax.plot([x, sx], [y, sy], 'r--', linewidth=2, alpha=0.7,
            label=f'Snap distance: {noise:.6f}')
    
    # Annotations
    ax.annotate(f'noise = {noise:.6f}', 
                xy=(sx, sy), xytext=(sx + 0.15, sy + 0.15),
                fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Set up axes
    limit = max(1.5, abs(x) + 0.3, abs(y) + 0.3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Vector Snap (density={density})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return {'input': (x, y), 'snapped': (sx, sy), 'noise': noise}


def plot_noise_heatmap(density=200, resolution=50, figsize=(10, 8), save_path=None):
    """
    Create a heatmap of snap noise across the input space.
    
    Parameters
    ----------
    density : int
        Manifold density
    resolution : int
        Grid resolution
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    manifold = PythagoreanManifold(density)
    
    # Create grid
    x = np.linspace(-1.5, 1.5, resolution)
    y = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute noise for each point
    noise_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            _, _, noise = manifold.snap(X[i, j], Y[i, j])
            noise_grid[i, j] = noise
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(noise_grid, extent=[-1.5, 1.5, -1.5, 1.5], 
                   origin='lower', cmap='viridis', aspect='equal')
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'w--', linewidth=2, alpha=0.7)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Snap Noise')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Noise Heatmap (density={density})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return noise_grid


def plot_noise_distribution(density=200, n_samples=10000, seed=42, 
                            figsize=(12, 4), save_path=None):
    """
    Plot the distribution of snap noise.
    
    Parameters
    ----------
    density : int
        Manifold density
    n_samples : int
        Number of random samples
    seed : int
        Random seed
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    np.random.seed(seed)
    manifold = PythagoreanManifold(density)
    
    # Generate random unit vectors
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Snap all vectors
    results = manifold.snap_batch(vectors)
    noises = np.array([noise for _, _, noise in results])
    
    # Create figure with multiple plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram
    axes[0].hist(noises, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(noises.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {noises.mean():.4f}')
    axes[0].set_xlabel('Snap Noise')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Histogram')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(noises, vert=True)
    axes[1].set_ylabel('Snap Noise')
    axes[1].set_title('Box Plot')
    axes[1].set_xticklabels([''])
    
    # CDF
    sorted_noises = np.sort(noises)
    cdf = np.arange(1, len(sorted_noises) + 1) / len(sorted_noises)
    axes[2].plot(sorted_noises, cdf, linewidth=2)
    axes[2].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='P50')
    axes[2].axhline(0.9, color='green', linestyle='--', alpha=0.7, label='P90')
    axes[2].set_xlabel('Snap Noise')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('CDF')
    axes[2].legend()
    
    fig.suptitle(f'Noise Distribution (density={density}, n={n_samples:,})', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nNoise Statistics:")
    print(f"  Mean:   {noises.mean():.6f}")
    print(f"  Std:    {noises.std():.6f}")
    print(f"  Min:    {noises.min():.6f}")
    print(f"  P25:    {np.percentile(noises, 25):.6f}")
    print(f"  P50:    {np.percentile(noises, 50):.6f}")
    print(f"  P75:    {np.percentile(noises, 75):.6f}")
    print(f"  P90:    {np.percentile(noises, 90):.6f}")
    print(f"  Max:    {noises.max():.6f}")
    
    return noises


def plot_density_comparison(densities=[50, 100, 200, 500], n_samples=5000,
                           figsize=(12, 5), save_path=None):
    """
    Compare noise distributions across different densities.
    
    Parameters
    ----------
    densities : list
        List of density values to compare
    n_samples : int
        Number of samples per density
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    np.random.seed(42)
    
    # Generate test vectors once
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Collect data
    data = []
    for density in densities:
        manifold = PythagoreanManifold(density)
        results = manifold.snap_batch(vectors)
        noises = [noise for _, _, noise in results]
        data.append({
            'density': density,
            'state_count': manifold.state_count,
            'noises': noises
        })
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Violin/box plot
    axes[0].boxplot([d['noises'] for d in data],
                    labels=[f"d={d['density']}\n({d['state_count']} s)" for d in data])
    axes[0].set_ylabel('Snap Noise')
    axes[0].set_title('Noise by Density')
    
    # Line plot of mean noise
    means = [np.mean(d['noises']) for d in data]
    stds = [np.std(d['noises']) for d in data]
    
    axes[1].errorbar(densities, means, yerr=stds, fmt='o-', capsize=5, 
                     markersize=10, linewidth=2)
    axes[1].set_xlabel('Density')
    axes[1].set_ylabel('Mean Noise')
    axes[1].set_title('Mean Noise vs Density')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print comparison table
    print("\nDensity Comparison:")
    print("-" * 65)
    print(f"{'Density':>8} {'States':>10} {'Mean':>10} {'Std':>10} {'P90':>10}")
    print("-" * 65)
    for d in data:
        print(f"{d['density']:>8} {d['state_count']:>10} "
              f"{np.mean(d['noises']):>10.6f} {np.std(d['noises']):>10.6f} "
              f"{np.percentile(d['noises'], 90):>10.6f}")
    
    return data


def plot_triple_distribution(max_c=200, figsize=(10, 6), save_path=None):
    """
    Visualize the distribution of Pythagorean triples.
    
    Parameters
    ----------
    max_c : int
        Maximum hypotenuse
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    triples = generate_triples(max_c)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot of a vs b
    a_vals = [t[0] for t in triples]
    b_vals = [t[1] for t in triples]
    c_vals = [t[2] for t in triples]
    
    scatter = axes[0].scatter(a_vals, b_vals, c=c_vals, cmap='viridis', 
                              alpha=0.6, s=30)
    axes[0].set_xlabel('a')
    axes[0].set_ylabel('b')
    axes[0].set_title('Pythagorean Triples (a² + b² = c²)')
    plt.colorbar(scatter, ax=axes[0], label='c (hypotenuse)')
    
    # Histogram of c values
    axes[1].hist(c_vals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Hypotenuse (c)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Distribution of Hypotenuse Values (n={len(triples)})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return triples


def main():
    """Run all visualization examples."""
    print("=" * 60)
    print("Constraint Theory - Visualization Examples")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("\nInstall matplotlib to run these examples:")
        print("  pip install matplotlib")
        return
    
    print("\n1. Manifold 2D Visualization")
    print("-" * 40)
    plot_manifold_2d(density=200)
    
    print("\n2. Single Snap Visualization")
    print("-" * 40)
    plot_snap_visualization(0.577, 0.816, density=200)
    
    print("\n3. Noise Heatmap")
    print("-" * 40)
    plot_noise_heatmap(density=200, resolution=50)
    
    print("\n4. Noise Distribution")
    print("-" * 40)
    plot_noise_distribution(density=200, n_samples=10000)
    
    print("\n5. Density Comparison")
    print("-" * 40)
    plot_density_comparison()
    
    print("\n6. Triple Distribution")
    print("-" * 40)
    plot_triple_distribution(max_c=200)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
