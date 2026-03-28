#!/usr/bin/env python3
"""
Data Export Examples for Constraint Theory Python bindings.

This example demonstrates how to export constraint theory data for use
in web applications, other tools, and cross-platform sharing.

Run with: python examples/data_export.py

Requirements:
    pip install constraint-theory numpy
"""

import json
import csv
import numpy as np
from datetime import datetime
from constraint_theory import PythagoreanManifold, generate_triples


def export_manifold_json(density=200, filename='manifold.json'):
    """
    Export manifold data as JSON for web visualization.
    
    Creates a JSON file with Pythagorean triple points that can be
    directly consumed by JavaScript visualization libraries like D3.js.
    
    Parameters
    ----------
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    dict
        The exported data structure
    """
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    # Build points structure
    points = []
    for a, b, c in triples:
        # Add points in all quadrants
        points.append({
            'x': a / c,
            'y': b / c,
            'quadrant': 1,
            'triple': {'a': a, 'b': b, 'c': c}
        })
        points.append({
            'x': -a / c,
            'y': b / c,
            'quadrant': 2,
            'triple': {'a': a, 'b': b, 'c': c}
        })
        points.append({
            'x': -a / c,
            'y': -b / c,
            'quadrant': 3,
            'triple': {'a': a, 'b': b, 'c': c}
        })
        points.append({
            'x': a / c,
            'y': -b / c,
            'quadrant': 4,
            'triple': {'a': a, 'b': b, 'c': c}
        })
    
    data = {
        'metadata': {
            'density': density,
            'state_count': manifold.state_count,
            'point_count': len(points),
            'exported_at': datetime.now().isoformat(),
            'version': '0.1.0'
        },
        'points': points
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(points)} manifold points to {filename}")
    return data


def export_snap_results_json(vectors, density=200, filename='snap_results.json'):
    """
    Export snap results as JSON for web visualization.
    
    Parameters
    ----------
    vectors : array-like
        Input vectors to snap
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    dict
        The exported data structure
    """
    manifold = PythagoreanManifold(density)
    results = manifold.snap_batch(vectors)
    
    data = {
        'metadata': {
            'density': density,
            'state_count': manifold.state_count,
            'vector_count': len(vectors),
            'exported_at': datetime.now().isoformat()
        },
        'results': [
            {
                'input': {'x': float(v[0]), 'y': float(v[1])},
                'snapped': {'x': float(r[0]), 'y': float(r[1])},
                'noise': float(r[2])
            }
            for v, r in zip(vectors, results)
        ]
    }
    
    # Add statistics
    noises = [r['noise'] for r in data['results']]
    data['statistics'] = {
        'mean_noise': float(np.mean(noises)),
        'std_noise': float(np.std(noises)),
        'min_noise': float(np.min(noises)),
        'max_noise': float(np.max(noises)),
        'median_noise': float(np.median(noises))
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(vectors)} snap results to {filename}")
    return data


def export_webgl_buffer(density=200, filename='manifold_webgl.json'):
    """
    Export manifold as interleaved buffer for WebGL.
    
    Creates an optimized data structure for GPU rendering with
    interleaved position and color data.
    
    Parameters
    ----------
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    dict
        The exported data structure
    """
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    # Create interleaved buffer: x, y, r, g, b, a
    # Each point has position (x, y) and color (r, g, b, a)
    vertices = []
    
    for a, b, c in triples:
        for sx, sy in [(a/c, b/c), (-a/c, b/c), (a/c, -b/c), (-a/c, -b/c)]:
            # Color based on angle for visual variety
            angle = np.arctan2(sy, sx)
            
            # HSV to RGB conversion for nice colors
            hue = (angle + np.pi) / (2 * np.pi)  # 0 to 1
            r, g, b = _hsv_to_rgb(hue, 0.7, 0.9)
            
            vertices.extend([float(sx), float(sy), r, g, b, 1.0])
    
    data = {
        'metadata': {
            'format': 'interleaved',
            'stride': 6,  # x, y, r, g, b, a
            'vertexCount': len(vertices) // 6,
            'density': density,
            'state_count': manifold.state_count
        },
        'vertices': vertices
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Exported {len(vertices) // 6} WebGL vertices to {filename}")
    return data


def export_threejs_points(density=200, filename='manifold_threejs.json'):
    """
    Export manifold in Three.js Points geometry format.
    
    Parameters
    ----------
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    dict
        The exported data structure
    """
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    positions = []
    colors = []
    
    for a, b, c in triples:
        for sx, sy in [(a/c, b/c), (-a/c, b/c), (a/c, -b/c), (-a/c, -b/c)]:
            positions.append(float(sx))
            positions.append(float(sy))
            positions.append(0.0)  # z = 0 for 2D
            
            # Color based on position
            angle = np.arctan2(sy, sx)
            hue = (angle + np.pi) / (2 * np.pi)
            r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(r)
            colors.append(g)
            colors.append(b)
    
    data = {
        'metadata': {
            'version': 4.6,
            'type': 'BufferGeometry',
            'generator': 'constraint-theory-export'
        },
        'data': {
            'attributes': {
                'position': {
                    'itemSize': 3,
                    'type': 'Float32Array',
                    'array': positions,
                    'normalized': False
                },
                'color': {
                    'itemSize': 3,
                    'type': 'Float32Array',
                    'array': colors,
                    'normalized': False
                }
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Exported Three.js geometry to {filename}")
    return data


def export_csv_vectors(vectors, density=200, filename='snap_results.csv'):
    """
    Export snap results as CSV for spreadsheet tools.
    
    Parameters
    ----------
    vectors : array-like
        Input vectors to snap
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    str
        The CSV content
    """
    manifold = PythagoreanManifold(density)
    results = manifold.snap_batch(vectors)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['input_x', 'input_y', 'snapped_x', 'snapped_y', 'noise'])
        
        # Data rows
        for v, r in zip(vectors, results):
            writer.writerow([v[0], v[1], r[0], r[1], r[2]])
    
    print(f"Exported {len(vectors)} results to {filename}")
    return filename


def export_numpy_arrays(vectors, density=200, prefix='manifold_data'):
    """
    Export data as NumPy binary files for efficient Python loading.
    
    Parameters
    ----------
    vectors : array-like
        Input vectors to snap
    density : int
        Manifold density
    prefix : str
        Filename prefix
    
    Returns
    -------
    dict
        Paths to exported files
    """
    manifold = PythagoreanManifold(density)
    results = manifold.snap_batch(vectors)
    
    # Convert to arrays
    input_array = np.array(vectors, dtype=np.float32)
    snapped_array = np.array([[r[0], r[1]] for r in results], dtype=np.float32)
    noise_array = np.array([r[2] for r in results], dtype=np.float32)
    
    # Save
    paths = {}
    paths['input'] = f'{prefix}_input.npy'
    paths['snapped'] = f'{prefix}_snapped.npy'
    paths['noise'] = f'{prefix}_noise.npy'
    
    np.save(paths['input'], input_array)
    np.save(paths['snapped'], snapped_array)
    np.save(paths['noise'], noise_array)
    
    print(f"Exported NumPy arrays to {prefix}_*.npy")
    return paths


def export_for_web_demo(density=200, filename='web_demo_data.json'):
    """
    Export complete data package for the interactive web demo.
    
    This creates a self-contained JSON file with everything needed
    for the constraint-theory.superinstance.ai demo.
    
    Parameters
    ----------
    density : int
        Manifold density
    filename : str
        Output filename
    
    Returns
    -------
    dict
        The exported data structure
    """
    manifold = PythagoreanManifold(density)
    triples = generate_triples(density)
    
    # Generate sample snap demonstrations
    sample_inputs = [
        (0.577, 0.816),
        (0.707, 0.707),
        (0.1, 0.995),
        (0.866, 0.5),
        (0.259, 0.966)
    ]
    
    demonstrations = []
    for x, y in sample_inputs:
        sx, sy, noise = manifold.snap(x, y)
        demonstrations.append({
            'input': {'x': x, 'y': y},
            'snapped': {'x': float(sx), 'y': float(sy)},
            'noise': float(noise),
            'description': f'Input ({x:.3f}, {y:.3f}) snaps to ({sx:.4f}, {sy:.4f})'
        })
    
    # Manifold points for visualization
    points = []
    for a, b, c in triples:
        points.append({'x': float(a/c), 'y': float(b/c), 'c': int(c)})
    
    data = {
        'version': '1.0',
        'metadata': {
            'density': density,
            'state_count': manifold.state_count,
            'point_count': len(points),
            'exported_at': datetime.now().isoformat(),
            'library_version': '0.1.0'
        },
        'manifold': {
            'points': points,
            'bounds': {'min': -1, 'max': 1}
        },
        'demonstrations': demonstrations,
        'documentation': {
            'description': 'Pythagorean manifold for deterministic vector snapping',
            'usage': 'Snap input vectors to nearest Pythagorean triple coordinates',
            'api': {
                'snap': {'params': ['x', 'y'], 'returns': ['snapped_x', 'snapped_y', 'noise']},
                'snap_batch': {'params': ['vectors'], 'returns': 'list of snap results'}
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported web demo data to {filename}")
    return data


def _hsv_to_rgb(h, s, v):
    """Convert HSV to RGB color space."""
    if s == 0.0:
        return (v, v, v)
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)
    return (v, v, v)


def main():
    """Run all export examples."""
    print("=" * 60)
    print("Constraint Theory - Data Export Examples")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, 100)
    sample_vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    
    print("\n1. Export Manifold as JSON")
    print("-" * 40)
    export_manifold_json(density=200, filename='manifold.json')
    
    print("\n2. Export Snap Results as JSON")
    print("-" * 40)
    export_snap_results_json(sample_vectors[:10], density=200, 
                            filename='snap_results.json')
    
    print("\n3. Export WebGL Buffer")
    print("-" * 40)
    export_webgl_buffer(density=200, filename='manifold_webgl.json')
    
    print("\n4. Export Three.js Points")
    print("-" * 40)
    export_threejs_points(density=200, filename='manifold_threejs.json')
    
    print("\n5. Export CSV")
    print("-" * 40)
    export_csv_vectors(sample_vectors[:20], density=200, 
                       filename='snap_results.csv')
    
    print("\n6. Export NumPy Arrays")
    print("-" * 40)
    export_numpy_arrays(sample_vectors, density=200, prefix='manifold_data')
    
    print("\n7. Export Web Demo Data")
    print("-" * 40)
    export_for_web_demo(density=200, filename='web_demo_data.json')
    
    print("\n" + "=" * 60)
    print("All exports complete!")
    print("=" * 60)
    
    print("""
Exported Files:
- manifold.json        : General-purpose JSON format
- snap_results.json    : Snap results with metadata
- manifold_webgl.json  : Optimized for WebGL rendering
- manifold_threejs.json: Three.js BufferGeometry format
- snap_results.csv     : Spreadsheet-compatible CSV
- manifold_data_*.npy  : NumPy binary arrays
- web_demo_data.json   : Complete package for web demo

Usage Tips:
- Use JSON for web applications and APIs
- Use CSV for Excel/Sheets analysis
- Use NumPy arrays for Python-only workflows
- Use WebGL/Three.js formats for 3D visualization
    """)


if __name__ == "__main__":
    main()
