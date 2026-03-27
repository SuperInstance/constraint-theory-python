//! Python bindings for Constraint Theory
//!
//! This module provides Python access to the Constraint Theory Rust library
//! via PyO3 bindings.

use pyo3::prelude::*;
use pyo3::types::PyList;

use constraint_theory_core::{PythagoreanManifold, snap as rust_snap};

/// A Pythagorean manifold for deterministic vector snapping
#[pyclass(name = "PythagoreanManifold")]
pub struct PyManifold {
    inner: PythagoreanManifold,
}

#[pymethods]
impl PyManifold {
    /// Create a new Pythagorean manifold with specified density
    ///
    /// Args:
    ///     density: Maximum value of m in Euclid's formula (controls resolution)
    ///
    /// Returns:
    ///     New manifold with pre-computed valid states
    #[new]
    pub fn new(density: usize) -> Self {
        PyManifold {
            inner: PythagoreanManifold::new(density),
        }
    }

    /// Get the number of valid states in the manifold
    #[getter]
    pub fn state_count(&self) -> usize {
        self.inner.state_count()
    }

    /// Snap a 2D vector to the nearest Pythagorean triple
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///
    /// Returns:
    ///     Tuple of (snapped_x, snapped_y, noise) where noise is 1 - resonance
    pub fn snap(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let (snapped, noise) = self.inner.snap([x, y]);
        (snapped[0], snapped[1], noise)
    }

    /// Snap multiple vectors at once using SIMD (more efficient for large batches)
    ///
    /// Args:
    ///     vectors: List of (x, y) tuples
    ///
    /// Returns:
    ///     List of (snapped_x, snapped_y, noise) tuples
    pub fn snap_batch_simd(&self, py: Python<'_>, vectors: &PyList) -> PyResult<Vec<(f32, f32, f32)>> {
        let input: Vec<[f32; 2]> = vectors
            .iter()
            .map(|item| {
                let tuple: (f32, f32) = item.extract()?;
                Ok([tuple.0, tuple.1])
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        py.allow_threads(|| {
            let results = self.inner.snap_batch_simd(&input);
            Ok(results.into_iter().map(|(s, n)| (s[0], s[1], n)).collect())
        })
    }

    /// Get all valid states in the manifold
    ///
    /// Returns:
    ///     List of (x, y) tuples representing valid Pythagorean coordinates
    pub fn states(&self) -> Vec<(f32, f32)> {
        self.inner.states().iter().map(|s| (s[0], s[1])).collect()
    }

    /// Get a human-readable string representation
    fn __repr__(&self) -> String {
        format!(
            "PythagoreanManifold(states={})",
            self.inner.state_count()
        )
    }

    /// Get a short string representation
    fn __str__(&self) -> String {
        format!("Manifold({} states)", self.inner.state_count())
    }
}

/// Snap a vector to the nearest Pythagorean triple using a default manifold
///
/// Args:
///     x: X coordinate
///     y: Y coordinate
///     density: Optional density parameter (default: 200)
///
/// Returns:
///     Tuple of (snapped_x, snapped_y, noise)
#[pyfunction]
#[pyo3(signature = (x, y, density=200))]
pub fn snap(x: f32, y: f32, density: usize) -> (f32, f32, f32) {
    let manifold = PythagoreanManifold::new(density);
    let (snapped, noise) = rust_snap(&manifold.inner, [x, y]);
    (snapped[0], snapped[1], noise)
}

/// Generate Pythagorean triples up to a maximum hypotenuse
///
/// Args:
///     max_c: Maximum hypotenuse value
///
/// Returns:
///     List of (a, b, c) tuples where a² + b² = c²
#[pyfunction]
pub fn generate_triples(max_c: i32) -> Vec<(i32, i32, i32)> {
    let mut triples = Vec::new();
    
    // Use Euclid's formula: a = m² - n², b = 2mn, c = m² + n²
    // where m > n, gcd(m,n) = 1, and m-n is odd for primitive triples
    let mut m: i32 = 2;
    while m * m + 1 <= max_c {
        for n in 1..m {
            let a = m * m - n * n;
            let b = 2 * m * n;
            let c = m * m + n * n;
            
            if c > max_c {
                break;
            }
            
            // Check if primitive (gcd condition)
            if gcd(m - n, m) == 1 && (m - n) % 2 == 1 {
                // Add primitive and all multiples
                let mut ka = a;
                let mut kb = b;
                let mut kc = c;
                while kc <= max_c {
                    // Ensure consistent ordering (a <= b)
                    if ka <= kb {
                        triples.push((ka, kb, kc));
                    } else {
                        triples.push((kb, ka, kc));
                    }
                    ka += a;
                    kb += b;
                    kc += c;
                }
            }
        }
        m += 1;
    }
    
    // Sort by c value
    triples.sort_by_key(|t| t.2);
    triples.dedup();
    triples
}

/// Helper function to compute GCD
fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// Python module definition
#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifold>()?;
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    
    // Add module docstring
    m.add("__doc__", "Python bindings for Constraint Theory - deterministic geometric snapping with O(log n) KD-tree lookup")?;
    
    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
