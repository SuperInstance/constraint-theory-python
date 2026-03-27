//! Python bindings for Constraint Theory
//!
//! This module provides Python access to the Constraint Theory Rust library
//! via PyO3 bindings.

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

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

    /// Snap multiple vectors at once (SIMD optimized)
    ///
    /// Args:
    ///     vectors: List of [x, y] pairs or Nx2 numpy array
    ///
    /// Returns:
    ///     List of (snapped_x, snapped_y, noise) tuples
    pub fn snap_batch(&self, py: Python<'_>, vectors: &PyAny) -> PyResult<Vec<(f32, f32, f32)>> {
        // Try to interpret as numpy array first
        if let Ok(arr) = vectors.extract::<PyReadonlyArray2<f32>>() {
            let array = arr.as_array();
            let mut results = Vec::with_capacity(array.nrows());
            
            for row in array.rows() {
                let (snapped, noise) = self.inner.snap([row[0], row[1]]);
                results.push((snapped[0], snapped[1], noise));
            }
            
            return Ok(results);
        }
        
        // Fall back to list of pairs
        let list: &PyList = vectors.extract()?;
        let mut results = Vec::with_capacity(list.len());
        
        for item in list.iter() {
            let pair: (f32, f32) = item.extract()?;
            let (snapped, noise) = self.inner.snap([pair.0, pair.1]);
            results.push((snapped[0], snapped[1], noise));
        }
        
        Ok(results)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("PythagoreanManifold(density={}, states={})", 
                self.inner.state_count() / 5, // Approximate density from state count
                self.inner.state_count())
    }

    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Snap a vector to a Pythagorean manifold
///
/// Convenience function for one-off snaps. For multiple snaps,
/// create a PythagoreanManifold and use its snap method.
///
/// Args:
///     manifold: The PythagoreanManifold to use
///     x: X coordinate
///     y: Y coordinate
///
/// Returns:
///     Tuple of (snapped_x, snapped_y, noise)
#[pyfunction]
pub fn snap(manifold: &PyManifold, x: f32, y: f32) -> (f32, f32, f32) {
    manifold.snap(x, y)
}

/// Generate primitive Pythagorean triples up to a maximum hypotenuse
///
/// Args:
///     max_c: Maximum value of c (hypotenuse)
///
/// Returns:
///     List of (a, b, c) tuples where a² + b² = c²
#[pyfunction]
pub fn generate_triples(max_c: usize) -> Vec<(u32, u32, u32)> {
    let mut triples = Vec::new();
    
    for m in 2..=((max_c as f64).sqrt() as usize) {
        for n in 1..m {
            if (m - n) % 2 == 1 && gcd(m, n) == 1 {
                let a = (m * m - n * n) as u32;
                let b = (2 * m * n) as u32;
                let c = (m * m + n * n) as u32;
                
                if c <= max_c as u32 {
                    triples.push((a, b, c));
                }
            }
        }
    }
    
    triples
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Python module definition
#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyManifold>()?;
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
