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
    density: usize,
}

#[pymethods]
impl PyManifold {
    /// Create a new Pythagorean manifold with specified density
    #[new]
    pub fn new(density: usize) -> Self {
        PyManifold { 
            inner: PythagoreanManifold::new(density),
            density,
        }
    }

    /// Get the number of valid states in the manifold
    #[getter]
    pub fn state_count(&self) -> usize {
        self.inner.state_count()
    }

    /// Get the density parameter
    #[getter]
    pub fn density(&self) -> usize {
        self.density
    }

    /// Snap a 2D vector to the nearest Pythagorean triple
    pub fn snap(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let (snapped, noise) = self.inner.snap([x, y]);
        (snapped[0], snapped[1], noise)
    }

    /// Snap multiple vectors at once (SIMD optimized)
    pub fn snap_batch(&self, py: Python<'_>, vectors: &PyList) -> PyResult<Vec<(f32, f32, f32)>> {
        let input: Vec<[f32; 2]> = vectors
            .iter()
            .map(|item| {
                let t: (f32, f32) = item.extract()?;
                Ok([t.0, t.1])
            })
            .collect::<PyResult<_>>()?;
        
        py.allow_threads(|| {
            let results = self.inner.snap_batch_simd(&input);
            Ok(results.into_iter().map(|(s, n)| (s[0], s[1], n)).collect())
        })
    }

    fn __repr__(&self) -> String {
        format!("PythagoreanManifold(density={}, states={})", self.density, self.inner.state_count())
    }
}

/// Snap a vector using a default manifold
#[pyfunction]
#[pyo3(signature = (x, y, density=200))]
pub fn snap(x: f32, y: f32, density: usize) -> (f32, f32, f32) {
    let m = PythagoreanManifold::new(density);
    let (s, n) = m.snap([x, y]);
    (s[0], s[1], n)
}

/// Generate Pythagorean triples up to max hypotenuse
#[pyfunction]
pub fn generate_triples(max_c: i32) -> Vec<(i32, i32, i32)> {
    let mut triples = Vec::new();
    let mut m: i32 = 2;
    while m * m + 1 <= max_c {
        for n in 1..m {
            let a = m * m - n * n;
            let b = 2 * m * n;
            let c = m * m + n * n;
            if c > max_c { break; }
            if (m - n) % 2 == 1 && gcd(m - n, m) == 1 {
                let (mut ka, mut kb, mut kc) = (a, b, c);
                while kc <= max_c {
                    triples.push(if ka <= kb {(ka,kb,kc)} else {(kb,ka,kc)});
                    ka += a; kb += b; kc += c;
                }
            }
        }
        m += 1;
    }
    triples.sort_by_key(|t| t.2);
    triples.dedup();
    triples
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifold>()?;
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
