//! Python bindings for Constraint Theory
//!
//! This module provides Python access to the Constraint Theory Rust library
//! via PyO3 bindings.
//!
//! # Schema Alignment (PASS 5)
//!
//! This Python API is designed to match the Rust core exactly:
//!
//! | Rust (constraint-theory-core) | Python (this module) |
//! |-------------------------------|----------------------|
//! | `PythagoreanManifold::new(density: usize)` | `PythagoreanManifold(density: int)` |
//! | `manifold.snap([x, y]) -> ([f32; 2], f32)` | `manifold.snap(x, y) -> (float, float, float)` |
//! | `manifold.snap_batch_simd(&vectors)` | `manifold.snap_batch(vectors)` |
//! | `manifold.state_count()` | `manifold.state_count` (property) |
//!
//! # Type Mapping (PASS 6)
//!
//! | Rust Type | Python Type | Notes |
//! |-----------|-------------|-------|
//! | `usize` | `int` | Density parameter |
//! | `f32` | `float` | 32-bit float on Rust, Python float is 64-bit |
//! | `[f32; 2]` | `Tuple[float, float]` | Rust array -> Python tuple |
//! | `Vec<([f32; 2], f32)>` | `List[Tuple[float, float, float]]` | Batch results |
//!
//! # Cross-Reference
//!
//! - Rust Core: https://github.com/SuperInstance/constraint-theory-core
//! - WASM Bindings: https://github.com/SuperInstance/constraint-theory-wasm
//! - Research: https://github.com/SuperInstance/constraint-theory-research

use pyo3::prelude::*;
use pyo3::types::PyList;
use constraint_theory_core::{PythagoreanManifold, snap as rust_snap};

/// A Pythagorean manifold for deterministic vector snapping
///
/// Schema Alignment (PASS 5):
/// ==========================
/// This class wraps the Rust `PythagoreanManifold` from constraint-theory-core.
///
/// # Rust API Reference
///
/// ```rust,ignore
/// // Rust constructor
/// let manifold = PythagoreanManifold::new(density: usize);
///
/// // Rust snap method - takes array, returns tuple
/// let (snapped, noise) = manifold.snap([x, y]);
/// // snapped: [f32; 2], noise: f32
///
/// // Rust batch method
/// let results = manifold.snap_batch_simd(&vectors);
/// // results: Vec<([f32; 2], f32)>
///
/// // Rust state count
/// let count = manifold.state_count();
/// // count: usize
/// ```
///
/// # Python Adaptation
///
/// Python convenience methods adapt the Rust API:
/// - `snap(x, y)` unpacks the array result to a 3-tuple `(x, y, noise)`
/// - `state_count` is a property, not a method
/// - `snap_batch` wraps `snap_batch_simd`
///
/// # Type Mapping
///
/// - Rust `usize` <-> Python `int`
/// - Rust `f32` <-> Python `float` (note: Python uses 64-bit floats)
/// - Rust `[f32; 2]` <-> Python `Tuple[float, float]`
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

    /// Get the density parameter used to create this manifold
    #[getter]
    pub fn density(&self) -> usize {
        self.density
    }

    /// Snap a 2D vector to the nearest Pythagorean triple
    pub fn snap(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let (snapped, noise) = self.inner.snap([x, y]);
        (snapped[0], snapped[1], noise)
    }

    /// Snap multiple vectors at once using SIMD
    ///
    /// # Arguments
    ///
    /// * `vectors` - List of (x, y) tuples or NumPy Nx2 array
    ///
    /// # Returns
    ///
    /// List of (snapped_x, snapped_y, noise) tuples
    ///
    /// # GIL Handling (PASS 7)
    ///
    /// This method releases the GIL during computation, allowing
    /// other Python threads to run concurrently. This is especially
    /// beneficial for large batches.
    ///
    /// # NumPy Array Shape (PASS 6)
    ///
    /// Expected shape: (N, 2) where N is the number of vectors
    /// - First column: x coordinates
    /// - Second column: y coordinates
    ///
    /// # Example
    ///
    /// ```python
    /// manifold = PythagoreanManifold(200)
    /// vectors = [[0.6, 0.8], [0.707, 0.707]]
    /// results = manifold.snap_batch(vectors)
    /// # results = [(0.6, 0.8, 0.0), (0.6, 0.8, 0.014)]
    /// ```
    pub fn snap_batch_simd(&self, py: Python<'_>, vectors: &PyList) -> PyResult<Vec<(f32, f32, f32)>> {
        let input: Vec<[f32; 2]> = vectors
            .iter()
            .map(|item| {
                let t: (f32, f32) = item.extract()?;
                Ok([t.0, t.1])
            })
            .collect::<PyResult<_>>()?;
        
        // Release GIL for long-running operations (PASS 7)
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
            if (m - n) % 2 == 1 { // primitive condition
                let mut ka = a; let mut kb = b; let mut kc = c;
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

#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifold>()?;
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
