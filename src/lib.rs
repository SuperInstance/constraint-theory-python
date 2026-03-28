//! Python bindings for Constraint Theory
//!
//! This module provides Python access to the Constraint Theory Rust library
//! via PyO3 bindings. It matches the Rust core API from Iteration 1.
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
//! # Modules Bound
//!
//! - `hidden_dimensions`: Hidden dimension encoding (k = ⌈log₂(1/ε)⌉)
//! - `quantizer`: PythagoreanQuantizer with TERNARY/POLAR/TURBO/HYBRID modes
//! - `holonomy`: Holonomy verification for constraint consistency
//! - `manifold`: Core PythagoreanManifold for 2D snapping
//!
//! # Cross-Reference
//!
//! - Rust Core: https://github.com/SuperInstance/constraint-theory-core
//! - WASM Bindings: https://github.com/SuperInstance/constraint-theory-wasm
//! - Research: https://github.com/SuperInstance/constraint-theory-research

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::f64;

// ============================================================================
// Hidden Dimensions Module Bindings
// ============================================================================

/// Compute the number of hidden dimensions required for a given precision.
///
/// Uses the GUCT formula: k = ⌈log₂(1/ε)⌉
///
/// Args:
///     epsilon (float): Target precision (must be > 0)
///
/// Returns:
///     int: Number of hidden dimensions needed
///
/// Example:
///     >>> from constraint_theory import hidden_dim_count
///     >>> hidden_dim_count(1e-10)
///     34
#[pyfunction]
pub fn hidden_dim_count(epsilon: f64) -> usize {
    if epsilon <= 0.0 {
        return usize::MAX;
    }
    if epsilon >= 1.0 {
        return 0;
    }
    (1.0 / epsilon).log2().ceil() as usize
}

/// Compute precision from hidden dimension count (inverse of hidden_dim_count).
#[pyfunction]
pub fn precision_from_hidden_dims(k: usize) -> f64 {
    2.0_f64.powi(-(k as i32))
}

/// Compute holographic accuracy for a given configuration.
///
/// Formula: accuracy(k, n) = k/n + O(1/log n)
#[pyfunction]
pub fn holographic_accuracy(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let base_accuracy = k as f64 / n as f64;
    let correction = if n > 1 {
        1.0 / (n as f64).ln()
    } else {
        0.0
    };
    (base_accuracy + correction).min(1.0)
}

/// Lift a point to higher dimensions by adding hidden dimensions.
#[pyfunction]
pub fn lift_to_hidden(point: Vec<f64>, k: usize) -> Vec<f64> {
    let n = point.len();
    let mut lifted = Vec::with_capacity(n + k);
    
    // Copy visible dimensions
    lifted.extend_from_slice(&point);
    
    // Initialize hidden dimensions with geometric progression
    for i in 0..k {
        let hidden_val = 2.0_f64.powi(-(i as i32 + 1));
        lifted.push(hidden_val);
    }
    
    lifted
}

/// Project a lifted point back to visible dimensions.
#[pyfunction]
pub fn project_to_visible(lifted: Vec<f64>, n: usize) -> Vec<f64> {
    lifted.iter().take(n).copied().collect()
}

/// Encode a point using hidden dimensions for exact constraint satisfaction.
#[pyfunction]
pub fn encode_with_hidden_dims(point: Vec<f64>, epsilon: f64) -> Vec<f64> {
    let n = point.len();
    let k = hidden_dim_count(epsilon);
    
    // Lift to hidden dimensions
    let lifted = lift_to_hidden(point.clone(), k);
    
    // Snap to lattice in lifted space
    let snapped = snap_to_lattice_internal(&lifted);
    
    // Project back to visible dimensions
    project_to_visible(snapped, n)
}

/// Internal function to snap to lattice
fn snap_to_lattice_internal(point: &[f64]) -> Vec<f64> {
    let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm < 1e-10 {
        return point.to_vec();
    }
    
    point.iter().map(|&x| snap_to_rational_internal(x / norm) * norm).collect()
}

/// Internal function to snap to Pythagorean ratio
fn snap_to_rational_internal(value: f64) -> f64 {
    let pythagorean_ratios: &[f64] = &[
        0.0, 1.0,
        3.0/5.0, 4.0/5.0,
        5.0/13.0, 12.0/13.0,
        8.0/17.0, 15.0/17.0,
        7.0/25.0, 24.0/25.0,
        20.0/29.0, 21.0/29.0,
        9.0/41.0, 40.0/41.0,
        0.5, 0.7071067811865476,
    ];
    
    let mut best = value;
    let mut min_dist = f64::MAX;
    
    for &ratio in pythagorean_ratios {
        let dist = (value - ratio).abs();
        if dist < min_dist {
            min_dist = dist;
            best = ratio;
        }
        // Also check negative
        let dist = (value - (-ratio)).abs();
        if dist < min_dist {
            min_dist = dist;
            best = -ratio;
        }
    }
    
    best
}

/// Configuration for hidden dimension encoding.
#[pyclass(name = "HiddenDimensionConfig")]
pub struct PyHiddenDimensionConfig {
    #[pyo3(get)]
    pub epsilon: f64,
    #[pyo3(get)]
    pub hidden_dims: usize,
    #[pyo3(get)]
    pub cross_plane_optimization: bool,
}

#[pymethods]
impl PyHiddenDimensionConfig {
    #[new]
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            hidden_dims: hidden_dim_count(epsilon),
            cross_plane_optimization: true,
        }
    }
    
    #[staticmethod]
    pub fn with_hidden_dims(hidden_dims: usize) -> Self {
        Self {
            epsilon: precision_from_hidden_dims(hidden_dims),
            hidden_dims,
            cross_plane_optimization: true,
        }
    }
    
    pub fn encode(&self, point: Vec<f64>) -> Vec<f64> {
        encode_with_hidden_dims(point, self.epsilon)
    }
    
    fn __repr__(&self) -> String {
        format!("HiddenDimensionConfig(epsilon={}, hidden_dims={})", self.epsilon, self.hidden_dims)
    }
}

// ============================================================================
// Quantizer Module Bindings
// ============================================================================

/// Quantization modes for PythagoreanQuantizer.
#[pyclass(name = "QuantizationMode")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyQuantizationMode {
    /// Ternary quantization (BitNet style): {-1, 0, 1}
    Ternary = 0,
    /// Polar coordinate quantization (PolarQuant style)
    Polar = 1,
    /// Near-optimal distortion quantization (TurboQuant style)
    Turbo = 2,
    /// Auto-select mode based on input characteristics
    Hybrid = 3,
}

/// A rational number for exact representation.
#[pyclass(name = "Rational")]
#[derive(Clone, Copy, Debug)]
pub struct PyRational {
    #[pyo3(get)]
    pub num: i64,
    #[pyo3(get)]
    pub den: u64,
}

#[pymethods]
impl PyRational {
    #[new]
    pub fn new(num: i64, den: u64) -> Self {
        Self { num, den }
    }
    
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
    
    pub fn is_pythagorean(&self) -> bool {
        let a = self.num.unsigned_abs() as u64;
        let c = self.den;
        
        if c == 0 || a > c {
            return false;
        }
        
        let b_sq = c * c - a * a;
        let b = (b_sq as f64).sqrt() as u64;
        b * b == b_sq
    }
    
    fn __repr__(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }
    
    fn __float__(&self) -> f64 {
        self.to_f64()
    }
}

/// Result of quantization operation.
#[pyclass(name = "QuantizationResult")]
#[derive(Clone, Debug)]
pub struct PyQuantizationResult {
    #[pyo3(get)]
    pub data: Vec<f64>,
    #[pyo3(get)]
    pub mode: String,
    #[pyo3(get)]
    pub bits: u8,
    #[pyo3(get)]
    pub mse: f64,
    #[pyo3(get)]
    pub constraints_satisfied: bool,
    #[pyo3(get)]
    pub unit_norm_preserved: bool,
}

#[pymethods]
impl PyQuantizationResult {
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    
    pub fn check_unit_norm(&self, tolerance: f64) -> bool {
        (self.norm() - 1.0).abs() < tolerance
    }
    
    fn __repr__(&self) -> String {
        format!(
            "QuantizationResult(mode={}, bits={}, mse={:.6}, unit_norm={})",
            self.mode, self.bits, self.mse, self.unit_norm_preserved
        )
    }
}

/// Pythagorean Quantizer - Unified quantization with constraint preservation.
#[pyclass(name = "PythagoreanQuantizer")]
pub struct PyPythagoreanQuantizer {
    mode: PyQuantizationMode,
    bits: u8,
    max_denominator: usize,
}

#[pymethods]
impl PyPythagoreanQuantizer {
    #[new]
    #[pyo3(signature = (mode=3, bits=4))]
    pub fn new(mode: usize, bits: u8) -> Self {
        let mode = match mode {
            0 => PyQuantizationMode::Ternary,
            1 => PyQuantizationMode::Polar,
            2 => PyQuantizationMode::Turbo,
            _ => PyQuantizationMode::Hybrid,
        };
        Self {
            mode,
            bits: bits.max(1),
            max_denominator: 100,
        }
    }
    
    #[staticmethod]
    pub fn for_llm() -> Self {
        Self::new(0, 1)
    }
    
    #[staticmethod]
    pub fn for_embeddings() -> Self {
        Self::new(1, 8)
    }
    
    #[staticmethod]
    pub fn for_vector_db() -> Self {
        Self::new(2, 4)
    }
    
    #[staticmethod]
    pub fn hybrid() -> Self {
        Self::new(3, 4)
    }
    
    #[getter]
    pub fn mode_name(&self) -> String {
        match self.mode {
            PyQuantizationMode::Ternary => "Ternary".to_string(),
            PyQuantizationMode::Polar => "Polar".to_string(),
            PyQuantizationMode::Turbo => "Turbo".to_string(),
            PyQuantizationMode::Hybrid => "Hybrid".to_string(),
        }
    }
    
    pub fn quantize(&self, data: Vec<f64>) -> PyQuantizationResult {
        let mode = self.select_mode(&data);
        
        let (quantized, mse) = match mode {
            PyQuantizationMode::Ternary => self.quantize_ternary(&data),
            PyQuantizationMode::Polar => self.quantize_polar(&data),
            PyQuantizationMode::Turbo => self.quantize_turbo(&data),
            PyQuantizationMode::Hybrid => self.quantize_turbo(&data),
        };
        
        let unit_norm_preserved = self.check_unit_norm(&quantized);
        
        PyQuantizationResult {
            data: quantized,
            mode: match mode {
                PyQuantizationMode::Ternary => "Ternary".to_string(),
                PyQuantizationMode::Polar => "Polar".to_string(),
                PyQuantizationMode::Turbo => "Turbo".to_string(),
                PyQuantizationMode::Hybrid => "Hybrid".to_string(),
            },
            bits: self.bits,
            mse,
            constraints_satisfied: unit_norm_preserved || mode != PyQuantizationMode::Polar,
            unit_norm_preserved,
        }
    }
    
    pub fn snap_to_pythagorean(&self, value: f64) -> f64 {
        snap_to_rational_internal(value)
    }
    
    pub fn snap_to_lattice(&self, value: f64, max_denominator: usize) -> (f64, i64, u64) {
        let mut best_val = value;
        let mut best_num = value.round() as i64;
        let mut best_den = 1u64;
        let mut best_err = f64::MAX;
        
        for c in 2..=max_denominator {
            for a in 1..c {
                let b_sq = (c * c - a * a) as f64;
                if b_sq > 0.0 {
                    let b = b_sq.sqrt() as usize;
                    if b * b == (c * c - a * a) {
                        let ratio_a = a as f64 / c as f64;
                        let ratio_b = b as f64 / c as f64;
                        
                        let err_a = (value - ratio_a).abs();
                        if err_a < best_err {
                            best_err = err_a;
                            best_val = ratio_a;
                            best_num = a as i64;
                            best_den = c as u64;
                        }
                        
                        let err_b = (value - ratio_b).abs();
                        if err_b < best_err {
                            best_err = err_b;
                            best_val = ratio_b;
                            best_num = b as i64;
                            best_den = c as u64;
                        }
                    }
                }
            }
        }
        
        (best_val, best_num, best_den)
    }
    
    fn __repr__(&self) -> String {
        format!("PythagoreanQuantizer(mode={:?}, bits={})", self.mode, self.bits)
    }
}

impl PyPythagoreanQuantizer {
    fn select_mode(&self, data: &[f64]) -> PyQuantizationMode {
        if self.mode != PyQuantizationMode::Hybrid {
            return self.mode;
        }
        
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let is_unit_norm = (norm - 1.0).abs() < 0.01;
        
        let threshold = 0.1;
        let sparse_count = data.iter().filter(|&&x| x.abs() < threshold).count();
        let sparsity = sparse_count as f64 / data.len().max(1) as f64;
        
        if is_unit_norm {
            PyQuantizationMode::Polar
        } else if sparsity > 0.5 {
            PyQuantizationMode::Ternary
        } else {
            PyQuantizationMode::Turbo
        }
    }
    
    fn quantize_ternary(&self, data: &[f64]) -> (Vec<f64>, f64) {
        let mean_abs: f64 = data.iter().map(|x| x.abs()).sum::<f64>() / data.len().max(1) as f64;
        let threshold = mean_abs * 0.1;
        
        let quantized: Vec<f64> = data.iter().map(|&x| {
            if x.abs() < threshold {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                -1.0
            }
        }).collect();
        
        let mse: f64 = data.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f64>() / data.len().max(1) as f64;
        
        (quantized, mse)
    }
    
    fn quantize_polar(&self, data: &[f64]) -> (Vec<f64>, f64) {
        let n = data.len();
        if n < 2 {
            return (data.to_vec(), 0.0);
        }
        
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return (vec![1.0], 0.0);
        }
        
        let normalized: Vec<f64> = data.iter().map(|&x| x / norm).collect();
        
        let mut quantized = vec![0.0; n];
        
        for i in (0..n).step_by(2) {
            if i + 1 < n {
                let angle = normalized[i + 1].atan2(normalized[i]);
                let snapped_angle = self.snap_angle_to_pythagorean(angle);
                quantized[i] = snapped_angle.cos();
                quantized[i + 1] = snapped_angle.sin();
            } else {
                quantized[i] = snap_to_rational_internal(normalized[i]);
            }
        }
        
        let q_norm: f64 = quantized.iter().map(|x| x * x).sum::<f64>().sqrt();
        if q_norm > 1e-10 {
            quantized = quantized.iter().map(|&x| x / q_norm).collect();
        }
        
        let mse: f64 = normalized.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f64>() / n as f64;
        
        (quantized, mse)
    }
    
    fn quantize_turbo(&self, data: &[f64]) -> (Vec<f64>, f64) {
        let n = data.len();
        if n == 0 {
            return (vec![], 0.0);
        }
        
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        
        if range < 1e-10 {
            return (vec![min_val; n], 0.0);
        }
        
        let levels = (1 << self.bits) as f64;
        
        let quantized: Vec<f64> = data.iter().map(|&x| {
            let scaled = ((x - min_val) / range * (levels - 1.0)).round();
            let snapped = snap_to_rational_internal(scaled / (levels - 1.0));
            min_val + snapped * range
        }).collect();
        
        let mse: f64 = data.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f64>() / n as f64;
        
        (quantized, mse)
    }
    
    fn snap_angle_to_pythagorean(&self, angle: f64) -> f64 {
        let pythagorean_angles: &[f64] = &[
            0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, -std::f64::consts::FRAC_PI_2,
            (4.0_f64 / 3.0).atan(),
            (3.0_f64 / 4.0).atan(),
            (12.0_f64 / 5.0).atan(),
            (5.0_f64 / 12.0).atan(),
            (15.0_f64 / 8.0).atan(),
            (8.0_f64 / 15.0).atan(),
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_3,
        ];
        
        let mut best = angle;
        let mut min_diff = f64::MAX;
        
        for &pyth_angle in pythagorean_angles {
            let diff = ((angle - pyth_angle).abs() % std::f64::consts::TAU)
                .min((pyth_angle - angle).abs() % std::f64::consts::TAU);
            if diff < min_diff {
                min_diff = diff;
                best = pyth_angle;
            }
        }
        
        best
    }
    
    fn check_unit_norm(&self, data: &[f64]) -> bool {
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        (norm - 1.0).abs() < 0.01
    }
}

// ============================================================================
// Holonomy Module Bindings
// ============================================================================

pub type RotationMatrix = [[f64; 3]; 3];

#[pyclass(name = "HolonomyResult")]
#[derive(Clone, Debug)]
pub struct PyHolonomyResult {
    #[pyo3(get)]
    pub matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub norm: f64,
    #[pyo3(get)]
    pub information: f64,
    #[pyo3(get)]
    pub is_identity: bool,
    #[pyo3(get)]
    pub tolerance: f64,
}

#[pymethods]
impl PyHolonomyResult {
    pub fn is_within_tolerance(&self, tolerance: f64) -> bool {
        self.norm < tolerance
    }
    
    pub fn angular_deviation(&self) -> f64 {
        let trace = self.matrix[0][0] + self.matrix[1][1] + self.matrix[2][2];
        let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        cos_angle.acos()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HolonomyResult(norm={:.6}, is_identity={}, information={:.2})",
            self.norm, self.is_identity, self.information
        )
    }
}

#[pyfunction]
pub fn identity_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]
}

#[pyfunction]
pub fn compute_holonomy(cycle: Vec<Vec<Vec<f64>>>) -> PyHolonomyResult {
    let tolerance = 1e-6;
    
    if cycle.is_empty() {
        return PyHolonomyResult {
            matrix: identity_matrix(),
            norm: 0.0,
            information: f64::INFINITY,
            is_identity: true,
            tolerance,
        };
    }
    
    let mut product = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for rotation in &cycle {
        let m = vec_to_matrix(rotation);
        product = matrix_multiply(&product, &m);
    }
    
    let norm = deviation_from_identity(&product);
    let information = if norm > 0.0 { -norm.log2() } else { f64::INFINITY };
    let is_identity = norm < tolerance;
    
    PyHolonomyResult {
        matrix: matrix_to_vec(&product),
        norm,
        information,
        is_identity,
        tolerance,
    }
}

#[pyfunction]
pub fn verify_holonomy(cycles: Vec<Vec<Vec<Vec<f64>>>>, tolerance: f64) -> bool {
    cycles.iter().all(|cycle| {
        let result = compute_holonomy(cycle.clone());
        result.norm < tolerance
    })
}

#[pyfunction]
pub fn rotation_x(angle: f64) -> Vec<Vec<f64>> {
    let c = angle.cos();
    let s = angle.sin();
    vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, c, -s],
        vec![0.0, s, c],
    ]
}

#[pyfunction]
pub fn rotation_y(angle: f64) -> Vec<Vec<f64>> {
    let c = angle.cos();
    let s = angle.sin();
    vec![
        vec![c, 0.0, s],
        vec![0.0, 1.0, 0.0],
        vec![-s, 0.0, c],
    ]
}

#[pyfunction]
pub fn rotation_z(angle: f64) -> Vec<Vec<f64>> {
    let c = angle.cos();
    let s = angle.sin();
    vec![
        vec![c, -s, 0.0],
        vec![s, c, 0.0],
        vec![0.0, 0.0, 1.0],
    ]
}

#[pyclass(name = "HolonomyChecker")]
pub struct PyHolonomyChecker {
    accumulated: RotationMatrix,
    step_count: usize,
    tolerance: f64,
}

#[pymethods]
impl PyHolonomyChecker {
    #[new]
    #[pyo3(signature = (tolerance=1e-6))]
    pub fn new(tolerance: f64) -> Self {
        Self {
            accumulated: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            step_count: 0,
            tolerance,
        }
    }
    
    pub fn apply(&mut self, rotation: Vec<Vec<f64>>) {
        let m = vec_to_matrix(&rotation);
        self.accumulated = matrix_multiply(&self.accumulated, &m);
        self.step_count += 1;
    }
    
    pub fn check_partial(&self) -> PyHolonomyResult {
        let norm = deviation_from_identity(&self.accumulated);
        let information = if norm > 0.0 { -norm.log2() } else { f64::INFINITY };
        
        PyHolonomyResult {
            matrix: matrix_to_vec(&self.accumulated),
            norm,
            information,
            is_identity: norm < self.tolerance,
            tolerance: self.tolerance,
        }
    }
    
    pub fn check_closed(&self) -> PyHolonomyResult {
        let inverse = transpose(&self.accumulated);
        let cycle = vec![
            matrix_to_vec(&self.accumulated),
            matrix_to_vec(&inverse),
        ];
        compute_holonomy(cycle)
    }
    
    pub fn reset(&mut self) {
        self.accumulated = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        self.step_count = 0;
    }
    
    #[getter]
    pub fn step_count(&self) -> usize {
        self.step_count
    }
    
    fn __repr__(&self) -> String {
        format!("HolonomyChecker(steps={}, tolerance={:.6})", self.step_count, self.tolerance)
    }
}

// Helper functions for matrix operations
fn vec_to_matrix(v: &Vec<Vec<f64>>) -> RotationMatrix {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if i < v.len() && j < v[i].len() {
                m[i][j] = v[i][j];
            }
        }
    }
    m
}

fn matrix_to_vec(m: &RotationMatrix) -> Vec<Vec<f64>> {
    m.iter().map(|row| row.to_vec()).collect()
}

fn matrix_multiply(a: &RotationMatrix, b: &RotationMatrix) -> RotationMatrix {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn transpose(matrix: &RotationMatrix) -> RotationMatrix {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = matrix[j][i];
        }
    }
    result
}

fn deviation_from_identity(matrix: &RotationMatrix) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = matrix[i][j] - expected;
            sum += diff * diff;
        }
    }
    sum.sqrt()
}

// ============================================================================
// Manifold Module (Legacy Compatibility)
// ============================================================================

#[pyclass(name = "PythagoreanManifold")]
pub struct PyManifold {
    density: usize,
    states: Vec<(f64, f64)>,
}

#[pymethods]
impl PyManifold {
    #[new]
    pub fn new(density: usize) -> Self {
        let states = generate_pythagorean_states(density);
        PyManifold { density, states }
    }

    #[getter]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    #[getter]
    pub fn density(&self) -> usize {
        self.density
    }

    pub fn snap(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let (snapped_x, snapped_y, noise) = snap_to_manifold(x, y, &self.states);
        (snapped_x, snapped_y, noise)
    }
    
    /// Snap multiple vectors at once (batch operation)
    pub fn snap_batch(&self, vectors: Vec<(f64, f64)>) -> Vec<(f64, f64, f64)> {
        vectors.iter().map(|(x, y)| self.snap(*x, *y)).collect()
    }

    fn __repr__(&self) -> String {
        format!("PythagoreanManifold(density={}, states={})", self.density, self.states.len())
    }
}

fn generate_pythagorean_states(max_c: usize) -> Vec<(f64, f64)> {
    let mut states = Vec::new();
    let mut m: i32 = 2;
    while m * m + 1 <= max_c as i32 {
        for n in 1..m {
            let a = m * m - n * n;
            let b = 2 * m * n;
            let c = m * m + n * n;
            if c > max_c as i32 { break; }
            if (m - n) % 2 == 1 {
                let mut ka = a; let mut kb = b; let mut kc = c;
                while kc <= max_c as i32 {
                    states.push((ka as f64 / kc as f64, kb as f64 / kc as f64));
                    states.push((kb as f64 / kc as f64, ka as f64 / kc as f64));
                    ka += a; kb += b; kc += c;
                }
            }
        }
        m += 1;
    }
    states.sort_by(|a, b| {
        let dist_a = a.0.hypot(a.1);
        let dist_b = b.0.hypot(b.1);
        dist_a.partial_cmp(&dist_b).unwrap()
    });
    states.dedup();
    states
}

fn snap_to_manifold(x: f64, y: f64, states: &[(f64, f64)]) -> (f64, f64, f64) {
    let norm = (x * x + y * y).sqrt();
    if norm < 1e-10 {
        return (0.0, 0.0, 0.0);
    }
    
    let nx = x / norm;
    let ny = y / norm;
    
    let mut best_dist = f64::MAX;
    let mut best_state = (0.0, 0.0);
    
    for &(sx, sy) in states {
        let dist = (nx - sx).hypot(ny - sy);
        if dist < best_dist {
            best_dist = dist;
            best_state = (sx, sy);
        }
    }
    
    let snapped_x = best_state.0 * norm;
    let snapped_y = best_state.1 * norm;
    let noise = ((snapped_x - x).hypot(snapped_y - y)) / norm;
    
    (snapped_x, snapped_y, noise)
}

#[pyfunction]
#[pyo3(signature = (x, y, density=200))]
pub fn snap(x: f64, y: f64, density: usize) -> (f64, f64, f64) {
    let states = generate_pythagorean_states(density);
    snap_to_manifold(x, y, &states)
}

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
            if (m - n) % 2 == 1 {
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

// ============================================================================
// Module Definition
// ============================================================================

#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Hidden dimensions module
    m.add_function(wrap_pyfunction!(hidden_dim_count, m)?)?;
    m.add_function(wrap_pyfunction!(precision_from_hidden_dims, m)?)?;
    m.add_function(wrap_pyfunction!(holographic_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(lift_to_hidden, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_visible, m)?)?;
    m.add_function(wrap_pyfunction!(encode_with_hidden_dims, m)?)?;
    m.add_class::<PyHiddenDimensionConfig>()?;
    
    // Quantizer module
    m.add_class::<PyQuantizationMode>()?;
    m.add_class::<PyRational>()?;
    m.add_class::<PyQuantizationResult>()?;
    m.add_class::<PyPythagoreanQuantizer>()?;
    
    // Holonomy module
    m.add_function(wrap_pyfunction!(identity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_holonomy, m)?)?;
    m.add_function(wrap_pyfunction!(verify_holonomy, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_x, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_y, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_z, m)?)?;
    m.add_class::<PyHolonomyResult>()?;
    m.add_class::<PyHolonomyChecker>()?;
    
    // Manifold module (legacy)
    m.add_class::<PyManifold>()?;
    m.add_function(wrap_pyfunction!(snap, m)?)?;
    m.add_function(wrap_pyfunction!(generate_triples, m)?)?;
    
    m.add("__version__", "0.3.0")?;
    Ok(())
}
