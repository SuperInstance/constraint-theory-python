# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Constraint Theory Python bindings seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to the [Security Advisories page](https://github.com/SuperInstance/constraint-theory-python/security/advisories)
   - Click "Report a vulnerability"
   - Fill in the details

2. **Email**
   - Send an email to: security@superinstance.ai
   - Include "SECURITY: constraint-theory-python" in the subject line

### What to Include

Please include the following information:

- Type of vulnerability (e.g., buffer overflow, memory safety, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Critical vulnerabilities within 30 days

### Disclosure Policy

- We follow responsible disclosure practices
- We will coordinate with you on the disclosure timeline
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using this library:

1. **Input Validation**: Always validate input vectors before processing
2. **Memory Management**: The library uses Rust's memory safety guarantees, but ensure you're using the latest version
3. **Batch Processing**: For large datasets, use `snap_batch()` which is optimized for performance

## Security Considerations for PyO3 Bindings

### Memory Safety

The Python bindings use PyO3, which provides safe FFI between Python and Rust:

| Security Feature | Implementation |
|------------------|----------------|
| **Memory Safety** | Rust's ownership system prevents use-after-free, buffer overflows, and dangling pointers |
| **Null Safety** | Rust's Option type enforces null checking at compile time |
| **Thread Safety** | `Send` and `Sync` traits ensure thread-safe data access |
| **Panic Handling** | Rust panics are caught and converted to Python exceptions |

### Attack Surface Analysis

| Component | Risk Level | Mitigation |
|-----------|------------|------------|
| `PythagoreanManifold(density)` | Low | Integer validation, bounded memory allocation |
| `snap(x, y)` | Low | Pure computation, no side effects, type checked at FFI |
| `snap_batch(vectors)` | Low | SIMD computation, GIL released, memory bounded by input |
| `generate_triples(max_c)` | Low | Integer math only, output size bounded |

### FFI Security Considerations

```python
# Type checking at FFI boundary
# PyO3 validates all types before passing to Rust
manifold = PythagoreanManifold(200)

# These raise TypeError before reaching Rust
manifold.snap("invalid", 0.8)    # TypeError
manifold.snap(None, 0.8)         # TypeError
manifold.snap([], 0.8)           # TypeError
```

### Panic Safety

Rust panics are caught and converted to Python exceptions:

```python
# This would panic in Rust, but raises Python exception instead
try:
    PythagoreanManifold(0)  # Invalid density
except (ValueError, RuntimeError) as e:
    print(f"Caught error: {e}")
```

### Thread Safety Guarantees

| Operation | Thread Safe | Notes |
|-----------|-------------|-------|
| Manifold construction | Yes | Each thread gets its own manifold |
| Shared manifold read | Yes | Immutable after construction |
| Concurrent `snap()` calls | Yes | No mutable state |
| Concurrent `snap_batch()` calls | Yes | GIL released, safe parallel access |

### Denial of Service Considerations

```python
# Large densities can cause slow construction
# Consider validating density in application code
MAX_REASONABLE_DENSITY = 10000

def create_manifold_safely(density: int) -> 'PythagoreanManifold':
    if not isinstance(density, int):
        raise TypeError("density must be integer")
    if density <= 0:
        raise ValueError("density must be positive")
    if density > MAX_REASONABLE_DENSITY:
        raise ValueError(f"density {density} exceeds maximum {MAX_REASONABLE_DENSITY}")
    return PythagoreanManifold(density)
```

### Known Security Considerations

- This library performs geometric calculations and does not handle cryptographic operations
- The library uses PyO3 for Python bindings, benefiting from Rust's memory safety
- No known security vulnerabilities in the current version

### Dependencies Security

| Dependency | Purpose | Security Notes |
|------------|---------|----------------|
| `pyo3` | Python bindings | Memory-safe FFI, actively maintained |
| `constraint-theory-core` | Core algorithm | Rust memory safety guarantees |

## Security Updates

Security updates will be released as patch versions. Subscribe to GitHub releases to be notified of updates.

---

Thank you for helping keep Constraint Theory Python bindings secure!
