# Contributing to Constraint Theory Python Bindings

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/constraint-theory-python.git`
3. Create a branch: `git checkout -b feature/my-feature`

## Development Setup

### Prerequisites

- Python 3.8+ (3.11 recommended)
- Rust 1.70+ (for building the native extension)
- pip or poetry for dependency management

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install maturin pytest numpy

# Build the Rust extension in development mode
maturin develop

# Verify installation
python -c "from constraint_theory import PythagoreanManifold; print('OK')"
```

### Building from Source

```bash
# Development build (faster compile, slower runtime)
maturin develop

# Release build (slower compile, faster runtime)
maturin develop --release
```

## Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or modifying tests

### Commit Messages

Follow conventional commits:

```
feat: add new batch processing method
fix: correct noise calculation for edge cases
docs: update installation instructions
test: add tests for numpy array input
refactor: simplify manifold initialization
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=constraint_theory --cov-report=html

# Run specific test file
pytest tests/test_bindings.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names

```python
def test_snap_exact_pythagorean_triple():
    """Test that exact Pythagorean triples snap with zero noise."""
    manifold = PythagoreanManifold(density=100)
    sx, sy, noise = manifold.snap(0.6, 0.8)  # 3-4-5 triangle
    assert noise < 0.001, "Exact triple should have near-zero noise"
```

## Pull Request Process

1. **Update Documentation**: Ensure README.md and docstrings are updated
2. **Add Tests**: New features need tests
3. **Run Tests**: All tests must pass
4. **Check Examples**: Ensure example scripts still work
5. **Submit PR**: Use the PR template

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No new warnings introduced

## Style Guidelines

### Python Code

- Follow PEP 8
- Use type hints where possible
- Write docstrings for public functions

```python
def process_vectors(
    manifold: PythagoreanManifold,
    vectors: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """Process a list of vectors through the manifold.
    
    Args:
        manifold: The PythagoreanManifold to use.
        vectors: List of (x, y) coordinate pairs.
    
    Returns:
        List of (snapped_x, snapped_y, noise) tuples.
    """
    return manifold.snap_batch(vectors)
```

### Rust Code

- Follow Rust standard formatting (`cargo fmt`)
- Document public APIs with doc comments
- Run clippy: `cargo clippy -- -D warnings`

## Questions?

- Open a [Discussion](https://github.com/SuperInstance/constraint-theory-python/discussions)
- Check existing [Issues](https://github.com/SuperInstance/constraint-theory-python/issues)

Thank you for contributing!
