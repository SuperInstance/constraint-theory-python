# Version Management

This document describes how versions are managed for the Constraint Theory Python package.

## Version Numbering Scheme

We follow [Semantic Versioning 2.0.0](https://semver.org/) with some Python-specific conventions.

### Version Format

```
MAJOR.MINOR.PATCH[-SUFFIX]
```

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible
- **SUFFIX**: Pre-release identifiers (optional)

### Examples

| Version | Meaning |
|---------|---------|
| `0.1.0` | Initial release |
| `0.1.1` | Bug fix release |
| `0.2.0` | New features, still pre-1.0 |
| `1.0.0` | First stable API release |
| `1.1.0` | New features added |
| `2.0.0` | Breaking changes |

### Pre-Release Versions

| Suffix | Meaning | Example |
|--------|---------|---------|
| `a1`, `a2`, ... | Alpha (internal testing) | `0.2.0a1` |
| `b1`, `b2`, ... | Beta (external testing) | `0.2.0b1` |
| `rc1`, `rc2`, ... | Release candidate | `0.2.0rc1` |
| `.dev0`, `.dev1`, ... | Development version | `0.2.1.dev0` |

### Version Ordering

```
0.2.0.dev0 < 0.2.0a1 < 0.2.0a2 < 0.2.0b1 < 0.2.0rc1 < 0.2.0 < 0.2.1.dev0
```

---

## Version Sources

The version is defined in multiple places and must be kept in sync:

| File | Version Location | Priority |
|------|------------------|----------|
| `pyproject.toml` | `version = "X.Y.Z"` | Primary source |
| `Cargo.toml` | `version = "X.Y.Z"` | Must match |
| `src/lib.rs` | `m.add("__version__", "X.Y.Z")` | Must match |
| `constraint_theory/__init__.py` | `__version__ = "X.Y.Z"` | Fallback |

### pyproject.toml

```toml
[project]
name = "constraint-theory"
version = "0.1.0"
```

### Cargo.toml

```toml
[package]
name = "constraint-theory-python"
version = "0.1.0"
```

### src/lib.rs

```rust
#[pymodule]
fn constraint_theory(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ...
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

### constraint_theory/__init__.py

```python
__version__ = "0.1.0"
```

---

## Version Bumping Process

### Manual Version Update

1. Update all version files:
   ```bash
   # Edit files
   vim pyproject.toml Cargo.toml src/lib.rs constraint_theory/__init__.py
   ```

2. Update the changelog:
   ```bash
   vim CHANGELOG.md
   ```

3. Commit and tag:
   ```bash
   git add -A
   git commit -m "Bump version to X.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```

### Automated Version Bumping (Recommended)

We use `bump-my-version` for automated version management:

```bash
# Install
pip install bump-my-version

# Bump version
bump-my-version bump patch  # 0.1.0 -> 0.1.1
bump-my-version bump minor  # 0.1.0 -> 0.2.0
bump-my-version bump major  # 0.1.0 -> 1.0.0
```

Configuration in `.bumpversion.toml`:

```toml
[tool.bumpversion]
current_version = "0.1.0"
commit = true
tag = true
tag_name = "v{new_version}"

[[tool.bumpversion.files]]
filename = "pyproject.toml"
search = 'version = "{current_version}"'
replace = 'version = "{new_version}"'

[[tool.bumpversion.files]]
filename = "Cargo.toml"
search = 'version = "{current_version}"'
replace = 'version = "{new_version}"'

[[tool.bumpversion.files]]
filename = "src/lib.rs"
search = 'm.add("__version__", "{current_version}")'
replace = 'm.add("__version__", "{new_version}")'

[[tool.bumpversion.files]]
filename = "constraint_theory/__init__.py"
search = '__version__ = "{current_version}"'
replace = '__version__ = "{new_version}"'
```

---

## Version Compatibility

### Rust Core Compatibility

Python bindings are compatible with specific versions of the Rust core:

```toml
# Cargo.toml
[dependencies]
constraint-theory-core = ">=1.0.0,<2.0.0"
```

### Compatibility Matrix

| Python Bindings | Rust Core | Python Versions |
|-----------------|-----------|-----------------|
| 0.1.x | 1.0.x | 3.8-3.12 |
| 0.2.x | 1.0.x-1.1.x | 3.8-3.12 |
| 1.0.x | 1.x.x | 3.8-3.13 |

### Version Checking at Runtime

```python
from constraint_theory import __version__, CORE_MIN_VERSION, CORE_MAX_VERSION

def check_compatibility():
    """Verify version compatibility."""
    major, minor, patch = map(int, __version__.split('.'))
    
    if major < 1:
        print("Warning: Pre-release version in use")
    
    return True
```

---

## Dependency Versioning

### Python Dependencies

```toml
# pyproject.toml

[project]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "numpy>=1.20",
]
```

### Rust Dependencies

```toml
# Cargo.toml

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
constraint-theory-core = "1.0.1"
```

---

## Changelog Management

### CHANGELOG.md Format

We follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Feature description

### Changed
- Change description

### Fixed
- Bug fix description

## [0.1.0] - 2024-01-15

### Added
- Initial release with core snapping functionality
- PythagoreanManifold class with snap and snap_batch methods
- generate_triples utility function
- NumPy integration support

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
```

### Changelog Categories

- **Added**: New features
- **Changed**: Changes to existing features
- **Deprecated**: Features to be removed
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| 0.1.0 | TBD | Initial release |
| 0.2.0 | TBD | Planned: Performance improvements |
| 1.0.0 | TBD | Planned: Stable API |

---

## Best Practices

### Do's

1. **Always update all version files** when bumping version
2. **Update CHANGELOG.md** before every release
3. **Tag releases** in git with `vX.Y.Z` format
4. **Test on multiple Python versions** before release
5. **Pin dependencies** for reproducibility

### Don'ts

1. **Don't skip version numbers** (0.1.0 -> 0.3.0 is confusing)
2. **Don't change released versions** (immutable releases)
3. **Don't mix version suffixes** (no `0.1.0-alpha1`, use `0.1.0a1`)
4. **Don't forget to push tags** (`git push --tags`)

---

## Troubleshooting

### Version Mismatch

If version files get out of sync:

```bash
# Check all versions
grep -r "version" pyproject.toml Cargo.toml src/lib.rs constraint_theory/__init__.py

# Fix mismatched versions
# Edit files manually or use bump-my-version
```

### Installation Shows Wrong Version

```python
# Check installed version
import constraint_theory
print(constraint_theory.__version__)

# If wrong, force reinstall
# pip install --force-reinstall constraint-theory
```

### Git Tag Issues

```bash
# List tags
git tag -l

# Delete local tag
git tag -d v0.1.0

# Delete remote tag
git push origin --delete v0.1.0

# Recreate tag
git tag v0.1.0
git push origin v0.1.0
```

---

## See Also

- [Release Checklist](RELEASE.md)
- [CI/CD Pipeline](../.github/workflows/ci.yml)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
