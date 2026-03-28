# Release Checklist

This checklist ensures consistent, high-quality releases of the Constraint Theory Python package.

## Pre-Release Checklist

### Code Quality

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code coverage >= 90%: `pytest tests/ --cov=constraint_theory`
- [ ] No linting errors: `black --check . && isort --check .`
- [ ] Type checking passes: `mypy constraint_theory/`
- [ ] Documentation builds: `mkdocs build` (if applicable)
- [ ] CHANGELOG.md updated with new version and changes

### Rust Core

- [ ] Rust code compiles without warnings: `cargo build --release`
- [ ] Rust tests pass: `cargo test`
- [ ] Rust clippy passes: `cargo clippy -- -D warnings`
- [ ] Rust documentation: `cargo doc --no-deps`
- [ ] Core version compatibility verified in Cargo.toml

### Python Bindings

- [ ] Python builds successfully: `maturin build --release`
- [ ] Wheel installs correctly: `pip install dist/*.whl`
- [ ] Import works: `python -c "from constraint_theory import PythagoreanManifold; print('OK')"`
- [ ] All examples run: `python examples/basic_usage.py`

### Cross-Platform Testing

- [ ] Linux x86_64 wheel builds and tests
- [ ] macOS ARM64 wheel builds and tests
- [ ] macOS x86_64 wheel builds and tests (optional for Apple Silicon)
- [ ] Windows x86_64 wheel builds and tests

### Version Updates

- [ ] `pyproject.toml` version updated
- [ ] `Cargo.toml` version updated
- [ ] `src/lib.rs` `__version__` constant updated
- [ ] `constraint_theory/__init__.py` `__version__` updated
- [ ] Git tag created: `git tag v0.X.0`

---

## Build Wheels

### Local Build (for testing)

```bash
# Install build tools
pip install maturin

# Build wheel for current platform
maturin build --release

# Check wheel contents
unzip -l target/wheels/*.whl
```

### CI/CD Build (for release)

The GitHub Actions workflow automatically builds wheels for all platforms.

1. Create a GitHub release with tag `v0.X.0`
2. Wait for CI to complete
3. Download wheels from the workflow artifacts

### Manual Multi-Platform Build

```bash
# Using cibuildwheel for all platforms
pip install cibuildwheel

# Build on appropriate runner
cibuildwheel --platform linux
cibuildwheel --platform macos
cibuildwheel --platform windows
```

---

## PyPI Publishing

### TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi target/wheels/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ constraint-theory
```

### PyPI (Production)

```bash
# Upload to PyPI
python -m twine upload target/wheels/*

# Verify installation
pip install constraint-theory
python -c "from constraint_theory import PythagoreanManifold; print(__version__)"
```

### Automated Publishing (GitHub Actions)

```yaml
# .github/workflows/release.yml handles this automatically
# Just create a GitHub release and it will:
# 1. Build wheels for all platforms
# 2. Upload to PyPI
# 3. Create GitHub release notes
```

---

## Post-Release Checklist

### Verification

- [ ] PyPI page looks correct: https://pypi.org/project/constraint-theory/
- [ ] Installation works: `pip install constraint-theory`
- [ ] Version is correct: `python -c "import constraint_theory; print(constraint_theory.__version__)"`
- [ ] Documentation links work
- [ ] All demos still work

### Communication

- [ ] GitHub release notes published
- [ ] Update website if applicable
- [ ] Announce on social media (optional)
- [ ] Update dependent projects

### Cleanup

- [ ] Merge release branch to main (if using release branches)
- [ ] Update version to next development version (e.g., 0.X.1.dev0)
- [ ] Close completed milestone

---

## Version Numbering

### Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Pre-Release Versions

```
0.2.0a1    # Alpha release (internal testing)
0.2.0b1    # Beta release (external testing)
0.2.0rc1   # Release candidate (final testing)
0.2.0      # Official release
```

### Development Versions

```
0.2.1.dev0  # Development version for 0.2.1
```

---

## Hotfix Process

For critical bugs requiring immediate fix:

1. Create hotfix branch from release tag
   ```bash
   git checkout -b hotfix/0.1.1 v0.1.0
   ```

2. Fix the bug and update version to 0.1.1

3. Run minimal test suite
   ```bash
   pytest tests/test_critical.py
   ```

4. Build and publish
   ```bash
   maturin build --release
   twine upload target/wheels/*
   ```

5. Merge back to main and update changelog

---

## Rollback Process

If a release has critical issues:

1. **Yank the release from PyPI**
   ```bash
   twine upload --repository pypi --comment "Critical bug in X" target/wheels/constraint_theory-0.X.0-*.whl
   pip yank constraint-theory==0.X.0
   ```

2. **Update documentation** with known issues

3. **Fix and release** a patch version

4. **Notify users** through GitHub issues/discussions

---

## Files to Update Per Release

| File | What to Update |
|------|----------------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `Cargo.toml` | `version = "X.Y.Z"` |
| `src/lib.rs` | `m.add("__version__", "X.Y.Z")` |
| `constraint_theory/__init__.py` | `__version__ = "X.Y.Z"` |
| `CHANGELOG.md` | Add release notes |
| `README.md` | Update badges (if needed) |
| GitHub Release | Create with tag `vX.Y.Z` |

---

## Example Release Sequence

```bash
# 1. Update version numbers
# Edit pyproject.toml, Cargo.toml, src/lib.rs, constraint_theory/__init__.py

# 2. Update changelog
# Edit CHANGELOG.md

# 3. Commit version bump
git add -A
git commit -m "Bump version to 0.2.0"

# 4. Create tag
git tag v0.2.0

# 5. Push
git push origin main --tags

# 6. Create GitHub release (triggers CI)

# 7. Verify PyPI upload

# 8. Test installation
pip install --upgrade constraint-theory
python -c "from constraint_theory import __version__; print(__version__)"
```

---

## Security Considerations

### API Token Security

- Never commit PyPI API tokens to git
- Use GitHub secrets for CI/CD
- Rotate tokens if compromised

### Supply Chain Security

- Pin dependencies in requirements.txt
- Use dependency scanning tools
- Review dependency updates carefully

### Release Signing

- Consider signing releases with GPG
- Use `sigstore` for artifact signing

```bash
# Install sigstore
pip install sigstore

# Sign wheel
sigstore sign target/wheels/constraint_theory-0.2.0-*.whl
```

---

## Contact

For release-related questions:
- GitHub Issues: https://github.com/SuperInstance/constraint-theory-python/issues
- PyPI Project: https://pypi.org/project/constraint-theory/
