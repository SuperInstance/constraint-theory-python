## Description

A clear and concise description of the changes in this pull request.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes:

- [ ] Unit tests pass (`pytest tests/`)
- [ ] Example scripts run successfully
- [ ] Manual testing (describe below)

```bash
# Test commands you ran
pytest tests/ -v
python examples/quickstart.py
```

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Changes to Core Library

If this PR modifies the Rust core library (`src/lib.rs` or `constraint_theory_core`):

- [ ] I have updated the Python bindings accordingly
- [ ] I have verified memory safety
- [ ] I have tested with NumPy arrays

## Additional Notes

Add any other notes or context about the pull request here.

## Related Issues

Closes #(issue number)
