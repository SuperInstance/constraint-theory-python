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

## Known Security Considerations

- This library performs geometric calculations and does not handle cryptographic operations
- The library uses PyO3 for Python bindings, benefiting from Rust's memory safety
- No known security vulnerabilities in the current version

## Security Updates

Security updates will be released as patch versions. Subscribe to GitHub releases to be notified of updates.

---

Thank you for helping keep Constraint Theory Python bindings secure!
