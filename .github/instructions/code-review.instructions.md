---
applyTo: "**"
excludeAgent: "coding-agent"
---

## Code Review Focus

Focus reviews on these categories, in priority order:

1. **Security** - Flag hardcoded secrets, SQL injection, XSS, command injection, path traversal, and insecure deserialization
2. **Silent failures** - Flag empty catch blocks, swallowed errors, missing error propagation, or functions that return `None`/null/undefined on failure without signaling
3. **Coupling violations** - Flag domain/model code that imports infrastructure (HTTP, DB, file I/O, framework decorators)
4. **Complexity violations** - Flag functions over 40 lines, files over 300 lines, functions with more than 4 parameters, nesting deeper than 3 levels
5. **Architectural violations** - Flag cross-slice internal imports, mutable shared state, God classes/modules
6. **Weak tests** - Flag tests that prove nothing: no assertions, `is_ok()`/`is_some()` without checking inner values, tautological assertions, mirror-implementation tests that re-derive expected values from the same logic, mock echo tests, `.len()`-only collection checks, assertions gated behind conditional branches that silently pass, and happy-path-only coverage with no error/boundary tests

## What NOT to Comment On

- Linting — handled by Ruff in CI
- Formatting and whitespace — handled by `ruff format` in CI
- Import ordering — handled by tooling
- Missing docstrings on internal or private functions
- Style preferences that are consistent with the rest of the codebase
- Nitpicks with no functional impact

## Review Style

- Only comment when confidence is high
- If a pattern is used consistently elsewhere in the codebase, do not flag it as wrong
- Suggest specific fixes, not vague improvements
- One comment per issue — do not repeat the same feedback on multiple occurrences
