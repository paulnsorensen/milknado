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

Mechanical slice-boundary breaches are enforced by import-linter (`lint-imports`, part of `just check-llm`) — do not re-flag plain forbidden imports; review what the checker cannot see. Test quality rules are scoped to `tests/**` in `tests.instructions.md`; deep slice review is scoped to `src/milknado/**` in `sliced-bread-review.instructions.md`; domain barrel diffs get `barrel.instructions.md` — do not duplicate those categories here.

## What NOT to Comment On

- Linting — handled by Ruff in CI
- Formatting and whitespace — handled by `ruff format` in CI
- Import ordering — handled by tooling
- Slice-boundary import contracts — enforced by import-linter in the gate
- Source file-size budget — enforced by `scripts/check_file_lengths.py` in the gate
- Missing docstrings on internal or private functions
- Style preferences that are consistent with the rest of the codebase
- Nitpicks with no functional impact

## Review Style

- Only comment when confidence is high
- If a pattern is used consistently elsewhere in the codebase, do not flag it as wrong
- Suggest specific fixes, not vague improvements
- One comment per issue — do not repeat the same feedback on multiple occurrences
