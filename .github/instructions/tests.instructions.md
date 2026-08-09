---
applyTo: "tests/**/*.py"
excludeAgent: "coding-agent"
---

## pytest test review

Tests run under pytest with xdist (`just test`). Live-API end-to-end tests carry the `e2e` marker (claude CLI + `ANTHROPIC_API_KEY`; slow) — new e2e tests must be marked. Coverage is gated at 95% for both the project and the diff (`codecov/patch` parity via `diff-cover` in `just check-llm`), so changed lines need real tests, not incidental execution.

Flag as weak — a test that passes even when the code is broken:

- No assertions, or existence/no-crash-only checks
- Tautological assertions, and mirror-implementation tests that re-derive the expected value from the same logic as the system under test
- Mock echo tests — asserting a mock returned what the mock was told to return
- Assertions behind conditional branches that silently pass when the branch is skipped
- Happy-path-only coverage — no error or boundary case for the behavior under test
- `len()`-only collection checks where content matters
- Mocking the system under test
- Shallow input-variation tests that duplicate one behavior instead of consolidating by behavior

Do not flag: missing type annotations, unused arguments, or broad exception assertions in tests — `tests/**` is exempted from those Ruff rules by design (see per-file-ignores in `pyproject.toml`); naming/style consistent with existing tests.
