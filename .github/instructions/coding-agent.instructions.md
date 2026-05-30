---
applyTo: "**"
excludeAgent: "code-review"
---

## Coding Agent Guidelines

When implementing changes:

- Read existing code before modifying it — understand the patterns in use
- Follow the existing code style of the file you are editing
- Keep changes minimal and focused on the issue at hand
- Do not refactor surrounding code unless the issue requires it
- Do not add docstrings to helper functions or private methods with clear names
- Do not introduce new dependencies without explicit approval
- Write tests for new functionality — match the existing test patterns in `tests/`
- Prefer editing existing files over creating new ones
- Run `just build` before opening a PR — it must pass (lint, tests, and the 90% coverage gate)

## Architecture Rules

- New domain concepts go in their own module under `src/milknado/domains/`
- Public API is exposed through `__init__.py` barrel files only
- Do not create abstract classes, interfaces, or factories unless there are 2+ concrete implementations
- Core models must not import infrastructure code
- Cross-slice translation belongs in an explicit bridge module, not inside the pure slice
