---
applyTo: "**/*.py"
---
- Use type hints on all function signatures and return types
- Use `uv` for dependency management, not pip directly
- Use pytest for tests, not unittest
- Use Ruff for linting and formatting (`just lint-fix`)
- Prefer frozen dataclasses over raw dicts for structured data
- Use `from __future__ import annotations` for forward references
- Raise specific exceptions; avoid `raise Exception`, and use bare `raise` only to re-raise the active exception
