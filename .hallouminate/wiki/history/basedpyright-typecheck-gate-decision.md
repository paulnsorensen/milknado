# Basedpyright type-check gate — decision

## ADR-basedpyright-typecheck-gate-001: basedpyright as the type checker of record  [status: accepted]

- **Decision:** Use basedpyright at its recommended tier for the full Python tree: `src/`, `scripts/`, and `tests/`.
- **Gate:** Run `uv run basedpyright` through `just typecheck`. Enforce it after dead-code checks in `just check-llm` and `just build-ci`.
- **CI:** Run a dedicated `typecheck` job that invokes `just typecheck`, matching the easy-cheese workflow pattern.
- **Rejected:** Retaining `ty>=0.0.38` or running two checkers was rejected. The extra checker duplicated diagnostics and allowed policy drift.
- **Consequence:** basedpyright is the sole type checker. The gate requires zero errors and zero warnings across the full tree.

Sources: `justfile`, `pyproject.toml`, `.github/workflows/ci.yml`, and `CLAUDE.md`.
