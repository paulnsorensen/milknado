# ADR — Cross-feature web UI state lives in `web/src/shared/`

Date: 2026-09-20 · Status: accepted · Issue: #464

## Decision

Shared state lives in `web/src/shared/<name>/` when more than one web feature reads it.
Each shared module exposes one public surface, its `index.ts`.
Features import the `index.ts` only, never a file inside the module.

The first module is `web/src/shared/node-detail/`.
It holds the active sidecar tab (`detailTab`), the paged node detail (`nodeDetail`), and the node detail wire types (`detailWire`).
The `node-sidecar`, `changes`, `session-input`, `shortcuts`, and `narrow` features import it.

## Why

The state started inside `features/node-sidecar/`.
Three other features imported its internal files, so a `node-sidecar` refactor could break them silently.

`web/src/app/` was rejected as the home.
That directory holds shell wiring (`slots`, `actions`, `registry`, `store`, `api`), not feature domain state.
The app layer must not depend on a feature's wire types.

## Limits

No lint rule enforces the boundary.
`web/eslint.config.js` has no import-boundary plugin, and adding one needs a new npm dependency.
A review must check that a feature imports `shared/<name>` and not `shared/<name>/<file>`.

A `vi.mock('../../shared/node-detail', ...)` replaces the whole public surface.
Use `importOriginal` and override only the exports the test isolates.

`features/narrow/NarrowDetail.tsx` still imports the `NodeSidecar` component from `features/node-sidecar/`.
That component import is a separate question and stays open.
