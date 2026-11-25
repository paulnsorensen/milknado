# Web dashboard (npm workspace)

These rules add to the root `AGENTS.md` for files under this directory.

## Commands (run from the repository root)

```bash
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web run test
npm --prefix web run build
```

`just check-llm` runs the first three. You run the build.

## Build output (narrows root "Web UI prerequisites")

- The build empties and rewrites `../src/milknado/web/static/`. Do not edit files there by hand.
- Commit the source change and the rebuilt static output together. `../tests/browser/test_stale_build.py` fails on drift.

## Vendored design system

- `vendor/milknado/` is a design-sync pull from the Milknado Design System project. Do not edit it by hand.
- Read `../.design-sync/NOTES.md` before a vendor update. After the update, rebuild and commit the static output.

## Toolchain

- Use Node `^22.12.0 || >=24.0.0`, as `package.json` declares.
- The Python server code is in `../src/milknado/web/`. `lint-imports` forbids it to import `milknado.cli`, `milknado.mcp`, or Textual.
