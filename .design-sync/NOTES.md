# design-sync notes

## Direction: project to repo

The Milknado design system lives in the claude.ai/design project
`68ad3112-43c2-4e42-b616-c77f726490ac` (Milknado Design System), not in this repo.
This repo consumes a compiled copy under `web/vendor/milknado/`.

Do not run the design-sync converter repo to project. The project holds the
`.jsx` sources, `tokens/*.css`, `components/mk-components.css`, guidelines and a
UI kit that the converter would overwrite or delete.

## What a pull updates

- `web/vendor/milknado/tokens.css` = `tokens/colors.css` + `typography.css` +
  `spacing.css` + `motion.css` from the project.
- `web/vendor/milknado/components/bundle.css` = `components/mk-components.css`.
- `web/vendor/milknado/components/bundle.js` stays the classic `window.Milknado`
  script; the project's `_ds_bundle.js` uses namespace `MilknadoDesignSystem_68ad31`
  and a bare `React` global, so it is not a drop-in.
- `tokens.json` and `index.d.ts` are already identical to the project copies.

## Gotchas

- 2026-09-27: the original vendored `bundle.css` read unprefixed variables
  (`--accent`) while `tokens.css` defined `--mk-*`; the dashboard rendered
  unstyled. The project CSS fixed this by prefixing everything `--mk-`.
- Fonts: the project loads Inter, Source Serif 4 and IBM Plex Mono from Google
  Fonts (`tokens/fonts.css`). The dashboard links the same stylesheet from
  `web/index.html`; offline it falls back to system-ui / Georgia / Menlo.
- After a vendor change run `npm --prefix web run build` and commit
  `src/milknado/web/static/`; `tests/browser/test_stale_build.py` gates it.
- Design canvas for the web UI: https://claude.ai/artifact/Gb9Yyqia1FrtZ3t5rxKaLg
  (15 artboards; uses the vendored `window.Milknado` bundle plus `mk-app.css`).

## Build captures uploaded to the project

- 2026-09-28: `docs/web-ui/build-2026-09-28-main-light.png` (1440x900, read-only
  viewer, light theme) shows the dashboard as built at PR #483. Captured through a
  temporary `milknado web --port 8765 --no-open` viewer with Playwright. Uploaded
  as a single-file plan; the converter did not run.
