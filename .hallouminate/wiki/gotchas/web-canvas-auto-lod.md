# Web canvas: measured width triggers vendor auto-LOD

Source: PR #476 cure (2026-09-27). Confidence: certain (reproduced with the Playwright suite).

## Gotcha

`MikadoGraph` (vendored in `web/vendor/milknado/components/bundle.js`) steps its level of detail down
(`full` → `pill` → `dot`) when a node row overflows the width it receives. An 8-wide card row needs
about 680px. On `main` the dashboard never passed a `width`, so auto-LOD never fired. The canvas
restyle passes the measured region size from `useRegionSize`; at the Playwright viewport the canvas
region is 672px wide, so every render auto-downgraded to pills and `.mk-node` disappeared.

The `onLayout` write-back (`setGraphView({ lod })`) then locks the downgrade, because the auto path
only steps forward.

## Rule

- Pass `autoLod={false}` to `MikadoGraph` in `web/src/app/DefaultLayout.tsx`. Zoom-driven LOD and the
  user's `Node style` picks still work; only the width-overflow auto step is off.
- Before you re-enable auto-LOD, widen the browser fixture viewport or make the write-back one-shot.
- `tests/browser/test_graph_view.py` asserts `.mk-node` counts and `pending <title>` buttons; those
  exist only at the `full` level.

## Related

- Transcript lines in the session console render as two grid cells (`.mk-line-time` + text), not a
  `"kind: text"` string. Assert both cells (`tests/browser/test_node_detail.py::_expect_transcript_line`).
- The auth-gated dashboard vendors fonts through `@fontsource` (`web/src/design-system/fonts.css`);
  do not add third-party font links to `web/index.html`.
- `scripts/capture_web_ui.py` is inside the basedpyright gate (`include = ["src", "scripts", "tests"]`);
  Playwright calls need typed `Page` params and `_ =` for unused results.
