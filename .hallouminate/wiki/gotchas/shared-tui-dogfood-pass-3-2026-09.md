# Shared TUI dogfood pass 3

The shared `milknado watch` and `milknado run` workspace passes the 80x24 compact and 120x40 standard dogfood scenarios. The scoped fixes cover paging guidance and refresh-safe changed-file selection.

## Defects

| ID | Surface | Finding | Status | Fix and regression |
| --- | --- | --- | --- | --- |
| TUI-1 | `watch`, `run`; 80x24 and 120x40 | Details help omitted the `[ ]` related-values and `( )` session-history paging hints. | Fixed | `src/milknado/app/run_view_app.py`; `tests/test_graph_navigation_ui.py::test_detail_help_exposes_paging_key_hints`, parameterized for both sizes and commands. |
| TUI-2 | `watch`, `run`; 120x40 Changes | Switching between runs that share a worktree cleared the selected changed-file path and selected the first diff again. | Fixed | `src/milknado/app/session_changes.py`; `tests/test_session_tui_scheduling.py::test_switching_shared_context_preserves_keyboard_file_choice`. The regression fails against `HEAD` base and passes with the fix. |

## Evidence

The reviewer-accessible evidence bundle is [published on GitHub Gist](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f). The implementation and evidence are tracked in [PR #480](https://github.com/paulnsorensen/milknado/pull/480). It contains the 12 before/after SVG captures, the deterministic fixture launcher, and the VHS tapes.

The capture pairs use the same fixture data, interaction state, terminal size, Catppuccin Mocha theme, and pinned `12:34:56` header clock:

- Help: `watch-80x24-before.svg` / `watch-80x24-after.svg`, `run-80x24-before.svg` / `run-80x24-after.svg`, `watch-120x40-before.svg` / `watch-120x40-after.svg`, and `run-120x40-before.svg` / `run-120x40-after.svg`.
- Changes: `watch-120x40-changes-before.svg` / `watch-120x40-changes-after.svg` and `run-120x40-changes-before.svg` / `run-120x40-changes-after.svg`.
- Reproduction: `watch-80x24-help.tape`, `run-80x24-help.tape`, `watch-120x40-help.tape`, `run-120x40-help.tape`, and the two 120x40 changes tapes. Each pair uses one tape with identical keys before and after.
- The fixture launcher supports `details` to focus `#details-panel` and `changes` to seed two runs with one shared worktree.

The VHS commands completed for the tapes with `CHROME_BIN` set to the local Playwright Chromium executable. This checkout's VHS renderer did not retain generated output files, so the pinned SVGs in the public bundle are the inspected captures.

## Residuals

No defect remained in tree-left navigation, inspector paging data, focused-session preservation, Events visibility, header/footer layout, key hints, status line, or watch/run presentation parity. The changes regression covers the shared-context transition that previously escaped the periodic-refresh test.
