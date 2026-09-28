# Shared TUI dogfood pass 3

The shared `milknado watch` and `milknado run` workspace passes the 80x24 compact and 120x40 standard dogfood scenarios. The scoped fixes cover paging guidance and refresh-safe changed-file selection.

## Defects

| ID | Surface | Finding | Status | Fix and regression |
| --- | --- | --- | --- | --- |
| TUI-1 | `watch`, `run`; 80x24 and 120x40 | Details help omitted the `[ ]` related-values and `( )` session-history paging hints. | Fixed | `src/milknado/app/run_view_app.py`; `tests/test_graph_navigation_ui.py::test_detail_help_exposes_paging_key_hints`, parameterized for both sizes and commands. |
| TUI-2 | `watch`, `run`; 120x40 Changes | Switching between runs that share a worktree cleared the selected changed-file path and selected the first diff again. | Fixed | `src/milknado/app/session_changes.py`; `tests/test_session_tui_scheduling.py::test_switching_shared_context_preserves_keyboard_file_choice` focuses the run table, switches with `j`, and fails on the base source. |

## Capture pairs

All pairs use Catppuccin Mocha, pinned `12:34:56`, the same fixture data, dimensions, focus, and keys.

Public evidence bundle: [GitHub Gist](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f).

| Defect | Size | Before | After |
| --- | --- | --- | --- |
| TUI-1 watch help | 80x24 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-80x24-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-80x24-after-svg) |
| TUI-1 run help | 80x24 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-80x24-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-80x24-after-svg) |
| TUI-1 watch help | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-after-svg) |
| TUI-1 run help | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-after-svg) |
| TUI-2 watch Changes | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-changes-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-changes-after-svg) |
| TUI-2 run Changes | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-changes-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-changes-after-svg) |

The bundle also contains the deterministic launcher and tapes. The help tapes focus `#details-panel` and press `?` in both states. The Changes tapes open Changes, select `second.txt`, return to the run table, and press `j` in both states.

## Reproduction

Use the launcher and the matching tape from the public bundle at the source revision under review. Run `vhs validate` on the six tapes before rendering.

The published SVG captures were inspected for dimensions, clipping, focus, key hints, status, pinned clock, and selected diff. VHS rendering in this checkout did not retain output files, so no fresh local render is claimed.

## Residuals

No defect remained in tree-left navigation, inspector paging data, focused-session preservation, Events visibility, header/footer layout, key hints, status line, or watch/run presentation parity. The Changes regression covers the shared-context transition that previously escaped the periodic-refresh test.
