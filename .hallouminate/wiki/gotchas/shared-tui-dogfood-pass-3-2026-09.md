# Shared TUI dogfood pass 3

The shared `milknado watch` and `milknado run` workspace passes the 80x24 compact and 120x40 standard dogfood scenarios. The scoped fixes cover paging guidance, refresh-safe changed-file selection, and focus-helper reuse.

## Defects

| ID | Surface | Finding | Status | Fix and regression |
| --- | --- | --- | --- | --- |
| TUI-1 | `watch`, `run`; 80x24 and 120x40 | Details help omitted the `[ ]` related-values and `( )` session-history paging hints. | Fixed | `src/milknado/app/run_view_app.py`; `tests/test_graph_navigation_ui.py::test_detail_help_exposes_paging_key_hints`, parameterized for both sizes and commands. |
| TUI-2 | `watch`, `run`; 120x40 Changes | Switching between runs that share a worktree cleared the selected changed-file path and selected the first diff again. | Fixed | `src/milknado/app/session_changes.py`; `tests/test_session_tui_scheduling.py::test_switching_shared_context_preserves_keyboard_file_choice` focuses the run table, switches with `j`, and fails on the base source. |
Note: TUI-3 is a code-quality follow-up, not an operator-facing defect. It has no capture pair.

## Capture pairs

Each pair uses the same fixture data, terminal dimensions, focus, interaction keys, and pinned `12:34:56` clock.

Public evidence bundle: [GitHub Gist](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f).

Capture source revisions:

- Before: `ac969ecfe791059a1ffb8a9d64849b7eafddb68a`
- After: `78f2976c5ab322b3523419a3370524a3d021f735`

The SVGs come from the published `capture_tui.py` pilot script, which runs at those revisions and calls Textual `App.save_screenshot(filename=...)`. The six parity VHS tapes and four dogfood VHS tapes reproduce the same interactions independently.

| Defect | Size | Before | After |
| --- | --- | --- | --- |
| TUI-1 watch help | 80x24 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-80x24-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-80x24-after-svg) |
| TUI-1 run help | 80x24 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-80x24-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-80x24-after-svg) |
| TUI-1 watch help | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-after-svg) |
| TUI-1 run help | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-after-svg) |
| TUI-2 watch Changes | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-changes-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-watch-120x40-changes-after-svg) |
| TUI-2 run Changes | 120x40 | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-changes-before-svg) | [capture](https://gist.github.com/paulnsorensen/340ed730f5485c4f8345c3ab155d195f#file-run-120x40-changes-after-svg) |

The VHS tapes set the Catppuccin Mocha terminal theme. The SVGs are Textual exports and retain Textual's application palette; the SVG captures make no Catppuccin palette claim.

## Reproduction

The published SVGs come from Textual Pilot and `App.save_screenshot`, not VHS. The
before capture revision is `ac969ecfe791059a1ffb8a9d64849b7eafddb68a`. The after
capture revision is `78f2976c5ab322b3523419a3370524a3d021f735`.

With the repository checkout in `$REPO` and the public bundle in `$BUNDLE`, reproduce the
published SVGs with the same command from the bundle README:

```bash
BASE=ac969ecfe791059a1ffb8a9d64849b7eafddb68a
AFTER=78f2976c5ab322b3523419a3370524a3d021f735
for pair in "before:$BASE" "after:$AFTER"; do
  state=${pair%%:*}
  revision=${pair#*:}
  work="/tmp/shared-tui-pass-3-$state"
  rm -rf "$work"
  mkdir -p "$work"
  git -C "$REPO" archive "$revision" | tar -x -C "$work"
  mkdir -p "$work/.cheese/tui-demo/dogfood"
  cp "$BUNDLE/launch_tui.py" "$BUNDLE/capture_tui.py" "$work/.cheese/tui-demo/dogfood/"
  (
    cd "$work"
    for kind in watch run; do
      for size in "80 24 details" "120 40 details" "120 40 changes"; do
        set -- $size
        columns=$1
        rows=$2
        scenario=$3
        if [ "$scenario" = details ]; then
          output=".cheese/tui-demo/dogfood/${kind}-${columns}x${rows}-${state}.svg"
        else
          output=".cheese/tui-demo/dogfood/${kind}-${columns}x${rows}-${scenario}-${state}.svg"
        fi
        PYTHONPATH=src:. uv run --no-project python \
          .cheese/tui-demo/dogfood/capture_tui.py \
          "$kind" "$scenario" "$state" "$output" "$columns" "$rows"
      done
    done
  )
done
```

The helper uses Textual Pilot and `App.save_screenshot`, with the pinned `12:34:56`
clock, identical fixture data, focus, keys, and 80x24 or 120x40 dimensions.
The details capture presses `?` in both revisions, so each help pair uses the same open-overlay state.
`vhs validate '.cheese/tui-demo/dogfood/*.tape' '.cheese/tui-demo/parity-hitl/*.tape'` validates all ten VHS tapes. Those tapes set Catppuccin Mocha and produce PNG/GIF output; the SVGs retain Textual's application palette and make no Catppuccin palette claim.

## Residuals

No defect remained in tree-left navigation, inspector paging data, focused-session preservation, Events visibility, header/footer layout, key hints, status line, or watch/run presentation parity. The Changes regression covers the shared-context transition that previously escaped the periodic-refresh test.
