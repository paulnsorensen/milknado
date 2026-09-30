# Deep-module reliability TUI evidence

These eight matched pairs compare base `37e942cc9102f63d9bf93c4c21c94fdcaf8716c8` with source `e6bc9030fba7514bb1ee2045028b8d466aed36a9`.
The after source includes the TUI change from `966c3df`.
Presentation files did not change between the requested checkpoint `51731144` and the captured revision.
The [after manifest](./after/manifest.json) records SHA-256 hashes for seven presentation inputs.

## Reproduce

Run the script from a checkout that contains this evidence directory.
Set `PYTHON` to an environment with Textual, Playwright Chromium, and the after source dependencies.
These commands create pinned source worktrees and regenerate the captures.

```bash
evidence=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
before="$work/before"
after="$work/after"
git worktree add --detach "$before" 37e942cc9102f63d9bf93c4c21c94fdcaf8716c8
git worktree add --detach "$after" e6bc9030fba7514bb1ee2045028b8d466aed36a9
python="${PYTHON:?Set PYTHON to an installed Python environment}"
script="$evidence/docs/tui-captures/deep-module-reliability/capture.py"
captures="$evidence/.context/deep-module-reliability"
published="$evidence/docs/tui-captures/deep-module-reliability"

PYTHONPATH="$before/src" "$python" "$script" \
  --source-root "$before" --output "$captures/before"
PYTHONPATH="$after/src" "$python" "$script" \
  --source-root "$after" --output "$captures/after" --warning
PYTHONPATH="$before/src" "$python" "$script" \
  --source-root "$before" --output "$captures/before" \
  --publish-to "$published/before"
PYTHONPATH="$after/src" "$python" "$script" \
  --source-root "$after" --output "$captures/after" \
  --publish-to "$published/after"
```

The baseline captures preceded the script's manifest-hash addition.
The baseline images remain unchanged.
The fixed fixture, `textual-dark` theme, `12:00:00` clock, and dimensions match across pairs.
The script validates each imported source root and source revision.
It checks after presentation hashes before publication.
The script drives `q` for run quit, `s` for graceful stop, and `q` for local watch exit.
It never presses `f` for run quit.

## Inspection

Every before and after PNG was opened at both sizes.
Main views show one active run, visible status totals, and selected-goal focus.
The standard split graph truncates several labels on both revisions.
The run footer shows `q Quit`, `f Force`, and `s Stop scheduling`.
The watch footer shows `q Quit` without stop or force controls.
The baseline `q` confirmation shows the graceful-stop message.
The after `q` confirmation says `Quit and force stop 1 active run?`.
The `s` confirmation remains graceful on both revisions.
Both confirmation choices show `y` and `n/Esc`; no message clips.
The compact modal footer still shows the underlying `Enter Open` hint.
Watch `q` exited locally with zero fixture stop calls at both sizes.

The warning appears on stderr after Textual exits.
The [standard transcript](./after/run-unconfirmed-stop-warning-120x40.txt) and [compact transcript](./after/run-unconfirmed-stop-warning-80x24.txt) contain that exact warning.
The fixture returned `False` from `force_stop_all()` once per transcript.
The transcript fixture patches `ExecutionApp.run`; it tests the wrapper's post-exit warning, not terminal rendering or live cleanup.

## Reviewer-accessible pairs

| State | 120x40 | 80x24 |
| --- | --- | --- |
| run main | [before](./before/run-main-120x40.png) / [after](./after/run-main-120x40.png) | [before](./before/run-main-80x24.png) / [after](./after/run-main-80x24.png) |
| run quit confirmation | [before](./before/run-quit-confirmation-120x40.png) / [after](./after/run-quit-confirmation-120x40.png) | [before](./before/run-quit-confirmation-80x24.png) / [after](./after/run-quit-confirmation-80x24.png) |
| run graceful stop confirmation | [before](./before/run-stop-confirmation-120x40.png) / [after](./after/run-stop-confirmation-120x40.png) | [before](./before/run-stop-confirmation-80x24.png) / [after](./after/run-stop-confirmation-80x24.png) |
| watch main | [before](./before/watch-main-120x40.png) / [after](./after/watch-main-120x40.png) | [before](./before/watch-main-80x24.png) / [after](./after/watch-main-80x24.png) |

## Limits

The fixture supplies synthetic snapshots.
These frames do not start a live worker, controller, provider, CLI process, or database.
The watch fixture checks local exit behavior, not remote worker state.
The warning fixture checks stderr output after a synthetic failed result, not a real failed cleanup.
Chromium renders exported Textual SVGs, so the frames do not prove terminal-emulator palette fidelity.