# PR 483: matched TUI review evidence

These 52 matched pairs cover run, read-only watch, and attached watch.
Every PNG was opened and inspected for clipping, focus, key hints, and visible status.
These are synthetic presentation captures, not live worker evidence.

## Revisions and environment

- Before: `d7fa73bcff47a099b315815534996eb547319a05`, the split execution-engine base.
- After: `7bae8d62bbd5af997727fda0f35ea60cf1ea79ce`, the original PR 483 source.
- Terminal dimensions: standard `120x40`; compact `80x24`.
- Theme: Textual `textual-dark`; header clock: `12:00:00`.
- Runtime: Python 3.13.13; Textual 8.2.8; Chromium through Playwright.
- Data: the same five-node snapshot from [the existing fixture](../agent-steering/fixture.py).
- Renderer: real `ExecutionApp` and `WatchApp` under `App.run_test`, then Chromium renders each exported SVG.

PNGs are the published evidence.
The script also creates SVG intermediates locally.
Manifests record source revisions, screen classes, focus identifiers, and pending review counts.

## Reproduce

Run these commands from a checkout that contains this evidence script and the existing fixture.
Use its installed Python environment with Playwright Chromium available.
Set `PYTHON` to that environment's Python executable.

```bash
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
git worktree add --detach "$work/before" d7fa73bcff47a099b315815534996eb547319a05
git worktree add --detach "$work/after" 7bae8d62bbd5af997727fda0f35ea60cf1ea79ce
PYTHON=${PYTHON:-"$repo/.venv/bin/python"}
for phase in before after; do
  PYTHONPATH="$work/$phase/src" "$PYTHON" "$repo/docs/tui-captures/pr483/capture.py" \
    --source-root "$work/$phase" --output "$repo/docs/tui-captures/pr483/$phase"
done
```

The script checks that imported Milknado files belong to the selected source checkout.
It waits for Textual message processing and submission workers.
Use fresh output directories for a complete rerun.
`--resume` retains previously inspected frames from the same revision and fills missing matrix cells.

## State controls

| State | Deterministic input |
|---|---|
| main | Mount with goal 12 selected. |
| help | Press F1. |
| confirmation | Open the force-stop confirmation for run-12. |
| stop-confirmation | Press s; baseline has no binding, while after opens the stop-scheduling confirmation. |
| session | Press i and set the draft to “Keep the change bounded.” |
| submitted | Use the session draft, press Enter, and wait for synthetic command admission. |
| permission | Press i; select approve and permission perm-1 without submission. |
| error | Supply a session error and a listener error in the snapshot. |
| owner-unavailable | Supply no live session actions and unavailable run controls. |
| review | Supply the same review event; after also supplies one pending durable review record. |
| no-worktree | Open detail and select the Changes tab with no session context. |

The before runtime has no `pending_goal_reviews` field.
The wrapper adds that field only when the runtime exposes it.
No fake review event substitutes for the after runtime's durable field.

## Inspection results and retained risks

- Stop-scheduling help and footer hints fit both layouts on the run surface.
- The stop confirmation fits at 120x40 and wraps at 80x24.
- Session drafts, permission identifiers, and Send remain visible in both layouts.
- Successful synthetic submission clears the input on both revisions.
- Help and confirmation screens show close or cancel keys.
- Read-only watch has no session input controls.
- Standard error frames show both the session error and listener error.
- Compact error frames show the listener error; session detail stays closed.
- Compact owner-unavailable frames stay on the graph; standard frames expose unavailable controls.
- Compact review frames show the full goal and review identifiers.
- **Retained PR 483 risk:** standard review headers truncate the durable review message to “Review pendin…”.
  This hides the goal and review identifiers on all three surfaces.
- **Retained risk:** no-worktree Changes shows an empty file list and diff on both revisions.
  These captures do not show an explanatory no-worktree message.
- **Retained hint issue:** compact confirmation footers also show “Enter Open”.
  The force-stop baseline already shows this hint.
- The fixture exposes cancel and force availability in attached-watch help.
  Attached watch does not gain those run-control bindings from this fixture.
- The header icon renders as a missing glyph in this browser environment on both revisions.

These findings document the existing source; this evidence change makes no production fixes.

## Limits

No capture starts a CLI process, live worker, provider connection, owner process, or database.
Attached watch uses an in-process synthetic admission callback.
Submission captures test successful UI clearing, not cross-process delivery or competing-draft races.
The permission state selects a request but does not approve it.
Confirmation captures do not confirm a destructive action.
No-worktree captures select the tab directly; they do not test a live Git worktree.
The fixture's old node review data does not prove durable review persistence.
Browser-rendered SVGs do not establish terminal-emulator palette fidelity.
These captures do not test 40-column layouts, resize transitions, rapid typing, or terminal cleanup.

## PNG pairs

| Surface | State | 120x40 | 80x24 |
|---|---|---|---|
| run | main | [before](./before/run-main-120x40.png) / [after](./after/run-main-120x40.png) | [before](./before/run-main-80x24.png) / [after](./after/run-main-80x24.png) |
| run | help | [before](./before/run-help-120x40.png) / [after](./after/run-help-120x40.png) | [before](./before/run-help-80x24.png) / [after](./after/run-help-80x24.png) |
| run | confirmation | [before](./before/run-confirmation-120x40.png) / [after](./after/run-confirmation-120x40.png) | [before](./before/run-confirmation-80x24.png) / [after](./after/run-confirmation-80x24.png) |
| run | stop-confirmation | [before](./before/run-stop-confirmation-120x40.png) / [after](./after/run-stop-confirmation-120x40.png) | [before](./before/run-stop-confirmation-80x24.png) / [after](./after/run-stop-confirmation-80x24.png) |
| run | session | [before](./before/run-session-120x40.png) / [after](./after/run-session-120x40.png) | [before](./before/run-session-80x24.png) / [after](./after/run-session-80x24.png) |
| run | submitted | [before](./before/run-submitted-120x40.png) / [after](./after/run-submitted-120x40.png) | [before](./before/run-submitted-80x24.png) / [after](./after/run-submitted-80x24.png) |
| run | permission | [before](./before/run-permission-120x40.png) / [after](./after/run-permission-120x40.png) | [before](./before/run-permission-80x24.png) / [after](./after/run-permission-80x24.png) |
| run | error | [before](./before/run-error-120x40.png) / [after](./after/run-error-120x40.png) | [before](./before/run-error-80x24.png) / [after](./after/run-error-80x24.png) |
| run | owner-unavailable | [before](./before/run-owner-unavailable-120x40.png) / [after](./after/run-owner-unavailable-120x40.png) | [before](./before/run-owner-unavailable-80x24.png) / [after](./after/run-owner-unavailable-80x24.png) |
| run | review | [before](./before/run-review-120x40.png) / [after](./after/run-review-120x40.png) | [before](./before/run-review-80x24.png) / [after](./after/run-review-80x24.png) |
| run | no-worktree | [before](./before/run-no-worktree-120x40.png) / [after](./after/run-no-worktree-120x40.png) | [before](./before/run-no-worktree-80x24.png) / [after](./after/run-no-worktree-80x24.png) |
| watch | main | [before](./before/watch-main-120x40.png) / [after](./after/watch-main-120x40.png) | [before](./before/watch-main-80x24.png) / [after](./after/watch-main-80x24.png) |
| watch | help | [before](./before/watch-help-120x40.png) / [after](./after/watch-help-120x40.png) | [before](./before/watch-help-80x24.png) / [after](./after/watch-help-80x24.png) |
| watch | error | [before](./before/watch-error-120x40.png) / [after](./after/watch-error-120x40.png) | [before](./before/watch-error-80x24.png) / [after](./after/watch-error-80x24.png) |
| watch | owner-unavailable | [before](./before/watch-owner-unavailable-120x40.png) / [after](./after/watch-owner-unavailable-120x40.png) | [before](./before/watch-owner-unavailable-80x24.png) / [after](./after/watch-owner-unavailable-80x24.png) |
| watch | review | [before](./before/watch-review-120x40.png) / [after](./after/watch-review-120x40.png) | [before](./before/watch-review-80x24.png) / [after](./after/watch-review-80x24.png) |
| watch | no-worktree | [before](./before/watch-no-worktree-120x40.png) / [after](./after/watch-no-worktree-120x40.png) | [before](./before/watch-no-worktree-80x24.png) / [after](./after/watch-no-worktree-80x24.png) |
| attached-watch | main | [before](./before/attached-watch-main-120x40.png) / [after](./after/attached-watch-main-120x40.png) | [before](./before/attached-watch-main-80x24.png) / [after](./after/attached-watch-main-80x24.png) |
| attached-watch | help | [before](./before/attached-watch-help-120x40.png) / [after](./after/attached-watch-help-120x40.png) | [before](./before/attached-watch-help-80x24.png) / [after](./after/attached-watch-help-80x24.png) |
| attached-watch | session | [before](./before/attached-watch-session-120x40.png) / [after](./after/attached-watch-session-120x40.png) | [before](./before/attached-watch-session-80x24.png) / [after](./after/attached-watch-session-80x24.png) |
| attached-watch | submitted | [before](./before/attached-watch-submitted-120x40.png) / [after](./after/attached-watch-submitted-120x40.png) | [before](./before/attached-watch-submitted-80x24.png) / [after](./after/attached-watch-submitted-80x24.png) |
| attached-watch | permission | [before](./before/attached-watch-permission-120x40.png) / [after](./after/attached-watch-permission-120x40.png) | [before](./before/attached-watch-permission-80x24.png) / [after](./after/attached-watch-permission-80x24.png) |
| attached-watch | error | [before](./before/attached-watch-error-120x40.png) / [after](./after/attached-watch-error-120x40.png) | [before](./before/attached-watch-error-80x24.png) / [after](./after/attached-watch-error-80x24.png) |
| attached-watch | owner-unavailable | [before](./before/attached-watch-owner-unavailable-120x40.png) / [after](./after/attached-watch-owner-unavailable-120x40.png) | [before](./before/attached-watch-owner-unavailable-80x24.png) / [after](./after/attached-watch-owner-unavailable-80x24.png) |
| attached-watch | review | [before](./before/attached-watch-review-120x40.png) / [after](./after/attached-watch-review-120x40.png) | [before](./before/attached-watch-review-80x24.png) / [after](./after/attached-watch-review-80x24.png) |
| attached-watch | no-worktree | [before](./before/attached-watch-no-worktree-120x40.png) / [after](./after/attached-watch-no-worktree-120x40.png) | [before](./before/attached-watch-no-worktree-80x24.png) / [after](./after/attached-watch-no-worktree-80x24.png) |
