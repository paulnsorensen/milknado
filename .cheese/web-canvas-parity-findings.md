# Web canvas parity findings

## Test contract

The browser contract uses these accessible names and exact copy:

| Surface | Accessible names and copy |
| --- | --- |
| Force stop | `alertdialog` name `Force stop the run?`; body `The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree.`; buttons `Keep the run`, `Force stop` |
| Stop scheduling | `alertdialog` name `Stop scheduling and stop N active runs?`; body `Milknado dispatches no more nodes. Each active run stops after its current turn. Done work stays in the graph.`; buttons `Keep running`, `Stop runs` |
| Help | `dialog` name `Keyboard shortcuts`; button `Close`; regions `Graph`, `Runs`, `Steering`; width `760px` |
| Add node | `dialog` name `Add node`; fields `Description`, `Parent`, `Prerequisites`, `Files`; `group` name `Flavor` with preset buttons `implement`, `spec`, `spike`, `prototype`, `research`; buttons `Cancel`, `Add node` (visible copy includes `+`) |
| Goal review | sidecar kicker `Goal review N`; buttons `Accept change`, `Reject change`; sidecar width `560px` |
| Permission | region `Permission requested`; visible `Permission requested`, request id (accessible name `Request ID <id>`), `Command line`, command text; buttons `Approve`, `Deny`; at-risk background |
| Toast | `alert`; at-risk glyph; text `Session input was rejected: the run is not active.`; button `Dismiss`; width `320px`, bottom-right |
| Watch | badge `Read-only`; no `Stop scheduling`; no `Session guidance`; Session tab caption `Read-only`; selected-node metrics `ETA`, `Attempt`, `guidance` each show `unavailable` |

## Findings

| Board | Before | After | Status |
| --- | --- | --- | --- |
| ForceStop | Generic body and `Dismiss` / `Confirm` actions. | Exact force-stop body and `Keep the run` / `Force stop`. | Fixed |
| StopScheduling | Generic prompt and actions. | Active-run count, exact body, and `Keep running` / `Stop runs`. | Fixed |
| Permission | Command text was read only from the selected node's current detail page and was often `unavailable`; controls rendered without a matching selection. | Owner capabilities carry each pending permission command; the block requires the selected owner node and renders the command line. | Fixed |
| Watch | Read-only still showed session-input fallback and omitted observer metrics. | Read-only header, Session caption, no session input, and unavailable observer metrics inside the Run group. Verified only for a host with no live owner: the branch keys on `capabilities.owner.available`, which also flips with the active-run count. Node 99 moves it to a host-level mode flag and adds the one-owner watch and two-run owner tests. | Partial (node 99) |
| Errors | Failed node exposes the at-risk error row and the rejected-input toast from the Errors board. | `role=alert` shows the run error above the session section; the toast shows `Session input was rejected: the run is not active.` with its at-risk glyph and `Dismiss`. | Fixed |
| AddNode | Flavor was a free-text input. | Flavor is a `group` with the five preset buttons required by the canvas. | Fixed |
| Help, GoalReview | No copy changes required after live Playwright comparison. | Existing structure retained and regression coverage added or preserved. | Verified |

## Evidence

Before/after captures use a 1440x900 Chromium viewport and deterministic seeded fixtures:

- `docs/web-ui/canvas-parity-before-force-stop.png`
- `docs/web-ui/canvas-parity-before-stop-scheduling.png`
- `docs/web-ui/canvas-parity-before-help.png`
- `docs/web-ui/canvas-parity-before-add-node.png`
- `docs/web-ui/canvas-parity-before-watch.png`
- `docs/web-ui/canvas-parity-before-permission.png`
- `docs/web-ui/canvas-parity-before-toast.png`
- `docs/web-ui/canvas-parity-before-goal-review.png`
- `docs/web-ui/canvas-parity-after-force-stop.png`
- `docs/web-ui/canvas-parity-after-stop-scheduling.png`
- `docs/web-ui/canvas-parity-after-help.png`
- `docs/web-ui/canvas-parity-after-add-node.png`
- `docs/web-ui/canvas-parity-after-watch.png`
- `docs/web-ui/canvas-parity-after-permission.png`
- `docs/web-ui/canvas-parity-after-toast.png`
- `docs/web-ui/canvas-parity-after-goal-review.png`

Regression checks:

- `web/src/features/graph-edits/AddNode.test.tsx`
- `web/src/features/run-controls/ConfirmDialog.test.tsx`
- `web/src/features/run-controls/RunModeHeader.test.tsx`
- `web/src/features/session-input/PermissionActions.test.tsx`
- `web/src/features/session-input/SessionInputSection.test.tsx`
- `web/src/features/node-sidecar/NodeSidecar.test.tsx`
- `web/src/features/errors/NoticeToasts.test.tsx`
- `tests/browser/test_session_input.py`
- `tests/browser/test_session_input.py::test_rejected_inactive_session_input_shows_reason`
- `tests/browser/test_confirmations.py`
- `tests/browser/test_error_states.py`
- `tests/browser/test_help_shortcuts.py`
- `tests/browser/test_graph_edits.py`
- `tests/browser/test_goal_review.py`
- `tests/browser/test_observer_mode.py`

Hardening:

- `EditNodeDialog` seeds the selected node before paint, so live snapshot updates cannot overwrite browser-entered descriptions.
- `tests/browser/test_graph_edits.py::test_edit_node_mutates_db_and_page` covers the live edit flow.
