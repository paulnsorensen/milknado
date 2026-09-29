# Campaign dogfood gotchas (structural-debt sweep, 2026-09-06)

Lessons from running Milknado on itself for goal node 107 (twenty `structural-debt` issues, eight `implement` tasks) on branch `sdw/campaign`.

## The reviewer command must work from the launching shell

`review_agent` inherits the environment of the `milknado run` process.
The dotfiles `claude` wrapper (`claude-guard`) refuses to start when more than eight claude sessions run or RAM is low.
A refused reviewer exits 1 with empty stdout.
Milknado records that as `reviewer produced no parseable <verdict> tag`, marks the node `blocked`, and preserves the worktree.
The worker's work is complete but unreviewed.

Check before a campaign: run the exact `review_agent` string once from the same shell and confirm it prints a `<verdict>` tag.
Launch the loop with `CLAUDE_GUARD=0` when the guard would refuse; the guard's own message names that override and the operator owns the decision.
Do not switch the reviewer to OpenRouter or any other paid API route as a workaround; the owner decides billing routes, not the agent.

## `milknado.toml` is snapshot-tested

`tests/test_meta_flavor_profiles.py` pins every field of every repository flavor profile, including the `review_agent` string.
A config change on a campaign branch must update `EXPECTED_PROFILES` in the same commit.
Otherwise every worker fails the gate and edits the test itself to make it pass, which pollutes every node diff with a test change.
Two workers in this campaign did exactly that with two different constant names.

## A reviewer error does not resume preserved work

A blocked node (reviewer error or reviewer timeout) can only return to `pending` by operator action.
Re-dispatch does not reuse the preserved worktree; `WorktreeManager` relocates to a `-2` suffix and the worker starts from the base again.
Save `git diff` from the preserved worktree before removing it if the earlier solution is worth comparing.

## Reviewer timeout was a hard-coded 30 minutes

`_drain_review_run` and the review `RunConfig` in `adapters/loop.py` used a literal 1800 seconds.
An Opus review of the largest node diff (three fence-invariant issues in graph persistence) exceeded it and was recorded as `reviewer timed out before producing a verdict`.
The campaign adds `review_timeout_seconds` (per flavor, inherits like `review_max_rounds`) so the cap is configurable.
Review logs used to live in a `tempfile.TemporaryDirectory`, so a timed-out review left no log to diagnose.
The campaign moves them to `<worktree>/.ralph-logs/review/` so they persist.

## Large briefs overflow one argv element

Linux limits a single argv string to `MAX_ARG_STRLEN` (131072 bytes).
A correction-round brief (task brief plus review findings plus prior context) crossed that limit.
The omp adapter passed the whole prompt as one argument, so `execve` failed with `E2BIG` before omp started.
The symptom is a 0-byte review or worker log and an instant `reviewer produced no parseable <verdict> tag`.
The omp adapter now delivers the prompt on stdin (`OmpAdapter.deliver_prompt`); tracked as GitHub issue #422.

## Hand corrected work back through the brief

When a reviewer error blocks a node whose corrections already passed the gate, save `git diff` from the preserved worktree to a handoff file.
Prepend a `PRIOR ATTEMPT` paragraph to the node description that names the file and tells the worker to `git apply --3way` it first, then re-verify each finding.
Node 113 landed on the next dispatch this way with no repeated implementation work.

## A multi-root database never completes the goal

`get_root()` returns the live root with the lowest id, and `complete_root()` requires every live non-root node to be `done`.
A project database that keeps older goals alive (here nodes 1, 71, 83 beside 107) therefore never flips the running goal to `done` on a specless run.
Archive finished campaigns before starting a new goal, or mark the goal done by hand after the loop exits.
Tracked as GitHub issue #433.

## Launch the run loop from an operator terminal

A run loop started as a background process of an agent harness inherits that harness's resource guards.
The Claude Code memory guard killed the loop twice with about 30 GB free.
Start `CLAUDE_GUARD=0 uv --directory <repo> run milknado run --project-root <repo>` from the operator's own shell and let the agent monitor the database.

## Killing a coordinator with dirty worktrees crashes it

`Executor.fail()` calls `WorktreeManager.remove` before `mark_failed`; a dirty worktree raises `UnlandedWorkError` and the exception escapes the run loop.
The node and its run row stay `running` with `pid=NULL`, and `run_inline_poll` cannot reconcile them.
Tracked as GitHub issue #420.
Manual recovery: `milknado_todo_set_status(node, "pending")`, finalize the orphaned `runs` rows by hand, remove the preserved worktrees and branches, relaunch.

## omp reviewer output is noisy

The review adapter drains `result_text` and `echo_stdout` events.
With an omp reviewer that means tool-call echoes and JSON-escaped text land in the findings file (`.cheese/age/<slug>.md`), around 65 KB for 2 KB of prose.
The verdict still parses, and the worker copes, but the findings brief is expensive.

## Pass 3: web and TUI live-run dogfood (2026-09-27)

The web dashboard and `milknado watch` consume the same observer projection. Pass 3 records the presentation, read-path, scheduling, and evidence defects below.

### Reproduction and evidence

- External-provider attempt: `/tmp/milknado-pass3-real4`, goal node 1, task node 2, run `node-2-20260928T035512Z-8e02e257`. The configured OMP worker remained active across two waits because nested worker context stopped at `milknado_deposit_result`; the [blocker record](./evidence/pass-3-real-worker-blocker.txt) documents the observation.
- Real deterministic run: `/tmp/milknado-pass3-real-final9`, goal node 1, task node 2, run `node-2-20260928T070051Z-0f8909e3`. `milknado web` and `milknado watch` were open before launch. The clean launching shell removed inherited `MILKNADO_*` context.
- The deterministic OMP worker accepted the attached-watch `proof` command, called the result sink, emitted `MILKNADO_NODE_COMPLETE`, and exited. The graph reached `2/2 complete`.
- Completion evidence: [wide web text](./evidence/pass-3-node-2-real-web-completion.txt), [wide web image](./evidence/pass-3-node-2-real-web-completion-wide.webp), [648px clipping capture](./evidence/pass-3-node-2-real-web-completion.webp), [watch](./evidence/pass-3-node-2-real-watch-completion.txt), [run summary](./evidence/pass-3-node-2-real-run.txt), and [receipts](./evidence/pass-3-node-2-real-command-receipts.txt).
- Archive evidence: [web](./evidence/pass-3-node-2-real-web-archive.txt) and [watch](./evidence/pass-3-node-2-real-archive.txt). After `milknado graph archive 1`, both surfaces showed no visible nodes or runs. The database recorded `archived_at` for goal 1 and task 2.
- The new final9 captures are distinct files. They use node 2 and the same run ID. The watch output contains no host hook text or personal session content.
- The 648px capture clips the canvas between 401px and 1179px, hiding the sidecar controls and receipt; the wide 1180px capture shows the selected node, Session tab, delivered `proof`, and completion line.



### PR 483 concern split and evidence limits (2026-09-29)

PR 483 is divided into engine/runtime, shared session state and TUI, then web dashboard concerns.
The latter layers depend on the preceding shared contracts.
Historical dogfood captures keep their original revision and fixture limits; they do not verify a newly split branch.
The historical gate below also applies only to its recorded run.

The split adds 52 matched TUI pairs at 120x40 and 80x24, using fixed snapshots, theme, and clock.[^split-evidence]
The before runtime is the engine-layer base; the after runtime is the original PR 483 source.
All application, domain, adapter, loop, and MCP files match the session layer byte-for-byte.
These captures use real app classes but no live worker, provider transport, or database.
They show retained limits: standard review headers truncate identifiers, no-worktree Changes stays blank, and compact confirmation has an extra Enter hint.

The historical web stop-scheduling pair shows only a scrim; it does not prove dialog copy or keyboard focus.[^split-web]
The inspected owner-mode, permission-ownership, and narrow-control pairs remain useful historical evidence.
Run each split branch's current gate instead of reusing the historical PASS.

[^split-evidence]: [Matched TUI evidence, commands, and limits](https://github.com/paulnsorensen/milknado/blob/7fc412d555cc36232c4bae1bd24221e27cc8cad3/docs/tui-captures/pr483/README.md).
[^split-web]: [Original PR 483](https://github.com/paulnsorensen/milknado/pull/483); `docs/web-ui/canvas-parity-before-stop-scheduling.png` and `canvas-parity-after-stop-scheduling.png`, inspected 2026-09-29.

_Source: PR 483 split and visual inspection · Updated: 2026-09-29 · Supersedes: no historical result_

### Durable source links

- [Observer read path](../../../src/milknado/domains/graph/observer.py)
- [Archive regression](../../../tests/test_graph_observer_snapshots.py)
- [OMP settled mapping](../../../src/milknado/loop/sessions/_omp.py)
- [OMP regression](../../../tests/test_session_omp.py)
- [Agent roster component](../../../web/src/features/agent-roster/AgentRosterSection.tsx)
- [Agent roster browser regression](../../../tests/browser/test_narrow_layout.py)
- [TUI footer actions](../../../src/milknado/app/run_overlays.py)
- [TUI diff path](../../../src/milknado/app/session_changes.py)
- [TUI session input scheduling](../../../src/milknado/app/session_commands.py)

### Pass-3 defect table

| Defect | Found | Fixed | Deferred | Evidence and regression |
| --- | --- | --- | --- | --- |
| `session_settled` rendered as `Unsupported OMP RPC event` | Yes | Yes | No | `_omp.py` maps `agent_settled` and `session_settled` to read-only `settled` status events. Regression: `tests/test_session_omp.py::test_non_terminal_turn_and_settled_event_are_read_only`. |
| Archived node runs remained in the observer run list | Yes | Yes | No | `_durable_runs` filters `n.archived_at IS NULL`. Regression: `tests/test_graph_observer_snapshots.py::test_observer_hides_runs_for_archived_nodes`. |
| Long agent descriptions overlapped the roster row | Yes | Yes | No | The roster clamps long descriptions. Regression: `tests/browser/test_narrow_layout.py::test_long_agent_description_is_clamped_in_the_roster`. |
| Footer action dispatch and focus behavior were reported as pass-3 fixes | No (base #473) | No (already on base) | No | Base commit `0063e632` (#473) already routes footer dispatch to the binding owner and keeps footer hints unfocusable. This pass adds no behavior change; `tests/test_execution_tui.py::test_mounted_footer_tracks_tree_selection_at_fixed_width` asserts the mounted hints cannot take focus. |
| Stale diff responses could replace a newly selected run | No (base #473) | No (already on base) | No | Base commit `0063e632` (#473) already rejects stale results by token and resets `_pending_diff` on selection changes; this branch only reformats that line in `src/milknado/app/session_changes.py`. Regression on base: `tests/test_execution_tui.py::test_stale_diff_response_cannot_replace_newly_selected_run`. |
| Session input could submit the widget value without updating the per-run draft revision | Yes | Yes | No | `_send_session_input` copies the widget value into the per-run draft and advances its revision before queueing. `_session_input_result` clears only the matching draft, even when the completed run is no longer selected. Regression: `tests/test_session_tui_scheduling.py::test_widget_value_is_submitted_when_draft_store_is_stale` and `::test_accepted_result_clears_draft_after_selection_changes`. |
| Web canvas clipped at viewport widths from 401px through 1179px | Yes | No | Yes | The linked 648px final9 capture demonstrates the clipped header and hidden sidecar controls. The 1180px capture proves the wide completion view. Follow-up node `103` owns the layout gap. |
| Real external OMP worker inherited nested context and stalled at `milknado_deposit_result` | Yes | No | Yes | The external-provider attempt remains blocked. Follow-up node `102` registers removal of the inherited nested-worker-context stall. The deterministic final9 run proves the clean worker path. |
| Controller cancellation recovery for a dead coordinator PID remains an engine gap | Yes | No | Yes | Existing follow-up node `79` (#309) owns coordinator-cancellation recovery. No execution-engine change was made in this task. |
| Published evidence duplicated captures or included host hook text | Yes | Yes | No | Final9 publishes distinct node-2 wide web, watch, receipt, archive, and clipping evidence. The watch fixture contains no host hook text. |
| Web shows a global error toast for a finished run whose worktree was removed, while watch shows the error inline | Yes | No | Yes | `pass-3-node-2-real-web-completion-wide.webp` shows the toast "git session changes failed: worktree is unavailable: ..." covering the session input. `GET /api/runs/<id>/changes` answers 409 from `src/milknado/web/routes/run_changes.py`; `milknado watch` renders the same error inline in the Changes pane (`src/milknado/app/session_changes.py`). Follow-up node `108` owns the web empty-state fix and its Playwright regression. |

### Residual gaps

- The external-provider OMP run remains blocked at `milknado_deposit_result`. Follow-up node `102` owns the inherited nested-worker-context stall.
- Controller-cancellation recovery for a dead coordinator remains deferred to existing follow-up node `79` (#309).
- Web widths from 401px through 1179px remain a presentation gap. Follow-up node `103` owns the layout fix.
- The dedicated `command_receipts` table is not a separate visual control. The wide web and watch completion views render the session receipt message; the `queued → submitted → delivered` rows are linked as durable evidence.

### Gate

Command: `just check-llm`

Observed output:

```text
✅ check:llm PASS — lint+format clean, no dead code, tests green, project+diff coverage ≥95%, typecheck clean
```
