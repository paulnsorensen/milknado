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
