---
kind: goal
slug: interactive-run-steering
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: [tmux-run-primitive]
---
# Non-headless agents: steer a running run mid-flight

## Intent

The user authorizes graph creation and execution after the Lavish review.[^execution-authorization]
This contract supersedes the historical tmux intent and acceptance below.
No tmux prerequisite applies to this native-session implementation.

## Acceptance

Provide one shared run/watch workspace for Codex, Claude, and OMP.
Place an interactive graph tree on the left.
Allow an attached watch process to command the existing live worker owner.
Expose all node data through compact groups and explicit disclosures.
Let agents create and change tasks through the existing follow-up tool.
Human review applies only to possible changes to the top-level GOAL.
Task descriptions, task criteria, implementation choices, and dependencies remain agent-owned within that goal.

- Reuse native session channels and provider adapters.
- Preserve Codex steer, Claude follow-up, and OMP steer/follow-up semantics.
- Keep view-only database connections read-only.
- Add an explicit graph-owned command inbox, separate from executable nodes.
- Fence commands by node, run, invocation, owner incarnation, and exact permission request.
- Use stable command IDs, bounded FIFO admission, expiry, and durable receipts.
- Publish fresh owner capabilities; persisted events alone do not establish available actions.
- Keep database transactions separate from vendor I/O.
- Distinguish queued, submitted, delivered, rejected, expired, and unconfirmed states.
- Never automatically replay a command after uncertain delivery.
- Preserve draft revisions, focus, selection, scroll, and run/watch quit semantics.
- Project DAG relationships without duplicate identities or changed dependencies.
- Expose all 21 current MikadoNode fields, including nulls and hydrated goal_run_id.
- Expose full descriptions, graph relations, reverse dependents, files, runs, reviews, sessions, goal claims, and artifacts.
- Page retained history; distinguish missing, unloaded, unstored, and expired data.
- Fence asynchronous detail responses by node and request generation.
- Render untrusted values as text; validate local artifact paths.
- Reuse milknado_track_follow_up with sibling defaults, atomic provenance, idempotency, and structured node links.
- Verify that tool in all three provider sessions without broad permission expansion.
- Bind reviews to the explicit top-level GOAL, never the discovery TASK, nearest parent, nested subgoal, or ROADMAP.
- Normal task changes require no human approval under an unchanged top-level goal.
- A revision identifies the reviewed goal; a hash cannot prove semantic scope.
- Enforce goal-impact admission across scheduler, direct claim, dispatch, and continuation paths.
- Do not use blocked status as an approval lock.

### Selected pause boundary

Pause affected work when evidence may change the top-level goal.[^pause-selection]
Pause the entire top-level goal if impact cannot be bounded.
Continue unaffected work when the boundary is known.
Gate new dispatch and continuation; request safe interruption where needed.
Pause does not roll back effects from tools already in progress.
Ordinary task changes never require goal review.

[^pause-selection]: User selects “Pause affected work; pause the whole goal if impact cannot be bounded.” on 2026-09-12.

### Execution and verification

The graph owns implementation task decomposition.
Separate snapshot data, command persistence, owner delivery, workspace presentation, discovery, goal admission, integration tests, and publication.
Split implementation from the final gate audit.
Run just check-llm; green means its PASS line.
Use exact-value tests and actual process-boundary tests.
Compare inspector coverage against dataclasses.fields(MikadoNode).
Test duplicates, stale owners, stale permissions, expiry, uncertain delivery, and draft revisions.
Test task-only changes without approval and top-level goal changes with review.
Capture run and watch with matched dimensions, theme, data, and state.
Include standard and compact layouts plus changed error and human-intervention states.
Open every capture and inspect clipping, focus, hints, and status.
Provide accessible capture links and reproducible fixtures in the PR.
Keep synthetic tests separate from live provider evidence.
Follow AGENTS.md and the local pythonic skill.
Do not change unrelated goals, worker models, billing routes, or permission policies.
Do not add remote multi-user control, historical backfills, or rich OMP editor widgets.

[^execution-authorization]: User instruction after draft 04: “the next step to do is to put this thing into a milknado graph and kick it off”.



### Graph handoff

The authorized scope maps to local top-level GOAL #34 with TASK nodes #35–#42.[^graph-kickoff]
Task #35 starts through the detached per-task runner from base 7dee80f78113b27b95f9a88459e5675716070ad1.
The remaining tasks form a prerequisite chain; starting one detached task does not start a goal-wide coordinator.
The current CLI run command has no goal selector, so do not launch it against this multi-goal database to continue this scope.
Dispatch only the ready task IDs under GOAL #34.
Task #40 applies the selected affected-work pause policy, with whole-goal fallback when impact cannot be bounded.
The existing unrelated goals remain untouched.

[^graph-kickoff]: milknado_todo_add returns GOAL 34 and TASK IDs 35–42. milknado_run_loop_start returns node-35-20260912T053957Z-834269ce. Current milknado run --help lists no goal selector.



The user requests full coordination of GOAL #34 and cleanup of completed graph work on 2026-09-12.
The coordinator archives 28 DONE nodes through the archive API; this preserves durable history and does not remove worktrees.
Old tasks #14 and #15 correspond to merged PRs #441 and #442 but retain legacy verification-state problems.
Do not bypass immutable dispatch-base verification to hide those records.
Old task #4 remains incomplete, and task #25 retains running status; neither belongs to this steering implementation.



### Recovery checkpoint — 2026-09-12

Task #35 completes after recovery in its original worktree, `milknado-35-complete-graph-and-node-detail`.
Its immutable base remains `7dee80f78113b27b95f9a88459e5675716070ad1`.
Commit `a6b2dee711f84ebad07427e01adb7175be750fad` includes reviewed snapshot and test synchronization corrections.
The feature branch `feat/tui-agent-steering` fast-forwards to that commit.
Genuine severity review approves at original run message sequence 13.
The completion verifier returns `ok=True` before commit; result sequence 15 records the recovery.
The verified status facade then marks #35 DONE.

The shell starts with a 256-file soft limit.
Use a process-local 4096 limit for the observed parallel-test descriptor exhaustion.
This does not fix separate test synchronization failures.
Textual worker waits snapshot the current worker set; a changes completion callback can start another diff worker.
The test now waits both phases and keeps exact diff and concurrency assertions.

The user requests completion and one PR for the intended steering feature, not a snapshot-only PR.
Task #36 starts as `node-36-20260912T141013Z-581c6cde` from base `a6b2dee`.
The standalone inbox phase fails source-only dead-code checks because the original task #37 owns every production caller.
The coordinator stops that detached run and preserves `milknado-36-fenced-durable-session-command` for recovery.
The runner process becomes a zombie; its OMP child exits.
Task #36 now includes production owner delivery and shared run/watch admission.
Task #37 retains independent delivery acceptance and two-process failure tests.
This changes task boundaries, not the top-level goal or verification gate.
Never hide an unconnected facade behind suppressions or fake caller aliases.
Tasks #37–#42 remain required.
Large-graph polling performance belongs to task #38.
Baseline synthetic run/watch captures use `a6b2dee` before the new steering presentation.



The task #36 recovery exposes a receipt-ordering defect through a separate-process admission test.
Provider turn IDs incorrectly replace the owner invocation fence before the command receipt.
An early correction marks successful writes delivered; later semantic review rejects that correction.
Preserve the baseline receipt contract: a pipe write alone does not prove vendor acknowledgement.
Codex steer and interrupt require correlated RPC responses; Codex approval requires `serverRequest/resolved`.
Claude follow-up requires the user echo; Claude interrupt requires its control response.
OMP steer and follow-up require correlated user echoes; OMP interrupt requires its successful response.
Claude and OMP permission protocols intentionally settle their decisions after successful writes.
Broken pipes remain unconfirmed; explicit vendor rejection remains rejected.
The owner invocation identifies one provider process execution and stays stable across its conversational turns.
Vendor turn IDs remain private protocol fences, separate from the process invocation identity.
New process executions require new opaque invocation IDs; close publishes empty actions after pending receipt cleanup.
These corrections remain required before task #36 completion.



Task #35 is now archived through the supported API; 29 completed nodes are archived in total.
The active database contains 13 unique open nodes: two goals, seven steering tasks, and four older tasks.
Task #36 review sequence 1 rejects the recovery implementation.
Blocking findings include missing attached-UI actions and incorrect provider permission IDs.
Further findings cover closed invocation availability, caller-owner validation, and duplicate delivery from concurrent drains.
Admission must distinguish expected fence rejection from operational storage faults.
A passing direct source admission test does not prove the attached TUI can select and submit an action.
Task #36 remains blocked; tasks #37–#42 remain pending.
No PR exists for the full steering feature.



Explicit channel member types resolve the typecheck failure; `just check-llm` passes.
The attached UI now projects durable owner actions and uses an admission controller.
A graph-backed Textual test selects follow-up, enters text, and verifies a queued durable command.
A separate test verifies read-only input rejection and hidden terminal capabilities.
The full gate also passes after this attached-UI correction.
Atomic owner claim now uses `BEGIN IMMEDIATE` before capability reads and a queued-only conditional update.
The caller supplies its owner incarnation; expired or mismatched invocation commands cannot enter the returned claim batch.
Direct run and attached watch now share durable admission and owner claim; nongraph channels retain in-memory admission.
The initial claim correction passes `just check-llm`; bounded severity review finds no remaining security or concurrency defect.
That review finds missing expiry receipts; the correction expires eligible commands for the whole caller run inside the claim transaction.
Five focused claim and expiry tests pass, including previous-invocation expiry history.
The shared expiry helper does not own a transaction, so it cannot commit the caller's claim lock early.
The final combined gate remains pending while permission identity work continues.
Permission identity, invocation closure, and stale displayed-fence findings still require correction.
All task #36 recovery changes remain uncommitted at base `a6b2dee711f84ebad07427e01adb7175be750fad`.
A green gate alone does not close the remaining severity findings.
The coordinator has no goal-wide background runner.

The user extends the objective to all relevant outstanding work, with an empty active graph as the secondary outcome.
Task #4 restarts through exact-node dispatch as `node-4-20260912T163927Z-79b2f60e`.
Its preserved old worktree remains untouched; native dispatch uses `milknado-4-bound-task-slugs-in-worktree-p-2` at base `a6b2dee`.
Exact process inspection confirms runner PID 69881 and OMP child 69905 during execution; an earlier agent absence report is incorrect.
The native run then completes with exit code 0 and `rebased=true`.
It integrates commit `769844312a71f8186a1e596bdc1b705fb02ebe4f` into `feat/tui-agent-steering`.
The node facade confirms task #4 DONE; the archive API archives it.
There are now 30 archived completed nodes.
Task #36 retains its original immutable base `a6b2dee`; do not reset or rebase its dirty recovery worktree.
Its later integration must account for the root branch advance rather than assume a fast-forward.




The process identity correction now generates one UUID per runtime execution and retains it across turns.
Close settles pending receipts before publishing empty capabilities; 16 focused lifecycle and channel tests pass.
Service-level duplicate admission now retains original expiry and known outcomes; 18 admission and graph tests pass.
Real attached-permission tests expose a Codex fixture mismatch: the generic shell launcher adds a positional script argument.
Codex correctly rejects that argument before process startup; use a bare executable test shim instead of weakening provider argument validation.
Provider receipt correlation remains under correction; the full feature is not ready for publication.



Attached approval delivery now passes all three fake-vendor process tests.
The tests assert one actual provider frame, duplicate admission identity, ordered durable receipts, and owner completion.
Two non-obvious corrections unlock this result: Codex resolution accepts numeric request IDs, and permission receipts precede capability removal.[^receipt-order-recovery]
Codex fixture completion uses an `agentMessage` item; a `text` item does not supply the completion result.
Direct run snapshots now overlay durable owner and invocation fences; the channel-only snapshot previously omitted them.
The UI queue regression verifies stale invocation rejection and draft retention.
Formatting passes after fixture decomposition; the combined gate and full task #36 review remain pending.

[^receipt-order-recovery]: Task #36 recovery worktree, `src/milknado/loop/sessions/_codex_approval.py`, `_runtime.py`, and `tests/test_attached_permission_delivery.py`; parent run reports `3 passed in 1.96s` on 2026-09-12.



The expanded approval suite now passes six approve/deny cases across Claude, Codex, and OMP.
A successful human denial is a delivered command, not a rejected command; the permission outcome remains denied.
Full review identifies missing durable unconfirmed receipts, concurrent permission-decision identity corruption, and a late-admission shutdown race.
The unconfirmed correction includes terminal user-event correlation; its final verification remains pending.
Concurrent-decision admission and explicit owner-shutdown finalization remain under correction.
The latest full gate does not establish readiness; tests change during that run, so the coordinator requires a frozen-tree rerun.
At 17:54 UTC, the execution service cannot create even `true` because it reports `Too many open files`.
This failure occurs before a shell starts, so a shell-local file limit cannot repair it.
Parent tilth and Hallouminate calls still work; source inspection remains possible.



### Execution-service recovery handoff

The execution service fails to spawn `true` across three consecutive goal turns with `Too many open files`.
The coordinator stops further unverified implementation and requires external process-service recovery before tests or publication.
Resume the preserved task #36 worktree at `/Users/paul/Dev/milknado/milknado-36-fenced-durable-session-command`.
Its immutable base remains `a6b2dee711f84ebad07427e01adb7175be750fad`; changes remain uncommitted.
Root `feat/tui-agent-steering` remains at `769844312a71f8186a1e596bdc1b705fb02ebe4f` from task #4 integration.
Do not reset either checkout or replace the original dispatch base.

Permission admission now acquires `BEGIN IMMEDIATE` before reads and uses one INSERT path.
It checks pending decisions inside that transaction instead of matching SQLite error strings.
`tests/test_permission_command_race.py` now starts simultaneous admissions on separate connections.
These last admission corrections remain unverified because process creation fails before the shell starts.
The unconfirmed regression now uses durable claim and an actual graph receipt sink; rerun it after service recovery.
Explicit shutdown finalization remains unimplemented; neither shutdown worker applies source edits.
Next: verify current corrections, implement shutdown finalization, finish task #36 review and native verification, then complete tasks #37–#42.
No full-feature PR exists, and the graph is not empty.



Process execution works again on 2026-09-12 at 19:58 UTC.
The resumed gate rejects three type errors, then rejects twenty type warnings after those errors are fixed.
Warnings are gate failures; zero type errors alone does not establish PASS.
The correction moves shared test setup into fixture modules and removes unchecked return values without new suppressions.
Task #36 remains uncommitted; explicit shutdown finalization and tasks #37–#42 remain required.



The frozen task #36 recovery tree passes `just check-llm` after the type corrections.[^recovered-gate]
This verifies the current concurrent-permission and durable-unconfirmed regression tests.
It does not resolve the known late-admission shutdown race or establish full feature completion.

[^recovered-gate]: Execution session 97839 exits 0 on 2026-09-12 and prints `check:llm PASS` for lint, format, dead code, tests, project coverage, diff coverage, and typecheck.



Explicit shutdown finalization now exists in the task #36 recovery tree.
The adapter installs an explicit channel-close callback; empty action snapshots do not imply shutdown.
The graph transaction clears exact-owner capabilities and rejects only matching queued commands.
It preserves submitted commands and replacement-owner state.
Channel close skips ordinary capability publication when this explicit callback owns revocation.
This prevents the closing channel from overwriting replacement-owner capabilities.
Two adapter-seam regressions fail before the fix and pass afterward, including an event-sink failure.
New graph tests cover duplicate close, replacement ownership, empty actions, and simultaneous close/admission connections.
The combined gate stops at one unused test-parameter warning; the parameter name is corrected.
A fresh gate cannot start because the execution service again reports `Too many open files` before shell creation.
The fresh shutdown taste-test passes all seven lenses.
A full task #36 severity review now runs read-only against the frozen recovery tree.
The execution service still rejects gate startup after a two-minute wait.
No task status or publication claim changes.



The renewed execution-service failure persists across three consecutive goal turns.
Fresh `true` probes fail before process creation, including with login disabled.
No gate process remains live; session 2377 is terminal with one corrected test warning.
The read-only reviewer `/root/task36_final_review` remains live without a completed report.
Preserve that reviewer and the frozen recovery worktree; do not restart either because an observation times out.
Resume with the same review handle and rerun `just check-llm` after process-service recovery.



After session recovery, a fresh `true` probe succeeds and the final task #36 gate starts as session 26844.
The former reviewer handle no longer exists; no report is found in the root `.cheese/age` search.
A replacement read-only reviewer, `/root/task36_recovered_review`, reviews the same preserved worktree and base.
Its report target is `.cheese/age/tui-agent-steering-task36-review.md` so another restart cannot lose its findings.



The current task #36 tree passes `just check-llm` in session 26844.
Native `milknado_node_verify` also returns `ok=True` for `node-36-20260912T141013Z-581c6cde` in session 80323.
This verification uses the preserved dirty worktree; its original base remains `a6b2dee711f84ebad07427e01adb7175be750fad`.
Final severity review remains required before task #36 integration.
The current tilth contract uses the server project for Git sources, regardless of `cwd`.
Absolute `a`/`b` file comparisons work for worktree diffs; do not interpret a root-only Git diff as an empty worktree.



Final task #36 review rejects with two findings, preserved in `.cheese/age/tui-agent-steering-task36-review.md` and original-run review sequence 3.
Known-command admission must distinguish absent records from terminal non-success and enforce supplied owner/invocation fences against the stored command.
Current-owner claim must reject fence-invalid rows within its transaction instead of leaving their receipt queued until expiry.
A transient Claude/OMP permission-settlement gap cannot produce a second wire write: the single owner thread removes capabilities before its next drain.
Fix the stale queued receipt, not an invented exactly-once delivery guarantee.
The bounded cure owns `_command_admission.py`, `_command_claim.py`, and their regression tests in the original task #36 worktree.



Task #36 completes on 2026-09-12 after both final severity findings close and a fresh seven-lens taste-test passes.
The final regression set retains separate new-admission and known-command fence tests; 16 focused cases pass.
Full gate session 50556 passes; native verifier session 84310 returns `ok=True` with empty feedback.
Original-run review sequence 4 records the genuine approval.
Task commit `12ed9418c792f66b4852b55f1d6b8a5f3d786042` integrates into the feature branch as `925cff36777eb40e9152bc995cb4992dc688f446`.
The supported status facade marks #36 DONE; the archive API hides its completed subtree.
Task #37 starts through exact-node dispatch as `node-37-20260912T205527Z-ff0d0ca0`, runner PID 13123.
It owns independent process-boundary acceptance, not a repeat implementation of the inbox.
Tasks #38–#42 remain required; no full-feature PR exists at this checkpoint.



Watch observes run status separately from graph node status.
After task #36 archival, its outer and inner stopped executions still had `running` run rows.
At 21:07 UTC, process inspection finds neither original runner PID 65264 nor matching task36 worker processes.
The coordinator finalizes both original rows as failed through `graph.runs.finish`, without a fabricated exit code.
Their detail records distinguish the abandoned execution from the verified recovery and identify the reconciliation timestamp.
The verified task remains DONE; worktrees and review evidence remain intact.
Remaining stale historical run rows require the same evidence-based cleanup before final publication.



The historical run cleanup completes for nodes #9, #12, #13, and #33.
Owner PIDs 40943 and 67640 are absent during process inspection.
Runs for #9, #12, and #13 retain stored successful native verification and publication results.
GitHub confirms PR #440 merged as `a0825220f59e855df9aac3963bb3e96a2b9034ec`; their in-session run rows become DONE.
Run `node-33-20260910T062541Z-779f6ead` has no stored result or verification and becomes failed, not fabricated success.
Every corrected run records an unknown exit code and an explicit reconciliation timestamp.
Graph nodes #9, #12, #13, #33, and #36 remain DONE.
The Watch snapshot now reports only task #37 active, through its outer runner and inner worker records.



Task #38 starts beside task #37 on 2026-09-12 after the production-delivery work in #36 completes.
The initial serial dependency is no longer necessary: #37 owns independent failure acceptance; #38 owns presentation.
The coordinator adds prerequisites 41→37 and 38→36 before removing 38→37 through the public graph API.
Final integration #41 still waits for both acceptance and presentation; no requirement or gate is removed.
Task #38 uses isolated native run `node-38-20260912T211631Z-ffd6ea1e`, runner PID 5741, at base `925cff36777eb40e9152bc995cb4992dc688f446`.
Its explicit runner command uses the checkout virtual environment to load the integrated runtime, not the MCP host's older cached package.
Worker models and permission policies remain unchanged.
If #37 finds a delivery defect that affects the presentation boundary, pause affected #38 work and coordinate the fix.



The coordinator exercises real OMP steering against task #38 through the shared attached admission service.
Command `ccd051aa47ad4df6bea6579f8c2fd597` carries the displayed owner and invocation fences for inner run `node-38-20260912T211632Z-e7989d81`.
It requests three scoped implementation cuts and preserves independent task #37 acceptance.
Durable history records queued at 21:17:47.319723, submitted at 21:17:47.821999, and delivered at 21:17:53.788945 UTC.
This is an installed OMP worker with real provider acknowledgement, not a fake-vendor fixture.
It proves this shared-service steering path, not TUI keyboard behavior, all OMP actions, or Codex/Claude compatibility.



The first task #38 runner fails at 21:19:55 UTC before source changes.
Its outer exit code is 1; inner status is failed without timeout, stored error event, result, review, or verification.
Process inspection confirms runner PID 5741 is a zombie and OMP 7086 is absent; native teardown removes the unchanged worktree.
The cause remains unknown; successful steering receipt delivery does not prove later worker completion.
Exact-node retry `node-38-20260912T212455Z-9c1c9478` starts with INFO logs and a process-local 4096 file limit.

Task #37's draft tests manually close a channel without a vendor write, abrupt owner failure, replacement owner, or retry.
The coordinator stops that attempt for this concrete acceptance mismatch, not for elapsed observation time.
Cancellation leaves runner PID 13123 as a zombie; OMP 13239 is absent.
Both run rows become failed through the supported facade with unknown exit codes and explicit reconciliation details.
Task #37 remains blocked while a bounded coder recovery replaces the draft in its original worktree at base `925cff36777eb40e9152bc995cb4992dc688f446`.
The recovery must use production owner/runtime wiring, actual process failure, observed fake-vendor frames, and a replacement-owner no-replay assertion.
Abrupt termination does not execute graceful close: tests must not invent unconfirmed receipts where only fenced queued/submitted state and later expiry exist.
Task #38 remains independent; final integration still requires both tasks.



The task #38 retry uses worktree `milknado-38-tree-left-shared-workspace-and-2`; runner PID 13598 and OMP PID 13921 are live.
Its startup log reports an occupied-orphan fallback for the former worktree name.
The former path is absent in a direct tilth directory check; do not treat the generic warning as proof that files remain there.




Task #37 process acceptance requires explicit cleanup of both owner and vendor process groups.
The vendor starts in a separate session; killing the owner group does not stop a blocked vendor.[^task37-process-review]
Process inspection confirms four leaked fake vendors from the rejected test draft; the coordinator terminates only those fixtures and verifies absence.
The draft also accepts ambiguous receipt histories and omits expiry and replacement-fence assertions.
The bounded cure addresses these findings in the preserved task #37 worktree; task #38 continues independently.
A focused test pass does not close these acceptance defects or replace the required full gate.

[^task37-process-review]: `.cheese/age/task37-process-acceptance.md`; `src/milknado/loop/sessions/_process.py:270-279`; process inspection on 2026-09-12 at 21:41 UTC.




Post-integration verification does not reproduce a clean publication gate.
Task #37 gate session 98135 passes tests but fails diff coverage: 83 of 1201 changed lines remain uncovered (93%).
Do not reuse the earlier pre-commit PASS as final publication evidence.
A clean task #36 comparison at commit `12ed9418c792f66b4852b55f1d6b8a5f3d786042` stops earlier in session 98389.
Its lifecycle tests `test_detached_pipe_holder_does_not_block_session_cleanup` and `test_stderr_output_callback_can_stop_runtime` cannot find their child PID markers.
That comparison does not establish the cause of the diff-coverage failure; diagnosis remains required.
Task #37 now separates subprocess fixtures from acceptance tests and checks a second replacement command as a replay barrier.
Final review and a current full gate remain required before task completion or publication.




Task #37 completes after the independent process and public command-boundary reviews close all findings.
The second fresh taste-test passes all seven lenses.
Gate session 21172 passes; final native verifier session 86852 returns `ok=True` after the last assertion correction.
Original-run review sequence 1 records approval; result sequence 3 records the verified recovery.
Commit `21eabdc5392b5d00b0ca4102887e214bb7b1b3f0` integrates as `18588b8c3d3a16ad2360c259c0c636819312fbed` on `feat/tui-agent-steering`.
The supported status facade marks #37 DONE; archival preserves its evidence.
The added public-boundary tests pass both coverage gates without private helper filler or threshold changes.
The original abandoned execution rows remain failed; do not relabel their process exit as successful recovery.
Task #38 continues in its existing native run and isolated worktree.
Tasks #38–#42 and the full-feature PR remain required.




Task #38 recovery preserves `milknado-38-tree-left-shared-workspace-and-2` at immutable base `925cff36777eb40e9152bc995cb4992dc688f446`.
The coordinator stops its second native attempt for concrete scope divergence, not elapsed observation time.
After delivered instructions forbid duplicate private coverage filler, the worker creates `tests/test_graph_boundary_coverage.py` with mocked persistence helpers and fabricated query results.
Cancellation reports no exit because runner PID 13598 is a zombie; OMP PID 13921 is absent.
The public run facade finalizes outer and inner rows as failed with unknown exit codes and explicit reconciliation details.
Task #38 remains blocked during bounded recovery; existing TUI edits remain intact.
Preparation removes only the rejected filler and overlays the six verified task #37 files without changing HEAD or the dispatch base.
A separate read-only severity review inspects the frozen TUI source, tests, and missing capture evidence.
Do not restart native task #38 or replace its dirty worktree.
Tasks #39–#42 remain required.




Task #38 review distinguishes absent session data from expired command receipts.
Current session event storage is append-only; source review finds no snapshot-retention expiry policy.[^task38-state-review]
A missing `run_sessions` row must not appear as loaded empty history.
Expose command receipt expiry separately; do not invent a retention policy merely to populate an expired snapshot label.
The same review finds that database-wide `PRAGMA data_version` cannot provide graph-specific cache reuse during unrelated session writes.
Graph revision tracking must cover nodes, edges, and goal claims while leaving unrelated session writes out of full graph invalidation.

[^task38-state-review]: `.cheese/age/task38-recovery-review.md`; `src/milknado/domains/graph/snapshot_history.py:182-201`; `src/milknado/domains/graph/_session_persistence.py`.



A temporary-database reproduction confirms a durable receipt visibility gap at root commit `18588b8`.
The public command facade admits a command, then claims at its exact expiry time.
Claim returns no command; durable history contains `queued` and `expired`.
The session projection contains no events, and the node detail contract contains no receipt field.[^task38-receipt-proof]
Therefore, transcript rendering alone cannot show all durable command outcomes.
Task #38 must expose bounded durable receipt history separately from session transcript events.

[^task38-receipt-proof]: Coordinator reproduction on 2026-09-12 at 23:33 UTC through `MikadoGraph.commands.admit`, `claim_pending`, `history`, `sessions.view`, and `get_node_detail_snapshot`; `src/milknado/domains/graph/_command_claim.py:75-87`; `src/milknado/adapters/_loop_session.py:123-143`.



Task #38 graph-cache repair receives independent severity approval and seven passing taste-test lenses.
It remains uncommitted in the original recovery worktree.
Parent gate 23600 fails on TUI type diagnostics; it does not prove publication readiness.
The repair's private-connection test warning is corrected through a forwarding connection-factory trace, without suppressions.

The reviewer rejects a persisted detail revision as unnecessary for the separate freshness repair.
Refresh the existing bounded selected-node pages after each source snapshot instead.
Keep the current detail visible until its generation-fenced replacement arrives.
Preserve page, history position, draft, and focus when the selected node identity stays unchanged.[^task38-detail-refresh]

[^task38-detail-refresh]: `.cheese/age/task38-recovery-review.md`, H2 correction; reviewer clarification on 2026-09-12 at 23:34 UTC. Existing detail reads use bounded relation pages in `src/milknado/domains/graph/snapshot.py:48-140`.



The H2 source passes severity review, but its regression proof needs nonzero history and explicit wrong-node/generation checks.
The first H2 draft uses reflection to bypass type diagnostics; the coordinator rejects that draft.
The replacement uses typed members. Public fixture extraction and stronger response-fence tests remain in progress.
Gate `11746` does not run typecheck: tests fail first, and `justfile:99-142` places typecheck last.
Do not reuse the coder's withdrawn typecheck claim.

A separate two-file fixture cure replaces environment-selected Python shebangs with the current test interpreter.
It addresses measured cold-start delay without changing deadlines, assertions, or runtime code.
Parent session `58565` runs both fixture files with four workers and coverage: 18 tests pass in 7.11 seconds.
Full gate `9439` still reports two oversize-frame timeouts near 2.05 seconds on worker `gw0`.
That full-suite-only failure remains unresolved; four workers plus coverage alone do not reproduce it.
The fixture cure and task #38 remain uncommitted. Final publication still requires the complete gate to pass.



Task #38 H2 closes after severity review and seven passing taste-test lenses.
The final response-fence tests await Textual worker completion before their negative assertions.
A local typed protocol narrows Textual's public worker API; no reflection or suppression bypass remains.
Focused H2 tests pass eight cases. Gate `38437` reaches typecheck and reports two formatter-test errors plus 19 other warnings.
The earlier oversize timeouts do not recur in this gate; no runtime timeout or threshold changes apply.

The next bounded cure owns only `app/graph_view.py` and `tests/test_graph_view.py`.
It proves dynamic disclosure of every node field and validates related artifact paths.
Receipt disclosure, aggregate pagination hints, focus, minimum layout, and rendered evidence remain separate unfinished cuts.
Task #38 stays blocked in the graph during this external recovery; no native runner restarts.
Tasks #39–#42 and one full-feature PR remain required.



The task #38 artifact-path correction passes severity review and fresh taste-test.
The all-field test still checks labels and values independently.
Both reviews require exact labeled values, including each null and the complete description block.
The selection correction now runs in the preserved worktree through `task38_selection_focus_cure`.
Snapshot refresh currently assigns another node's run to a selected node without a run.
The correction must preserve node/run identity together and retain input focus on ordinary refresh.[^selection-recovery]

Lifecycle failures also reproduce without coverage or parallel test workers.
Focused session 4355 reports 16 passes and three missing PID failures.
Thus, the earlier full-suite-only explanation does not describe the current evidence.
Direct fake-worker execution remains slower than explicit interpreter execution after the interpreter-header correction.
Session 14490 measures direct startup at 1.518, 1.805, and 1.225 seconds; explicit startup takes 0.032, 0.032, and 0.028 seconds.
Escalated session 25288 shows the same pattern: direct startup takes 1.703, 1.848, and 1.337 seconds.
Its explicit startup takes 0.020, 0.020, and 0.023 seconds.
These tests bypass session runtime and use identical generated scripts with two input frames.
Sandbox removal does not resolve this measured difference.
The underlying host cause remains unproven; do not change production runtime or increase deadlines from this evidence alone.[^startup-recovery]

[^selection-recovery]: Canonical `.cheese/age/task38-recovery-review.md`; inspector taste-test round 1 and selection source inspection on 2026-09-13.
[^startup-recovery]: Parent execution sessions 4355, 14490, and 25288 on 2026-09-13; task #38 recovery base `925cff36777eb40e9152bc995cb4992dc688f446`.



The same sole writer also owns a separate tests-only correction in `tests/test_graph_view.py` after its selection and focus cuts.
This correction binds every field label to its expected value and preserves the approved artifact validation.
M4 next requires taste-test round 2; no third round is permitted.
Parent session 67378 also compares virtual-environment and resolved interpreter shebangs.
Both direct-execution variants remain slow in later samples; resolving the interpreter path does not remove the measured delay.
The selection writer remains live at the final observation; do not start a duplicate writer.



The coordinator now permits two disjoint writers in the preserved task #38 worktree.
This replaces the earlier single-writer restriction without changing the immutable base.
`task38_selection_focus_cure` owns graph selection, focus, and the labeled-field test correction.
`task38_explicit_fixture_cure` owns only the lifecycle, failure-path, and runtime fixture test files.
The latter applies the reviewed explicit-interpreter fixture shape from diagnostic 70356.
Neither writer may start the full gate before both report frozen source.
The coordinator grants one combined gate after that barrier.
No production runtime change or timeout increase is authorized.



Frozen task #38 gate 78850 now prints `check:llm PASS`.
The preceding gate 52606 reports 3800 passes, one terminal-session TUI failure, and three skips.
That single failure does not reproduce in focused session 89726 or the unchanged-tree gate 78850; its cause remains unproven.
The explicit-interpreter fixtures and warning cleanup remain reviewed or inspected without weaker deadlines, assertions, or gate thresholds.
M4 completes taste-test round 2 with all seven lenses passing.

Task #38 still requires source corrections and rendered evidence; the green gate does not complete it.
Selection taste-test round 1 requires graphless snapshot fallback to synchronize node identity.
The current correction also must assert the exact modal focused widget, not just modal screen identity.
The selection writer retains these obligations before its final taste-test round.
The empty-graph hidden-run keyboard defect now has real run/watch key regressions.

`task38_receipt_disclosure_cure` now owns missing-session states, bounded durable receipt disclosure, and aggregate pagination.
It uses the same preserved worktree and immutable base.
Its graph-navigation ownership covers pagination methods only; the selection writer owns selection and focus methods.
Both preserve anchored edits in their separate sections.
The full gate waits until both writers report frozen source.
Minimum layout, matched captures, tasks #39–#42, publication, and final graph cleanup remain required.



The final selection correction passes 34 focused tests and an isolated clean typecheck.
Severity review closes selection, M1 focus, the associated test gaps, and M4.
Selection taste-test round 2 passes all seven lenses; no third round is permitted.
Graphless snapshot fallback now aligns both IDs, and the modal regression checks the exact non-null focused widget.
The latest selection source still needs the combined full gate after the next cures freeze.

Two disjoint task #38 cures now remain active: receipt disclosure with aggregate pagination, and minimum-size layout.
`task38_minimum_layout_cure` owns only minimum presentation/control behavior, focused tests, and bounded local captures.
It must preserve the approved selection methods and the other writer's graph snapshot and pagination changes.
Both use the original preserved worktree at base `925cff36777eb40e9152bc995cb4992dc688f446`.
The parent owns the next combined full gate and the final matched capture audit.




Task #38 receipt correction now includes separate receipt-query and shared pagination modules.
The final worker attempt cannot start its focused tests because process creation returns `Too many open files (os error 24)`.
The coordinator reproduces this failure with normal execution, elevated execution, and the required full gate on 2026-09-13.
No command starts in these failed attempts; no test or gate result exists for this source.
The process or descriptor leak responsible remains unknown; do not kill unrelated processes or weaken project gates.
Root MCP tilth reads and writes still work, while replacement workers lack the connected tilth methods.
The coordinator uses root MCP to correct receipt regression expectations without another shell process.[^task38-fd-recovery]

The receipt regression now compares all 13 fields in exact durable order, rather than independent ID and status sets.
For 11 receipts at limit 3, page offsets 0, 3, and 6 have more; offsets 9 and 12 do not.
Claim-time expiry records `command expired`, not the separate transition-time message.
The tests preserve all command histories across read-only reads and assert no session event from queued-command expiry.
The receipt-only UI fixture now uses consistent limit 1 and total 2.
These coordinator edits remain untested until process creation recovers.
Minimum-layout source remains unchanged; its replacement worker cannot start the tilth CLI.
Tasks #38–#42, matched captures, the full gate, the full-feature PR, and graph cleanup remain incomplete.

[^task38-fd-recovery]: Root tool calls on 2026-09-13 around 01:41–01:44 UTC; preserved task #38 base `925cff36777eb40e9152bc995cb4992dc688f446`; `tests/test_graph_receipt_snapshot.py`; `tests/graph_navigation_fixtures.py`; `src/milknado/domains/graph/_command_claim.py:23-52`.




The next recovery turn reproduces the process-creation failure before a trivial command and before `just check-llm`.
The coordinator adds `tests/test_minimum_layout.py` through the working root tilth MCP.
These tests cover minimum workspace replacement, restored controls, restricted help, draft/focus preservation, and no hidden submission.
A recording source checks the actual controller call and proves submission after usable dimensions return.
A separate key test preserves run quit confirmation and cancellation.
The tests remain unexecuted; the minimum-layout implementation remains unchanged pending the red test run.
Do not call these regressions passing or use their presence as rendered evidence.[^task38-minimum-red-pending]

[^task38-minimum-red-pending]: `milknado-38-tree-left-shared-workspace-and-2/tests/test_minimum_layout.py`; coordinator tool observations around 2026-09-13 01:44–01:46 UTC.




The third consecutive recovery turn still cannot start a trivial command: process creation returns `Too many open files (os error 24)`.
The agent inventory confirms all phase agents are terminal; none supplies a live verification handle.
Receipt corrections and minimum-layout tests remain saved but unverified.
The coordinator marks the thread goal blocked pending command-runner recovery, without changing the goal or completing graph tasks.
Resume from the same dirty task #38 worktree after command execution works.
Run the minimum regressions before their implementation, then complete receipt review and the required full gate.
Do not reuse earlier passing gates for the current source.




A later runner probe succeeds and reports a soft file limit of 256.
This supersedes the active process-creation blocker; the underlying cause remains unconfirmed.
The user requests `/wheypoint` before further implementation or tests.
The checkpoint runtime saves work `tui-agent-steering`, revision `rev-19fd9fd9012b`, with `repo-snapshot` durability.
Its generated projection is `.cheese/notes/tui-agent-steering.md`.
Resume through `/cheese --continue tui-agent-steering`; do not treat checkpoint status `ok` as feature completion.
No tests or captures run after command execution recovers.

## Current-state correction — 2026-09-11

This goal records the original tmux proposal, not the current TUI baseline.
The native `run` TUI already exposes structured session input through the controller and session channel.[^native-controls]
Codex exposes steer; Claude exposes follow-up and rejects native steer; OMP exposes steer and follow-up.[^native-families]
The `watch` TUI remains a read-only observer.[^watch-boundary]
Plan improvements against these existing controls instead of rebuilding input transport.
Separate-process control requires a separate ownership design; this note does not approve that change.
See [native worker sessions](../../architecture/execution.md) and [watch observer](../../architecture/watch-observer.md).

[^native-controls]: src/milknado/app/session_panels.py:69-102; src/milknado/app/session_commands.py:123-195; src/milknado/app/run.py:345-349
[^native-families]: src/milknado/loop/sessions/_codex.py:25-29,125-156; src/milknado/loop/sessions/_claude.py:95-106; src/milknado/loop/sessions/_omp_state.py:24-54
[^watch-boundary]: architecture/watch-observer.md, Presentation; src/milknado/app/watch_tui.py



### Follow-up design scope

The user selects control from `watch`, with shared run/watch presentation and a graph tree on the left.[^scope-feedback]
Agents must turn discovered work into new graph nodes from the current session.
The existing follow-up tool is the starting point; this scope does not call for another node-creation mechanism.
A graph-owned command inbox is proposed for cross-process input, separate from executable task nodes.
The user selects a hybrid execution policy and complete node inspection.[^hybrid-feedback]
Agents choose, create, and change tasks without human approval while the top-level goal stays unchanged.[^goal-level-feedback]
Human review applies only to information that should or could change the top-level GOAL, not ordinary task changes.
Task descriptions, task criteria, implementation choices, and dependencies remain agent-owned within that goal.
The proposed review record targets the explicit top-level GOAL ID; a discovery task supplies evidence, not the approval subject.
The proposed goal contract records the top-level outcome and agreed boundaries, not an immutable task plan.
Use compact inspector groups with full descriptions and null values in disclosures; preserve access to every field.

[^goal-level-feedback]: User feedback on `.lavish/agent-steering.html`, draft 03, received 2026-09-11 America/Los_Angeles. The user ends the review session after this clarification.
Its revision identifies the assessed contract but does not prove an agent's semantic judgment.
The later pause selection supersedes this draft question; see Selected pause boundary.
The UI must expose every current node field plus related durable records, with explicit pagination and retention limits.

[^hybrid-feedback]: User feedback on `.lavish/agent-steering.html`, draft 02, prompts 1, 3, and 4, received 2026-09-11 America/Los_Angeles.
These are planning decisions, not an approved implementation or a claim that detached control already exists.

[^scope-feedback]: User feedback on `.lavish/agent-steering.html`, draft 01, prompts 1, 3, and 4, received 2026-09-11 America/Los_Angeles.

## Original intent

The original proposal assumes headless workers with only cancel and wait controls. Firstmate's most-loved UX affordance is the
opposite — every crew agent runs interactively in its tmux window, and the
captain can type into it mid-run to redirect, answer a question, or unblock
it, without restarting the task.

With [[tmux-run-primitive]] providing the window, this goal makes the agent
inside it steerable: a run dispatched in interactive mode runs the harness in
its normal (non-headless) mode, so a human attaching to the window can
converse with the worker directly. The ralph loop contract still holds — the
run ends when quality gates pass — but a human can participate in getting it
there.

This is the second half of the firstmate-usability adoption track. Together
with ralph trees and wiki roadmaps, it is what lets milknado be the golden
path over firstmate: the same casual working-with-sub-agents feel, backed by
a real dependency graph, deterministic batching, and hard verification gates
instead of prose rules.

Open questions to resolve during design (not silently):

- Harness support matrix: which of the allowlisted harnesses (claude, codex,
  cursor-agent, gemini) can run interactively under tmux with the worker
  hook still firing? Interactive mode may be per-harness opt-in.
- Loop semantics: in interactive mode, does the ralph iteration boundary
  survive (harness exits, loop re-invokes), or does the loop become a single
  long-lived interactive session with gate checks on idle?
- How does human input mid-run interact with the run's brief/RALPH.md
  contract — is a human redirection recorded (e.g. as a `run_messages` row)
  so the audit trail stays truthful?
- Supervision interplay: does typing into a window suppress
  [[zero-token-run-supervision]] stall detection for that run?

## Acceptance

- A run can be dispatched in interactive mode (opt-in, per-run), launching
  the harness non-headless inside the run's tmux window; headless remains
  the default.
- A human attaching to the window can send input to the running agent and
  the agent acts on it, demonstrated by an e2e or scripted-tmux test.
- Quality-gate verification is unchanged: an interactive run still cannot
  reach DONE without a passing `milknado_node_verify` — human participation
  does not weaken the gate.
- Human mid-run input leaves a durable trace (documented mechanism, e.g. a
  `run_messages` row or status event), so a harvested run shows whether it
  was steered.
- Unsupported-harness dispatch in interactive mode fails fast with a clear
  message listing supported harnesses.
- `architecture/execution.md` documents interactive mode, its harness
  matrix, and its interaction with supervision and verification.
