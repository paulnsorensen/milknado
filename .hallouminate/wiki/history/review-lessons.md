# Review Lessons — Recurring Findings & Gotchas

Distilled from review/cure cycles on PRs #96, #109, #110, #114, #244. Grouped by theme —
things that bit us, don't repeat them.

## Lifecycle: completion guards must fail loud, not swallow

In the executor, `Executor.complete()` originally short-circuited on **any**
non-RUNNING status (`if node.status != RUNNING: return no-op`). That made
duplicate completions idempotent — but it also silently swallowed *invalid*
completions of non-terminal nodes (PENDING/BLOCKED). Worse, the run loop
(`src/milknado/domains/execution/run_loop/_completion.py`) then printed a green ✓ and counted the node as
`completed` even though no terminal transition ever happened. (#109)

**Rule:** an idempotency guard must distinguish "already terminal, safe to
re-run" from "illegal state, must fail". Narrow the no-op to terminal statuses
only (`DONE`, `FAILED`); `raise InvalidTransition` for everything else. A
no-op that masks an invalid transition violates fail-fast and produces a false
success signal downstream. Test every branch of the guard — the terminal no-op,
*and* the loud-raise path.

## Config: round-trips and layer ordering silently corrupt data

The layered-config work (#96) surfaced a cluster of round-trip / merge-order
defects in `config.py`:

- **Suppress-on-equality, not suppress-always.** `save_config` dropped an
  explicit `execution_agent` whenever a structured `worker_tools` override
  existed, so a load→save→reload swapped the user's custom command for the
  derived `--allowedTools` one. Fix: suppress the field *only when it equals*
  the derived command; otherwise emit the explicit value.
- **Resolve relative paths against their own layer.** Relative prompt `_path`
  values from the *global* config were resolved against `project_root` because
  resolution happened after the global/local merge. A global path then read a
  project-local decoy or raised `FileNotFoundError`. Fix: absolutize relative
  global paths against the global config dir *before* the merge.
- **Validate at the boundary.** `_load_prompt_prepend` stringified non-string
  values (`["x"]` → `"['x']"`), inconsistent with the worker-side
  `_coerce_tool_list`. Fix: reject non-string `inline`/`_path` with a clear
  `ValueError` before coercion.

**Rule:** for any config that serializes, write a load→save→reload round-trip
test for the *explicit-value-plus-override* combo. Resolve relative paths
against the layer they came from, not the active dir. Validate types at the
read boundary.

## Tests that encode the bug, not the intent

While fixing #96, two pre-existing tests turned out to assert the *old buggy*
behavior — and they only "passed" because they built a `MilknadoConfig`
directly with an `execution_agent` set to the dataclass default, a state
`load_config` never actually produces. They were corrected to build input
through `load_config` so the value is genuinely derived.

**Rule:** a test that constructs an unrealistic object to satisfy itself is
testing an impossible state. Build fixtures through the real entry point
(`load_config`), not by hand-constructing internals.

## CI: codecov/patch is the gate that catches under-tested branches

Both #96 and #109 had `codecov/patch` red while every other check was green
(merge state shows `UNSTABLE`, which is the red coverage check — **not** a
conflict). In #109 the new guard's non-DONE branches were simply untested. The
patch-coverage gap was not a separate task — it was the same fix: add the
branch tests the correctness finding already demanded.

**Rule:** new guard/branch code needs a test per branch, or `codecov/patch`
fails the patch. Fold the coverage fix into the correctness fix.

## Review signal: bot grading beat the fresh human-style pass

On #96, a fresh `/age` pass produced false positives and *missed* three real
config bugs that Copilot's inline comments caught. On #109/#110, fresh `/age`
findings deduped cleanly against the bot comments.

**Rule:** don't discount inline bot comments on a PR's own new code — on
config/validation surfaces they were the stronger signal. But verify reviewer
claims against intent: on #110 Copilot flagged the broadened Claude allowlist as
a security regression when the broadening *was* the PR's titled intent
(structured `worker.tools` as the new single source of truth, pinned by
`test_cli.py:1420`). Correct mechanics + wrong intent = reject the finding.

## Triage: open issues outlive their fix — verify against HEAD

The May 2026 MCP steel-thread review (#81) filed 43 findings. They were
remediated across PRs #103/#104/#105/#106/#107 (plus #99 and #21) and merged —
but the **issues were never closed**. A later triage pass took "open issue" as
"unfixed work" and nearly fanned out a 6-curd `/cheese-factory` run to
re-implement fixes already in `main`. Verifying each finding against HEAD found
**35 of 38** "high-priority" tickets already done; only #111 finding-3 was
genuinely unfixed.

**Rule:** before triaging or dispatching a finding backlog, verify each item
against current HEAD — read the cited `file:line` and check whether the
recommendation is already applied. A remediation PR routinely lands the fix
without closing the tracking issue, and area labels mislead on attribution
(#69's `tomli_w` fix was #99 not the graph-core batch; #58's rebase split was
#21). The #81 remediation map:

- **#104** — batch-planner (#66 #68 #71 #74–77 #79 #80)
- **#103** — graph-core (#41 #43 #45 #48 #59 #63–65 #67 #70 #72)
- **#105** — worker-dispatch (#38 #39 #40 #44 #47 #53)
- **#106** — detached-ralph (#46 #50 #52 #56 #57 #60 #62)
- **#107** — MCP enhancements (#82 run_list/run_cancel, #83 unified `RunDict`, #89 logging)

The cheap signal: a `/cheese-factory` decomposer that reads the code before
fanning out catches this for free — its first pass found 5/6 curds already
implemented. When a "fix" backlog is suspiciously large, suspect stale-open
issues before suspecting unfixed work.

## Review scope: diff-scoped review can't see an inherited encapsulation smell

On #111, `/age` reviewed the diff that promoted `_check_mega_batch` →
public `check_mega_batch` and routed it through the planning crust. The
encapsulation dimension passed it clean — *"helper promoted to public within its
owning slice … no cross-slice leak."* That grade was locally correct and
globally wrong. The real smell is structural and **predates the diff**:
`check_mega_batch` is a `BatchPlan` invariant that lives in `domains/planning`
but is **never called by the planner** — only by the MCP entrypoint
`_plan_batches_impl`. The CLI (`Planner.launch`) silently skips it, so the
invariant is enforced by *convention at call sites*, not by the producer. The
producer already owns a parallel size signal (`Batch.oversized` /
`PlanResult.oversized_count`), making the guard redundant as well as misplaced.

Why it slipped past:

1. **Diff-scoped review grades the delta, not the design.** The violation came
   in earlier (#27 added the guard, #104 wired the MCP path); the #111 diff only
   renamed + deduped, and within that frame encapsulation got *locally better*.
2. **The fix matched the "good" side of a hygiene pattern.** "Private symbol
   crossing a slice boundary → promote to public + crust-export" is a real
   Sliced-Bread rule. Promoting `check_mega_batch` pattern-matched the *good*
   side and masked the structural smell — the public surface only exists to be
   called from above the domain layer.
3. **Nobody challenged the spec's premise.** The mini-spec said "promote to
   public"; cook and age both asked "is that done well?" — never "should this
   be public, or should the validation move into the producer?"

**Rule:** when a spec's premise is "make X public" or "share X across the
boundary," challenge whether X should exist at the seam at all. A guard/
validation that **all callers must invoke** (or that lives in a slice but is
only called from outside it) signals the **domain isn't validating its own
invariant** — move the invariant into the producer so every caller inherits it,
and the public surface + cross-slice import both disappear. Asymmetry (one
caller applies the check, another skips it) is the proof it's enforced by
convention, not construction. This pattern is being added to `/age`'s
encapsulation dimension as a `high` trigger — see
[easy-cheese#110](https://github.com/paulnsorensen/easy-cheese/issues/110).
Diff-scoped review is the right default, but its blind spot is an inherited
violation that the diff *tidies*.

## Lifecycle: the stop_on_error signal must equal the iteration classification (#244)

`_run_agent_phase` (`src/milknado/loop/engine.py`) derives two things from one
`AgentResult`: the emitted event via `_classify_iteration_outcome` (the
authority), and the boolean fed to `stop_on_error`. #244 first fixed a
disagreement — a turn cap classifies COMPLETED, but `stop_on_error` saw
`agent.success == False` (the streaming path SIGKILLs the capped child) and
marked the run FAILED — by returning `agent.success or agent.turn_capped`. That
reintroduced the *opposite* disagreement. On the **blocking path**,
`_count_tool_uses_post_hoc` sets `turn_capped=True` from the tool-use count
*after* the child exits, so a run that both exceeds `max_turns` and times out
returns `timed_out=True` **and** `turn_capped=True`. `_classify_iteration_outcome`
checks `timed_out` first (→ TIMED_OUT), but `agent.success or agent.turn_capped`
was `True`, so a real timeout stopped tripping `stop_on_error` and the run could
finish COMPLETED with `failed > 0`. Fix: derive the signal from the
classification — `iteration_succeeded = event_type == EventType.ITERATION_COMPLETED`.

**Rule:** when one result object feeds both a status classification and a
control-flow boolean, compute the boolean *from* the classification, never
recompute it from the raw fields — two independent derivations of "did this
succeed" drift the moment a new field (`turn_capped`) can co-occur with an old
one (`timed_out`). The streaming path makes those flags mutually exclusive; the
blocking post-hoc path does not, so test the co-occurring case
(`timed_out and turn_capped`), not just the happy cap.

## Tests: completion sentinels belong to the process being observed (#325)

A force-stop regression originally handled `SIGTERM` by touching an empty sentinel, spawning a detached helper to fill it after a delay, and exiting. Python 3.11 CI could observe the file forever empty even though the handler ran: the assertion depended on scheduling and survival of a second process rather than completion of the event under test.[^sentinel-failure] The reliable test writes the completed `stopped` value directly in the signal handler before `SystemExit`, then polls for that content because joining the manager thread does not prove its descendant finished handling the signal.[^sentinel-fix]

The corrected sentinel still exposed a production race: after the process-group leader exited, `_terminate_lingering_group` sent SIGTERM and immediately SIGKILL, so the descendant could die before its handler wrote the completed sentinel. The cleanup now uses the existing SIGTERM grace period to poll group existence with signal `0`, returning if the group disappears and escalating only when the deadline expires.[^lingering-fix] Focused tests mock the clock and sleep so both graceful exit and SIGKILL-race behavior are deterministic without wall-clock delays.[^lingering-tests]

**Rule:** a completion sentinel must be finalized by the process whose completion it represents. Do not touch an incomplete marker and delegate its final value to a detached helper; that tests helper scheduling, not the lifecycle event. When the production join covers only an ancestor, poll for completed sentinel content rather than file existence. Cleanup must also allow the descendant's SIGTERM handler to finish before escalation: poll group liveness throughout the grace period, then SIGKILL only if it remains.

[^sentinel-failure]: PR #325 Actions job 89758215559; `tests/loop/test_manager.py:311-367` before commit `5889c635a07627461d6acab44a92adb6ae90ff17`
[^sentinel-fix]: `tests/loop/test_manager.py:311-360`; PR #325 commit `5889c635a07627461d6acab44a92adb6ae90ff17`
[^lingering-fix]: `src/milknado/loop/_agent.py:971-990`
[^lingering-tests]: `tests/loop/test_agent.py:2107-2145`

_Source: PR #325 affinage · Updated: 2026-07-26 · Supersedes: none_

## Execution: a swallowed signal reads as success, not as "unknown" (#355, #356, #359)

Three sibling fixes from the same review cycle share one root cause: a
failure signal was captured correctly but then either miscompared, dropped
on the floor, or folded into the wrong caller-facing verdict.

- **Literal-string drift between writer and filter.** `runs.status` is
  written lowercase (`'running'`) everywhere but was filtered
  uppercase (`status != 'RUNNING'`) in one query. SQLite's BINARY collation
  makes that comparison case-sensitive, so the "terminal runs only" prune
  guard matched every run and could delete a still-running node's message
  history.[^status-casing] No test exercised pruning at all before the fix.
- **A swallowed exception was misattributed to the wrong party.** The
  completion verifier's git probes caught their own failures
  (`CalledProcessError`/`TimeoutExpired`/`OSError`, plus an unhandled
  `UnicodeDecodeError`) and returned `False`, which the caller read as "clean
  tree" and then as "worker produced no change" — feeding that verbatim back
  into the worker's next-iteration rejection feedback. A non-zero exit, a
  timeout, or a missing `git` binary was attributed to the agent, and the
  node was rejected and redispatched against a still-broken
  environment.[^git-probe] The fix gives infra failure its own exception
  (`_GitProbeFailure`) and its own caller-facing message, distinct from a
  genuinely clean tree.
- **A written-but-unread result field hid a real failure as a clean
  completion.** `CompletionResult.review_notification_failed` was set on
  three paths in `executor.py` but had no consumer in the run loop — a
  review whose verdict never reached the worker rendered as an ordinary
  completed node. Found via a `vulture` dead-code sweep, not a
  reported bug.[^notify-field]

**Rule:** when a result/status value is computed on a failure path,
grep for every sibling field on the same result type and confirm each has a
consumer — an unread field is a silent failure waiting to be misread as
success. When an except-block converts an infrastructure failure into a
caller-facing verdict, give "I couldn't tell" its own return value or
exception; collapsing it into the same `False`/`None` as the negative case
misattributes the failure to whichever party the caller blames for a
negative result. For any hand-written SQL literal compared against a
column written elsewhere in the module, bind one named constant instead of
inlining the string at each call site — casing/spelling drift between a
writer and a filter is invisible in review because the permissive branch is
the one that fires.

[^status-casing]: `src/milknado/domains/graph/_persistence.py:537` (`_RUN_STATUS_RUNNING`); issue #329, PR #355 commit `b1713c95d22be260de8bedd64023403cf0aac988`
[^git-probe]: `src/milknado/domains/execution/completion.py` (`_GitProbeFailure`, `_change_rejection`); issue #330, PR #356 commit `3d2c46d58fdd76ab17640954e3d64201dff517d6`
[^notify-field]: `src/milknado/domains/execution/run_loop/_completion.py`, `src/milknado/domains/execution/executor.py`; PR #359 commit `0a094144d14042741d963472d67203703487b88b`

_Source: PRs #355, #356, #359 · Updated: 2026-08-16 · Supersedes: none_

## Gate: a coverage-gate zero-caller finding can mean "silently disconnected," not "confirmed dead" (#416)

The whole-symbol dead-code gate ([[dead-code-coverage-gate-decision]]) correctly
flagged `complete_root` as zero-caller, and issue #365 AC-2 directed deleting it
as confirmed dead. The gate was right about the symptom and wrong about the
cause: the `run_loop.py` → `run_loop/` package split had silently dropped the
only live call to `complete_root`, not deliberately removed it. Every headless
and TUI `milknado run` left the root goal stranded `PENDING` even after every
child node finished `DONE`, because the sole completion path
(`_maybe_verify_spec`) early-returns on a `spec_text` that no production caller
ever supplies — only tests ever set it. Deleting the symbol would have
cemented the bug and erased the evidence of the dropped wiring.

**Rule:** before actioning a dead-code-gate finding as "delete", read why the
symbol lost its last caller. A symbol that shipped *with* a caller and now has
none is a candidate for "wiring dropped in a refactor", not just "confirmed
dead" — restoring the call site can be the correct fix. The commit that
introduced the symbol and the commit that removed its last call tell you which
case you are in. The fix here also added a guard the original code lacked:
structural root-completion refuses to fire on a root with no non-root nodes,
since completion over an empty child set is vacuously true.

_Source: PR #416 (commit `ca93d305b3008009b80134533fdb8553d76a009e`) · Updated:
2026-09-06 · Supersedes: issue #365 AC-2 (which had directed deletion)_
