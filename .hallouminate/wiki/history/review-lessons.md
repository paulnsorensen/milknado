# Review Lessons — Recurring Findings & Gotchas

Distilled from review/cure cycles on PRs #96, #109, #110. Grouped by theme —
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
