# T3 Code

T3 Tools Inc.'s T3 Code repository, last verified 2026-11-17, provides an agent runtime with an outside-MCP thread-control interface.
Canonical source: [T3 Code](https://github.com/pingdotgg/t3code).
The inspected revision is `6ab633f6d6e8273f08a060137ab64463638787ce`.[^runtime]

## Runtime contribution

T3 Code owns provider sessions, turn operations, event streams, and optional provider-specific continuation.
Its command path commits events, projections, receipts, and effect intent before executing side effects.[^runtime][^intent]
Application-owned delegation can create child threads and request isolated worktrees.[^delegation]
These mechanisms support evaluating T3 as Milknado's worker runtime.
They do not establish a replacement for Milknado's dependency admission or goal acceptance policy.

## Outside MCP control and limits

T3 Code's outside-agent interface lets authorized clients launch, read, message, wait for, and interrupt threads.
Outside callers launch top-level threads rather than use in-thread-only delegation.[^outside]
A launch accepts an explicit workspace strategy and a client request key for retries.
Dependent work must use the correct base; uncommitted edits do not transfer.[^launch]

T3 Code rejects a browser Origin on its MCP endpoint before authentication.[^mcp]
Its pending-request MCP tool handles user questions, not provider permission approval.[^permission]
MCP-client credentials cannot access ordinary HTTP or WebSocket routes.[^auth]
A separate web orchestration interface therefore needs more than direct browser MCP calls.
Permission handling needs T3's human-facing surface or another separately authorized integration.

## Recovery and audit limits

T3 Code recovery reconciles interrupted process-bound work instead of guaranteeing transparent replay of every provider action.[^recovery]
Its retained event history cannot rebuild all historical projections after pruning.
Clients can need a fresh snapshot.[^overview]
A goal controller must not equate dispatch acknowledgement or provider completion with accepted code.

## Relevance to Milknado

T3 Code could supply runtime execution while Milknado retains planning, reviewed graph application, dependency admission, and acceptance.
This is a source-grounded hypothesis, not an approved integration decision.
See [planning](../architecture/planning.md) for the existing batch-plan boundary.
See [web metaharness view](../architecture/web-metaharness-view.md) for web integration constraints.

T3's inspected execution schema records nested runtime work.
The inspected contracts do not establish Milknado-style prerequisite and file-conflict scheduling.[^graph]
Five Sol agents supplied the first research pass and independent checks; the parent verified decisive implementation sections.
No live OAuth, provider, restart, worktree, or browser integration check ran.

## Backend robustness and complexity

T3 Code's backend is substantially larger than Milknado's selected runtime, but it is not uniformly more robust.
A second source pass measures 252,903 code lines in T3's server and 45,510 in Milknado's selected runtime: 5.56×.
The narrower orchestration/execution scopes contain 82,161 and 30,516 code lines: 2.69×.[^backend-size]
These are implementation-size ratios, not feature-normalized complexity or reliability scores.
T3's separate provider packages, contracts, and client runtime are outside the headline count.
The parent reproduces the counts with Tokei 15.0.0 at the pinned revisions.[^backend-size]

T3 Code centralizes command receipts, events, projections, and outbox intent in one transaction.[^intent]
Milknado already has bounded durable commands, same-ID checks, receipts, and run/invocation/owner fences.[^milk-commands]
T3's advantage here is a broader common atomic boundary, not durable storage versus volatile storage.

Both backends contain substantial process safeguards.
T3's ACP runtime tracks descendant identities, escalates TERM to KILL, checks survivors, and supports cgroup or platform-specific cleanup.[^backend-process]
T3 also persists provider identities and tests restart reaping with real child processes.[^backend-reaper]
Milknado's dispatch recovery requires confirmed covered-worker exit before closing evidence records.[^milk-reaper]
The inspected T3 provider reaper retains unkillable records and permits startup; Milknado's inspected cancellation path preserves unresolved state.[^backend-reaper][^milk-reaper]
This is a specific availability-versus-fail-closed difference, not a whole-backend ranking.

T3 Code adds per-RPC authority and selected stream ACK windows for its broader remote/client surface.[^backend-rpc][^backend-stream]
Milknado already checks Host, cookie, and write Origin, and evicts slow snapshot clients.[^milk-web]
T3's stream byte thresholds are not universal hard memory limits; a single large snapshot can exceed them.[^backend-stream]

T3 explicitly accepts possible recent-commit loss on OS crash or power loss with SQLite `synchronous=NORMAL`.[^backend-sqlite]
Milknado quarantines corrupt databases and recreates them; this does not recover graph data.[^milk-db]
No comparative reliability, load, provider, or crash test runs during this research.
Six Sol agents cover research, independent verification, and wiki location; the parent checks decisive source and metrics.
The independent pass corrects an unsupported initial claim that Milknado has uniformly stronger process cleanup.

Runtime reuse could reduce Milknado's owned infrastructure, while retaining integration and upstream-maintenance costs.
This remains a hypothesis, not approval to replace Milknado's goal admission or acceptance policy.

## Provenance

T3 Code evidence comes from the pinned upstream clone and its matching documentation.
The comparison uses Milknado revision `c9250b8f039e4c06b38df728c3aade180a489614`.
The first comparison source hash is `d20e8a1cdd3fddcc`.
The backend second-pass source hash is `e0499de28f6ab053`.
Source mechanisms are verified at these revisions; integration feasibility and comparative reliability remain untested.

[^runtime]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/provider-core/src/server/ProviderAdapter.ts (verified 2026-11-17).
[^intent]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/orchestration-v2/EventSink.ts (verified 2026-11-17).
[^delegation]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/OrchestratorMcpService.ts (verified 2026-11-17).
[^outside]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/docs/user/outside-agents.md (verified 2026-11-17).
[^launch]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/toolkits/project/tools.ts (verified 2026-11-17).
[^mcp]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/McpHttpServer.ts (verified 2026-11-17).
[^permission]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/toolkits/thread/tools.ts (verified 2026-11-17).
[^auth]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/docs/internals/environment-auth.md (verified 2026-11-17).
[^recovery]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/orchestration-v2/ProviderRuntimeRecoveryService.ts (verified 2026-11-17).
[^overview]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/docs/internals/overview.md (verified 2026-11-17).
[^graph]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/contracts/src/orchestrationV2.ts (verified 2026-11-17).


[^backend-size]: Parent Tokei 15.0.0 measurements on 2026-11-17 at the pinned revisions. T3 broad scope: `tokei apps/server/src --types TypeScript,TSX -e '*.test.ts' -e '*.test.tsx' -e '*.spec.ts' -e '*.spec.tsx' -e '*.testkit.ts' -e testUtils -e testkit -e fixtures -e __fixtures__ -e _generated`. Milknado broad scope: `tokei src/milknado/domains src/milknado/adapters src/milknado/loop src/milknado/mcp src/milknado/web src/milknado/app --types Python -e static`. Narrow scopes: T3 `apps/server/src/orchestration-v2`; Milknado `domains/coordinator`, `domains/dispatch`, `domains/execution`, `domains/graph`, `loop`, and `adapters` under `src/milknado`. Code LOC excludes comments and blanks; languages and capabilities differ.
[^milk-commands]: `src/milknado/domains/graph/_command_persistence.py:107-177`; `src/milknado/domains/graph/_command_claim.py:72-140`; `src/milknado/adapters/_loop_session.py:126-172`.
[^backend-process]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/provider-acp/src/server/AcpSessionRuntime.ts (lines 885–1023 and 1614–1652, verified 2026-11-17).
[^backend-reaper]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/process/providerProcesses.ts (lines 93–135); https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/process/providerProcesses.test.ts (lines 58–126, inspected, not run).
[^milk-reaper]: `src/milknado/domains/dispatch/reap.py:205-300`; `src/milknado/domains/dispatch/cancel.py:169-227`.
[^backend-rpc]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/auth/RpcAuthorization.ts (lines 39–96, verified 2026-11-17).
[^backend-stream]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/terminal/OutputProtocol.ts (lines 16–34, 115–139, and 185–205, verified 2026-11-17).
[^milk-web]: `src/milknado/web/guards.py:39-63`; `src/milknado/web/fanout.py:124-145`.
[^backend-sqlite]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/persistence/Sqlite.ts (lines 67–94, verified 2026-11-17).
[^milk-db]: `src/milknado/domains/graph/_connection.py:36-68`.

_Source: T3 Code pinned source, reproduced metrics, and independent parent-checked verification · Updated: 2026-11-17 · Supersedes: none._
