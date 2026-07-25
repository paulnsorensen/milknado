# ADR — Node-scoped agent sessions with worktree pinning (adversarial-review-loops-002)

Date: 2026-07-19 · Status: accepted · Spec: adversarial-review-loops

## Decision

`session_mode = fresh | resume` is a per-flavor setting. `resume` introduces a new node-scoped entity (`NodeAgentSession`) — explicitly NOT a field on the per-dispatch `run` record — capturing the CLI session id from first-turn JSON output and resuming it across review/redispatch rounds. Resume-mode nodes pin their worktree until final approval, always exec from that worktree cwd, and re-pass model/effort flags every round. `resume` + `cursor-agent` fails config validation.

## Rationale

The 2026-07-25 local spike verified Claude Code 2.1.218 and Codex 0.145.0 retaining prior conversational context both in an unchanged worktree and after deletion/recreation at the same path. That result does not validate historical tool calls against a replacement filesystem, so the engine must preserve the original path until no resume remains. The locally installed Cursor CLI failed to start before version/authentication/session capture, so the Cursor validation refusal remains necessary.[^1]

The `run` ≠ `session` aliasing is stated up front because milknado's `run` means one dispatch.

## Consequences

`Executor.complete` runs the review gate before merge-back; rejection redispatches the same worktree and only final approval reaches landed-worktree removal. Fresh mode remains the default and keeps today's behavior byte-identical.[^2]

[^1]: docs/agent-cli-headless-resume.md
[^2]: src/milknado/domains/execution/executor.py:925-987, 1306-1341, 337-375