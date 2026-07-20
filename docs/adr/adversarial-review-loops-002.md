# ADR — Node-scoped agent sessions with worktree pinning (adversarial-review-loops-002)

Date: 2026-07-19 · Status: accepted · Spec: adversarial-review-loops

## Decision

`session_mode = fresh | resume` is a per-flavor setting. `resume` introduces a new node-scoped entity (`NodeAgentSession`) — explicitly NOT a field on the per-dispatch `run` record — capturing the CLI session id from first-turn JSON output and resuming it across review/redispatch rounds. Resume-mode nodes pin their worktree until node approval, always exec from that worktree cwd, and re-pass model/effort flags every round. `resume` + `cursor-agent` fails config validation.

## Rationale

Cited research (`.cheese/research/agent-cli-headless-resume/agent-cli-headless-resume.md`): every CLI ties sessions to the original cwd — Claude hard-fails cross-cwd resume, Codex silently rebases cwd to the caller's dir, Gemini requires the original project dir; no CLI restores model flags on resume; cursor-agent resume has no primary-source documentation. The `run` ≠ `session` aliasing is stated up front because milknado's `run` means one dispatch.

## Consequences

Worktree teardown ordering for resume-mode nodes must wait for node approval (spike: milknado F002 issue). Fresh mode remains the default and keeps today's behavior byte-identical.
