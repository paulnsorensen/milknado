# ADR — Engine-native adversarial reviewers (adversarial-review-loops-001)

Date: 2026-07-19 · Status: accepted · Spec: adversarial-review-loops

## Decision

Milknado spawns plan-critic and node-reviewer subprocesses itself, from config (`plan_reviewer_agent`, per-flavor `review_agent`), rather than exposing verdict-injection MCP tools for a coordinator session to drive.

## Rationale

- CLI and MCP-driven runs behave identically — a bare `milknado run` gets the same adversarial gates as a coordinator-orchestrated one.
- The coordinator/worker boundary never inverts: the engine owns loop mechanics, the coordinator observes state (review rounds/verdicts in `status`) and receives structured deposits (`milknado_deposit_review`).
- Rejected alternative (MCP-coordinator-driven, and the hybrid) recorded in `.cheese/.out-of-scope/adversarial-review-loops-001.md`.

## Consequences

Reviewer/critic agents are configured as full CLI command strings reusing the existing per-family argv builders and tool allowlists; multi-model workflows (Claude plans, Codex reviews) are pure config.
