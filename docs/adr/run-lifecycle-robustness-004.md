# ADR — Hybrid verdict durability: node_reviews table plus in-memory handoff (run-lifecycle-robustness-004)

Date: 2026-07-23 · Status: accepted · Spec: run-lifecycle-robustness

## Decision

Review verdicts become first-class graph state: migration v2 creates `node_reviews(node_id, round, verdict, findings, created_at)` with FK to `nodes(id)` only — deliberately not `runs`. `_notify_review` dual-writes: the `node_reviews` row for audit/queryability, and the in-memory findings thread (run-lifecycle-robustness-002) for the redispatch handoff.

## Rationale

- Verdicts were ephemeral log lines in `run_messages` — fragile (FK-failed when the `runs` row was missing, #296) and unqueryable.
- The redispatch handoff must not become DB-dependent: #297 exists precisely because the DB can be down mid-loop; the in-memory thread delivers findings regardless.
- Migration v2 dogfoods the new automigrate machinery (run-lifecycle-robustness-003) on a real, low-risk table.

## Alternatives

- Seam patches only (runs row + in-memory thread, no table) — smallest correct fix, but verdicts stay ephemeral with no audit trail.
- Table-only handoff (redispatch reads findings back from `node_reviews`) — simplest data flow, but makes the review loop DB-dependent, the exact class #297 heals.
- FK from `node_reviews` to `runs` — reintroduces the #296 bug class.

## Consequences

Verdicts gain a durable, queryable audit trail keyed by `(node_id, round)`. `_notify_review` writes two places; the table write must tolerate absence of the corresponding `runs` row by construction.
