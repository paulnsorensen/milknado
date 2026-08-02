# ADR — Lifecycle is a separate document field, not a NodeStatus extension (formalize-roadmap-graph-schema-001)

Date: 2026-08-02 · Status: accepted · Spec: formalize-roadmap-graph-schema

## Decision

Represent authoring-lifecycle judgments (`deprecated`) as a new goal-file frontmatter field `lifecycle: active | deprecated`, mapped to `MikadoNode.archived_at` on import. The `status:` field stays exclusively execution-owned with the NodeStatus vocabulary (`pending|running|done|blocked|failed`).

## Rationale

NodeStatus is an execution state machine with a transition table (DONE terminal, BLOCKED/FAILED→PENDING); "deprecated" is an authoring judgment with no place in run transitions. Hand-authored wikis were already writing `status: deprecated` into a field the exporter overwrites — a collision, not a vocabulary gap.

## Alternatives

- Extend NodeStatus with `deprecated` — pollutes the run-state machine with a non-execution state.
- No field; deprecation = archiving via the graph — hand-authored wikis without a graph could not express it.

## Consequences

A hand-authored `status:` value outside NodeStatus fails parse loudly. Existing wikis using `status: deprecated` (cheez-wiki) must migrate to `lifecycle: deprecated` (follow-up F001). Deprecated documents are exempt from Intent/Acceptance requirements.
