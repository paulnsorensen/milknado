# ADR — Preserve live output iteration

Date: 2026-09-30 · Status: accepted

## Context

Generic blocking workers can retain a reader after bounded cleanup expires.
Result extraction can then iterate the output buffer while that reader appends another line.

## Decision

Share one list-backed output buffer between generic agents and native sessions.
Keep each caller's existing character limit.
Preserve iteration that tolerates appends and can observe late lines.

## Alternatives

A deque improves oldest-line removal but raises when the buffer changes during iteration.
A snapshot prevents that exception but hides lines appended after iteration starts.
Neither alternative preserves the generic path's existing behavior.

## Consequences

The shared buffer retains linear-time oldest-line removal.
A deterministic interleaving test protects late-line visibility without timing or process mocks.
Process cleanup and platform containment remain unchanged.

## Evidence

- `src/milknado/loop/_agent.py`: `_drain_readers`, `_cleanup_agent`, and `_run_agent_blocking`.
- `src/milknado/loop/_output.py`: `BoundedOutput`.
- `tests/loop/test_agent.py`: `TestBoundedOutput`.

This file records the decision because the wiki ingestion service is unavailable in this session.
