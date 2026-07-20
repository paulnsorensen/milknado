# ADR — Ordered curds: package → extract → import-linter (sliced-bread-crust-002)

Date: 2026-07-20 · Status: accepted · Spec: sliced-bread-crust

## Decision

Ship the redesign as three stacked PRs (curds): (1) behavior-identical package move, (2) extract into `app/`, (3) import-linter in `check-llm`.

## Rationale

Aggressive extract-everything in one PR mixes renames with behavior-preserving moves and is hard to review; a mid-extract failure leaves half-moved imports. Stacking isolates blast radius while keeping one approved spec.

## Alternatives

- Single mega-PR — faster wall-clock, worse review and bisect.
- Extract-only without packaging — leaves root flood.

## Consequences

Merge bottom-up. Each curd must pass `just check-llm`. Follow-ups F001–F004 cover sharpening after the stack lands.
