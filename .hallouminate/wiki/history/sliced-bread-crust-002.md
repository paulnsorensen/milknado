# ADR — Ordered curds for crust redesign (sliced-bread-crust-002)

Date: 2026-07-20 · Status: accepted · Spec: sliced-bread-crust

Ship as three stacked PRs: (1) package move, (2) `app/` extract, (3) import-linter — not one mega-PR. Isolates rename blast from behavior-preserving extract and keeps each curd `just check-llm` green.

Rejected: single mega-PR; extract-without-packaging.
