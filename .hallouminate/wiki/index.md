# Milknado Wiki

Cross-session knowledge for **milknado** — a Mikado execution engine that
decomposes goals into dependency graphs and runs them as parallel ralph loops.

This wiki is the canonical home for durable knowledge that does not belong in
code comments: architecture, conventions, and the lessons behind past decisions.
Each section's `index.md` is auto-maintained by hallouminate (the
`HALLOUMINATE:INDEX` block) — add a topic file under a section and it appears in
that section's list automatically.

## Sections

- [Architecture](./architecture/index.md) — how the engine is built: the
  Mikado graph, planning, batching, execution/dispatch, adapters, entry points.
- [Conventions](./conventions/index.md) — the build gate (`just check-llm`),
  code style, complexity budgets, and engineering principles.
- [History](./history/index.md) — distilled decisions and recurring review
  lessons (the durable residue of `.cheese/` review artifacts).

## Start here

- New to the codebase → [Architecture › Entry Points](./architecture/entry-points.md),
  then [Graph Domain](./architecture/graph.md).
- About to open a PR → [Build Gate — just check-llm](./conventions/build-and-gates.md).
- Touching the runner/executor → [Execution & Dispatch](./architecture/execution.md)
  and [Review Lessons](./history/review-lessons.md).

## Authoring conventions

- One topic per file (`<section>/<slug>.md`), first line `# Title`.
- Record the *why* and the *shape*, not the *what* — the code already says what.
- Read before overwriting; prune what goes stale.
- Don't hand-edit a section `index.md` between the `HALLOUMINATE:INDEX` markers —
  it's regenerated on every `add_markdown`.
