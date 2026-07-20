# ADR — Easy-cheese compatibility via the durable-doc boundary (adversarial-review-loops-003)

Date: 2026-07-19 · Status: accepted · Spec: adversarial-review-loops

## Decision

Milknado integrates with the easy-cheese pack at the artifact/format layer, not via runtime coupling: coder-flavored briefs embed the resolved spec markdown path (durable corpus `$XDG_DATA_HOME/cheese/<project>/specs/`, `.cheese/specs/` fallback); reviewer deposits persist as `/age`-format findings (`## Blocker/High/Medium/Low`, `**[<dimension>]** path:line` bullets) at `.cheese/age/<node-slug>.md`; handbacks carry the four-field block (`status/next/artifact/orientation`). `review` and `plate` join `BUILTIN_FLAVORS`, mirroring the `/age` and `/plate` phases.

## Rationale

Easy-cheese's own phase contract is markdown + YAML frontmatter over fixed paths — no runtime API exists to couple to. Conforming at the doc boundary lets `/cure`, easy-cheese coders, and human readers consume milknado worker output directly, and lets milknado workers consume `/mold` specs unchanged, while keeping the engine harness-agnostic (workers stay claude/codex/gemini subprocesses).

## Consequences

Briefs teach the formats; the engine persists deposits at the phase paths but does not deep-validate schemas (parse failures on the `<verdict>` tag fail fast after one re-prompt; findings markdown is stored as-is).
