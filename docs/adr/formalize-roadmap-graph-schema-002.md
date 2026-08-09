# ADR — Model owns read/interchange; surgical writes stay (formalize-roadmap-graph-schema-002)

Date: 2026-08-02 · Status: accepted · Spec: formalize-roadmap-graph-schema

## Decision

The canonical pydantic document model is authoritative for reading and interchange (JSON Schema, JSON instances, renderers). Wiki files stay authoritative for presentation: all writes remain surgical (`set_frontmatter_field`, `replace_harvest_block`, atomic writes); full canonical render is used only for the existing new-file orphan-scaffold path.

## Rationale

Canonical re-render on export would destroy hand-authored formatting, comments, footnotes, and any section the model does not cover. Byte preservation is an existing, tested exporter property users rely on.

## Alternatives

- Canonical re-render — perfect model round-trip, unacceptable loss of hand-authored bytes.

## Consequences

The model can lag presentation content it does not parse; mitigated by whole-document coverage (frontmatter + H1 + Intent/Acceptance/Outcome). Renderer/JSON codecs read through the model only, never re-derive from raw markdown.
