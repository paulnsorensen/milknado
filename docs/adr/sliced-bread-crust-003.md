# ADR — App owns policy; entrypoints stay thin I/O (sliced-bread-crust-003)

Date: 2026-07-20 · Status: accepted · Spec: sliced-bread-crust

## Decision

Known application/policy/composition symbols (19 B-candidates and peers) move into `milknado.app`. CLI/MCP modules only parse, call `app.*`, and format/register. Clear domain leaks go into existing `domains/` barrels during extract; exhaustive hunt is follow-up F001.

## Rationale

User asked for better crusts and for adapter-wiring/policy in crusts to live in `app/`, with entrypoints invoking those app files. Classification found policy densest in `cli_plan` and `mcp_node`; no clear domain leaks yet.

## Alternatives

- Composition-only extract (project open + branch guard) — leaves fat entry modules.
- Push everything into `domains/` — wrong layer for orchestration/wiring.

## Consequences

`app/` grows beyond the #121 stub. Cross-cutting seams start with `open_graph` / `resolve_project_root` (~10 callers).
