# ADR — Application and MCP boundary ownership (sliced-bread-crust-003)

Date: 2026-07-20–21 · Status: accepted · Spec: sliced-bread-crust

## Application policy

Known application/policy/composition symbols (19 B-candidates and peers) move into `milknado.app`. CLI/MCP modules only parse, call `app.*`, and format/register. Clear domain leaks go into existing `domains/` barrels during extract; exhaustive hunt is follow-up F001.

### Rationale

User asked for better crusts and for adapter-wiring/policy in crusts to live in `app/`, with entrypoints invoking those app files. Classification found policy densest in `cli_plan` and `mcp_node`; no clear domain leaks yet.

### Alternatives

- Composition-only extract (project open + branch guard) — leaves fat entry modules.
- Push everything into `domains/` — wrong layer for orchestration/wiring.

### Consequences

`app/` grows beyond the #121 stub. Cross-cutting seams start with `open_graph` / `resolve_project_root` (~10 callers).

## MCP boundary parsers

Keep `_parse_kind`, `_parse_todo_status`, and `_parse_flavor` in `milknado.mcp._core`.

These functions normalize external MCP arguments at the transport boundary. They do not own application policy or flavor-profile resolution, so moving them into `app/` or `domains/` would widen the policy surface without creating a shared caller.

### Consequences

MCP handlers must call the boundary parsers for enum-like inputs and expose their `ValueError` messages as structured tool errors. Domain graph validation and configuration-registry construction remain responsible for their own invariants.
