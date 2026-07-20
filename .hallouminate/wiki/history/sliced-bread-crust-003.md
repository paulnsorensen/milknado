# ADR — App owns policy; entrypoints thin I/O (sliced-bread-crust-003)

Date: 2026-07-20 · Status: accepted · Spec: sliced-bread-crust

Nineteen known B-candidates (policy/composition) move into `milknado.app`; CLI/MCP only parse → call `app.*` → format. Domain leaks go into existing `domains/` barrels when found; exhaustive hunt is follow-up F001.

Rejected: composition-only extract; pushing orchestration into `domains/`.
