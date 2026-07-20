# ADR — Sibling cli/mcp packages (sliced-bread-crust-001)

Date: 2026-07-20 · Status: accepted · Spec: [[sliced-bread-crust]][^1]

Delivery modules live in `milknado.cli` and `milknado.mcp`; application/policy in `milknado.app`. Thin-root-only failed the registration-surface test (8 `mcp_*` + 7 `cli_*` modules remain after extract).[^2]

Rejected: kill-app-stub-only; thin root of two files; `crust/` umbrella package.

[^1]: Durable spec corpus: sliced-bread-crust
[^2]: Shape check: 32 @mcp.tool across 8 modules
