# ADR — Sibling cli/ + mcp/ packages over thin root or crust/ umbrella (sliced-bread-crust-001)

Date: 2026-07-20 · Status: accepted · Spec: sliced-bread-crust

## Decision

Delivery modules live in `milknado.cli` and `milknado.mcp` packages. Application/policy lives in `milknado.app`. Package root is not a flat flood of `cli_*` / `mcp_*` files.

## Rationale

- User preferred thin root files unless too fat; after extraction, 8 `mcp_*` + 7 `cli_*` registration modules remain — thin root of two files is not enough.
- `crust/` umbrella was rejected in favour of sibling packages so `app/` stays the application layer name already started in #121.
- Separating `app/` from delivery packages makes "entrypoints invoke app" structurally obvious.

## Alternatives

- Kill `app/` stub only (Do Nothing on packaging) — leaves the root flood.
- Thin root only (`cli.py` + `mcp_server.py`) — fails registration-surface test.
- `milknado/crust/{cli,mcp}` umbrella — extra nesting; conflicts with finishing `app/` as the application layer.

## Consequences

pyproject scripts retarget to `milknado.cli:app` and `milknado.mcp.server:main`. Binary names unchanged. Tests and patches update import paths.
