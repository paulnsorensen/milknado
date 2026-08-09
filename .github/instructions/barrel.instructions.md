---
applyTo: "src/milknado/domains/*/__init__.py"
excludeAgent: "coding-agent"
---

## Barrel contract review — strictest tier

A diff to a domain `__init__.py` barrel is a public API change for its slice. Review it as a contract change, not a code change: default severity one tier higher than the same finding elsewhere.

For every **added or changed** export:

- Name the out-of-slice caller that needs it — in this PR's diff, or an existing call site in `app/`, `cli/`, `mcp/`, `adapters/`, or another domain. No caller, no export: request removal, not a TODO.
- Re-exporting an underscore-private module's type publishes that internal's API — flag it unless the type genuinely is the slice's public product; prefer a function that hides it.
- Implementation logic accreting in the barrel itself belongs in an internal module — the barrel stays a thin facade over deep internals.
- An export that 1:1 mirrors one internal function's shape adds interface without hiding anything — question whether it should exist at that granularity.

For every **removed or renamed** export: check all consumers moved with it — `app/`, `cli/`, `mcp/`, `adapters/`, sibling domains, and tests. import-linter will not catch a stale caller of a renamed export; flag it now.
