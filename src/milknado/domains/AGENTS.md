# Domain slices

These rules add to the root `AGENTS.md` for files under this directory.

## Purity (narrows root principle 3)

- Keep domain code free of I/O. Do not add `subprocess`, `os.kill`, `psutil`, network calls, or file reads.
- Define a `Protocol` port for each external need, in `common/protocols.py` or a slice `ports.py`. Implement it in `../adapters/`.
- Do not import `milknado.loop`. The existing imports in `execution/` and `coordinator/` are audited debt (issue #227). Do not add more.
- Do not import `milknado.app`, `milknado.cli`, `milknado.mcp`, `milknado.web`, or `rich`. `lint-imports` enforces this.
- `graph/` must not import `execution/`. The graph is a state crust, not a policy consumer.

## Slice crusts

- Each slice exports its public API from its `__init__.py`. Import another slice only through that barrel.
- Put cross-slice translation in a bridge module, for example `planning/batching_bridge.py`. Keep it out of the pure slice.
- Private `_module` files are not an import surface for app, cli, mcp, adapters, or web. A new direct import needs a documented exception in the `lint-imports` contract.

## Size (narrows root "Code Style")

- `execution/executor.py`, `graph/graph.py`, `graph/_persistence.py`, and `common/config.py` exceed 300 lines. Do not grow them. Extract a sibling module.

For fast feedback on imports, run `uv run lint-imports`. `just check-llm` stays the gate.
