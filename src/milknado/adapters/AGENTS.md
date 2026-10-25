# Adapters

These rules add to the root `AGENTS.md` for files under this directory.

## Scope (narrows root principle 3)

- One adapter wraps one external system: git, tmux, gh, CRG, hallouminate, OS processes, or the loop engine.
- An adapter implements a port from `../domains/common/protocols.py` or a slice `ports.py`. Domain code depends on the port.
- An adapter reports facts. Retry, replan, accept, and refuse decisions belong in the domain or in `../app/`. Example: `LoopAdapter.verify_spec` returns `done`, `gaps`, or `unavailable`, and the run loop decides.
- Fail loud on an external error. Raise with the exit code and stderr. Do not return an empty default.

## Imports

- Do not import `milknado.app`. `lint-imports` enforces this.
- Import domain types from slice barrels, not from private `_module` files. `lint-imports` enforces this.
- Import the loop engine only from `milknado.loop` (its `__all__`). Do not add an import of a private `milknado.loop._*` module. The existing ones are audited debt.
- Export each new public adapter from `__init__.py`. Consumers import it from `milknado.adapters`.

## Dead code and waivers (narrows root "Build Gate")

- Do not put `# noqa: V1xx` on code that has no production caller. Delete the code and its tests.
- `loop.py` and `git.py` exceed 300 lines and carry per-file ruff waivers (issue #331). Do not grow them or add waivers.
