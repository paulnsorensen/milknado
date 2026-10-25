# CLI delivery

These rules add to the root `AGENTS.md` for the `milknado` CLI modules.
Source: `../../../docs/adr/sliced-bread-crust-003.md`.

## Shape: parse, call app, format

- A command parses Typer options, calls a public `milknado.app` function, and renders the result.
- Do not construct adapters, open the database, or run dispatch policy in a command. Add or extend an `app` function.
- Do not import a `_private` name from `milknado.app`.
- Open the project with `open_project_for_cli` in `_helpers.py`. Close the graph in a `finally` block.
- Rich output, prompts, and exit codes belong here. Map an expected `ValueError` or app error to a red message and `typer.Exit(code=1)`, as sibling commands do. Do not show a traceback.
- Put a new command group in its own module and mount it in `__init__.py`. `__init__.py` already exceeds 300 lines.

## No aliases (narrows root "No Migration Code")

- Do not add alias names for renamed helpers in `_helpers.py` or any other module. Change the callers.
- Do not list a name in `__all__` only to hide it from the dead-code gate.
