# MCP delivery

These rules add to the root `AGENTS.md` for the `milknado-mcp` tool modules.
Source: `../../../docs/adr/sliced-bread-crust-003.md`.

## Shape: parse, call app, return

- A tool parses its arguments, calls a public `milknado.app` function, and returns a structured value.
- Do not construct adapters or run dispatch policy in a tool module. Add or extend an `app` function.
- Do not import a `_private` name from `milknado.app`. Do not add a pyright suppression to make one importable.
- Do not add application composition here. `_loop_node_runner.py` is audited debt in this package.

## Boundary

- Normalize enum-like inputs with the parsers in `_core.py` (`_parse_kind`, `_parse_todo_status`, `_parse_flavor`). Keep them there. Return their `ValueError` text as a structured tool error.
- The server speaks stdio. A tool must not write to stdout or stderr.
- Register a tool with `@mcp.tool()` on the shared instance from `_core.py`. Import each new tool module in `main()` in `server.py`.
- Tool return types such as `RunDict` use `typing_extensions.TypedDict`. FastMCP needs it on Python 3.11.
- Test tools through the registered transport in `../../../tests/meta/mcp_contracts.py`.
