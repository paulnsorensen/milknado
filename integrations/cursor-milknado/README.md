# Milknado — Cursor plugin

Cursor plugin layout per [Plugins reference](https://cursor.com/docs/reference/plugins).

## Install (development)

1. Add `milknado` to your app’s `pyproject.toml` (path or PyPI) so `uv run milknado` and `uv run milknado-mcp` work from the workspace root.
2. In Cursor: install the plugin from this directory (or publish via marketplace).

## MCP

`mcp.json` starts `uv run milknado-mcp` with `MILKNADO_PROJECT_ROOT=${workspaceFolder}`. If your Cursor build does not expand `${workspaceFolder}` inside MCP `env`, remove the `env` block and rely on opening the terminal from the project root, or set `MILKNADO_PROJECT_ROOT` in your shell profile.

## Commands

Markdown commands under `commands/` wrap the Typer CLI (`plan`, `run`, `status`, `agents check`).

These files are generated from `integrations/spec/commands.yaml` via:

```bash
uv run python integrations/spec/generate_integrations.py
```
