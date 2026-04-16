# Milknado + OpenAI Codex

Codex does not use Claude-style `plugin.json`. Use **MCP** and/or shell integration.

Official reference: [Codex MCP](https://developers.openai.com/codex/mcp).

## CLI registration

From the repository that depends on `milknado`:

```bash
codex mcp add milknado --env MILKNADO_PROJECT_ROOT=$PWD -- uv run milknado-mcp
```

Adjust `uv run` if you use a virtualenv or global install.

## Project-scoped `config.toml`

For a **trusted** project, add `.codex/config.toml` (see example file in this directory). Tables use `[mcp_servers.NAME]` per Codex docs.

## `milknado.toml`

Set `agent_family = "codex"` for built-in defaults, then optionally override
`planning_agent` and `execution_agent`. Run:

```bash
uv run milknado agents check --project-root .
```

## Leaf workers (ralphify)

Ralphify receives `execution_agent` from `milknado.toml`. Ensure your Codex
invocation matches ralphify’s non-interactive / stdin contract for the version
you ship.
