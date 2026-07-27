# Multi-host integrations

| Directory | Target |
|-----------|--------|
| [cursor-milknado](cursor-milknado/) | Cursor IDE plugin (`.cursor-plugin`, commands, MCP) |
| [codex](codex/) | OpenAI Codex MCP snippets and `codex mcp add` example |
| [gemini-milknado](gemini-milknado/) | Gemini CLI extension (`gemini-extension.json`, commands) |

Shared **MCP** entrypoint: `milknado-mcp` (stdio), implemented in `src/milknado/mcp_server.py`.

Runtime agent selection lives in `milknado.toml`: `agent_family`, plus optional `planning_agent` and `execution_agent` overrides. OMP planning uses `agent_family = "omp"` and requires an explicit `execution_agent` with its OMP `--tools` scope; planning is forced to read-only tools and forwards only `OPENROUTER_API_KEY`. Use `uv run milknado agents check` to verify.

Command docs for Cursor and Gemini are generated from one source of truth:

- Spec: `integrations/spec/commands.yaml`
- Templates: `integrations/spec/templates/`
- Generator: `integrations/spec/generate_integrations.py`

Regenerate with:

```bash
uv run python integrations/spec/generate_integrations.py
```
