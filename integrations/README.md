# Multi-host integrations

| Directory | Target |
|-----------|--------|
| [cursor-milknado](cursor-milknado/) | Cursor IDE plugin (`.cursor-plugin`, commands, MCP) |
| [codex](codex/) | OpenAI Codex MCP snippets and `codex mcp add` example |
| [gemini-milknado](gemini-milknado/) | Gemini CLI extension (`gemini-extension.json`, commands) |

Shared **MCP** entrypoint: `milknado-mcp` (stdio), implemented in `src/milknado/mcp_server.py`.

Runtime agent selection lives in `milknado.toml`: `agent_preset` and optional `agent_command` override. Use `uv run milknado agents check` to verify.
