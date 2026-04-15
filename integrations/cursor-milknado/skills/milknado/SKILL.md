---
name: milknado
description: >
  Mikado-method execution with milknado: dependency graphs, parallel ralph
  loops in worktrees, and code-review-graph context. Use when planning goals,
  running autonomous loops, or inspecting graph state.
---

# Milknado

## When to use

- Decomposing a goal into ordered work (`milknado plan`)
- Running ready leaves as parallel agent loops (`milknado run`)
- Inspecting node status or adding prerequisites (`milknado status`, `milknado add-node`)

## Configuration

- `milknado.toml`: `agent_preset` (`claude` | `cursor` | `gemini` | `codex` | `custom`) and optional `agent_command` override.
- Built-in presets use stdin for planning prompts and ralphify-compatible print-mode binaries for execution.

## MCP

This plugin ships `mcp.json` registering the `milknado-mcp` stdio server (`milknado_graph_summary`, `milknado_add_node`). Install the `milknado` package in the workspace (e.g. path dep in `pyproject.toml`) so `uv run milknado-mcp` resolves.
