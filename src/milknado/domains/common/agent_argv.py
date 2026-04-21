"""Agent CLI argv helpers for planning (subprocess) and execution."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Final

DEFAULT_PLANNING_AGENT_BY_FAMILY: Final[dict[str, str]] = {
    "claude": "claude --model opus -p --dangerously-skip-permissions",
    "cursor": "cursor-agent --model opus -p",
    "gemini": "gemini --model gemini-3.1-pro-preview -p --yolo",
    "codex": "codex exec --model gpt-5.4 --sandbox workspace-write",
}

# Per-family tool allowlists for deny-by-default execution workers.
# Families absent from this dict have no CLI-level tool restriction:
#   cursor-agent: --allowedTools not implemented for headless (config-file only, broken Apr 2026)
#   codex: --sandbox workspace-write scopes files, not tools; no allowlist flag available
WORKER_ALLOWED_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "claude": (
        "mcp__tilth__*",
        "Bash(rtk:*)",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "MultiEdit",
    ),
    # Gemini uses raw MCP tool names (no mcp__server__ prefix) + ShellTool(pattern) for shell.
    "gemini": (
        "tilth_search",
        "tilth_read",
        "tilth_files",
        "tilth_deps",
        "tilth_diff",
        "tilth_edit",
        "ShellTool(rtk *)",
        "read_file",
        "write_file",
        "edit_file",
    ),
}

_CLAUDE_TOOLS_CSV = ",".join(WORKER_ALLOWED_TOOLS["claude"])
_GEMINI_TOOLS_CSV = ",".join(WORKER_ALLOWED_TOOLS["gemini"])

DEFAULT_EXECUTION_AGENT_BY_FAMILY: Final[dict[str, str]] = {
    "claude": f"claude --model sonnet -p --allowedTools '{_CLAUDE_TOOLS_CSV}'",
    "cursor": "cursor-agent --model sonnet -p",
    # --allowed-tools replaces --yolo for execution; workers get deny-by-default, not trust-all.
    "gemini": f"gemini --model gemini-2.5-flash -p --allowed-tools '{_GEMINI_TOOLS_CSV}'",
    "codex": "codex exec --model gpt-5.4-mini --sandbox workspace-write",
}


def resolve_planning_agent_command(
    family: str,
    *,
    planning_agent: str | None = None,
) -> str:
    """Return planning agent command for one-shot planning subprocess."""
    override = (planning_agent or "").strip()
    if override:
        return override
    return DEFAULT_PLANNING_AGENT_BY_FAMILY[family]


def resolve_execution_agent_command(
    family: str,
    *,
    execution_agent: str | None = None,
) -> str:
    """Return execution agent command for ralph loop workers."""
    override = (execution_agent or "").strip()
    if override:
        return override
    return DEFAULT_EXECUTION_AGENT_BY_FAMILY[family]


def build_planning_subprocess(
    context_path: Path,
    planning_agent_command: str,
    *,
    allow_external_mcp: bool = False,
    project_root: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build argv + kwargs for one-shot planning subprocess."""
    body = context_path.read_text(encoding="utf-8")
    parts = shlex.split(planning_agent_command, posix=True)
    if not parts:
        parts = shlex.split(DEFAULT_PLANNING_AGENT_BY_FAMILY["claude"], posix=True)
    if "-" not in parts:
        parts.append("-")
    mcp_config = project_root / ".mcp.json" if project_root else None
    if mcp_config and mcp_config.exists():
        parts.extend(["--mcp-config", str(mcp_config)])
    extra: dict[str, Any] = {"input": body, "text": True}
    if not allow_external_mcp:
        extra["env"] = build_minimal_mcp_env()
    return parts, extra


def build_minimal_mcp_env() -> dict[str, str]:
    """Return subprocess env with external MCP injection stripped by default."""
    env = dict(os.environ)
    for key in list(env.keys()):
        upper = key.upper()
        if "MCP" not in upper:
            continue
        if upper.startswith("MILKNADO_"):
            continue
        # Keep CRG env controls available to planning.
        if upper.startswith("CRG_"):
            continue
        del env[key]
    return env
