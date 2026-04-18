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

DEFAULT_EXECUTION_AGENT_BY_FAMILY: Final[dict[str, str]] = {
    "claude": "claude --model sonnet -p --dangerously-skip-permissions",
    "cursor": "cursor-agent --model sonnet -p",
    "gemini": "gemini --model gemini-2.5-flash -p --yolo",
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
) -> tuple[list[str], dict[str, Any]]:
    """Build argv + kwargs for one-shot planning subprocess."""
    body = context_path.read_text(encoding="utf-8")
    parts = shlex.split(planning_agent_command, posix=True)
    if not parts:
        parts = shlex.split(DEFAULT_PLANNING_AGENT_BY_FAMILY["claude"], posix=True)
    if "-" not in parts:
        parts.append("-")
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
