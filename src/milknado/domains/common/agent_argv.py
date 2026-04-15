"""Agent CLI argv helpers for planning (subprocess) and execution (ralphify string)."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Final

AGENT_PRESETS: Final[tuple[str, ...]] = (
    "custom",
    "claude",
    "cursor",
    "gemini",
    "codex",
)

# Strings passed to ralphify RunConfig.agent (full shell command, one string).
EXECUTION_AGENT_BY_PRESET: Final[dict[str, str]] = {
    "claude": "claude -p --dangerously-skip-permissions",
    "cursor": "cursor-agent -p",
    "gemini": "gemini -p --yolo",
    # Codex flags evolve; users can override via agent_command when preset=codex
    # or set agent_preset=codex without agent_command for this default.
    "codex": "codex exec --sandbox workspace-write",
}

# Planning: built-in presets send the planning markdown on stdin (text mode).
_PLANNING_ARGV_BY_PRESET: Final[dict[str, list[str]]] = {
    "claude": ["claude", "-p", "--dangerously-skip-permissions", "-"],
    "cursor": ["cursor-agent", "-p", "-"],
    "gemini": ["gemini", "-p", "--yolo", "-"],
    "codex": ["codex", "exec", "--sandbox", "workspace-write", "-"],
}


def normalize_preset(value: str | None) -> str:
    p = (value or "custom").strip().lower()
    if p not in AGENT_PRESETS:
        return "custom"
    return p


def resolve_execution_agent_command(
    preset: str,
    *,
    agent_command: str,
    agent_command_key_present: bool,
) -> str:
    """Return the shell command string used by ralphify for loop workers."""
    p = normalize_preset(preset)
    if p == "custom":
        return agent_command.strip() if agent_command.strip() else "claude"
    if agent_command_key_present and agent_command.strip():
        return agent_command.strip()
    return EXECUTION_AGENT_BY_PRESET[p]


def build_planning_subprocess(
    context_path: Path,
    preset: str,
    agent_command: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build argv and extra kwargs for subprocess.run when launching planning.

    - Built-in presets: prompt is read from ``context_path`` and sent on stdin
      (``text=True``, ``input=...``).
    - ``custom``: ``shlex.split(agent_command)`` plus the full file body as the
      last argv element (legacy behavior).
    """
    p = normalize_preset(preset)
    body = context_path.read_text(encoding="utf-8")
    if p == "custom":
        parts = shlex.split(agent_command, posix=True)
        if not parts:
            parts = ["claude"]
        argv = parts + [body]
        return argv, {}

    argv = list(_PLANNING_ARGV_BY_PRESET[p])
    return argv, {"input": body, "text": True}
