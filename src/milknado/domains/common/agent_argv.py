"""Agent CLI argv helpers for planning (subprocess) and execution."""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence
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
        # serena layer: read, symbol edits, and content edits. serena's
        # shell-exec tool is intentionally omitted; the only Bash workers get
        # is the rtk-hooked Bash(rtk:*) below.
        "mcp__serena__get_symbols_overview",
        "mcp__serena__find_symbol",
        "mcp__serena__find_referencing_symbols",
        "mcp__serena__find_declaration",
        "mcp__serena__find_implementations",
        "mcp__serena__get_diagnostics_for_file",
        "mcp__serena__replace_symbol_body",
        "mcp__serena__insert_before_symbol",
        "mcp__serena__insert_after_symbol",
        "mcp__serena__replace_content",
        "mcp__serena__rename_symbol",
        "mcp__serena__safe_delete_symbol",
        "mcp__milknado__milknado_track_follow_up",
        "mcp__milknado__milknado_deposit_result",
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
        # serena symbol layer (raw names; shell-exec intentionally omitted).
        "get_symbols_overview",
        "find_symbol",
        "find_referencing_symbols",
        "find_declaration",
        "find_implementations",
        "get_diagnostics_for_file",
        "replace_symbol_body",
        "insert_before_symbol",
        "insert_after_symbol",
        "replace_content",
        "rename_symbol",
        "safe_delete_symbol",
        "milknado_track_follow_up",
        "milknado_deposit_result",
        "ShellTool(rtk *)",
        "read_file",
        "write_file",
        "edit_file",
    ),
}

# Worker subprocesses may only invoke a known AI-agent CLI. Match on the bare
# executable name (basename of argv[0]) so neither a prefix trick
# (`claude-evil`) nor an absolute path (`/usr/bin/claude`) slips the check.
_ALLOWED_WORKER_EXECUTABLES: frozenset[str] = frozenset(
    {"claude", "codex", "cursor-agent", "gemini"}
)


def resolve_worker_tools(
    family: str,
    tools: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve a single-list tool spec (with optional \"...\" sentinel) to the final tool list.

    - ``None`` returns the built-in family default (``WORKER_ALLOWED_TOOLS[family]``).
    - A list without ``\"...\"`` replaces the family default entirely.
    - A list with ``\"...\"`` expands it in place to the built-in default, then dedupes
      first-wins across the full list.
    - At most one ``\"...\"`` is allowed; validated by the caller at config load.
    """
    if tools is None:
        return WORKER_ALLOWED_TOOLS.get(family, ())
    base = WORKER_ALLOWED_TOOLS.get(family, ())
    expanded: list[str] = []
    for item in tools:
        if item == "...":
            expanded.extend(base)
        else:
            expanded.append(item)
    # Dedupe: first occurrence wins.
    seen: set[str] = set()
    result: list[str] = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return tuple(result)


def _default_execution_command(family: str, tools: Sequence[str]) -> str:
    """Build the family-specific default execution command from a tool list."""
    csv = ",".join(tools)
    if family == "claude":
        return f"claude --model sonnet -p --allowedTools '{csv}'"
    if family == "gemini":
        # --allowed-tools replaces --yolo for execution; workers get
        # deny-by-default, not trust-all.
        return f"gemini --model gemini-2.5-flash -p --allowed-tools '{csv}'"
    if family == "cursor":
        return "cursor-agent --model sonnet -p"
    if family == "codex":
        return "codex exec --model gpt-5.4-mini --sandbox workspace-write"
    raise KeyError(family)


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
    tools: Sequence[str] | None = None,
) -> str:
    """Return execution agent command for ralph loop workers.

    Precedence:
      1. ``execution_agent`` (explicit full CLI string) — opt out of the
         allowlist machinery entirely.
      2. Family default command, built around ``tools`` (or the family default
         allowlist when ``tools`` is None).
    """
    override = (execution_agent or "").strip()
    if override:
        return override
    effective_tools = tuple(tools) if tools is not None else WORKER_ALLOWED_TOOLS.get(family, ())
    return _default_execution_command(family, effective_tools)


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
