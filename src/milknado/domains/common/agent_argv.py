"""Agent CLI argv helpers for planning (subprocess) and execution."""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PLANNING_ALLOWED_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "claude": ("Read", "Glob", "Grep"),
    "gemini": ("read_file", "glob", "search_file_content"),
}

DEFAULT_PLANNING_AGENT_BY_FAMILY: Final[dict[str, str]] = {
    "claude": "claude --model opus -p",
    "cursor": "cursor-agent --model opus -p",
    "gemini": "gemini --model gemini-3.1-pro-preview -p",
    "codex": "codex exec --model gpt-5.4 --sandbox read-only",
}

# Per-family tool allowlists for deny-by-default execution workers. Families
# without CLI-level tool restriction receive no host-shell capability here.
WORKER_ALLOWED_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "claude": (
        "mcp__tilth__*",
        # serena shell execution is intentionally omitted.
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
        "read_file",
        "write_file",
        "edit_file",
    ),
}

# Worker subprocesses may only invoke a known AI-agent CLI. Match on the bare
# executable name (basename of argv[0]) so neither a prefix trick
# (`claude-evil`) nor an absolute path (`/usr/bin/claude`) slips the check.
ALLOWED_WORKER_EXECUTABLES: frozenset[str] = frozenset(
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


def _replace_option(parts: list[str], option: str, value: str) -> None:
    while option in parts:
        index = parts.index(option)
        del parts[index : index + 2]
    parts.extend([option, value])


def _sandbox_planning_argv(parts: list[str]) -> list[str]:
    executable = Path(parts[0]).name
    if executable not in ALLOWED_WORKER_EXECUTABLES:
        raise ValueError(f"unsupported planning agent executable: {executable!r}")
    unsafe_flags = {
        "--dangerously-skip-permissions",
        "--dangerously-bypass-approvals-and-sandbox",
        "--full-auto",
        "--yolo",
    }
    parts = [part for part in parts if part not in unsafe_flags]
    if executable == "claude":
        _replace_option(parts, "--permission-mode", "plan")
        _replace_option(parts, "--allowedTools", ",".join(PLANNING_ALLOWED_TOOLS["claude"]))
    elif executable == "gemini":
        _replace_option(parts, "--allowed-tools", ",".join(PLANNING_ALLOWED_TOOLS["gemini"]))
    elif executable == "codex":
        _replace_option(parts, "--sandbox", "read-only")
    else:
        raise ValueError("cursor-agent cannot enforce read-only planning capabilities")
    return parts


def build_planning_subprocess(
    context_path: Path,
    planning_agent_command: str,
    *,
    allow_external_mcp: bool = False,
    project_root: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build a read-only argv and minimal environment for one-shot planning."""
    body = context_path.read_text(encoding="utf-8")
    parts = shlex.split(planning_agent_command, posix=True)
    if not parts:
        parts = shlex.split(DEFAULT_PLANNING_AGENT_BY_FAMILY["claude"], posix=True)
    parts = _sandbox_planning_argv(parts)
    parts = [part for part in parts if part != "-"]
    mcp_config = project_root / ".mcp.json" if project_root else None
    if allow_external_mcp and mcp_config and mcp_config.exists():
        parts.extend(["--mcp-config", str(mcp_config)])
    parts.append("-")
    extra: dict[str, Any] = {
        "env": build_minimal_mcp_env(),
        "input": body,
        "text": True,
    }
    return parts, extra


def build_minimal_mcp_env() -> dict[str, str]:
    """Return only process essentials; never forward coordinator credentials."""
    allowed = ("HOME", "LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TERM", "TMPDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


@dataclass(frozen=True)
class NodeAgentSession:
    """Node-scoped agent session (NOT a run field) pinned across review/redispatch rounds."""

    node_id: int
    family: str
    session_id: str
    worktree_path: str
    created_at: str


def capture_session_id(family: str, first_turn_json: str) -> str:
    """Extract the session id from a family's first-turn JSON stdout.

    claude: top-level ``session_id``. codex: top-level ``session_id`` (or the
    ``session_configured`` event's ``session_id``). gemini: plain
    ``--output-format json`` output does not carry a session id (see spec open
    question) — fail fast rather than guess.
    """
    try:
        payload = json.loads(first_turn_json)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{family}: could not parse first-turn JSON output for session id") from exc
    session_id = None
    if isinstance(payload, dict):
        session_id = payload.get("session_id")
        if session_id is None and family == "codex":
            nested = payload.get("msg")
            if isinstance(nested, dict):
                session_id = nested.get("session_id")
    if not session_id or not isinstance(session_id, str):
        raise ValueError(
            f"{family}: first-turn JSON output carried no session_id — cannot resume "
            "(gemini plain JSON output does not expose one; see spec open question)"
        )
    return session_id


def build_resume_argv(
    family: str,
    session_id: str,
    prompt: str,
    model_flags: Sequence[str],
) -> list[str]:
    """Build the resume argv for `family`, always re-passing model/effort flags.

    No family CLI restores model/effort flags on resume, so callers must supply
    them fresh every round via ``model_flags``.
    """
    if family == "claude":
        return ["claude", "-p", "--resume", session_id, *model_flags, prompt]
    if family == "codex":
        return ["codex", "exec", "resume", session_id, "--json", *model_flags, prompt]
    if family == "gemini":
        return ["gemini", "--resume", session_id, *model_flags, "--prompt", prompt]
    raise ValueError(f"unsupported resume family: {family!r}")
