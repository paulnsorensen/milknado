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
    "omp": ("read", "grep", "glob", "lsp"),
}

DEFAULT_PLANNING_AGENT_BY_FAMILY: Final[dict[str, str]] = {
    "claude": "claude --model opus -p",
    "cursor": "cursor-agent --model opus -p",
    "gemini": "gemini --model gemini-3.1-pro-preview -p",
    "codex": "codex exec --model gpt-5.4 --sandbox read-only",
    "omp": "omp -p",
}

# Per-family tool allowlists for deny-by-default execution workers. Families
# without CLI-level tool restriction receive no host-shell capability here.
WORKER_ALLOWED_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "claude": (
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
        "mcp__milknado__milknado_deposit_review",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "MultiEdit",
    ),
    # Gemini uses raw MCP tool names (no mcp__server__ prefix) + ShellTool(pattern) for shell.
    "gemini": (
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
        "milknado_deposit_review",
        "read_file",
        "write_file",
        "edit_file",
    ),
}

# Worker subprocesses may only invoke a known AI-agent CLI. Matching the exact
# bare token rejects both prefix tricks (`claude-evil`) and absolute/relative
# paths (`/usr/bin/claude`, `./claude`).
ALLOWED_WORKER_EXECUTABLES: frozenset[str] = frozenset(
    {"claude", "codex", "copilot", "crush", "cursor-agent", "gemini", "omp", "opencode"}
)


def validate_worker_argv(argv: Sequence[str]) -> None:
    """Reject worker argv unless its executable is an allowed bare token.

    Path-bearing tokens are rejected instead of being normalized to a basename:
    the exact token that passes validation is the token the subprocess executes.
    """
    executable = argv[0] if argv else ""
    if (
        not executable
        or executable not in ALLOWED_WORKER_EXECUTABLES
        or "/" in executable
        or "\\" in executable
    ):
        shown = Path(executable).name if executable else ""
        raise ValueError(
            "worker_cmd must start with one of "
            f"{sorted(ALLOWED_WORKER_EXECUTABLES)!r}; got {shown!r}"
        )


# Agents in this set receive the node brief as a trailing positional argument
# rather than on stdin. Kept alongside ALLOWED_WORKER_EXECUTABLES so the brief-
# delivery trait lives with the executable identities it keys off, not as a
# magic string in the dispatch runner.
POSITIONAL_BRIEF_EXECUTABLES: frozenset[str] = frozenset({"omp"})


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
    """Return the execution agent command for ralph loop workers.

    Explicit commands remain configurable, but every consuming subprocess
    boundary validates their executable with :func:`validate_worker_argv`.
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


def _remove_omp_planning_options(parts: list[str]) -> list[str]:
    value_options = {
        "--approval-mode",
        "--config",
        "--extension",
        "--hook",
        "--mode",
        "--plugin-dir",
        "--tools",
        "-e",
    }
    flag_options = {
        "--auto-approve",
        "--no-extensions",
        "--no-lsp",
        "--no-pty",
        "--no-rules",
        "--no-session",
        "--no-skills",
        "--no-tools",
        "--plan-yolo",
        "--print",
        "-p",
    }
    kept = [parts[0]]
    index = 1
    while index < len(parts):
        option = parts[index]
        name = option.partition("=")[0]
        if name in flag_options:
            index += 1
        elif name in value_options:
            index += 1 if "=" in option else 2
        else:
            kept.append(option)
            index += 1
    return kept


def _sandbox_planning_argv(parts: list[str]) -> list[str]:
    executable = parts[0] if parts else ""
    # Bare non-allowlisted tokens get a planning-specific message before the
    # shared validator runs; the distinct wording is intentional and pinned by
    # test_custom_agent_command. validate_worker_argv below is the authority for
    # path-bearing/prefix-spoofed tokens.
    if (
        executable not in ALLOWED_WORKER_EXECUTABLES
        and "/" not in executable
        and "\\" not in executable
    ):
        raise ValueError(f"unsupported planning agent executable: {executable!r}")
    validate_worker_argv(parts)
    unsafe_flags = {
        "--auto-approve",
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
    elif executable == "omp":
        parts = _remove_omp_planning_options(parts)
        parts.extend(
            (
                "-p",
                "--no-session",
                "--no-pty",
                "--no-extensions",
                "--no-skills",
                "--no-rules",
                "--mode",
                "text",
                "--tools",
                ",".join(PLANNING_ALLOWED_TOOLS["omp"]),
            )
        )
    else:
        raise ValueError(f"{executable} cannot enforce read-only planning capabilities")
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
        "env": build_minimal_mcp_env(executable=parts[0]),
        "input": body,
        "text": True,
    }
    return parts, extra


def build_minimal_mcp_env(*, executable: str | None = None) -> dict[str, str]:
    """Return process essentials plus the OMP OpenRouter credential when needed."""
    allowed = ["HOME", "LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TERM", "TMPDIR"]
    if executable == "omp":
        allowed.append("OPENROUTER_API_KEY")
    return {key: os.environ[key] for key in allowed if key in os.environ}


@dataclass(frozen=True)
class NodeAgentSession:
    """Node-scoped agent session retained across review/redispatch rounds."""

    node_id: int
    family: str
    session_id: str
    worktree_path: str
    created_at: str


def capture_session_id(family: str, first_turn_json: str) -> str:
    """Extract a resumable session id from one adapter's first-turn JSON."""
    try:
        payload: Any = json.loads(first_turn_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{family}: could not parse first-turn JSON output for session id"
        ) from exc
    candidates: list[Any] = []
    keys = ("session_id", "sessionId") if family == "omp" else ("session_id",)
    if isinstance(payload, dict):
        candidates.extend(payload.get(key) for key in keys)
        if family == "codex":
            for key in ("msg", "event", "data"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested.get("session_id"))
    elif isinstance(payload, list):
        candidates.extend(
            item.get(key) for item in payload if isinstance(item, dict) for key in keys
        )
    session_id = next(
        (value for value in candidates if isinstance(value, str) and value.strip()),
        None,
    )
    if session_id is None:
        raise ValueError(f"{family}: first-turn JSON output carried no session_id; cannot resume")
    return session_id.strip()


def build_resume_command(command: str, family: str, session_id: str) -> str:
    """Add a session resume selector without dropping model/effort flags."""
    argv = shlex.split(command)
    if not argv:
        raise ValueError("agent command must not be empty")
    if family in {"claude", "gemini", "omp"}:
        argv.extend(["--resume", session_id])
    elif family == "codex":
        try:
            exec_index = argv.index("exec")
        except ValueError:
            exec_index = len(argv) - 1
        argv[exec_index + 1 : exec_index + 1] = ["resume", session_id]
    else:
        raise ValueError(f"unsupported resume family: {family!r}")
    return shlex.join(argv)
