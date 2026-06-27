"""FlavorIdentity resolution and capability classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.types import TaskFlavor

if TYPE_CHECKING:
    from milknado.domains.common.config import FlavorOverride, MilknadoConfig

# Serena write-symbol operation names (matched with and without mcp__serena__ prefix).
_SERENA_WRITE_OPS: frozenset[str] = frozenset(
    {
        "replace_symbol_body",
        "insert_before_symbol",
        "insert_after_symbol",
        "replace_content",
        "rename_symbol",
        "safe_delete_symbol",
    }
)

# Direct write/edit tool names used by various harnesses.
_DIRECT_WRITE_TOOLS: frozenset[str] = frozenset(
    {"Edit", "Write", "tilth_edit", "write_file", "edit_file"}
)

# Per-family --model fallback when no --model flag is present in execution_agent.
_FAMILY_DEFAULT_MODEL: dict[str, str] = {
    "claude": "sonnet",
    "gemini": "gemini-2.5-flash",
    "codex": "gpt-5.4-mini",
    "cursor": "sonnet",
}


def _is_write_capable(tools: tuple[str, ...]) -> bool:
    """True if any tool in the list grants write or exec capability."""
    for tool in tools:
        if tool in _DIRECT_WRITE_TOOLS:
            return True
        if tool.startswith("Bash(") or tool.startswith("ShellTool("):
            return True
        # Match both bare op name and mcp__serena__<op> form.
        bare = tool.removeprefix("mcp__serena__")
        if bare in _SERENA_WRITE_OPS:
            return True
    return False


def _has_bash(tools: tuple[str, ...]) -> bool:
    """True if any Bash(...) or ShellTool(...) pattern is in the tool list."""
    return any(t.startswith("Bash(") or t.startswith("ShellTool(") for t in tools)


def _parse_model(exec_cmd: str, family: str) -> str:
    """Extract --model <x> or --model=<x> from execution command; fall back to family default."""
    m = re.search(r"--model[=\s]+(\S+)", exec_cmd)
    if m:
        return m.group(1)
    return _FAMILY_DEFAULT_MODEL.get(family, "sonnet")


def _resolve_exec_cmd(
    cfg: MilknadoConfig, override: FlavorOverride | None, tools: tuple[str, ...]
) -> str:
    """Resolve the execution command string for model extraction."""
    from milknado.domains.common.agent_argv import resolve_execution_agent_command

    if override is not None and override.execution_agent:
        return override.execution_agent
    if override is not None and override.tools is not None:
        return resolve_execution_agent_command(cfg.agent_family, tools=tools)
    return cfg.execution_agent


@dataclass(frozen=True)
class FlavorIdentity:
    """Resolved, ready-to-use per-flavor agent identity for def generation."""

    flavor: TaskFlavor
    tools: tuple[str, ...]
    model: str
    max_turns: int
    is_write_capable: bool
    user_controlled_tools: bool  # True when execution_agent is full-custom with no tools override


@dataclass(frozen=True)
class RenderedDef:
    """A generated agent-definition file for one harness and flavor."""

    harness: str  # "claude" | "opencode" | "codex"
    path: Path
    content: str


def resolve_flavor_identity(cfg: MilknadoConfig, flavor: TaskFlavor) -> FlavorIdentity:
    """Resolve a TaskFlavor to its agent-def identity against cfg.

    Precedence mirrors resolve_flavor_profile:
      - tools: override.tools if set, else family default.
      - model: parsed from override.execution_agent → derived cmd → cfg.execution_agent.
      - max_turns: override.max_turns if set, else cfg.max_turns.
    """
    from milknado.domains.common.agent_argv import resolve_worker_tools

    override = cfg.flavors.get(flavor)
    if override is not None and override.tools is not None:
        tools_input = list(override.tools)
    else:
        tools_input = None
    tools = resolve_worker_tools(cfg.agent_family, tools_input)

    # Full-custom: execution_agent set by user with no tools override → user controls tools.
    user_controlled_tools = (
        override is not None and bool(override.execution_agent) and override.tools is None
    )

    exec_cmd = _resolve_exec_cmd(cfg, override, tools)
    model = _parse_model(exec_cmd, cfg.agent_family)
    max_turns = (
        override.max_turns
        if override is not None and override.max_turns is not None
        else cfg.max_turns
    )

    return FlavorIdentity(
        flavor=flavor,
        tools=tools,
        model=model,
        max_turns=max_turns,
        is_write_capable=_is_write_capable(tools),
        user_controlled_tools=user_controlled_tools,
    )
