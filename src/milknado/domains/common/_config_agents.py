"""Resolve configured planning and execution agent commands."""

from __future__ import annotations

from milknado.domains.common.agent_argv import (
    resolve_execution_agent_command,
    resolve_planning_agent_command,
    resolve_worker_tools,
)


def resolve_agents(
    family: str,
    planning_command: str | None,
    execution_command: str | None,
    worker_tools: dict[str, tuple[str, ...]],
) -> tuple[str, str]:
    planning_agent = resolve_planning_agent_command(family, planning_agent=planning_command)
    family_tools = worker_tools.get(family)
    resolved_tools = (
        resolve_worker_tools(family, list(family_tools)) if family_tools is not None else None
    )
    execution_agent = resolve_execution_agent_command(
        family, execution_agent=execution_command, tools=resolved_tools
    )
    return planning_agent, execution_agent
