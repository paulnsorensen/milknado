"""Pure flavor-profile resolution seam — no identity, no side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.config import Gate, MilknadoConfig
    from milknado.domains.common.types import TaskFlavor


@dataclass(frozen=True)
class FlavorProfile:
    """Resolved, ready-to-use per-flavor configuration. No identity."""

    execution_agent: str
    quality_gates: tuple[Gate, ...] | None
    brief_prepend: str | None


def resolve_flavor_profile(
    cfg: MilknadoConfig,
    flavor: TaskFlavor | None,
) -> FlavorProfile:
    """Resolve a TaskFlavor to its execution profile against cfg.

    Precedence rules (all pure; no side effects):
      - No flavor / no entry in cfg.flavors -> cfg defaults.
      - execution_agent: explicit override wins; elif flavor.tools set, derive
        from resolve_worker_tools(family, flavor.tools); else cfg.execution_agent.
      - quality_gates: flavor value if not None, else cfg.quality_gates.
      - brief_prepend: flavor value if not None, else cfg.worker_brief_prepend
        (replace, not concat).
    """
    from milknado.domains.common.agent_argv import (
        resolve_execution_agent_command,
        resolve_worker_tools,
    )

    override = cfg.flavors.get(flavor) if flavor is not None else None

    if override is None:
        return FlavorProfile(
            execution_agent=cfg.execution_agent,
            quality_gates=cfg.quality_gates,
            brief_prepend=cfg.worker_brief_prepend,
        )

    # Resolve execution_agent.
    if override.execution_agent is not None:
        execution_agent = override.execution_agent
    elif override.tools is not None:
        # Single-winner: flavor tools start from the built-in family default
        # (via resolve_worker_tools), NOT the family-level override.
        resolved_tools = resolve_worker_tools(cfg.agent_family, list(override.tools))
        execution_agent = resolve_execution_agent_command(cfg.agent_family, tools=resolved_tools)
    else:
        execution_agent = cfg.execution_agent

    quality_gates = (
        override.quality_gates if override.quality_gates is not None else cfg.quality_gates
    )
    brief_prepend = (
        override.brief_prepend if override.brief_prepend is not None else cfg.worker_brief_prepend
    )

    return FlavorProfile(
        execution_agent=execution_agent,
        quality_gates=quality_gates,
        brief_prepend=brief_prepend,
    )
