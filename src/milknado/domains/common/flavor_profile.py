"""Pure flavor-profile resolution seam — no identity, no side effects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from milknado.domains.common.flavor_codec import Gate

if TYPE_CHECKING:
    from milknado.domains.common.config import MilknadoConfig


class FlavorProfile(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved, ready-to-use per-flavor configuration. No identity.

    ``worker_agent_type`` / ``loop_mode`` / ``max_iterations`` / ``max_turns``
    drive the native Workflow backend; the subprocess dispatcher ignores them.
    """

    execution_agent: str
    quality_gates: tuple[Gate, ...] | None
    brief_prepend: str | None
    worker_agent_type: str
    loop_mode: str
    max_iterations: int
    max_turns: int
    worktree: bool = True
    session_mode: str = "fresh"
    review: bool = False
    review_agent: str | None = None
    review_max_rounds: int = 2
    on_reject: str = "block"


def resolve_flavor_profile(
    cfg: MilknadoConfig,
    flavor: str | None,
) -> FlavorProfile:
    """Resolve a flavor name to its execution profile against cfg.

    Precedence rules (all pure; no side effects):
      - No flavor / no entry in cfg.flavors -> cfg defaults.
      - execution_agent: explicit override wins; elif flavor.tools set, derive
        from resolve_worker_tools(family, flavor.tools); else cfg.execution_agent.
      - quality_gates: flavor value if not None, else cfg.quality_gates.
      - brief_prepend: flavor value if not None, else cfg.worker_brief_prepend
        (replace, not concat).
      - worker_agent_type / loop_mode / max_iterations / max_turns: flavor value
        if not None, else the cfg global default (native Workflow backend).
      - worktree: flavor value if not None, else cfg.worktree.
    """
    from milknado.domains.common.agent_argv import (
        resolve_execution_agent_command,
        resolve_worker_tools,
    )

    override = cfg.flavors.get(flavor) if flavor is not None else None
    config_worktree = getattr(cfg, "worktree", True)
    default_review = flavor in ("implement", "spec")

    if override is None:
        _validate_session_mode_family(cfg.agent_family, "fresh", None)
        return FlavorProfile(
            execution_agent=cfg.execution_agent,
            quality_gates=cfg.quality_gates,
            brief_prepend=cfg.worker_brief_prepend,
            worker_agent_type=cfg.worker_agent_type,
            loop_mode=cfg.loop_mode,
            max_iterations=cfg.max_iterations,
            max_turns=cfg.max_turns,
            worktree=config_worktree,
            session_mode="fresh",
            review=default_review,
            review_agent=None,
            review_max_rounds=2,
            on_reject="block",
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
    agent_type = override.agent_type if override.agent_type is not None else cfg.worker_agent_type
    loop_mode = override.loop_mode if override.loop_mode is not None else cfg.loop_mode
    max_iterations = (
        override.max_iterations if override.max_iterations is not None else cfg.max_iterations
    )
    max_turns = override.max_turns if override.max_turns is not None else cfg.max_turns
    worktree = override.worktree if override.worktree is not None else config_worktree
    session_mode = override.session_mode
    review = override.review if override.review is not None else default_review
    review_agent = override.review_agent
    review_max_rounds = override.review_max_rounds
    on_reject = override.on_reject
    _validate_session_mode_family(cfg.agent_family, session_mode, execution_agent)

    return FlavorProfile(
        execution_agent=execution_agent,
        quality_gates=quality_gates,
        brief_prepend=brief_prepend,
        worker_agent_type=agent_type,
        loop_mode=loop_mode,
        max_iterations=max_iterations,
        max_turns=max_turns,
        worktree=worktree,
        session_mode=session_mode,
        review=review,
        review_agent=review_agent,
        review_max_rounds=review_max_rounds,
        on_reject=on_reject,
    )


def _validate_session_mode_family(
    agent_family: str,
    session_mode: str,
    execution_agent: str | None,
) -> None:
    """Fail config validation when session_mode='resume' targets cursor-agent.

    Headless resume mechanics for cursor-agent are unverified (F001); resolve
    the effective family from an explicit execution_agent, else agent_family.
    """
    if session_mode != "resume":
        return
    import shlex
    from pathlib import Path

    from milknado.domains.common.agent_argv import ALLOWED_WORKER_EXECUTABLES

    family = agent_family
    if execution_agent:
        argv = shlex.split(execution_agent)
        if argv:
            name = Path(argv[0]).name
            if name in ALLOWED_WORKER_EXECUTABLES:
                family = name
    if family == "cursor-agent" or family == "cursor":
        raise ValueError(
            "session_mode 'resume' is not supported for the cursor-agent family "
            + "(headless resume mechanics are unverified — see "
            + "adversarial-review-loops-F001); use session_mode = 'fresh' or a "
            + "different agent"
        )
