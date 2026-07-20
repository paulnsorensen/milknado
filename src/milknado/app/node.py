"""Application-layer node claim provisioning and tool/model resolution."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common import FlavorProfile, MilknadoConfig

from milknado.domains.common import NodeStatus, resolve_flavor_profile, resolve_worker_tools
from milknado.domains.dispatch import create_isolated_worktree, now_iso
from milknado.domains.graph import CLAIM_ROLE

_logger = logging.getLogger(__name__)

GOAL_OWNER_ENV_VAR = "MILKNADO_GOAL_OWNER"

_MODEL_FLAG_RE = re.compile(r"--model[= ]+(\S+)")
_NODE_VERIFY_TOOL_BY_FAMILY: dict[str, str] = {
    "claude": "mcp__milknado__milknado_node_verify",
    "gemini": "milknado_node_verify",
}


def _resolve_model(execution_agent: str) -> str:
    """Extract the model from the resolved execution command, defaulting to sonnet."""
    match = _MODEL_FLAG_RE.search(execution_agent)
    return match.group(1) if match else "sonnet"


def _resolve_node_tools(
    cfg: MilknadoConfig,
    profile: FlavorProfile,
    override,  # noqa: ANN001
) -> tuple[str, ...]:
    """Resolve the worker tool allowlist for a native-backend node."""
    if override is not None and override.tools is not None:
        tools = resolve_worker_tools(cfg.agent_family, list(override.tools))
    else:
        family_tools = cfg.worker_tools.get(cfg.agent_family)
        tools = resolve_worker_tools(
            cfg.agent_family, list(family_tools) if family_tools is not None else None
        )
    if profile.loop_mode == "single":
        verify_tool = _NODE_VERIFY_TOOL_BY_FAMILY.get(cfg.agent_family)
        if verify_tool is not None and verify_tool not in tools:
            tools = (*tools, verify_tool)
    return tools


def _resolve_owner(owner: str) -> str:
    """Resolve the session owner token: explicit owner wins, else GOAL_OWNER_ENV_VAR."""
    explicit = owner.strip()
    if explicit:
        return explicit
    return os.environ.get(GOAL_OWNER_ENV_VAR, "").strip()


def _provision_claim_run(
    graph,  # noqa: ANN001
    root: Path,
    node,  # noqa: ANN001
    run_id: str,
    worktree: bool | None,
    cfg,  # noqa: ANN001
) -> Path | None:
    """Provision the claimed node's worktree (or in-place run) and stamp CLAIM_ROLE."""
    from milknado.adapters import GitAdapter

    profile = resolve_flavor_profile(cfg, node.flavor)
    use_worktree = profile.worktree if worktree is None else worktree
    wt_path: Path | None = None
    try:
        if use_worktree:
            isolated = create_isolated_worktree(
                GitAdapter(root), root, node.id, node.description, cfg.worktree_pattern
            )
            wt_path = isolated.worktree_path
            graph.set_worktree(node.id, run_id, str(wt_path), isolated.worker_branch)
            graph.start_run(run_id, node.id, str(wt_path), now_iso(), None)
        else:
            graph.start_run(run_id, node.id, str(root), now_iso(), None)
        graph.deposit_run_message(run_id, CLAIM_ROLE, "", now_iso())
    except Exception:
        graph.mark_terminal(node.id, run_id, NodeStatus.FAILED)
        raise
    return wt_path
