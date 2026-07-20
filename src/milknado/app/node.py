"""Application-layer policy for native (in-session Workflow) node claiming.

The MCP ``milknado_goal_*`` / ``milknado_todo_claim`` / ``milknado_node_verify``
tools stay thin registration veneers; the model/tool resolution, session-owner
token resolution, and worktree provisioning (the policy + adapter wiring) live
here so the entry module constructs no adapters and holds no policy inline.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from milknado.domains.common import (
    FlavorProfile,
    NodeStatus,
    resolve_flavor_profile,
    resolve_worker_tools,
)
from milknado.domains.dispatch import create_isolated_worktree, now_iso
from milknado.domains.graph import CLAIM_ROLE

_logger = logging.getLogger(__name__)

# Native claims carry CLAIM_ROLE so domain completion transitions can distinguish
# them from subprocess runs that use the same run table.
_MODEL_FLAG_RE = re.compile(r"--model[= ]+(\S+)")

# Session owner-token env var (ADR-003): a coordinator claiming a GOAL passes an
# explicit `owner`, but sub-agents it dispatches inherit process env (subprocess
# semantics, MILKNADO_NODE_ID precedent) rather than being told the token
# out-of-band. Reading it as a fallback lets a sub-agent call milknado_goal_release
# (or re-claim) without the coordinator threading `owner` through every dispatch.
GOAL_OWNER_ENV_VAR = "MILKNADO_GOAL_OWNER"

# Family-specific tool name the worker uses to self-verify in loop_mode="single".
# claude allowlists use the mcp__server__ prefix; gemini uses bare MCP tool names.
_NODE_VERIFY_TOOL_BY_FAMILY: dict[str, str] = {
    "claude": "mcp__milknado__milknado_node_verify",
    "gemini": "milknado_node_verify",
}


def _resolve_model(execution_agent: str) -> str:
    """Extract the model from the resolved execution command, defaulting to sonnet.

    The native backend returns `model` as a structured field so the Workflow can
    route `agent({model: ...})` correctly instead of inferring it — closing the
    per-flavor model-routing gap.
    """
    match = _MODEL_FLAG_RE.search(execution_agent)
    return match.group(1) if match else "sonnet"


def _resolve_node_tools(cfg, profile: FlavorProfile, override) -> tuple[str, ...]:  # noqa: ANN001
    """Resolve the worker tool allowlist for a native-backend node.

    Mirrors the subprocess execution_agent precedence (resolve_flavor_profile):
    a per-flavor ``override.tools`` (built from the family default) wins, else the
    GLOBAL ``[milknado.worker.tools]`` for the family, else the family default. An
    ``override.tools`` of ``[]`` is intentional (empty allowlist) and is honored —
    only ``None`` inherits. In loop_mode="single" the worker self-verifies, so
    ``milknado_node_verify`` is appended when the family exposes it.
    """
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
    """Resolve the session owner token: explicit `owner` wins, else GOAL_OWNER_ENV_VAR."""
    explicit = owner.strip()
    if explicit:
        return explicit
    return os.environ.get(GOAL_OWNER_ENV_VAR, "").strip()


def _provision_claim_run(
    graph, root, node, run_id: str, worktree: bool | None, cfg
) -> Path | None:  # noqa: ANN001
    """Provision the claimed node's worktree (or in-place run) and stamp CLAIM_ROLE.

    On any failure the claim is released with a fenced terminal write so the
    node is not stranded RUNNING; the exception is re-raised for the caller.
    """
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
        # Stamp the native-backend marker so the done-transition gate fires
        # for this run even before its first verify (fail-closed), while
        # leaving subprocess runs — which never carry it — exempt.
        graph.deposit_run_message(run_id, CLAIM_ROLE, "", now_iso())
    except Exception:
        # Claim succeeded but worktree/run setup failed: release the claim with a
        # fenced terminal write so the node is not stranded RUNNING.
        graph.mark_terminal(node.id, run_id, NodeStatus.FAILED)
        raise
    return wt_path
