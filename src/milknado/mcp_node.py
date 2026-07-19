"""Native Workflow ("ultracode") backend MCP tools — coordinator-only.

These tools back the in-session Claude Code dynamic-Workflow driver. They run
ALONGSIDE the subprocess dispatcher in `mcp_run.py` and never spawn a process:
the harness-side Workflow is the driver, milknado is the graph/state backend.

- `milknado_goal_claim` / `milknado_goal_release`: claim/release a GOAL node's
  subtree as a session's task list (ADR-003), wrapping the existing
  goal_claims persistence with an env-inherited owner token and dead-pid reclaim.
- `milknado_todo_claim`: atomically claim a node RUNNING, provision its
  worktree per the resolved flavor policy (or an explicit override, ADR-005),
  write the `runs` row, and return the structured config the Workflow needs to
  dispatch one `agent()` per ready node. Spawns nothing.
- `milknado_node_verify`: run the node's resolved quality_gates in its
  worktree_path or the project_root (ADR-005/006) and return the existing
  `CompletionVerdict` shape. The verdict is persisted as a `run_messages` row
  (role="verify") so the server-side completion gate in `mcp_todo_mutate` can
  reject a "done" transition that was never verified.

Coordinator-only: none of these tools belong in WORKER_ALLOWED_TOOLS.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from milknado._mcp_core import (
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.adapters import GitAdapter
from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.agent_argv import resolve_worker_tools
from milknado.domains.common.flavor_profile import FlavorProfile, resolve_flavor_profile
from milknado.domains.dispatch import RUN_ID_RE, create_isolated_worktree, make_run_id, now_iso
from milknado.domains.execution.completion import build_completion_verifier
from milknado.domains.graph.status_flow import CLAIM_ROLE, VERIFY_ROLE

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


def _resolve_model(execution_agent: str) -> str:
    """Extract the model from the resolved execution command, defaulting to sonnet.

    The native backend returns `model` as a structured field so the Workflow can
    route `agent({model: ...})` correctly instead of inferring it — closing the
    per-flavor model-routing gap.
    """
    match = _MODEL_FLAG_RE.search(execution_agent)
    return match.group(1) if match else "sonnet"


# Family-specific tool name the worker uses to self-verify in loop_mode="single".
# claude allowlists use the mcp__server__ prefix; gemini uses bare MCP tool names.
_NODE_VERIFY_TOOL_BY_FAMILY: dict[str, str] = {
    "claude": "mcp__milknado__milknado_node_verify",
    "gemini": "milknado_node_verify",
}


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


@mcp.tool()
def milknado_goal_claim(goal_id: int, owner: str = "", project_root: str = "") -> dict:
    """Claim a GOAL node's subtree as this session's task list (ADR-003).

    Acquires or reclaims the owner token and coordinator PID in one transaction.
    `owner` falls back to GOAL_OWNER_ENV_VAR so inherited coordinator sessions
    can act under the same token.
    """
    owner = _resolve_owner(owner)
    if not owner:
        raise ValueError(f"owner is required (or set {GOAL_OWNER_ENV_VAR})")

    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        won = graph.claim_or_reclaim_goal(goal_id, owner, os.getpid(), now=now_iso())
        if not won:
            node = graph.get_node(goal_id)
            current_owner = node.goal_run_id if node is not None else None
            raise ValueError(f"goal {goal_id} is already claimed by owner {current_owner!r}")
        _logger.info("milknado_goal_claim: goal=%d owner=%s", goal_id, owner)
        return {"goal_id": goal_id, "owner": owner}
    finally:
        graph.close()


@mcp.tool()
def milknado_goal_release(goal_id: int, owner: str = "", project_root: str = "") -> dict:
    """Release a GOAL claim held by `owner` (ADR-003); wraps release_goal_row.

    `owner` falls back to GOAL_OWNER_ENV_VAR when omitted. Fenced on the owning
    token: releasing with a non-owning or already-released token is a no-op.
    """
    owner = _resolve_owner(owner)
    if not owner:
        raise ValueError(f"owner is required (or set {GOAL_OWNER_ENV_VAR})")

    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        released = graph.release_goal(goal_id, owner)
        _logger.info(
            "milknado_goal_release: goal=%d owner=%s released=%s", goal_id, owner, released
        )
        return {"goal_id": goal_id, "released": released}
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_claim(
    node_id: int, worktree: bool | None = None, project_root: str = ""
) -> dict:
    """Claim a task node for native (in-session Workflow) execution; spawn nothing.

    Atomically CAS-claims the node PENDING/FAILED/BLOCKED -> RUNNING, writes a
    status='running' runs row, and honors goal_claims subtree fencing. Worktree
    provisioning is per-node policy (ADR-005): `worktree=None` (default) uses
    the resolved flavor profile's `worktree` knob; `worktree=True`/`False`
    overrides it for this call. When a worktree is provisioned, its durable
    path + branch are recorded on the node (the carry-forward state); an
    in-place claim (worktree=False) leaves worktree_path unset and runs the
    node in the current checkout. Returns the full structured config the
    Workflow node-runner needs to dispatch one agent() per node: brief,
    flavor, model, tools, worktree_path, agent_type, loop_mode,
    max_iterations, max_turns. NO process is spawned — the harness-side
    Workflow is the driver.
    """
    from milknado.domains.dispatch import render_brief

    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.kind != NodeKind.TASK:
            raise ValueError(
                f"node {node_id} has kind={node.kind.value}; only task nodes can be claimed"
            )

        run_id = make_run_id(node_id)

        profile = resolve_flavor_profile(cfg, node.flavor)
        brief = render_brief(graph, node_id, prepend=profile.brief_prepend)

        graph.claim_node_for_dispatch(node_id, run_id, now=now_iso())

        wt_path = _provision_claim_run(graph, root, node, run_id, worktree, cfg)

        override = cfg.flavors.get(node.flavor) if node.flavor is not None else None
        tools = _resolve_node_tools(cfg, profile, override)
        _logger.info("milknado_todo_claim: node=%d run_id=%s (native, no spawn)", node_id, run_id)
        return {
            "run_id": run_id,
            "node_id": node_id,
            "brief": brief,
            "flavor": node.flavor,
            "model": _resolve_model(profile.execution_agent),
            "tools": list(tools),
            "worktree_path": str(wt_path) if wt_path is not None else None,
            "agent_type": profile.worker_agent_type,
            "loop_mode": profile.loop_mode,
            "max_iterations": profile.max_iterations,
            "max_turns": profile.max_turns,
        }
    finally:
        graph.close()


def _provision_claim_run(
    graph, root, node, run_id: str, worktree: bool | None, cfg
) -> Path | None:  # noqa: ANN001
    """Provision the claimed node's worktree (or in-place run) and stamp CLAIM_ROLE.

    On any failure the claim is released with a fenced terminal write so the
    node is not stranded RUNNING; the exception is re-raised for the caller.
    """
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


@mcp.tool()
def milknado_node_verify(run_id: str, project_root: str = "") -> dict:
    """Run the node's resolved quality_gates in worktree_path or project_root; return the verdict.

    Exposes the tier-2 completion verifier as a callable: runs every resolved
    quality gate in the node's worktree (or, for an in-place claim with no
    worktree, the project root — ADR-005/006) and returns the existing
    CompletionVerdict shape {ok, feedback}. Completion evidence is
    flavor-appropriate (ADR-006): a worktree node keeps the stageable-change
    check; an in-place node with explicitly-empty gates and an artifact_path is
    checked for that artifact existing and being non-empty instead. The verdict
    is persisted as a role="verify" run message so the deposit/mark-terminal
    gate can confirm a "done" transition was actually verified.
    """
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        run = graph.get_run(run_id)
        if run is None:
            raise ValueError(f"run {run_id!r} not found")
        node = graph.get_node(run["node_id"])
        if node is None:
            raise ValueError(f"node {run['node_id']} for run {run_id!r} not found")

        cwd = Path(node.worktree_path) if node.worktree_path is not None else root
        profile = resolve_flavor_profile(cfg, node.flavor)
        verifier = build_completion_verifier(
            cwd,
            profile.quality_gates,
            in_place=node.worktree_path is None,
            artifact_path=node.artifact_path,
        )
        verdict = verifier()
        graph.deposit_run_message(
            run_id,
            VERIFY_ROLE,
            json.dumps({"ok": verdict.ok, "feedback": verdict.feedback}),
            now_iso(),
        )
        _logger.info("milknado_node_verify: run_id=%s ok=%s", run_id, verdict.ok)
        return {"ok": verdict.ok, "feedback": verdict.feedback}
    finally:
        graph.close()
