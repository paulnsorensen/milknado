"""Native Workflow ("ultracode") backend MCP tools — coordinator-only.

These tools back the in-session Claude Code dynamic-Workflow driver. They run
ALONGSIDE the subprocess dispatcher in `mcp.run` and never spawn a process:
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
  (role="verify") so the server-side completion gate in `mcp.todo_mutate` can
  reject a "done" transition that was never verified.

The model/tool/owner resolution and worktree provisioning policy lives in
`milknado.app.node`; this module only parses I/O and registers the tools.

Coordinator-only: none of these tools belong in WORKER_ALLOWED_TOOLS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from milknado.app.node import (
    GOAL_OWNER_ENV_VAR,
    _provision_claim_run,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]
    _resolve_model,  # pyright: ignore[reportPrivateUsage]
    _resolve_node_tools,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]
    _resolve_owner,  # pyright: ignore[reportPrivateUsage]
)
from milknado.domains.common import NodeKind, resolve_flavor_profile
from milknado.domains.dispatch import RUN_ID_RE, make_run_id, now_iso, render_brief
from milknado.domains.execution import build_completion_verifier
from milknado.domains.graph import VERIFY_ROLE
from milknado.mcp._core import Response, mcp, open_graph, resolve_project_root

_logger = logging.getLogger(__name__)

__all__ = [
    "GOAL_OWNER_ENV_VAR",
    "milknado_goal_claim",
    "milknado_goal_release",
    "milknado_node_verify",
    "milknado_todo_claim",
]


@mcp.tool()
def milknado_goal_claim(goal_id: int, owner: str = "", project_root: str = "") -> Response:
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
def milknado_goal_release(goal_id: int, owner: str = "", project_root: str = "") -> Response:
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
) -> Response:
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
        brief = render_brief(
            graph,
            node_id,
            prepend=profile.brief_prepend,
            project_root=root,
        )

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


@mcp.tool()
def milknado_node_verify(run_id: str, project_root: str = "") -> Response:
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
        _ = graph.deposit_run_message(
            run_id,
            VERIFY_ROLE,
            json.dumps({"ok": verdict.ok, "feedback": verdict.feedback}),
            now_iso(),
        )
        _logger.info("milknado_node_verify: run_id=%s ok=%s", run_id, verdict.ok)
        return {"ok": verdict.ok, "feedback": verdict.feedback}
    finally:
        graph.close()
