"""Native Workflow ("ultracode") backend MCP tools — coordinator-only.

These two tools back the in-session Claude Code dynamic-Workflow driver. They run
ALONGSIDE the subprocess dispatcher in `mcp_run.py` and never spawn a process:
the harness-side Workflow is the driver, milknado is the graph/state backend.

- `milknado_todo_claim`: atomically claim a node RUNNING, create its durable
  worktree, write the `runs` row, and return the structured config the Workflow
  needs to dispatch one `agent()` per ready node. Spawns nothing.
- `milknado_node_verify`: run the node's resolved quality_gates in its worktree
  and return the existing `CompletionVerdict` shape. The verdict is persisted as a
  `run_messages` row (role="verify") so the server-side completion gate in
  `mcp_todo_mutate` can reject a "done" transition that was never verified.

Coordinator-only: neither tool belongs in WORKER_ALLOWED_TOOLS.
"""

from __future__ import annotations

import json
import logging
import re

from milknado._mcp_core import (
    _check_ancestor_goal_not_claimed,
    _claim_ancestor_goal_for_dispatch,
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.agent_argv import resolve_worker_tools
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.dispatch import RUN_ID_RE, make_run_id, now_iso

_logger = logging.getLogger(__name__)

VERIFY_ROLE = "verify"
# Native-backend marker: milknado_todo_claim stamps this run message at claim
# time. The subprocess executor starts its runs via start_run and never deposits
# it, so its presence is what distinguishes a native-claimed run from a
# subprocess run that merely shares the run_id + runs-row shape.
CLAIM_ROLE = "claim"
_MODEL_FLAG_RE = re.compile(r"--model[= ]+(\S+)")


def _resolve_model(execution_agent: str) -> str:
    """Extract the model from the resolved execution command, defaulting to sonnet.

    The native backend returns `model` as a structured field so the Workflow can
    route `agent({model: ...})` correctly instead of inferring it — closing the
    per-flavor model-routing gap.
    """
    match = _MODEL_FLAG_RE.search(execution_agent)
    return match.group(1) if match else "sonnet"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:30]


@mcp.tool()
def milknado_todo_claim(node_id: int, project_root: str = "") -> dict:
    """Claim a task node for native (in-session Workflow) execution; spawn nothing.

    Atomically CAS-claims the node PENDING/FAILED/BLOCKED -> RUNNING, creates its
    durable worktree (the carry-forward state), records worktree_path + branch,
    writes a status='running' runs row, and honors goal_claims subtree fencing.
    Returns the full structured config the Workflow node-runner needs to dispatch
    one agent() per node: brief, flavor, model, tools, worktree_path, agent_type,
    loop_mode, max_iterations, max_turns. NO process is spawned — the harness-side
    Workflow is the driver.
    """
    from milknado.adapters import GitAdapter
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
        _check_ancestor_goal_not_claimed(graph, node_id)
        _claim_ancestor_goal_for_dispatch(graph, node_id, run_id)

        profile = resolve_flavor_profile(cfg, node.flavor)
        brief = render_brief(graph, node_id, prepend=profile.brief_prepend)

        if not graph.claim_node(node_id, run_id, now=now_iso()):
            current = graph.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )

        try:
            wt_path, branch = _create_node_worktree(
                GitAdapter(root), root, node_id, node.description
            )
            graph.set_worktree(node_id, run_id, str(wt_path), branch)
            graph.start_run(run_id, node_id, str(wt_path), now_iso(), None)
            # Stamp the native-backend marker so the done-transition gate fires
            # for this run even before its first verify (fail-closed), while
            # leaving subprocess runs — which never carry it — exempt.
            graph.deposit_run_message(run_id, CLAIM_ROLE, "", now_iso())
        except Exception:
            # Claim succeeded but worktree/run setup failed: release the claim with a
            # fenced terminal write so the node is not stranded RUNNING.
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise

        override = cfg.flavors.get(node.flavor) if node.flavor is not None else None
        tools = resolve_worker_tools(
            cfg.agent_family, list(override.tools) if override and override.tools else None
        )
        _logger.info("milknado_todo_claim: node=%d run_id=%s (native, no spawn)", node_id, run_id)
        return {
            "run_id": run_id,
            "node_id": node_id,
            "brief": brief,
            "flavor": node.flavor.value if node.flavor is not None else None,
            "model": _resolve_model(profile.execution_agent),
            "tools": list(tools),
            "worktree_path": str(wt_path),
            "agent_type": profile.worker_agent_type,
            "loop_mode": profile.loop_mode,
            "max_iterations": profile.max_iterations,
            "max_turns": profile.max_turns,
        }
    finally:
        graph.close()


def _create_node_worktree(git, root, node_id: int, description: str):  # noqa: ANN001
    """Create the node's durable worktree under project_root, mirroring the executor.

    Reuses GitAdapter.create_worktree and the executor's path/branch convention
    (worktree_pattern + milknado/<id>-<slug>), with the same path-traversal guard.
    """
    slug = _slugify(description)
    wt_path = root / f"milknado-{node_id}-{slug}"
    resolved_root = root.resolve()
    if not wt_path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"worktree path resolves outside project_root: {wt_path!r}")
    branch = f"milknado/{node_id}-{slug}"
    git.create_worktree(wt_path, branch)
    return wt_path, branch


@mcp.tool()
def milknado_node_verify(run_id: str, project_root: str = "") -> dict:
    """Run the node's resolved quality_gates in its worktree; return the verdict.

    Exposes the tier-2 completion verifier as a callable: runs every resolved
    quality gate in the node's worktree_path and returns the existing
    CompletionVerdict shape {ok, feedback}. The verdict is persisted as a
    role="verify" run message so the deposit/mark-terminal gate can confirm a
    "done" transition was actually verified.
    """
    from pathlib import Path

    from milknado.adapters.loop import _build_completion_verifier

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
        if node.worktree_path is None:
            raise ValueError(f"node {node.id} has no worktree_path; claim it before verifying")

        profile = resolve_flavor_profile(cfg, node.flavor)
        verifier = _build_completion_verifier(
            Path(node.worktree_path), list(profile.quality_gates)
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


def latest_verify_ok(graph, run_id: str) -> bool:  # noqa: ANN001
    """Return True iff the latest role='verify' message for run_id had ok=True.

    The server-side completion precondition: a "done" transition is rejected
    unless the node's owning run has a passing verify on record. A missing or
    malformed verdict reads as not-ok (fail-closed).
    """
    body = graph.latest_run_message(run_id, VERIFY_ROLE)
    if not body:
        return False
    try:
        return bool(json.loads(body).get("ok") is True)
    except (ValueError, TypeError, AttributeError):
        return False


def _is_native_claimed(graph, run_id: str) -> bool:  # noqa: ANN001
    """True iff the run was claimed via the native backend (milknado_todo_claim).

    Discriminates on the CLAIM_ROLE marker that milknado_todo_claim stamps — a
    NATIVE-only signal. A subprocess run carries the same run_id + runs-row shape
    but never deposits it, so it reads as non-native and is exempt from the gate.
    The verify message is also accepted as a marker for forward-safety: any run
    that has been through milknado_node_verify is native by construction.
    """
    return (
        graph.latest_run_message(run_id, CLAIM_ROLE) is not None
        or graph.latest_run_message(run_id, VERIFY_ROLE) is not None
    )


def assert_done_verified(graph, node) -> None:  # noqa: ANN001
    """Reject a "done" transition for a native-claimed node lacking a passing verify.

    Scoped to native-backend nodes only: the gate fires only when the node's
    owning run carries the native CLAIM_ROLE marker (or a verify message).
    Manual nodes (no owning run) and subprocess runs (run_id + runs row but no
    native marker) are left untouched — this gate is the precondition the spec
    puts on `milknado_node_verify`, not a new constraint on every status change.
    Fail-closed: a natively-claimed node with no passing verdict on record cannot
    be marked done, even before its first verify.
    """
    if node.run_id is None or graph.get_run(node.run_id) is None:
        return
    if not _is_native_claimed(graph, node.run_id):
        return
    if not latest_verify_ok(graph, node.run_id):
        raise ValueError(
            f"node {node.id} cannot be marked done: milknado_node_verify(run_id="
            f"{node.run_id!r}) has not returned ok=True. Run milknado_node_verify first."
        )
