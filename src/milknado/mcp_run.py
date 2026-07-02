"""Milknado MCP worker-run tools — sync and async headless dispatch."""

from __future__ import annotations

import logging
import shlex

from milknado._mcp_core import (
    _check_ancestor_goal_not_claimed,
    _claim_ancestor_goal_for_dispatch,
    mcp,
    open_graph,
    resolve_project_root,
)
from milknado.adapters import TmuxAdapter
from milknado.domains.common import NodeKind, NodeStatus
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.dispatch import (
    RUN_ID_RE,
    cancel_run,
    dispatch_node_sync,
    ensure_tmux_ready,
    make_run_id,
    now_iso,
    poll_async_run,
    reclaim_stale_node,
    reconcile_node_status,
    reconcile_run_window,
    render_brief,
    start_headless_async,
    validate_worker_argv,
)

_logger = logging.getLogger(__name__)


def _validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable isn't an allowed AI agent CLI.

    Eager pre-check on the MCP arg; the env fallback and built-in default are
    validated again where they're resolved (`runner._resolve_worker_cmd`). Both
    routes share `validate_worker_argv`, so the allowlist lives in one place.
    """
    if not worker_cmd or not worker_cmd.strip():
        return
    validate_worker_argv(shlex.split(worker_cmd))


@mcp.tool()
def milknado_run_once(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    project_root: str = "",
) -> dict:
    """Spawn a subprocess worker with the task brief on stdin; capture log and update status.

    worker_cmd defaults to profile.execution_agent (resolved from the node's flavor).
    On exit 0 the node is marked done; on nonzero/timeout it is marked failed.
    Blocks for up to timeout_seconds (default 600). For non-blocking dispatch
    use milknado_run_once_start / milknado_run_once_poll.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    _logger.info("milknado_run_once: node=%d timeout=%ds", node_id, timeout_seconds)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.kind != NodeKind.TASK:
            raise ValueError(
                f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
            )
        _check_ancestor_goal_not_claimed(graph, node_id)
        _claim_ancestor_goal_for_dispatch(graph, node_id, make_run_id(node_id))
        profile = resolve_flavor_profile(cfg, node.flavor)
        return dispatch_node_sync(
            graph,
            node_id,
            root,
            worker_cmd,
            timeout_seconds,
            default_cmd=profile.execution_agent,
            brief_prepend=profile.brief_prepend,
        )
    finally:
        graph.close()


@mcp.tool()
def milknado_run_once_start(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    use_tmux: bool = False,
    project_root: str = "",
) -> dict:
    """Start a worker asynchronously; returns immediately with a run_id for polling.

    Refuses if the node is already running. Use milknado_run_once_poll(run_id) to
    check progress; node status is reconciled to done/failed on the first poll
    after the worker exits. use_tmux=True runs the worker inside a named tmux
    window (`milknado attach <run_id>`); fails fast if tmux is unavailable.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    tmux: TmuxAdapter | None = None
    if use_tmux:
        # Fail closed BEFORE any claim: tmux was explicitly requested, so a
        # missing binary or unstartable server fails the dispatch loudly —
        # never a silent fallback to the in-process subprocess path.
        tmux = TmuxAdapter(root)
        ensure_tmux_ready(tmux)
    graph, cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.kind != NodeKind.TASK:
            raise ValueError(
                f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
            )
        run_id = make_run_id(node_id)
        _check_ancestor_goal_not_claimed(graph, node_id)
        _claim_ancestor_goal_for_dispatch(graph, node_id, run_id)
        if node.status == NodeStatus.RUNNING:
            # Before the claim can succeed, reconcile any orphaned terminal runs
            # for this node — a prior worker may have finished without anyone
            # polling, leaving the node stuck on RUNNING. Also sweep runs stuck on
            # "running" past their timeout: the worker thread vanished (e.g. server
            # crash) without writing a terminal state, which would lock the node
            # just as permanently. (The headless worker is an in-process thread
            # with no pid, so there is no pid-liveness fast path here — it relies
            # on this reconcile plus the stale-timeout backstop.)
            reclaim_stale_node(graph, node_id, fence_run_id=node.run_id)
        profile = resolve_flavor_profile(cfg, node.flavor)
        brief = render_brief(graph, node_id, prepend=profile.brief_prepend)
        # Atomic optimistic claim: the cross-process mutual-exclusion point that
        # replaces the in-process dispatch lock. The PENDING/FAILED/BLOCKED ->
        # RUNNING transition and the run_id fence are written in one statement, so a
        # second concurrent caller sees RUNNING and loses. The same run_id keys the
        # run row, keeping the node row and the run row in agreement.
        if not graph.claim_node(node_id, run_id, now=now_iso()):
            current = graph.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )
        try:
            ref = start_headless_async(
                root,
                node_id,
                brief,
                worker_cmd,
                timeout_seconds,
                run_id=run_id,
                default_cmd=profile.execution_agent,
                tmux=tmux,
            )
        except Exception:
            # Startup failed after the claim (bad worker cmd resolved in
            # start_headless_async, or a state-file write): release the claim with a
            # fenced terminal write so the node is not stranded RUNNING until the
            # stale-timeout backstop, then re-raise for a clean caller failure.
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise
        _logger.info(
            "milknado_run_once_start: node=%d run_id=%s timeout=%ds",
            node_id,
            ref.run_id,
            timeout_seconds,
        )
        return {
            "run_id": ref.run_id,
            "node_id": node_id,
            "status": "running",
            "exit_code": None,
            "timed_out": None,
            "rebased": None,
            "log_path": str(ref.log_path),
            "summary": None,
        }
    finally:
        graph.close()


@mcp.tool()
def milknado_run_once_poll(run_id: str, project_root: str = "") -> dict:
    """Poll an async run; reconciles node status to done/failed once the worker exits.

    Returns the deposited result payload (latest role='result' run message) under
    `result` when one exists, alongside the log-tail `summary`.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        state = poll_async_run(graph, root, run_id)
        if state["status"] in ("done", "failed"):
            # Fence on the polled run's run_id so polling an older run cannot
            # clobber a newer RUNNING owner that re-claimed this node.
            reconcile_node_status(
                graph, state["node_id"], state["status"], run_id=state.get("run_id")
            )
            # Per-row window reconcile backstop: a completed run's window is
            # normally self-cleaned; kill a straggler (e.g. after a restart).
            reconcile_run_window(root, state)
        state["result"] = graph.latest_run_message(run_id, "result")
    finally:
        graph.close()
    state.setdefault("rebased", None)
    return state


@mcp.tool()
def milknado_run_list(project_root: str = "", limit: int = 50) -> list[dict]:
    """List active and recent runs from the db, sorted newest first.

    Returns superset-schema dicts. summary is always None — use
    milknado_run_once_poll or milknado_run_loop_poll for log tails. Bounded to
    the `limit` most-recently-started runs so read cost stays flat as run history
    grows; raise `limit` to widen the window.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        return [_state_to_run_dict(r) for r in graph.recent_runs(limit)]
    finally:
        graph.close()


def _state_to_run_dict(state: dict) -> dict:
    return {
        "run_id": state.get("run_id"),
        "node_id": state.get("node_id"),
        "status": state.get("status"),
        "exit_code": state.get("exit_code"),
        "timed_out": state.get("timed_out"),
        "rebased": state.get("rebased"),
        "log_path": state.get("log_path"),
        "summary": None,
    }


@mcp.tool()
def milknado_run_cancel(run_id: str, project_root: str = "") -> dict:
    """Cancel a run, reconcile node status, and prune any worktree.

    Detached-ralph runs are signalled; async-headless runs receive a cancel
    sentinel and are given a bounded window to finalize cooperatively. No-ops
    if the run is already terminal. Returns the final run state.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    graph, _cfg = open_graph(root)
    try:
        final = cancel_run(graph, root, run_id)
    finally:
        graph.close()
    return _state_to_run_dict(final)


@mcp.tool()
def milknado_deposit_result(run_id: str, payload: str, project_root: str = "") -> dict:
    """Persist a worker's complete deliverable to the run's durable result sink.

    Called by a worker as its final step with its *complete* deliverable in
    `payload` (the full text, not a reference to content that lives only in the
    worker's own context). Stored as a role='result' run message so it survives the
    process boundary; milknado_run_once_poll returns it under `result`. Returns the
    run_id and the assigned message seq.
    """
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        if graph.get_run(run_id) is None:
            raise ValueError(f"run {run_id!r} not found")
        seq = graph.deposit_run_message(run_id, "result", payload, now_iso())
    finally:
        graph.close()
    return {"run_id": run_id, "seq": seq}
