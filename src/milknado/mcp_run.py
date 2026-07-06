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
from milknado.domains.common import NodeKind, NodeStatus, WorktreeMode
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.dispatch import (
    RUN_ID_RE,
    IsolateContext,
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
    resolve_feature_branch,
    setup_isolated_worktree,
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


def _prepare_isolation(
    graph,  # noqa: ANN001
    root,  # noqa: ANN001
    node,  # noqa: ANN001
    run_id: str,
    worktree: WorktreeMode,
    merge_back: bool,
    worktree_pattern: str,
):  # noqa: ANN201
    """Resolve the async worker's cwd + deferred merge-back context.

    ISOLATE: create the node's worktree + branch, record it on the (already
    run_id-claimed) node, run there, and — when merge_back — build the
    IsolateContext the worker thread uses to rebase-merge the branch back after it
    exits. THIS_BRANCH: run in the shared checkout with nothing to merge back.
    Runs after claim_node so set_worktree's run_id fence matches.
    """
    if worktree != WorktreeMode.ISOLATE:
        return root, None
    wt_path, worker_branch = setup_isolated_worktree(graph, root, node, run_id, worktree_pattern)
    merge_ctx = None
    if merge_back:
        merge_ctx = IsolateContext(
            worktree_path=wt_path,
            worker_branch=worker_branch,
            feature_branch=resolve_feature_branch(root),
            node_id=node.id,
            description=node.description,
        )
    return wt_path, merge_ctx


@mcp.tool()
def milknado_run_inline(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    worktree: WorktreeMode = WorktreeMode.ISOLATE,
    merge_back: bool = True,
    project_root: str = "",
) -> dict:
    """Spawn a subprocess worker with the task brief on stdin; capture log and update status.

    worker_cmd defaults to profile.execution_agent (resolved from the node's flavor).
    On exit 0 the node is marked done; on nonzero/timeout it is marked failed.
    Blocks for up to timeout_seconds (default 600). For non-blocking dispatch
    use milknado_run_inline_start / milknado_run_inline_poll.

    worktree controls isolation (safe by default): ISOLATE runs the worker in a
    fresh git worktree + branch, and on exit 0 (when merge_back, the default)
    rebase-merges the branch back into the caller's dispatch branch and tears the
    worktree down. merge_back=False leaves the branch/worktree for the caller to
    review or PR (teardown only via milknado_run_cancel). THIS_BRANCH runs the
    worker in the shared checkout — the diff lands in the current working tree —
    and ignores merge_back.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    _logger.info(
        "milknado_run_inline: node=%d timeout=%ds worktree=%s merge_back=%s",
        node_id,
        timeout_seconds,
        worktree.value,
        merge_back,
    )
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
            worktree_mode=worktree,
            merge_back=merge_back,
            worktree_pattern=cfg.worktree_pattern,
        )
    finally:
        graph.close()


@mcp.tool()
def milknado_run_inline_start(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    use_tmux: bool = False,
    worktree: WorktreeMode = WorktreeMode.ISOLATE,
    merge_back: bool = True,
    project_root: str = "",
) -> dict:
    """Start a worker asynchronously; returns immediately with a run_id for polling.

    Refuses if the node is already running. Use milknado_run_inline_poll(run_id) to
    check progress; node status is reconciled to done/failed on the first poll
    after the worker exits. use_tmux=True runs the worker inside a named tmux
    window (`milknado attach <run_id>`); fails fast if tmux is unavailable.

    worktree controls isolation (safe by default): ISOLATE runs the worker in a
    fresh git worktree + branch; on exit 0 (when merge_back, the default) the
    branch is rebase-merged back into the caller's dispatch branch and the worktree
    torn down. merge_back=False leaves the branch/worktree for the caller (teardown
    only via milknado_run_cancel). THIS_BRANCH runs in the shared checkout and
    ignores merge_back.
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
            worker_cwd, merge_ctx = _prepare_isolation(
                graph, root, node, run_id, worktree, merge_back, cfg.worktree_pattern
            )
            ref = start_headless_async(
                root,
                node_id,
                brief,
                worker_cmd,
                timeout_seconds,
                run_id=run_id,
                default_cmd=profile.execution_agent,
                tmux=tmux,
                cwd=worker_cwd,
                merge_ctx=merge_ctx,
            )
        except Exception:
            # Startup failed after the claim (bad worker cmd resolved in
            # start_headless_async, or a state-file write): release the claim with a
            # fenced terminal write so the node is not stranded RUNNING until the
            # stale-timeout backstop, then re-raise for a clean caller failure.
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise
        _logger.info(
            "milknado_run_inline_start: node=%d run_id=%s timeout=%ds",
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
def milknado_run_inline_poll(run_id: str, project_root: str = "") -> dict:
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
    # An ISOLATE merge-back that could not tear its worktree down persists the path
    # in the run row's `detail` column (async dispatch's only use of it); surface it
    # under the same key the sync dispatch and milknado_run_cancel return.
    state["worktree_preserved"] = state.get("detail")
    return state


@mcp.tool()
def milknado_run_list(project_root: str = "", limit: int = 50) -> list[dict]:
    """List active and recent runs, newest first.

    Returns superset-schema dicts for up to `limit` runs. Use poll tools for log
    tails; list results keep summary as None.
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
    """Cancel a run and return its final superset-schema state.

    No-ops for terminal runs. Active runs stop and finalize when safe. An owned
    worktree is removed unless fail-closed teardown preserves a dirty or unlanded
    one; `worktree_preserved` then holds that path, else None.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    graph, _cfg = open_graph(root)
    try:
        final = cancel_run(graph, root, run_id)
    finally:
        graph.close()
    result = _state_to_run_dict(final)
    result["worktree_preserved"] = final.get("worktree_preserved")
    return result


@mcp.tool()
def milknado_deposit_result(run_id: str, payload: str, project_root: str = "") -> dict:
    """Persist a worker's complete deliverable to the run's durable result sink.

    Called by a worker as its final step with its *complete* deliverable in
    `payload` (the full text, not a reference to content that lives only in the
    worker's own context). Stored as a role='result' run message so it survives the
    process boundary; milknado_run_inline_poll returns it under `result`. Returns the
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
