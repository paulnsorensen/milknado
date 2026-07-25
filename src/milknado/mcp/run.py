"""Milknado MCP worker-run tools — thin registration over milknado.app.run."""

from __future__ import annotations

import logging

from milknado.adapters import GitAdapter, ProcessAdapter
from milknado.app.run import (
    InlineRunRequest,
    run_inline,
    run_inline_start,
    validate_worker_cmd,
)
from milknado.domains.common import WorktreeMode
from milknado.domains.dispatch import (
    RUN_ID_RE,
    cancel_run,
    now_iso,
    poll_async_run,
    reconcile_node_status,
)
from milknado.mcp._core import (
    build_run_dict,
    mcp,
    open_graph,
    require_worker_run,
    resolve_project_root,
)

_logger = logging.getLogger(__name__)


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
    review or PR (teardown while the run is active happens via milknado_run_cancel;
    a completed merge_back=False worktree must be removed manually via
    `git worktree remove`). THIS_BRANCH runs the worker in the shared checkout —
    the diff lands in the current working tree — and ignores merge_back.
    """
    validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        state = run_inline(
            graph,
            cfg,
            root,
            InlineRunRequest(node_id, worker_cmd, timeout_seconds, worktree, merge_back),
        )
        return build_run_dict(state)
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
    while the run is active happens via milknado_run_cancel; a completed
    merge_back=False worktree must be removed manually via `git worktree remove`).
    THIS_BRANCH runs in the shared checkout and ignores merge_back.
    """
    validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        state = run_inline_start(
            graph,
            cfg,
            root,
            InlineRunRequest(node_id, worker_cmd, timeout_seconds, worktree, merge_back),
            use_tmux,
        )
        return build_run_dict(state)
    finally:
        graph.close()


@mcp.tool()
def milknado_run_inline_poll(run_id: str, project_root: str = "") -> dict:
    """Poll an async run from durable state and attach its deposited result."""
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        state = poll_async_run(graph, root, run_id)
        if state["status"] in ("done", "failed"):
            reconcile_node_status(
                graph, state["node_id"], state["status"], run_id=state.get("run_id")
            )
        state["result"] = graph.latest_run_message(run_id, "result")
    finally:
        graph.close()
    state["worktree_preserved"] = state.get("detail")
    return build_run_dict(state)


@mcp.tool()
def milknado_run_list(project_root: str = "", limit: int = 50) -> list[dict]:
    """List active and recent runs, newest first.

    Returns superset-schema dicts for up to `limit` runs. Use poll tools for log
    tails; list results keep summary as None.
    """
    root = resolve_project_root(project_root or None)
    graph, _cfg = open_graph(root)
    try:
        return [build_run_dict(r) for r in graph.recent_runs(limit)]
    finally:
        graph.close()


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
        final = cancel_run(graph, GitAdapter(root), ProcessAdapter(), root, run_id)
    finally:
        graph.close()
    return build_run_dict(final)


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
    require_worker_run(run_id)
    graph, _cfg = open_graph(root)
    try:
        if graph.get_run(run_id) is None:
            raise ValueError(f"run {run_id!r} not found")
        seq = graph.deposit_run_message(run_id, "result", payload, now_iso())
    finally:
        graph.close()
    return {"run_id": run_id, "seq": seq}


_VALID_REVIEW_VERDICTS = frozenset({"approve", "reject"})


@mcp.tool()
def milknado_deposit_review(
    run_id: str, verdict: str, findings_md: str, project_root: str = ""
) -> dict:
    """Persist a reviewer worker's structured verdict alongside the free-text result.

    Called by a reviewer worker as its final step with `verdict` ("approve" or
    "reject") and `findings_md` (the review findings in markdown). Stored as a
    role='review' run message, distinct from milknado_deposit_result's role='result'.
    Returns the run_id and the assigned message seq.
    """
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    if verdict not in _VALID_REVIEW_VERDICTS:
        raise ValueError(
            f"invalid verdict {verdict!r}: must be one of {sorted(_VALID_REVIEW_VERDICTS)}"
        )
    root = resolve_project_root(project_root or None)
    require_worker_run(run_id)
    graph, _cfg = open_graph(root)
    try:
        if graph.get_run(run_id) is None:
            raise ValueError(f"run {run_id!r} not found")
        body = f"{verdict}\n{findings_md}"
        seq = graph.deposit_run_message(run_id, "review", body, now_iso())
    finally:
        graph.close()
    return {"run_id": run_id, "seq": seq}
