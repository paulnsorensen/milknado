"""Milknado MCP worker-run tools — sync and async headless dispatch."""

from __future__ import annotations

import logging
import os
import shlex
import signal
import time
from pathlib import Path

from milknado._mcp_core import mcp, open_graph, resolve_project_root
from milknado.adapters import GitAdapter
from milknado.domains.common import NodeStatus
from milknado.domains.dispatch import (
    RUN_ID_RE,
    clear_cancel,
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    latest_terminal_run,
    make_run_id,
    now_iso,
    poll_async_run,
    reconcile_node_status,
    render_brief,
    request_cancel,
    run_headless,
    runs_dir,
    start_headless_async,
)
from milknado.domains.dispatch.runner import _validate_worker_argv

_logger = logging.getLogger(__name__)

# How long milknado_run_cancel waits for an async worker to observe the cancel
# sentinel and write its own terminal state before falling back to writing it
# directly. Must exceed the worker's worst-case finalize (poll interval +
# SIGTERM→SIGKILL grace in runner.py) so a live worker reliably wins the write.
_CANCEL_FINALIZE_TIMEOUT_SECS = 8.0
_CANCEL_FINALIZE_POLL_SECS = 0.25


def _validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable isn't an allowed AI agent CLI.

    Eager pre-check on the MCP arg; the env fallback and built-in default are
    validated again where they're resolved (`runner._resolve_worker_cmd`). Both
    routes share `_validate_worker_argv`, so the allowlist lives in one place.
    """
    if not worker_cmd or not worker_cmd.strip():
        return
    _validate_worker_argv(shlex.split(worker_cmd))


def _ensure_running(graph, node_id: int) -> bool:
    """Move a node to RUNNING so a (re)run's terminal transition (RUNNING ->
    DONE/FAILED) is always valid. BLOCKED/FAILED are reset through PENDING
    first. A DONE node cannot be reopened by the state machine and is left
    untouched — its prior result stands and callers must not record a new
    terminal status for it. Returns True when the node is RUNNING afterward.
    """
    node = graph.get_node(node_id)
    if node is None or node.status == NodeStatus.DONE:
        return False
    if node.status == NodeStatus.RUNNING:
        return True
    if node.status in (NodeStatus.BLOCKED, NodeStatus.FAILED):
        graph.mark_pending(node_id)
    graph.mark_running(node_id)
    return True


def _run_and_update_status(
    graph,
    node_id: int,
    project_root: Path,
    worker_cmd: str | None,
    timeout_seconds: int,
    *,
    brief_prepend: str | None = None,
) -> dict:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    brief = render_brief(graph, node_id, prepend=brief_prepend)
    running = _ensure_running(graph, node_id)
    run_id = make_run_id(node_id)
    started_at = now_iso()
    rdir = runs_dir(project_root)
    log_path = rdir / f"{run_id}.log"
    if running:
        # Insert a rescuable "running" run row BEFORE blocking in run_headless.
        # The sync call blocks up to timeout_seconds; if the client times out and
        # the server is killed mid-run, the post-run terminal write never lands,
        # stranding the node RUNNING forever. With this row fail_stale_running_runs
        # releases the node past timeout, matching start_headless_async.
        graph.start_run(run_id, node_id, str(log_path), started_at, timeout_seconds)
    # Only inject MILKNADO_RUN_ID when a 'running' run row exists. On a DONE-node
    # re-run no row is inserted (start_run is gated on `running`), so passing the
    # run_id would have the worker deposit into a row that doesn't exist —
    # milknado_deposit_result raises "run not found". Pass None instead.
    result = run_headless(
        project_root,
        node_id,
        brief,
        worker_cmd,
        timeout_seconds,
        run_id=run_id if running else None,
    )
    worker_terminal = "done" if result.exit_code == 0 and not result.timed_out else "failed"
    if running:
        if worker_terminal == "done":
            graph.mark_done(node_id)
        else:
            graph.mark_failed(node_id)
    final = graph.get_node(node_id)
    if final is None:
        raise RuntimeError(f"node {node_id} not found after run completed")
    if running:
        graph.finish_run(
            run_id,
            status=worker_terminal,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            ended_at=now_iso(),
        )
    graph.set_run_id(node_id, run_id)
    return {
        "run_id": run_id,
        "node_id": node_id,
        "status": final.status.value,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "rebased": None,
        "log_path": str(result.log_path),
        "summary": result.summary,
    }


@mcp.tool()
def milknado_todo_run(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    project_root: str = "",
) -> dict:
    """Spawn a subprocess worker with the task brief on stdin; capture log and update status.

    worker_cmd defaults to $MILKNADO_WORKER_CMD then `claude -p`.
    On exit 0 the node is marked done; on nonzero/timeout it is marked failed.
    Blocks for up to timeout_seconds (default 600). For non-blocking dispatch
    use milknado_todo_run_start / milknado_todo_run_poll.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    _logger.info("milknado_todo_run: node=%d timeout=%ds", node_id, timeout_seconds)
    try:
        return _run_and_update_status(
            graph,
            node_id,
            root,
            worker_cmd,
            timeout_seconds,
            brief_prepend=cfg.worker_brief_prepend,
        )
    finally:
        graph.close()


@mcp.tool()
def milknado_todo_run_start(
    node_id: int,
    worker_cmd: str | None = None,
    timeout_seconds: int = 600,
    project_root: str = "",
) -> dict:
    """Start a worker asynchronously; returns immediately with a run_id for polling.

    Refuses if the node is already running. Use milknado_todo_run_poll(run_id) to
    check progress; node status is reconciled to done/failed on the first poll
    after the worker exits.
    """
    _validate_worker_cmd(worker_cmd)
    root = resolve_project_root(project_root or None)
    graph, cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")
        if node.status == NodeStatus.RUNNING:
            # Before the claim can succeed, reconcile any orphaned terminal runs
            # for this node — a prior worker may have finished without anyone
            # polling, leaving the node stuck on RUNNING. Also sweep runs stuck on
            # "running" past their timeout: the worker thread vanished (e.g. server
            # crash) without writing a terminal state, which would lock the node
            # just as permanently. (The headless worker is an in-process thread
            # with no pid, so there is no pid-liveness fast path here — it relies
            # on this reconcile plus the stale-timeout backstop.)
            fail_stale_running_runs(graph, node_id)
            # Filter terminal files to this node's fence BEFORE picking the latest,
            # so a stale run with a later ended_at cannot mask the current owner's
            # file. reconcile_node_status then routes the write through the atomic
            # run_id + status fence (graph.mark_terminal), closing the TOCTOU a
            # pre-check-then-unfenced-write left open: a node re-claimed under a
            # newer run cannot be clobbered. A legacy node with no run_id has no
            # fence to honour and falls back to the unconditional reconcile.
            winner = latest_terminal_run(
                find_terminal_runs_for_node(graph, node_id, run_id=node.run_id)
            )
            if winner is not None:
                reconcile_node_status(
                    graph, node_id, winner["status"], run_id=winner.get("run_id")
                )
        brief = render_brief(graph, node_id, prepend=cfg.worker_brief_prepend)
        # Atomic optimistic claim: the cross-process mutual-exclusion point that
        # replaces the in-process dispatch lock. The PENDING/FAILED/BLOCKED ->
        # RUNNING transition and the run_id fence are written in one statement, so a
        # second concurrent caller sees RUNNING and loses. The same run_id keys the
        # run row, keeping the node row and the run row in agreement.
        run_id = make_run_id(node_id)
        if not graph.claim_node(node_id, run_id, now=now_iso()):
            current = graph.get_node(node_id)
            status = current.status.value if current is not None else "gone"
            raise ValueError(
                f"node {node_id} is already {status}; set status back to pending to retry"
            )
        try:
            ref = start_headless_async(
                root, node_id, brief, worker_cmd, timeout_seconds, run_id=run_id
            )
        except Exception:
            # Startup failed after the claim (bad worker cmd resolved in
            # start_headless_async, or a state-file write): release the claim with a
            # fenced terminal write so the node is not stranded RUNNING until the
            # stale-timeout backstop, then re-raise for a clean caller failure.
            graph.mark_terminal(node_id, run_id, NodeStatus.FAILED)
            raise
        _logger.info(
            "milknado_todo_run_start: node=%d run_id=%s timeout=%ds",
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
def milknado_todo_run_poll(run_id: str, project_root: str = "") -> dict:
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
        state["result"] = graph.latest_run_message(run_id, "result")
    finally:
        graph.close()
    state.setdefault("rebased", None)
    return state


@mcp.tool()
def milknado_run_list(project_root: str = "", limit: int = 50) -> list[dict]:
    """List active and recent runs from the db, sorted newest first.

    Returns superset-schema dicts. summary is always None — use
    milknado_todo_run_poll or milknado_ralph_run_poll for log tails. Bounded to
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


def _finalize_cancelled(graph, run_id: str) -> dict:  # noqa: ANN001
    """Write the cancelled terminal run row and return its refreshed dict.

    The direct-finalize path (pid run, or the fallback when the async worker never
    responds within the bound).
    """
    graph.finish_run(
        run_id,
        status="failed",
        exit_code=-1,
        timed_out=False,
        ended_at=now_iso(),
        error="cancelled",
    )
    return graph.get_run(run_id)


def _reconcile_cancel(root: Path, node_id, run_id: str, status: str = "failed") -> None:  # noqa: ANN001
    """Prune the run's worktree and reconcile its node to `status`, fenced on run_id.

    Routes through the run_id-fenced `reconcile_node_status` (graph.mark_terminal)
    so a node re-claimed under a newer run cannot be clobbered by a stale cancel —
    the merged worker-dispatch fencing invariant. A node with no run_id falls back
    to the unconditional RUNNING→terminal transition. `status` is the run's actual
    terminal status: a worker that finished `done` inside the cancel window must
    reconcile its node `done`, not be force-failed.
    """
    if node_id is None:
        return
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is not None and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                try:
                    GitAdapter(root).remove_worktree(wt)
                except Exception as exc:
                    _logger.warning("Failed to remove worktree %s on cancel: %s", wt, exc)
        reconcile_node_status(graph, node_id, status, run_id=run_id)
    finally:
        graph.close()
    try:
        GitAdapter(root).prune_worktrees()
    except Exception as exc:
        _logger.warning("git worktree prune failed on cancel: %s", exc)


def _await_cancel_finalize(graph, run_id: str) -> dict | None:  # noqa: ANN001
    """Poll the run row for the async worker's terminal write, up to a bound.

    Returns the terminal run dict once the worker finalizes, or None if the bound
    elapses (the worker is wedged, dead, or never existed) so the caller can take
    over the terminal write itself.
    """
    deadline = time.monotonic() + _CANCEL_FINALIZE_TIMEOUT_SECS
    while time.monotonic() < deadline:
        time.sleep(_CANCEL_FINALIZE_POLL_SECS)
        state = graph.get_run(run_id)
        if state is not None and state.get("status") != "running":
            return state
    return None


@mcp.tool()
def milknado_run_cancel(run_id: str, project_root: str = "") -> dict:
    """Cancel a run, reconcile node status, and prune any worktree.

    A detached-ralph run (has a recorded process-group pid) is signalled with
    SIGTERM and finalized immediately. An async-headless run shares the MCP
    server's process group, so it cannot be signalled — instead a cancel sentinel
    is written and the worker cooperatively terminates its own subprocess and
    writes the terminal state, which closes the state-clobber race. Cancel waits a
    bounded window for that finalize, then takes over the terminal write only if
    the worker never responds (wedged or crashed). No-ops cleanly if the run is
    already terminal. Returns the final run state in the unified superset schema.
    """
    root = resolve_project_root(project_root or None)
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    graph, _cfg = open_graph(root)
    try:
        state = graph.get_run(run_id)
        if state is None:
            raise ValueError(f"run {run_id!r} not found")
        if state.get("status") != "running":
            return _state_to_run_dict(state)
        if state.get("pid") is not None:
            final = _cancel_pid_run(graph, root, state, run_id)
        else:
            final = _cancel_async_run(graph, root, state, run_id)
    finally:
        graph.close()
    return _state_to_run_dict(final)


def _cancel_pid_run(graph, root: Path, state: dict, run_id: str) -> dict:  # noqa: ANN001
    """Cancel a detached-ralph run: its own process group → killpg is safe, and the
    run owns no in-process worker, so finalize the terminal state here directly."""
    pid = state["pid"]
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # process already gone — fall through to finalize the terminal state
    # Any other OSError (e.g. EPERM) means the process may still be alive: let it
    # propagate rather than reporting a run cancelled that we never signalled.
    final = _finalize_cancelled(graph, run_id)
    _reconcile_cancel(root, state.get("node_id"), run_id)
    _logger.info("milknado_run_cancel(pid): run_id=%s node_id=%s", run_id, state.get("node_id"))
    return final


def _cancel_async_run(graph, root: Path, state: dict, run_id: str) -> dict:  # noqa: ANN001
    """Cancel an async-headless run (or a crashed run stuck "running"). Request
    cooperative cancellation and let the worker own the terminal write; only take
    over if the worker never finalizes within the bound."""
    rdir = runs_dir(root)
    request_cancel(rdir, run_id)
    final = _await_cancel_finalize(graph, run_id)
    if final is None:
        # Worker never finalized within the bound: it is wedged, dead, or was never
        # alive (crashed sync run). Take over the terminal write so the node is not
        # stranded RUNNING, and clear the sentinel we wrote.
        latest = graph.get_run(run_id)
        if latest is not None and latest.get("status") != "running":
            final = latest  # worker finalized in the last poll gap — don't clobber
        else:
            final = _finalize_cancelled(graph, run_id)
        clear_cancel(rdir, run_id)

    # Reconcile to the run's ACTUAL terminal status: a worker that finished `done`
    # inside the finalize window must not have its node force-failed.
    _reconcile_cancel(root, final.get("node_id"), run_id, final.get("status", "failed"))
    _logger.info("milknado_run_cancel(async): run_id=%s node_id=%s", run_id, state.get("node_id"))
    return final


@mcp.tool()
def milknado_deposit_result(run_id: str, payload: str, project_root: str = "") -> dict:
    """Persist a worker's complete deliverable to the run's durable result sink.

    Called by a worker as its final step with its *complete* deliverable in
    `payload` (the full text, not a reference to content that lives only in the
    worker's own context). Stored as a role='result' run message so it survives the
    process boundary; milknado_todo_run_poll returns it under `result`. Returns the
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
