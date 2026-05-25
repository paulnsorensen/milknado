"""Headless single-node ralph loop: dispatch -> wait-for-completion -> merge.

The TUI-free twin of one `RunLoop` iteration. `RunLoop.run()` drives a
`rich.live.Live` display and a keyboard-input thread, so it cannot run inside a
headless server process or a detached subprocess. This function reuses the same
`Executor` + `RalphPort` primitives and mirrors the success/failure branching of
`run_loop._completion.handle_completion`, but with no terminal dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common.errors import CompletionTimeout

if TYPE_CHECKING:
    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor


@dataclass(frozen=True)
class HeadlessOutcome:
    node_id: int
    success: bool
    detail: str | None = None


def run_node_to_completion(
    executor: Executor,
    ralph: RalphPort,
    node_id: int,
    exec_config: ExecutionConfig,
    feature_branch: str,
    timeout: float,
) -> HeadlessOutcome:
    """Dispatch one node into its worktree, wait for the ralph run to finish, then
    rebase-merge it back. Returns success only when the run completed AND the
    rebase merged cleanly; a timeout, a non-completed run, or a rebase conflict
    are all failures (the node is marked failed by `Executor.fail`/`complete`)."""
    if feature_branch in ("", "HEAD"):
        # current_branch() returns "HEAD" on a detached HEAD — not a valid
        # rebase-onto target. Refuse before dispatching a worktree rather than
        # rebase-merging onto the literal ref "HEAD", and mark the node failed
        # for parity with the timeout / non-completion / conflict branches below
        # (PENDING -> FAILED is a valid transition; reset to pending to retry).
        executor.fail(node_id)
        return HeadlessOutcome(
            node_id,
            success=False,
            detail=f"invalid feature branch {feature_branch!r}; refusing to dispatch",
        )
    dispatch = executor.dispatch(node_id, exec_config)
    try:
        _run_id, completed = ralph.wait_for_next_completion({dispatch.run_id}, timeout=timeout)
    except CompletionTimeout:
        executor.fail(node_id)
        return HeadlessOutcome(node_id, success=False, detail="completion timeout")

    if not completed:
        executor.fail(node_id)
        return HeadlessOutcome(node_id, success=False, detail="worker run did not complete")

    result = executor.complete(node_id, feature_branch)
    if result.rebase_conflict is not None:
        rc = result.rebase_conflict
        detail = rc.detail or ("conflicts: " + ", ".join(rc.conflicting_files))
        return HeadlessOutcome(node_id, success=False, detail=detail)
    return HeadlessOutcome(node_id, success=result.rebased)
