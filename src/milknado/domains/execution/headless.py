"""Headless single-node ralph loop: dispatch -> wait-for-completion -> merge.

The TUI-free twin of one `RunLoop` iteration. `RunLoop.run()` drives a
`rich.live.Live` display and a keyboard-input thread, so it cannot run inside a
headless server process or a detached subprocess. This function reuses the same
`Executor` + `LoopPort` primitives and mirrors the success/failure branching of
`run_loop._completion.handle_completion`, but with no terminal dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from milknado.domains.common import ProgressEvent
from milknado.domains.common.errors import CompletionTimeout

if TYPE_CHECKING:
    from milknado.domains.common.protocols import TerminalRunOutcome
    from milknado.domains.execution.executor import (
        CompletionResult,
        DispatchResult,
        ExecutionConfig,
    )


@dataclass(frozen=True)
class HeadlessOutcome:
    node_id: int
    success: bool
    detail: str | None = None


class _ExecutorLike(Protocol):
    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> DispatchResult: ...
    def cancel(self, node_id: int) -> None: ...
    def complete(self, node_id: int, feature_branch: str) -> CompletionResult: ...
    def fail(self, node_id: int) -> None: ...


class _RalphLike(Protocol):
    def wait_for_next_completion(
        self, active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]: ...
    def stop_run(self, run_id: str, timeout: float | None = None) -> bool: ...


def run_node_to_completion(
    executor: _ExecutorLike,
    ralph: _RalphLike,
    node_id: int,
    exec_config: ExecutionConfig,
    feature_branch: str,
    timeout: float,
    *,
    base_oid: str | None = None,
    parent_run_id: str | None = None,
) -> HeadlessOutcome:
    """Dispatch one node into its worktree, wait for the ralph run to finish, then
    rebase-merge it back. Returns success only when the run completed AND the
    rebase merged cleanly; a stopped run is cancelled, while a timeout, failed
    run, or rebase conflict marks the node failed.
    """
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
    if parent_run_id is None:
        dispatch = executor.dispatch(node_id, exec_config, base_oid=base_oid)
    else:
        dispatch = executor.dispatch(
            node_id, exec_config, base_oid=base_oid, parent_run_id=parent_run_id
        )
    deadline = time.monotonic() + timeout
    while True:
        try:
            remaining_timeout = deadline - time.monotonic()
            _run_id, outcome = ralph.wait_for_next_completion(
                {dispatch.run_id}, timeout=remaining_timeout
            )
        except CompletionTimeout:
            if not ralph.stop_run(dispatch.run_id, timeout=10.0):
                return HeadlessOutcome(
                    node_id,
                    success=False,
                    detail="completion timeout; worker did not exit, ownership preserved",
                )
            executor.fail(node_id)
            return HeadlessOutcome(node_id, success=False, detail="completion timeout")

        if isinstance(outcome, ProgressEvent):
            continue

        if outcome == "stopped":
            executor.cancel(node_id)
            return HeadlessOutcome(node_id, success=False, detail="worker run stopped")
        if outcome == "failed":
            if not ralph.stop_run(dispatch.run_id, timeout=10.0):
                return HeadlessOutcome(
                    node_id,
                    success=False,
                    detail="worker run did not complete or exit; ownership preserved",
                )
            executor.fail(node_id)
            return HeadlessOutcome(node_id, success=False, detail="worker run did not complete")

        result = executor.complete(node_id, feature_branch)
        if result.redispatch is not None:
            dispatch = result.redispatch
            deadline = time.monotonic() + timeout
            continue
        if result.blocked:
            return HeadlessOutcome(
                node_id,
                success=False,
                detail="adversarial review blocked the node",
            )
        if result.rebase_conflict is not None:
            rc = result.rebase_conflict
            detail = rc.detail or ("conflicts: " + ", ".join(rc.conflicting_files))
            return HeadlessOutcome(node_id, success=False, detail=detail)
        return HeadlessOutcome(node_id, success=result.rebased)
