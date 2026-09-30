from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC
from typing import TYPE_CHECKING

from milknado.domains.common import ProgressEvent, TerminalRunOutcome
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.execution._models import PreservedWorkerRun
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.execution.run_loop._logging import ts
from milknado.domains.execution.run_loop._protocols import RunLoopState
from milknado.domains.execution.run_loop._result import NodeLoopOutcome

if TYPE_CHECKING:
    from milknado.domains.execution._models import RebaseConflict
    from milknado.domains.execution.executor import ExecutionConfig

_logger = logging.getLogger("milknado")
IDLE_RESCAN_SECONDS = 1.0


class NodeDriverMixin(RunLoopState, ABC):
    _failure_triggered: bool

    def run_node(  # noqa: PLR0913
        self,
        node_id: int,
        config: ExecutionConfig,
        feature_branch: str,
        timeout: float,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> NodeLoopOutcome:
        """Drive only the selected task through the shared loop lifecycle."""
        if feature_branch in ("", "HEAD"):
            self._executor.fail(node_id)
            return NodeLoopOutcome(
                node_id, False, f"invalid feature branch {feature_branch!r}; refusing to dispatch"
            )
        try:
            dispatch = self._executor.dispatch(
                node_id, config, base_oid=base_oid, parent_run_id=parent_run_id
            )
        except PreservedWorkerRun as exc:
            return NodeLoopOutcome(
                node_id,
                False,
                str(exc),
                ownership_preserved=True,
                worker_run_id=exc.run_id,
                owner_run_id=exc.owner_run_id,
            )
        self._active[dispatch.run_id] = node_id
        started = time.monotonic()
        self._dispatched_at[dispatch.run_id] = started
        deadline = started + timeout * (config.max_iterations or 1)
        while self._active:
            run_id = next(iter(self._active))
            try:
                _, outcome = self._loop.wait_for_next_completion(
                    {run_id}, timeout=deadline - time.monotonic()
                )
            except CompletionTimeout:
                stopped = self._stop_timed_out_run(run_id)
                detail = (
                    "completion timeout"
                    if stopped
                    else ("completion timeout; worker did not exit, ownership preserved")
                )
                return NodeLoopOutcome(
                    node_id, False, detail, timed_out=True, ownership_preserved=not stopped
                )
            if isinstance(outcome, ProgressEvent):
                continue
            timed_out = outcome.timed_out
            transition = self._handle_terminal(run_id, outcome, feature_branch)
            if transition is None:
                return NodeLoopOutcome(
                    node_id,
                    False,
                    "worker run did not complete or exit; ownership preserved",
                    timed_out,
                    ownership_preserved=True,
                )
            completed, _failed, conflicts = transition
            if self._active:
                continue
            if conflicts:
                conflict = conflicts[0]
                detail = conflict.detail or "conflicts: " + ", ".join(conflict.conflicting_files)
                return NodeLoopOutcome(node_id, False, detail, timed_out)
            if completed:
                return NodeLoopOutcome(node_id, True)
            detail = {
                "stopped": "worker run stopped",
                "failed": "worker run did not complete",
            }.get(outcome.status, "adversarial review blocked the node")
            return NodeLoopOutcome(node_id, False, detail, timed_out)
        raise RuntimeError(f"node {node_id} ended without a terminal outcome")

    def confirm_preserved_stop(self, outcome: NodeLoopOutcome) -> NodeLoopOutcome:
        """Keep detached supervision alive until the worker exit is confirmed."""
        if not outcome.ownership_preserved:
            return outcome
        run_id = outcome.worker_run_id or next(iter(self._active), None)
        if run_id is None:
            return dataclasses.replace(outcome, ownership_preserved=False)
        while True:
            try:
                stopped = self._executor.force_stop_run(run_id, timeout=10.0)
            except Exception:
                _logger.exception("stop retry failed for run_id=%s", run_id)
                stopped = False
            if stopped:
                _ = self._active.pop(run_id, None)
                if outcome.owner_run_id is not None:
                    self._executor.finish_preserved_abort(
                        outcome.node_id, outcome.owner_run_id, run_id
                    )
                else:
                    self._executor.fail(outcome.node_id)
                return dataclasses.replace(outcome, ownership_preserved=False)
            time.sleep(IDLE_RESCAN_SECONDS)

    def _handle_terminal(
        self,
        run_id: str,
        outcome: TerminalRunOutcome,
        feature_branch: str,
    ) -> tuple[int, int, list[RebaseConflict]] | None:
        if outcome.status == "failed" and not self._executor.stop_run(run_id, timeout=10.0):
            _logger.error("worker did not exit; preserving ownership run_id=%s", run_id)
            return None
        try:
            return handle_completion(self, run_id, outcome, feature_branch)
        except PreservedWorkerRun as exc:
            _ = self.confirm_preserved_stop(
                NodeLoopOutcome(
                    exc.node_id,
                    False,
                    str(exc),
                    ownership_preserved=True,
                    worker_run_id=exc.run_id,
                    owner_run_id=exc.owner_run_id,
                )
            )
            self._attempts[exc.node_id] = self._attempts.get(exc.node_id, 0) + 1
            if self._strict:
                self._failure_triggered = True
            return 0, 1, []

    def _stop_timed_out_run(self, run_id: str) -> bool:
        node_id = self._active[run_id]
        if not self._executor.force_stop_run(run_id, timeout=10.0):
            _logger.error(
                "worker did not exit after stop; preserving ownership node_id=%d run_id=%s",
                node_id,
                run_id,
            )
            return False
        _ = self._active.pop(run_id)
        self._executor.fail(node_id)
        self._logs.append(f"[{ts()}] ⏱ node {node_id} timeout")
        return True

    def _handle_completion_timeout(self, ct: CompletionTimeout) -> int:
        _logger.warning(
            "Completion timeout after %.1fs; active runs: %s",
            ct.waited_seconds,
            sorted(ct.active_run_ids),
        )
        newly_failed = 0
        for run_id in list(self._active):
            newly_failed += self._stop_timed_out_run(run_id)
        if self._strict:
            self._failure_triggered = True
        return newly_failed
