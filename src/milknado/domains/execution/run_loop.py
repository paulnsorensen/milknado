from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import get_dispatchable_nodes

if TYPE_CHECKING:
    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph.graph import MikadoGraph


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int


class RunLoop:
    def __init__(
        self,
        executor: Executor,
        graph: MikadoGraph,
        ralph: RalphPort,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._ralph = ralph
        self._active: dict[str, int] = {}

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
        poll_interval: float = 2.0,
    ) -> RunLoopResult:
        dispatched_total = 0
        completed_total = 0
        failed_total = 0

        dispatched_total += self._dispatch_batch(config, concurrency_limit)

        while self._active:
            time.sleep(poll_interval)
            completed, failed = self._check_completions(
                feature_branch,
            )
            completed_total += completed
            failed_total += failed
            dispatched_total += self._dispatch_batch(
                config, concurrency_limit,
            )

        root = self._graph.get_root()
        root_done = root is not None and root.status == NodeStatus.DONE
        return RunLoopResult(
            root_done=root_done,
            dispatched_total=dispatched_total,
            completed_total=completed_total,
            failed_total=failed_total,
        )

    def _dispatch_batch(
        self, config: ExecutionConfig, concurrency_limit: int,
    ) -> int:
        available = concurrency_limit - len(self._active)
        if available <= 0:
            return 0
        dispatchable = get_dispatchable_nodes(self._graph)
        batch = dispatchable[:available]
        for node_id in batch:
            result = self._executor.dispatch(node_id, config)
            self._active[result.run_id] = node_id
        return len(batch)

    def _check_completions(
        self, feature_branch: str,
    ) -> tuple[int, int]:
        completed = 0
        failed = 0
        done_runs: list[str] = []
        for run_id in self._active:
            if self._ralph.is_run_complete(run_id):
                done_runs.append(run_id)

        for run_id in done_runs:
            node_id = self._active.pop(run_id)
            if self._ralph.is_run_success(run_id):
                self._executor.complete(node_id, feature_branch)
                completed += 1
            else:
                self._executor.fail(node_id)
                failed += 1

        return completed, failed
