from __future__ import annotations

from typing import TYPE_CHECKING, final

from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.run_loop._result import VerifyOutcome
from milknado.domains.execution.run_loop._scheduler import Scheduler

if TYPE_CHECKING:
    from pathlib import Path

    from milknado.domains.common.protocols import LoopPort
    from milknado.domains.execution.executor import ExecutionConfig
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning import Planner


@final
class RunVerifier:
    def __init__(
        self, graph: MikadoGraph, loop: LoopPort, scheduler: Scheduler, planner: Planner | None
    ) -> None:
        self._graph = graph
        self._loop = loop
        self._scheduler = scheduler
        self._planner = planner

    def complete_root_if_settled(self) -> None:
        state = self._scheduler.view()
        if state.failure_triggered or state.active:
            return
        root = self._graph.get_root()
        if root is None:
            return
        if not any(node.id != root.id for node in self._graph.get_all_nodes()):
            return
        _ = self._graph.complete_root()

    def maybe_verify_spec(
        self, spec_text: str | None, spec_path: Path | None, config: ExecutionConfig
    ) -> VerifyOutcome | None:
        state = self._scheduler.view()
        if not spec_text or state.failure_triggered or state.active:
            return None
        root = self._graph.get_root()
        if root is None or root.status == NodeStatus.DONE:
            return None
        non_root_all_done = all(
            node.status == NodeStatus.DONE
            for node in self._graph.get_all_nodes()
            if node.id != root.id
        )
        if not non_root_all_done:
            return None
        result = self._loop.verify_spec(spec_text, str(self._graph))
        outcome = VerifyOutcome(done=result.outcome == "done", goal_delta=result.goal_delta)
        if result.outcome == "done":
            self._graph.mark_running(root.id)
            self._graph.mark_done(root.id)
        elif result.outcome == "gaps" and self._planner and result.goal_delta:
            _ = self._planner.replan_with_delta(result.goal_delta, config.project_root, spec_path)
        return outcome
