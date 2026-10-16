"""A sole task retries through the normal scheduler, except under strict stop."""

from __future__ import annotations

from pathlib import Path

import pytest
from typing_extensions import override

from milknado.domains.common import NodeStatus
from milknado.domains.common.config import Gate
from milknado.domains.execution import ExecutionConfig, Executor, RunLoop
from milknado.domains.graph import MikadoGraph
from tests.test_run_loop import FakeCrg, FakeGit, FakeLoop


def _proof_loop(graph: MikadoGraph) -> FakeLoop:
    class ProofLoop(FakeLoop):
        @override
        def start_run(self, run_id: str) -> None:
            row = graph.runs.get(run_id)
            assert row is not None
            node = graph.get_node(row["node_id"])
            assert node is not None and node.run_id is not None
            attempt_id = row["attempt_id"]
            assert attempt_id is not None
            assert graph.task_attempts.confirm_execution(attempt_id, node.run_id, run_id)
            super().start_run(run_id)

    return ProofLoop()


@pytest.mark.parametrize("strict", [False, True])
def test_sole_failed_task_retry_obeys_strict_stop(tmp_path: Path, strict: bool) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    loop_adapter = _proof_loop(graph)
    loop_adapter.set_run_fails("run-1")
    executor = Executor(graph=graph, git=FakeGit(), loop=loop_adapter, crg=FakeCrg())
    run_loop = RunLoop(executor=executor, graph=graph, loop=loop_adapter)
    config = ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate(command="uv run pytest"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
        max_node_attempts=2,
    )
    root = graph.add_node("root")
    task = graph.add_node("sole task", parent_id=root.id)

    result = run_loop.run(config, "main", strict=strict)

    assert result.failed_total == 1
    assert graph.task_attempts.failed_count(task.id) == 1
    if strict:
        assert result.dispatched_total == 1
        assert result.strict_exit is True
        current = graph.get_node(task.id)
        assert current is not None and current.status is NodeStatus.PENDING
    else:
        assert result.dispatched_total == 2
        assert result.completed_total == 1
        current = graph.get_node(task.id)
        assert current is not None and current.status is NodeStatus.DONE
    graph.close()
