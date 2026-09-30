from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import msgspec
import pytest

from milknado.domains.common import NodeKind, NodeSpec, NodeStatus, default_config
from milknado.domains.graph import ConcurrencyLimitReached, MikadoGraph
from milknado.project import open_project_graph

_NOW = "2026-01-01T00:00:00+00:00"


def test_independent_connections_admit_one_task_at_capacity_one(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    with closing(MikadoGraph(db_path, concurrency_limit=1)) as graph:
        first = graph.add_node("first").id
        second = graph.add_node("second").id

    barrier = Barrier(2)

    def claim(node_id: int) -> bool:
        with closing(MikadoGraph(db_path, concurrency_limit=1)) as graph:
            _ = barrier.wait(timeout=5)
            try:
                return graph.claim_node(node_id, f"run-{node_id}", now=_NOW)
            except ConcurrencyLimitReached as exc:
                assert (exc.running, exc.limit) == (1, 1)
                return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, (first, second)))

    assert outcomes.count(True) == 1
    with closing(MikadoGraph(db_path, concurrency_limit=1)) as graph:
        nodes = [graph.get_node(node_id) for node_id in (first, second)]
        assert sum(node is not None and node.status is NodeStatus.RUNNING for node in nodes) == 1
        assert sum(node is not None and node.status is NodeStatus.PENDING for node in nodes) == 1
        winner, loser = (first, second) if outcomes[0] else (second, first)
        assert graph.release(winner, f"run-{winner}")
        assert graph.claim_node(loser, "next", now=_NOW)
        assert graph.mark_terminal(loser, "next", NodeStatus.DONE)
        assert graph.claim_node(winner, "last", now=_NOW)


def test_run_rows_and_unowned_running_nodes_do_not_consume_extra_slots(tmp_path: Path) -> None:
    with closing(MikadoGraph(tmp_path / "graph.db", concurrency_limit=2)) as graph:
        first = graph.add_node("first").id
        second = graph.add_node("second").id
        third = graph.add_node("third").id
        graph.mark_running(third)
        assert graph.claim_node(first, "parent", now=_NOW)
        graph.runs.start("parent", first, "parent.log", _NOW, None)
        graph.runs.start("worker", first, "worker.log", _NOW, None)
        assert graph.claim_node(second, "second", now=_NOW)


def test_running_goal_does_not_consume_task_capacity(tmp_path: Path) -> None:
    with closing(MikadoGraph(tmp_path / "graph.db", concurrency_limit=1)) as graph:
        goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
        graph.mark_running(goal.id)
        task = graph.add_node("task", parent_id=goal.id)
        assert graph.claim_node(task.id, "task", now=_NOW)


def test_capacity_refusal_releases_new_ancestor_claim(tmp_path: Path) -> None:
    with closing(MikadoGraph(tmp_path / "graph.db", concurrency_limit=1)) as graph:
        busy = graph.add_node("busy").id
        goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
        waiting = graph.add_node("waiting", parent_id=goal.id)
        assert graph.claim_node(busy, "busy", now=_NOW)
        with pytest.raises(ConcurrencyLimitReached):
            graph.claim_node_for_dispatch(waiting.id, "waiting", now=_NOW)
        current = graph.get_node(waiting.id)
        assert current is not None
        assert current.status is NodeStatus.PENDING
        assert current.run_id is None
        current_goal = graph.get_node(goal.id)
        assert current_goal is not None
        assert current_goal.goal_run_id is None


def test_project_open_uses_configured_limit(tmp_path: Path) -> None:
    config = msgspec.structs.replace(default_config(tmp_path), concurrency_limit=1)
    with closing(open_project_graph(config)) as graph:
        first = graph.add_node("first").id
        second = graph.add_node("second").id
        assert graph.claim_node(first, "first", now=_NOW)
        with pytest.raises(ConcurrencyLimitReached):
            _ = graph.claim_node(second, "second", now=_NOW)


def test_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        _ = MikadoGraph(tmp_path / "graph.db", concurrency_limit=0)
