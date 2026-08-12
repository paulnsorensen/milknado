import os
from pathlib import Path

import pytest

from milknado.domains.common import NodeKind, NodeSpec, NodeStatus
from milknado.domains.graph import MikadoGraph


def test_node_summaries_filter_and_page_in_sql(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    first = graph.add_node("first", parent_id=goal.id, spec=NodeSpec(flavor="implement"))
    second = graph.add_node("second", parent_id=goal.id, spec=NodeSpec(flavor="research"))
    graph.mark_running(first.id)

    queries: list[str] = []
    graph._conn.set_trace_callback(queries.append)
    summaries = graph.get_node_summaries(
        status=NodeStatus.PENDING,
        kind=NodeKind.TASK,
        flavor="research",
        page=(1, 0),
    )
    graph._conn.set_trace_callback(None)

    # Filter the self-heal chokepoint's SELECT 1 health probes (#297); the
    # contract under test is that the domain read is a single SQL query.
    queries = [q for q in queries if q != "SELECT 1"]
    assert summaries == [
        {"id": second.id, "status": NodeStatus.PENDING.value, "description": "second"}
    ]
    assert len(queries) == 1
    assert "LIMIT 1 OFFSET 0" in queries[0]
    graph.close()


@pytest.mark.parametrize("limit", [0, 101])
def test_ready_nodes_reject_unbounded_limits(tmp_path: Path, limit: int) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        graph.get_ready_nodes(limit=limit)
    graph.close()


def test_ready_nodes_filter_kind_and_flavor(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    graph.add_node("implement", parent_id=goal.id, spec=NodeSpec(flavor="implement"))
    research = graph.add_node("research", parent_id=goal.id, spec=NodeSpec(flavor="research"))

    ready = graph.get_ready_nodes(kind=NodeKind.TASK, flavor="research")

    assert [node.id for node in ready] == [research.id]
    graph.close()


@pytest.mark.parametrize(
    ("page", "message"),
    [
        ((0, 0), "limit must be between 1 and 100"),
        ((101, 0), "limit must be between 1 and 100"),
        ((1, -1), "offset must be non-negative"),
    ],
)
def test_node_summaries_reject_invalid_pages(
    tmp_path: Path, page: tuple[int, int], message: str
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with pytest.raises(ValueError, match=message):
        graph.get_node_summaries(page=page)
    graph.close()


def test_next_runnable_returns_first_ready_or_none(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", parent_id=goal.id)
    assert graph.get_ready_nodes(kind=NodeKind.TASK, limit=1) == [task]
    graph.mark_running(task.id)
    graph.mark_done(task.id)
    assert graph.get_ready_nodes(kind=NodeKind.TASK, limit=1) == []
    graph.close()


def test_opening_current_schema_installs_query_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    graph._conn.execute("DROP INDEX idx_nodes_ready")
    graph._conn.execute("DROP INDEX idx_edges_child")
    graph._conn.commit()
    graph.close()

    reopened = MikadoGraph(db_path)
    indexes = {row[1] for row in reopened._conn.execute("PRAGMA index_list(nodes)").fetchall()} | {
        row[1] for row in reopened._conn.execute("PRAGMA index_list(edges)").fetchall()
    }
    assert {"idx_nodes_ready", "idx_edges_child"} <= indexes
    reopened.close()


def test_bulk_node_and_latest_result_queries_preserve_contract(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    first = graph.add_node("first")
    second = graph.add_node("second")
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "owner", os.getpid(), now="now")
    graph.start_run("run-first", first.id, "log", "2026-01-01T00:00:00+00:00", None)
    graph.start_run("run-second", second.id, "log", "2026-01-01T00:00:00+00:00", None)
    graph.deposit_run_message("run-first", "result", "old", "2026-01-01T00:00:01+00:00")
    graph.deposit_run_message("run-first", "result", "new", "2026-01-01T00:00:02+00:00")
    graph.deposit_run_message("run-second", "result", "second", "2026-01-01T00:00:01+00:00")

    loaded = graph.get_nodes([goal.id, second.id, first.id, second.id, 9999])
    results = graph.latest_results_for_nodes([first.id, second.id, 9999])

    assert [(node.id, node.goal_run_id) for node in loaded] == [
        (goal.id, "owner"),
        (second.id, None),
        (first.id, None),
        (second.id, None),
    ]
    assert results == {first.id: "new", second.id: "second"}
    graph.close()
