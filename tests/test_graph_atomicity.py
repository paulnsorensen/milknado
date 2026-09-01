from __future__ import annotations

import os
import sqlite3

# These tests intentionally exercise protected graph internals.
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import NoReturn, Protocol, cast

import pytest

from milknado.domains.common import (
    GraphExecutionSnapshot,
    GraphReadPort,
    MikadoNode,
    NodeKind,
    NodeSpec,
)
from milknado.domains.execution import get_execution_overview
from milknado.domains.graph import MikadoGraph
from tests.graph_helpers import graph_conn, graph_lock


def _raise_probe_failure(_pid: int) -> NoReturn:
    raise RuntimeError("probe failed")


class _OwnableLock(Protocol):
    def _is_owned(self) -> bool: ...


def _lock_owned(lock: object) -> bool:
    return cast(_OwnableLock, lock)._is_owned()  # pyright: ignore[reportPrivateUsage]


def _run_pair(operation: Callable[[str], None]) -> list[tuple[type[Exception] | None, str]]:
    barrier = Barrier(2)

    def run(value: str) -> tuple[type[Exception] | None, str]:
        _ = barrier.wait()
        try:
            operation(value)
        except Exception as exc:  # returned for exact losing-writer assertions
            return type(exc), str(exc)
        return None, "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(run, ("left", "right")))


def test_concurrent_add_node_loser_leaves_no_partial_node_or_edges(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    setup = MikadoGraph(db_path)
    goal = setup.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    prerequisite = setup.add_node("prerequisite", parent_id=goal.id)
    setup.close()

    def add(label: str) -> None:
        graph = MikadoGraph(db_path)
        try:
            _ = graph.add_node(
                f"candidate-{label}",
                parent_id=goal.id,
                spec=NodeSpec(wiki_ref="unique-ref", prereqs=(prerequisite.id,)),
            )
        finally:
            graph.close()

    results = _run_pair(add)
    assert sorted(result[1] == "ok" for result in results) == [False, True]
    assert {result[0] for result in results if result[0] is not None} == {sqlite3.IntegrityError}

    graph = MikadoGraph(db_path)
    candidates = [node for node in graph.get_all_nodes() if node.wiki_ref == "unique-ref"]
    assert len(candidates) == 1
    candidate = candidates[0]
    edges = cast(
        list[tuple[int, int]],
        graph_conn(graph)
        .execute(
            """SELECT parent_id, child_id FROM edges
            WHERE parent_id = ? OR child_id = ?
            ORDER BY parent_id, child_id""",
            (candidate.id, candidate.id),
        )
        .fetchall(),
    )
    assert {tuple(edge) for edge in edges} == {
        (goal.id, candidate.id),
        (candidate.id, prerequisite.id),
    }
    graph.close()


def test_concurrent_opposite_edges_cannot_commit_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    setup = MikadoGraph(db_path)
    left = setup.add_node("left")
    right = setup.add_node("right")
    setup.close()

    def add(label: str) -> None:
        graph = MikadoGraph(db_path)
        try:
            edge = (left.id, right.id) if label == "left" else (right.id, left.id)
            _ = graph.add_edge(*edge)
        finally:
            graph.close()

    results = _run_pair(add)
    assert sorted(result[1] == "ok" for result in results) == [False, True]
    assert {result[0] for result in results if result[0] is not None} == {ValueError}
    graph = MikadoGraph(db_path)
    assert len(graph_conn(graph).execute("SELECT 1 FROM edges").fetchall()) == 1
    graph.close()


def test_concurrent_goal_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    setup = MikadoGraph(db_path)
    goal = setup.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    setup.close()

    def claim(owner: str) -> None:
        graph = MikadoGraph(db_path)
        try:
            if not graph.claim_or_reclaim_goal(goal.id, owner, os.getpid(), now=owner):
                raise RuntimeError("lost")
        finally:
            graph.close()

    results = _run_pair(claim)
    assert sorted(result[1] == "ok" for result in results) == [False, True]
    assert {result for result in results if result[1] != "ok"} == {(RuntimeError, "lost")}
    graph = MikadoGraph(db_path)
    claimed = graph.get_node(goal.id)
    assert claimed is not None
    assert claimed.goal_run_id in {"left", "right"}
    claim_pid = cast(
        int,
        graph_conn(graph)
        .execute("SELECT pid FROM goal_claims WHERE goal_id = ?", (goal.id,))
        .fetchone()[0],
    )
    assert claim_pid == os.getpid()
    graph.close()


def test_missing_goal_claim_rolls_back_before_next_claim(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    with pytest.raises(ValueError, match=r"node 999 not found"):
        _ = graph.claim_or_reclaim_goal(999, "owner", os.getpid(), now="missing")

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "owner", os.getpid(), now="claimed")
    claimed = graph.get_node(goal.id)
    assert claimed is not None
    assert claimed.goal_run_id == "owner"
    graph.close()


def test_goal_claim_atomic_reclaim_requires_dead_pid(graph: MikadoGraph) -> None:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    now = "2026-01-01T00:00:00+00:00"
    assert graph.claim_or_reclaim_goal(goal.id, "run-a", 2**31 - 1, now=now)
    assert graph.claim_or_reclaim_goal(goal.id, "run-a", 2**31 - 1, now=now) is False
    assert graph.claim_or_reclaim_goal(goal.id, "run-b", now=now, pid=2**31 - 1)
    from milknado.domains.graph import _goal_claims

    claim = _goal_claims.get_goal_claim(graph_conn(graph), goal.id)
    assert claim is not None
    assert claim["run_id"] == "run-b"


def test_goal_claim_keeps_null_pid_claim_blocking(graph: MikadoGraph) -> None:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "run-a", now="2026-01-01T00:00:00+00:00")
    assert (
        graph.claim_or_reclaim_goal(goal.id, "run-b", now="2026-01-01T00:00:00+00:00", pid=123)
        is False
    )


def test_goal_claim_reclaim_surfaces_pid_liveness_failure(
    graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "run-a", 42, now="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(
        "milknado.domains.graph._goal_claims.pid_alive",
        _raise_probe_failure,
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        _ = graph.claim_or_reclaim_goal(goal.id, "run-b", now="2026-01-01T00:00:00+00:00", pid=123)


def test_goal_claim_row_handles_missing_row_and_dead_owner(graph: MikadoGraph) -> None:
    from milknado.domains.graph import _goal_claims

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "dead", 2**31 - 1, now="now")
    assert graph.claim_or_reclaim_goal(goal.id, "new", 2**31 - 1, now="now")

    class Cursor:
        rowcount: int = 0

        def fetchone(self) -> None:
            return None

    class Connection:
        def execute(self, _sql: str, _params: tuple[object, ...] = ()) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    assert _goal_claims.claim_goal_row(Connection(), 1, "run", "now", pid=123) is False


def test_goal_claim_or_reclaim_handles_invalid_and_missing_claims(graph: MikadoGraph) -> None:
    from milknado.domains.graph import _goal_claims

    unclaimed = _goal_claims.claim_or_reclaim_goal(graph_conn(graph), 1, "run", None, now="now")
    assert unclaimed is False
    task = graph.add_node("task")
    with pytest.raises(ValueError, match="only goal nodes"):
        _ = _goal_claims.claim_or_reclaim_goal(graph_conn(graph), task.id, "run", 123, now="now")

    class Cursor:
        def __init__(self, rowcount: int = 0, row: dict[str, str] | None = None) -> None:
            self.rowcount: int = rowcount
            self._row: dict[str, str] | None = row

        def fetchone(self) -> dict[str, str] | None:
            return self._row

    class Connection:
        def execute(self, sql: str, _params: tuple[object, ...] = ()) -> Cursor:
            if sql.startswith("SELECT kind"):
                return Cursor(row={"kind": "goal"})
            if sql.startswith("SELECT run_id"):
                return Cursor(row=None)
            return Cursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    assert _goal_claims.claim_or_reclaim_goal(Connection(), 1, "run", 123, now="now") is False


def test_goal_claim_or_reclaim_same_owner_and_try_reclaim_error(
    graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.graph import _goal_claims

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "run", 123, now="now")
    assert graph.claim_or_reclaim_goal(goal.id, "run", 123, now="now") is False
    monkeypatch.setattr(_goal_claims, "pid_alive", _raise_probe_failure)
    with pytest.raises(RuntimeError, match="probe failed"):
        _ = graph.try_reclaim_goal(goal.id, now="now")


def test_ancestor_claim_refuses_foreign_pid(graph: MikadoGraph) -> None:
    import os

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", parent_id=goal.id)
    assert graph.claim_or_reclaim_goal(goal.id, "run-a", os.getpid(), now="now")
    with pytest.raises(ValueError, match="ancestor goal"):
        _ = graph.claim_ancestor_goal(task.id, "run-b", 456, now="now")


def test_execution_overview_returns_one_public_atomic_projection(graph: MikadoGraph) -> None:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    first = graph.add_node("first", parent_id=goal.id)
    second = graph.add_node("second", parent_id=goal.id)

    assert get_execution_overview(graph, [goal.id, first.id, 999]) == (
        "goal",
        {goal.id: "goal", first.id: "first"},
        2,
    )
    assert get_execution_overview(graph, [second.id], {first.id}) == (
        "goal",
        {second.id: "second"},
        1,
    )
    assert not hasattr(graph, "get_execution_overview")


def test_execution_overview_accepts_graph_read_port(graph: MikadoGraph) -> None:
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    node = graph.add_node("task", parent_id=goal.id)
    snapshot = graph.get_execution_snapshot([node.id])

    class ReadPort:
        def get_node(self, node_id: int) -> MikadoNode | None:  # pyright: ignore[reportUnusedParameter]
            raise NotImplementedError

        def get_children(self, node_id: int) -> list[MikadoNode]:  # pyright: ignore[reportUnusedParameter]
            raise NotImplementedError

        def get_file_ownership(self, node_id: int) -> list[str]:  # pyright: ignore[reportUnusedParameter]
            raise NotImplementedError

        def get_execution_snapshot(self, node_ids: list[int]) -> GraphExecutionSnapshot:
            assert node_ids == [node.id]
            return snapshot

    read_port: GraphReadPort = ReadPort()
    assert get_execution_overview(read_port, [node.id]) == (
        "goal",
        {node.id: "task"},
        1,
    )


def test_execution_overview_reads_one_lock_held_graph_snapshot(
    graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.graph import _reads

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    task = graph.add_node("task", parent_id=goal.id)
    original_get_root = _reads.get_root
    original_get_nodes = _reads.get_nodes
    observed: list[tuple[str, bool]] = []

    def locked_get_root(conn: sqlite3.Connection) -> MikadoNode | None:
        observed.append(("root", _lock_owned(graph_lock(graph))))
        return original_get_root(conn)

    def locked_get_nodes(conn: sqlite3.Connection, node_ids: Iterable[int]) -> list[MikadoNode]:
        observed.append(("nodes", _lock_owned(graph_lock(graph))))
        return original_get_nodes(conn, node_ids)

    monkeypatch.setattr(_reads, "get_root", locked_get_root)
    monkeypatch.setattr(_reads, "get_nodes", locked_get_nodes)

    assert get_execution_overview(graph, [task.id]) == (
        "goal",
        {task.id: "task"},
        1,
    )
    assert observed == [("root", True), ("nodes", True)]


def test_execution_overview_uses_one_cross_connection_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.graph import _reads

    db_path = tmp_path / "overview.db"
    reader = MikadoGraph(db_path)
    writer = MikadoGraph(db_path)
    try:
        goal = reader.add_node("before", spec=NodeSpec(kind=NodeKind.GOAL))
        task = reader.add_node("task", parent_id=goal.id)
        original_get_nodes = _reads.get_nodes

        def update_between_reads(
            conn: sqlite3.Connection, node_ids: Iterable[int]
        ) -> list[MikadoNode]:
            writer.update_node(goal.id, description="after")
            _ = writer.add_node("new task", parent_id=goal.id)
            return original_get_nodes(conn, node_ids)

        monkeypatch.setattr(_reads, "get_nodes", update_between_reads)

        assert get_execution_overview(reader, [goal.id, task.id]) == (
            "before",
            {goal.id: "before", task.id: "task"},
            1,
        )
        updated_root = writer.get_root()
        assert updated_root is not None
        assert updated_root.description == "after"
    finally:
        reader.close()
        writer.close()


def test_inherited_analytics_operations_hold_graph_lock(
    graph: MikadoGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.graph import _analytics_facade

    def locked_get_spec_hash(_conn: sqlite3.Connection) -> str:
        assert _lock_owned(graph_lock(graph))
        return "locked"

    monkeypatch.setattr(_analytics_facade, "get_spec_hash", locked_get_spec_hash)

    assert graph.get_spec_hash() == "locked"
