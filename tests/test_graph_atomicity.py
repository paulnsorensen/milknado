from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import MikadoGraph


def _run_pair(operation):
    barrier = Barrier(2)

    def run(value: str):
        barrier.wait()
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
            graph.add_node(
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
    edges = graph._conn.execute(
        "SELECT parent_id, child_id FROM edges "
        "WHERE parent_id = ? OR child_id = ? ORDER BY parent_id, child_id",
        (candidate.id, candidate.id),
    ).fetchall()
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
            graph.add_edge(*edge)
        finally:
            graph.close()

    results = _run_pair(add)
    assert sorted(result[1] == "ok" for result in results) == [False, True]
    assert {result[0] for result in results if result[0] is not None} == {ValueError}
    graph = MikadoGraph(db_path)
    assert len(graph._conn.execute("SELECT 1 FROM edges").fetchall()) == 1
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
    claim_pid = graph._conn.execute(
        "SELECT pid FROM goal_claims WHERE goal_id = ?", (goal.id,)
    ).fetchone()[0]
    assert claim_pid == os.getpid()
    graph.close()


def test_missing_goal_claim_rolls_back_before_next_claim(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")

    with pytest.raises(ValueError, match=r"node 999 not found"):
        graph.claim_or_reclaim_goal(999, "owner", os.getpid(), now="missing")

    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    assert graph.claim_or_reclaim_goal(goal.id, "owner", os.getpid(), now="claimed")
    claimed = graph.get_node(goal.id)
    assert claimed is not None
    assert claimed.goal_run_id == "owner"
    graph.close()
