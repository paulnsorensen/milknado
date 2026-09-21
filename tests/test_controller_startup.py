from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV, NodeKind, NodeSpec
from milknado.domains.graph import (
    ControllerAuthorizationError,
    GoalReviewDecision,
    GoalReviewDecisionRequest,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.domains.graph import controller_capability as capability


def _review(graph: MikadoGraph, goal_id: int, revision: str) -> int:
    """Request a goal review and return its identifier."""
    record = graph.request_goal_review(
        GoalReviewRequest(
            goal_id=goal_id,
            goal_revision=revision,
            evidence="controller evidence",
            proposed_change=f"change {revision}",
            reviewer="worker",
        )
    )
    return record.review_id


def _child(
    db_path: Path, state_home: Path, code: str, *args: object
) -> subprocess.CompletedProcess[str]:
    """Run isolated controller code against a graph and managed state root."""
    env = os.environ.copy()
    _ = env.pop(CONTROLLER_MASTER_ENV, None)
    env["XDG_STATE_HOME"] = str(state_home)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(db_path), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_concurrent_registration_converges_and_restarts_in_a_new_process(tmp_path: Path) -> None:
    """Converge concurrent registration on one credential reusable after restart."""
    db_path = tmp_path / "graph.db"
    state_home = tmp_path / "state"
    graph = MikadoGraph(db_path)
    graph.close()
    code = """
        import hashlib
        import sys
        from pathlib import Path
        from milknado.domains.graph import MikadoGraph
        graph = MikadoGraph(Path(sys.argv[1]))
        try:
            print(hashlib.sha256(graph._controller_master or b'').hexdigest())
            graph.register_controller_master()
            print(hashlib.sha256(graph._controller_master or b'').hexdigest())
        finally:
            graph.close()
    """

    def launch(_index: int) -> subprocess.CompletedProcess[str]:
        """Launch one registration contender."""
        return _child(db_path, state_home, code)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch, range(4)))

    assert all(result.returncode == 0 for result in results), results
    hashes = [result.stdout.splitlines()[-1] for result in results]
    assert len(set(hashes)) == 1
    restart = _child(db_path, state_home, code)
    assert restart.returncode == 0, restart.stderr
    assert restart.stdout.splitlines()[-1] == hashes[0]
    records = list((state_home / "milknado" / "controllers").iterdir())
    assert len(records) == 1
    assert hashlib.sha256(records[0].read_bytes()).hexdigest() == hashes[0]


def test_separate_operator_process_approves_with_managed_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow a separate operator process to reuse managed authority."""
    db_path = tmp_path / "graph.db"
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "operator-secret")
    graph = MikadoGraph(db_path)
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    review_id = _review(graph, goal.id, "revision-1")
    graph.register_controller_master()
    graph.close()
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    code = """
        import sys
        from pathlib import Path
        from milknado.domains.graph import (
            GoalReviewDecision,
            GoalReviewDecisionRequest,
            MikadoGraph,
        )
        graph = MikadoGraph(Path(sys.argv[1]))
        try:
            graph.register_controller_master()
            record = graph.decide_goal_review(
                GoalReviewDecisionRequest(int(sys.argv[2]), GoalReviewDecision.ACCEPTED),
                decided_by="second-terminal",
            )
            print(record.decided_by)
        finally:
            graph.close()
    """
    result = _child(db_path, state_home, code, review_id)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "second-terminal"
    graph = MikadoGraph(db_path)
    try:
        record = graph.get_goal_review(review_id)
        assert record is not None
        assert record.decision is GoalReviewDecision.ACCEPTED
    finally:
        graph.close()


def test_matching_legacy_import_preserves_nodes_reviews_and_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import the matching legacy secret without changing existing graph data."""
    db_path = tmp_path / "graph.db"
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "legacy-secret")
    graph = MikadoGraph(db_path)
    goal = graph.add_node("legacy goal", spec=NodeSpec(kind=NodeKind.GOAL))
    decided_id = _review(graph, goal.id, "decided")
    graph.register_controller_master()
    _ = graph.decide_goal_review(
        GoalReviewDecisionRequest(decided_id, GoalReviewDecision.REJECTED), decided_by="legacy"
    )
    pending_id = _review(graph, goal.id, "pending")
    graph.close()
    record_path = next((state_home / "milknado" / "controllers").iterdir())
    record_path.unlink()
    graph = MikadoGraph(db_path)
    graph.register_controller_master()
    try:
        assert graph.get_node(goal.id) is not None
        decided = graph.get_goal_review(decided_id)
        pending = graph.get_goal_review(pending_id)
        assert decided is not None and decided.decision is GoalReviewDecision.REJECTED
        assert pending is not None and pending.decision is GoalReviewDecision.PENDING
    finally:
        graph.close()
    with sqlite3.connect(db_path) as conn:
        stored_hash = cast(
            tuple[str], conn.execute("SELECT master_hash FROM controller_master").fetchone()
        )
        consumed = cast(
            tuple[int],
            conn.execute("SELECT COUNT(*) FROM consumed_controller_capabilities").fetchone(),
        )
    assert stored_hash == (hashlib.sha256(b"legacy-secret").hexdigest(),)
    assert consumed == (1,)


def test_storage_failure_rolls_back_registration_without_graph_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Roll back registration when credential publication fails."""
    db_path = tmp_path / "graph.db"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "rollback-secret")
    graph = MikadoGraph(db_path)
    goal = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL))
    review_id = _review(graph, goal.id, "revision-1")
    graph.close()
    graph = MikadoGraph(db_path)

    def fail_publish(*_args: object) -> None:
        """Simulate a credential publication failure."""
        raise ControllerAuthorizationError("publish failed")

    monkeypatch.setattr(capability, "_publish_if_missing", fail_publish)

    with pytest.raises(ControllerAuthorizationError, match="publish failed"):
        graph.register_controller_master()
    graph.close()
    with sqlite3.connect(db_path) as conn:
        registration = cast(
            tuple[int], conn.execute("SELECT COUNT(*) FROM controller_master").fetchone()
        )
        reviews = cast(tuple[int], conn.execute("SELECT COUNT(*) FROM goal_reviews").fetchone())
        nodes = cast(tuple[int], conn.execute("SELECT COUNT(*) FROM nodes").fetchone())
    assert registration == (0,)
    assert reviews == (1,)
    assert nodes == (1,)
    assert review_id > 0
