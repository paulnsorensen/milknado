from __future__ import annotations

from pathlib import Path

from milknado.app.run import ExecutionRunStatus
from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import NodeKind, NodeSpec, NodeStatus, RunResult
from milknado.domains.graph import MikadoGraph


def _finish(graph: MikadoGraph, run_id: str, node_id: int) -> None:
    graph.finish_run(
        run_id,
        RunResult(
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-09-03T12:01:30+00:00",
        ),
    )
    graph.mark_terminal(node_id, run_id, NodeStatus.DONE)


def test_watch_snapshot_projects_durable_runs_and_log_tails(tmp_path: Path) -> None:
    db_path = tmp_path / ".milknado" / "milknado.db"
    db_path.parent.mkdir()
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Ship observer", spec=NodeSpec(kind=NodeKind.GOAL))
    active = graph.add_node("Watch active work", parent_id=goal.id)
    terminal = graph.add_node("Retain finished work", parent_id=goal.id)
    graph.add_node("Ready next work", parent_id=goal.id)

    active_log = tmp_path / ".milknado" / "runs" / "active.log"
    active_log.parent.mkdir()
    active_log.write_text("first line\nlatest line\n", encoding="utf-8")
    assert graph.claim_node(
        active.id,
        "active-run",
        now="2026-09-03T12:00:00+00:00",
    )
    graph.start_run(
        "active-run",
        active.id,
        str(active_log),
        "2026-09-03T12:00:00+00:00",
        600,
    )

    assert graph.claim_node(
        terminal.id,
        "done-run",
        now="2026-09-03T12:00:00+00:00",
    )
    graph.start_run(
        "done-run",
        terminal.id,
        str(active_log),
        "2026-09-03T12:00:00+00:00",
        600,
    )
    _finish(graph, "done-run", terminal.id)
    graph.close()

    snapshot = WatchSnapshotSource(tmp_path, db_path).snapshot()

    assert snapshot.goal == "Ship observer"
    assert snapshot.available == 1
    assert snapshot.completed == 1
    assert snapshot.failed == 0
    assert snapshot.stopped == 0
    assert [run.run_id for run in snapshot.active_runs] == ["active-run"]
    assert snapshot.active_runs[0].description == "Watch active work"
    assert snapshot.active_runs[0].status is ExecutionRunStatus.RUNNING
    assert snapshot.active_runs[0].output == ("first line", "latest line")
    assert snapshot.active_runs[0].actions.can_cancel is False
    assert [run.run_id for run in snapshot.terminal_runs] == ["done-run"]
    assert snapshot.terminal_runs[0].status is ExecutionRunStatus.COMPLETED
    assert snapshot.terminal_runs[0].duration_seconds == 90.0


def test_watch_snapshot_refresh_sees_new_committed_runs(tmp_path: Path) -> None:
    db_path = tmp_path / ".milknado" / "milknado.db"
    db_path.parent.mkdir()
    writer = MikadoGraph(db_path)
    node = writer.add_node("Observe commits")
    source = WatchSnapshotSource(tmp_path, db_path)

    assert source.snapshot().active_runs == ()

    assert writer.claim_node(node.id, "fresh-run", now="2026-09-03T12:00:00+00:00")
    writer.start_run(
        "fresh-run",
        node.id,
        str(tmp_path / "missing.log"),
        "2026-09-03T12:00:00+00:00",
        600,
    )

    assert [run.run_id for run in source.snapshot().active_runs] == ["fresh-run"]
    writer.close()
