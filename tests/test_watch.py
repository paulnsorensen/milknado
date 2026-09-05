from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from milknado.app.run import ExecutionRunStatus
from milknado.app.run_view import summary_text
from milknado.app.watch import (
    WatchSnapshotSource,
    _tail_open_file,  # pyright: ignore[reportPrivateUsage] -- wrapping the tail helper directly to assert its cache-hit count
)
from milknado.domains.common import NodeKind, NodeSpec, NodeStatus, RunResult
from milknado.domains.graph import MikadoGraph, read_observer_snapshot


def _finish(graph: MikadoGraph, run_id: str, node_id: int) -> None:
    _ = graph.finish_run(
        run_id,
        RunResult(
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at="2026-09-03T12:01:30+00:00",
        ),
    )
    _ = graph.mark_terminal(node_id, run_id, NodeStatus.DONE)


def _observed_runs(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    db_path = project_root / ".milknado" / "milknado.db"
    db_path.parent.mkdir(parents=True)
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Ship observer", spec=NodeSpec(kind=NodeKind.GOAL))
    active = graph.add_node("Watch active work", parent_id=goal.id)
    terminal = graph.add_node("Retain finished work", parent_id=goal.id)
    _ = graph.add_node("Ready next work", parent_id=goal.id)
    log_dir = project_root / ".milknado" / "runs" / "active"
    log_dir.mkdir(parents=True)
    _ = (log_dir / "001.log").write_text("first line\nlatest line\n", encoding="utf-8")
    secret = tmp_path / "secret.log"
    _ = secret.write_text("must not render\n", encoding="utf-8")
    (log_dir / "999.log").symlink_to(secret)
    assert graph.claim_node(active.id, "active-run", now="2026-09-03T12:00:00+00:00")
    graph.start_run("active-run", active.id, str(log_dir), "2026-09-03T12:00:00+00:00", 600)
    assert graph.claim_node(terminal.id, "done-run", now="2026-09-03T12:00:00+00:00")
    graph.start_run("done-run", terminal.id, str(log_dir), "2026-09-03T12:00:00+00:00", 600)
    _finish(graph, "done-run", terminal.id)
    graph.close()
    return project_root, db_path


def test_watch_snapshot_projects_safe_cached_durable_state(tmp_path: Path) -> None:
    project_root, db_path = _observed_runs(tmp_path)

    source = WatchSnapshotSource(project_root, db_path)
    with patch("milknado.app.watch._tail_open_file", wraps=_tail_open_file) as read_tail:
        snapshot = source.snapshot()
        _ = source.snapshot()

    assert read_tail.call_count == 1
    assert snapshot.goal == "Ship observer"
    assert snapshot.available == 1
    assert (snapshot.completed, snapshot.failed, snapshot.stopped) == (1, 0, 0)
    assert [run.run_id for run in snapshot.active_runs] == ["active-run"]
    active_run = snapshot.active_runs[0]
    assert active_run.description == "Watch active work"
    assert active_run.status is ExecutionRunStatus.RUNNING
    assert active_run.output == ("first line", "latest line")
    assert active_run.pending_guidance is None
    assert (active_run.attempt, active_run.max_attempts) == (None, None)
    assert "attempt unavailable" in summary_text(active_run)
    assert "Pending guidance: unavailable" in summary_text(active_run)
    assert active_run.actions.can_cancel is False
    assert [run.run_id for run in snapshot.terminal_runs] == ["done-run"]
    assert snapshot.terminal_runs[0].status is ExecutionRunStatus.COMPLETED
    assert snapshot.terminal_runs[0].pending_guidance is None
    assert snapshot.terminal_runs[0].duration_seconds == 90.0


def test_watch_rejects_log_replaced_by_out_of_root_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, db_path = _observed_runs(tmp_path)
    candidate = project_root / ".milknado" / "runs" / "active" / "001.log"
    secret = tmp_path / "replacement-secret.log"
    _ = secret.write_text("must not render after swap\n", encoding="utf-8")
    real_open = os.open
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    swapped = False

    def swap_then_open(path: Path, flags: int) -> int:
        nonlocal swapped
        if Path(path) == candidate:
            candidate.unlink()
            candidate.symlink_to(secret)
            swapped = True
        return real_open(path, flags)

    source = WatchSnapshotSource(project_root, db_path)
    with patch("milknado.app.watch.os.open", side_effect=swap_then_open):
        snapshot = source.snapshot()

    assert swapped is True
    assert snapshot.active_runs[0].output == ()


def test_watch_snapshot_refresh_uses_read_only_observer_query(tmp_path: Path) -> None:
    db_path = tmp_path / ".milknado" / "milknado.db"
    db_path.parent.mkdir()
    writer = MikadoGraph(db_path)
    node = writer.add_node("Observe commits")
    source = WatchSnapshotSource(tmp_path, db_path)

    with (
        patch.object(MikadoGraph, "__init__", side_effect=AssertionError("writer opened")),
        patch(
            "milknado.app.watch.read_observer_snapshot", wraps=read_observer_snapshot
        ) as observer_query,
    ):
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

    assert observer_query.call_count == 2
    writer.close()


def test_observer_counts_exact_dispatch_availability_without_conflict_pairs(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "milknado.db"
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Observe availability", spec=NodeSpec(kind=NodeKind.GOAL))
    active = graph.add_node("active", parent_id=goal.id)
    blocked = graph.add_node("blocked by active", parent_id=goal.id)
    first = graph.add_node("first shared candidate", parent_id=goal.id)
    later = graph.add_node("later shared candidate", parent_id=goal.id)
    _ = graph.add_node("unowned candidate", parent_id=goal.id)
    graph.set_file_ownership(active.id, ["active.py"])
    graph.set_file_ownership(blocked.id, ["active.py"])
    graph.set_file_ownership(first.id, ["shared.py"])
    graph.set_file_ownership(later.id, ["shared.py"])
    graph.mark_running(active.id)

    with (
        patch(
            "milknado.domains.graph._reads.get_ready_nodes",
            side_effect=AssertionError("materialized ready nodes"),
        ),
        patch(
            "milknado.domains.graph._persistence.check_parallel_safety",
            side_effect=AssertionError("materialized conflict pairs"),
        ),
    ):
        observed = read_observer_snapshot(db_path)

    assert observed.goal == "Observe availability"
    assert observed.available == 2
    graph.close()
