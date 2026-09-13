from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from milknado.app.run import ExecutionController, ExecutionRunStatus, ExecutionSnapshot
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_view import summary_text
from milknado.app.watch import (
    AttachedWatchSource,
    WatchSnapshotSource,
    _tail_open_file,  # pyright: ignore[reportPrivateUsage] -- wrapping the tail helper directly to assert its cache-hit count
)
from milknado.domains.common import (
    NodeKind,
    NodeSpec,
    NodeStatus,
    RunResult,
    SessionContext,
    SessionEvent,
    SessionInput,
)
from milknado.domains.execution import get_execution_overview
from milknado.domains.graph import (
    GoalReviewRequest,
    MikadoGraph,
    NodeDetailResponse,
    read_observer_snapshot,
)
from milknado.domains.graph import snapshot as graph_snapshot


def _finish(graph: MikadoGraph, run_id: str, node_id: int) -> None:
    _ = graph.runs.finish(
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
    graph.runs.start("active-run", active.id, str(log_dir), "2026-09-03T12:00:00+00:00", 600)
    assert graph.claim_node(terminal.id, "done-run", now="2026-09-03T12:00:00+00:00")
    graph.runs.start("done-run", terminal.id, str(log_dir), "2026-09-03T12:00:00+00:00", 600)
    _finish(graph, "done-run", terminal.id)
    graph.close()
    return project_root, db_path


def test_watch_reuses_graph_snapshot_until_graph_revision_changes(tmp_path: Path) -> None:
    _, db_path = _observed_runs(tmp_path)
    source = WatchSnapshotSource(tmp_path, db_path)
    with patch(
        "milknado.domains.graph.observer.read_graph_snapshot_connection",
        wraps=graph_snapshot.read_graph_snapshot_connection,
    ) as read_graph:
        first = source.snapshot()
        second = source.snapshot()
        assert first.graph is second.graph
        assert read_graph.call_count == 1

    writer = MikadoGraph(db_path)
    _ = writer.add_node("new graph node")
    writer.close()
    with patch(
        "milknado.domains.graph.observer.read_graph_snapshot_connection",
        wraps=graph_snapshot.read_graph_snapshot_connection,
    ) as read_graph:
        changed = source.snapshot()
        assert read_graph.call_count == 1
    assert changed.graph is not first.graph
    assert changed.graph is not None
    assert any(node.description == "new graph node" for node in changed.graph.nodes)
    source.close()


def test_watch_does_not_rehydrate_large_unchanged_graph(tmp_path: Path) -> None:
    db_path = tmp_path / "large.db"
    graph = MikadoGraph(db_path)
    root = graph.add_node("large graph")
    for index in range(200):
        _ = graph.add_node(f"node {index}", parent_id=root.id)
    graph.close()
    source = WatchSnapshotSource(tmp_path, db_path)
    with patch(
        "milknado.domains.graph.observer.read_graph_snapshot_connection",
        wraps=graph_snapshot.read_graph_snapshot_connection,
    ) as read_graph:
        first = source.snapshot()
        second = source.snapshot()
        assert first.graph is not None
        assert len(first.graph.nodes) == 201
        assert second.graph is first.graph
        assert read_graph.call_count == 1
    source.close()


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
        writer.runs.start(
            "fresh-run",
            node.id,
            str(tmp_path / "missing.log"),
            "2026-09-03T12:00:00+00:00",
            600,
        )
        assert [run.run_id for run in source.snapshot().active_runs] == ["fresh-run"]

    assert observer_query.call_count == 2
    writer.close()


def test_observer_counts_exact_dispatch_availability(tmp_path: Path) -> None:
    db_path = tmp_path / "milknado.db"
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Observe availability", spec=NodeSpec(kind=NodeKind.GOAL))
    active = graph.add_node("active", parent_id=goal.id)
    blocked = graph.add_node("blocked by active", parent_id=goal.id)
    first = graph.add_node("first shared candidate", parent_id=goal.id)
    later = graph.add_node("later shared candidate", parent_id=goal.id)
    _ = graph.add_node("unowned candidate", parent_id=goal.id)
    graph.files.claim(active.id, ["active.py"])
    graph.files.claim(blocked.id, ["active.py"])
    graph.files.claim(first.id, ["shared.py"])
    graph.files.claim(later.id, ["shared.py"])
    graph.mark_running(active.id)

    observed = read_observer_snapshot(db_path)
    _, _, run_available = get_execution_overview(graph, [])

    assert observed.goal == "Observe availability"
    assert observed.available == run_available == 2
    graph.close()


def test_observer_counts_beyond_ready_page_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "milknado.db"
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Large queue", spec=NodeSpec(kind=NodeKind.GOAL))
    for index in range(1001):
        _ = graph.add_node(f"candidate-{index}", parent_id=goal.id)

    observed = read_observer_snapshot(db_path)
    _, _, run_available = get_execution_overview(graph, [])
    assert observed.available == run_available == 1001
    graph.close()


@pytest.mark.parametrize(("review_scope", "expected"), [("unbounded", 0), ("bounded", 1)])
def test_observer_available_matches_execution_admission(
    tmp_path: Path, review_scope: str, expected: int
) -> None:
    db_path = tmp_path / "milknado.db"
    graph = MikadoGraph(db_path)
    goal = graph.add_node("Observe review availability", spec=NodeSpec(kind=NodeKind.GOAL))
    first = graph.add_node("first candidate", parent_id=goal.id)
    later = graph.add_node("later candidate", parent_id=goal.id)
    _ = graph.add_node("unowned candidate", parent_id=goal.id)
    graph.files.claim(first.id, ["shared.py"])
    graph.files.claim(later.id, ["shared.py"])
    affected = None if review_scope == "unbounded" else (first.id, later.id)
    _ = graph.request_goal_review(
        GoalReviewRequest(
            goal_id=goal.id,
            goal_revision="sha256:goal",
            evidence="Review evidence",
            proposed_change="Review proposed change",
            affected_node_ids=affected,
            reviewer="worker",
            assessed_at="2026-09-13T12:00:00+00:00",
        )
    )

    observed = read_observer_snapshot(db_path)
    _, _, run_available = get_execution_overview(graph, [])

    assert observed.available == run_available == expected
    graph.close()


def test_watch_source_assembles_requested_detail_with_graph(tmp_path: Path) -> None:
    db_path = tmp_path / "milknado.db"
    writer = MikadoGraph(db_path)
    node = writer.add_node("Observe detail")
    writer.close()

    source = WatchSnapshotSource(tmp_path, db_path)
    request = NodeSnapshotRequest(node.id, request_generation=7, limit=1)
    snapshot = source.snapshot(request)

    assert snapshot.graph is not None
    assert snapshot.node is not None
    assert snapshot.node.matches(node.id, 7)
    assert snapshot.node.detail is not None
    assert snapshot.node.detail.description == "Observe detail"
    with patch("milknado.app.watch.read_observer_snapshot", side_effect=AssertionError):
        assert source.node_snapshot(request) == snapshot.node


def test_watch_source_forwards_session_event_page(tmp_path: Path) -> None:
    db_path = tmp_path / "milknado.db"
    writer = MikadoGraph(db_path)
    node = writer.add_node("Watch session detail")
    _ = writer.runs.start(
        "watch-run",
        node.id,
        str(tmp_path / "run.log"),
        "2026-09-12T00:00:00+00:00",
        60,
    )
    _ = writer.sessions.start("watch-run", SessionContext(family="codex", cwd=str(tmp_path)))
    _ = writer.sessions.append("watch-run", SessionEvent(kind="status", text="first"))
    _ = writer.sessions.append("watch-run", SessionEvent(kind="status", text="second"))
    writer.close()
    source = WatchSnapshotSource(tmp_path, db_path)

    request = NodeSnapshotRequest(node.id, request_generation=8, limit=1, session_event_page=1)
    snapshot = source.snapshot(request)

    assert snapshot.node is not None and snapshot.node.detail is not None
    sessions = snapshot.node.detail.sessions.items
    assert sessions is not None and sessions[0].event_history.items is not None
    assert tuple(event.text for event in sessions[0].event_history.items) == ("first",)


def test_attached_watch_source_forwards_snapshots_and_admission() -> None:
    snapshot_marker = object()
    detail_marker = object()
    listeners: list[Callable[[ExecutionSnapshot], None]] = []
    commands: list[tuple[str, SessionInput]] = []

    class Source:
        def snapshot(self, request: NodeSnapshotRequest | None = None) -> ExecutionSnapshot:
            del request
            return cast(ExecutionSnapshot, snapshot_marker)

        def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
            del request
            return cast(NodeDetailResponse, detail_marker)

        def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
            listeners.append(listener)
            return lambda: None

    source = AttachedWatchSource(
        Source(), lambda run_id, command: commands.append((run_id, command)) or True
    )
    request = NodeSnapshotRequest(1, request_generation=1)
    command = SessionInput(action="steer", text="redirect")

    assert source.snapshot() is snapshot_marker
    assert source.node_snapshot(request) is detail_marker

    def listener(snapshot: ExecutionSnapshot) -> None:
        del snapshot

    assert source.subscribe(listener) is not None
    assert source.session_input("run-1", command) is True
    assert listeners == [listener]
    assert commands == [("run-1", command)]

    class Controller(Source):
        def session_input(self, run_id: str, value: SessionInput) -> bool:
            return run_id == "run-2" and value.action == "steer"

    fake_controller = cast(ExecutionController, cast(object, Controller()))
    attached = cast(
        AttachedWatchSource, ExecutionController.attached_watch_source(fake_controller)
    )
    assert attached.session_input("run-2", command) is True
