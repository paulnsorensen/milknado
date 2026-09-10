from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import pytest

from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import (
    NodeStatus,
    RunResult,
    SessionContext,
    SessionEvent,
)
from milknado.domains.graph import MikadoGraph, read_observer_snapshot
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session
from tests.graph_helpers import graph_conn


def _run(graph: MikadoGraph, run_id: str = "run-1") -> int:
    node = graph.add_node("structured session")
    assert graph.claim_node(node.id, run_id, now="2026-09-09T10:00:00+00:00")
    graph.runs.start(run_id, node.id, "missing.log", "2026-09-09T10:00:00+00:00", 600)
    return node.id


def test_session_start_append_and_view_normalize_input_trace(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "session.db")
    _ = _run(graph)
    context = SessionContext(family="omp", cwd=str(tmp_path), base_oid="base-1")

    graph.sessions.start("run-1", context)
    graph.sessions.append(
        "run-1", SessionEvent(kind="user", event_id="request-1", text="ship it", state="queued")
    )
    graph.sessions.append(
        "run-1", SessionEvent(kind="user", event_id="request-1", text="", state="submitted")
    )
    graph.sessions.append(
        "run-1", SessionEvent(kind="user", event_id="request-1", text="", state="delivered")
    )
    graph.sessions.append(
        "run-1", SessionEvent(kind="assistant", event_id="reply-1", text="hello ", delta=True)
    )
    graph.sessions.append(
        "run-1", SessionEvent(kind="assistant", event_id="reply-1", text="world", delta=True)
    )

    view = graph.sessions.view("run-1")
    assert view.context == context
    assert view.active is True
    assert view.events == (
        SessionEvent(kind="user", event_id="request-1", text="ship it", state="delivered"),
        SessionEvent(kind="assistant", event_id="reply-1", text="hello world"),
    )
    rows: list[tuple[str]] = (
        graph_conn(graph)
        .execute(
            "SELECT body FROM run_messages WHERE run_id = ? AND role = 'session' ORDER BY seq",
            ("run-1",),
        )
        .fetchall()
    )
    persisted = [msgspec.json.decode(row[0], type=SessionEvent) for row in rows]
    assert [event.state for event in persisted[:3]] == ["queued", "submitted", "delivered"]
    assert all(event.delta is False for event in persisted)
    graph.close()


_LARGE_PROMPT_WORKER = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('type') == 'user':
        Path('received.txt').write_text(frame['message']['content'], encoding='utf-8')
        result = {'type': 'result', 'subtype': 'success', 'result': 'complete'}
        print(json.dumps(result), flush=True)
        break
"""


def test_large_prompt_reaches_worker_with_bounded_durable_transcript(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "large-prompt.db")
    _ = _run(graph)
    graph.sessions.start("run-1", SessionContext(family="claude", cwd=str(tmp_path)))
    worker = tmp_path / "claude"
    _ = worker.write_text(
        _LARGE_PROMPT_WORKER,
        encoding="utf-8",
    )
    worker.chmod(0o755)
    prompt = "prefix" + "\n界\\" * 70000
    channel = SessionChannel(lambda event: graph.sessions.append("run-1", event))
    try:
        result = run_session(
            AgentRunSpec(
                cmd=[str(worker)],
                prompt=prompt,
                timeout=5.0,
                cwd=tmp_path,
                log_dir=None,
                iteration=1,
                capture_result_text=True,
            ),
            channel,
        )
        assert result.returncode == 0
        assert result.result_text == "complete"
        assert (tmp_path / "received.txt").read_text(encoding="utf-8") == prompt
        recorded = next(
            event for event in graph.sessions.view("run-1").events if event.kind == "user"
        )
        assert len(recorded.text) <= 8192
        assert recorded.text.endswith(prompt[-100:])
    finally:
        graph.close()


def test_session_append_sequences_across_graph_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    graph = MikadoGraph(db_path)
    _ = _run(graph)
    graph.sessions.start("run-1", SessionContext(family="codex", cwd=str(tmp_path)))
    graph.close()

    def append(index: int) -> int:
        worker = MikadoGraph(db_path)
        try:
            worker.sessions.append(
                "run-1",
                SessionEvent(kind="tool", event_id=f"tool-{index}", text=str(index)),
            )
            return index
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences = list(pool.map(append, range(32)))

    assert sorted(sequences) == list(range(32))
    reader = MikadoGraph(db_path)
    rows: list[tuple[int]] = (
        graph_conn(reader)
        .execute(
            "SELECT seq FROM run_messages WHERE run_id = ? AND role = 'session' ORDER BY seq",
            ("run-1",),
        )
        .fetchall()
    )
    assert [row[0] for row in rows] == list(range(1, 33))
    reader.close()


def test_session_view_is_bounded_and_isolated(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "bounded.db")
    _ = _run(graph, "run-a")
    second = graph.add_node("other session")
    graph.runs.start("run-b", second.id, "missing.log", "2026-09-09T10:00:00+00:00", 600)
    graph.sessions.start("run-a", SessionContext(family="omp", cwd="/a"))
    graph.sessions.start("run-b", SessionContext(family="claude", cwd="/b"))
    for index in range(4):
        graph.sessions.append(
            "run-a", SessionEvent(kind="assistant", event_id=f"event-{index}", text=str(index))
        )
    graph.sessions.append("run-b", SessionEvent(kind="assistant", event_id="only", text="b"))

    bounded = graph.sessions.view("run-a", limit=2)
    assert [event.event_id for event in bounded.events] == ["event-2", "event-3"]
    assert graph.sessions.view("run-b").events == (
        SessionEvent(kind="assistant", event_id="only", text="b"),
    )
    assert graph.sessions.view("missing").context is None
    assert graph.sessions.view("missing").events == ()
    with pytest.raises(ValueError, match="limit"):
        _ = graph.sessions.view("run-a", limit=-1)
    with pytest.raises(ValueError, match="limit"):
        _ = graph.sessions.view("run-a", limit=501)
    graph.close()


def test_session_view_bounds_logical_events_after_many_snapshots(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "logical-bound.db")
    _ = _run(graph)
    graph.sessions.start("run-1", SessionContext(family="omp", cwd=str(tmp_path)))
    graph.sessions.append(
        "run-1", SessionEvent(kind="user", event_id="request", text="continue", state="delivered")
    )
    graph.sessions.append("run-1", SessionEvent(kind="tool", text="tool result"))
    for index in range(1100):
        graph.sessions.append(
            "run-1",
            SessionEvent(kind="assistant", event_id="stream", text=f"{index} ", delta=True),
        )
    graph.sessions.append("run-1", SessionEvent(kind="status", text="running"))
    view = graph.sessions.view("run-1", limit=4)
    assert [event.kind for event in view.events] == ["user", "tool", "assistant", "status"]
    assert view.events[2].text == "".join(f"{index} " for index in range(1100))
    graph.close()


def test_session_view_keeps_empty_id_kinds_separate(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "empty-ids.db")
    _ = _run(graph)
    graph.sessions.start("run-1", SessionContext(family="codex", cwd=str(tmp_path)))
    graph.sessions.append("run-1", SessionEvent(kind="status", text="running"))
    graph.sessions.append("run-1", SessionEvent(kind="error", text="failed"))
    view = graph.sessions.view("run-1")
    assert [(event.kind, event.text) for event in view.events] == [
        ("status", "running"),
        ("error", "failed"),
    ]
    graph.close()


def test_observer_and_watch_retain_terminal_session_transcript(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    db_path = project_root / ".milknado" / "milknado.db"
    db_path.parent.mkdir(parents=True)
    graph = MikadoGraph(db_path)
    node_id = _run(graph, "terminal-run")
    context = SessionContext(family="claude", cwd=str(project_root), base_oid="base")
    graph.sessions.start("terminal-run", context)
    graph.sessions.append(
        "terminal-run",
        SessionEvent(kind="error", event_id="failure", text="worker failed", state="complete"),
    )
    graph.runs.finish(
        "terminal-run",
        RunResult(
            status="failed",
            exit_code=1,
            timed_out=False,
            ended_at="2026-09-09T10:01:00+00:00",
            error="worker failed",
        ),
    )
    _ = graph.mark_terminal(node_id, "terminal-run", NodeStatus.FAILED)
    graph.close()

    observed = read_observer_snapshot(db_path)
    assert observed.runs[0].session.context == context
    assert observed.runs[0].session.active is False
    assert observed.runs[0].session.events == (
        SessionEvent(kind="error", event_id="failure", text="worker failed", state="complete"),
    )
    snapshot = WatchSnapshotSource(project_root, db_path).snapshot()
    assert snapshot.terminal_runs[0].session.events[0].text == "worker failed"
    assert '{"kind"' not in snapshot.terminal_runs[0].session.events[0].text


def test_session_append_surfaces_missing_run_write_failure(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "failure.db")
    with pytest.raises(sqlite3.IntegrityError):
        graph.sessions.append("missing", SessionEvent(kind="user", text="must persist"))
    graph.close()
