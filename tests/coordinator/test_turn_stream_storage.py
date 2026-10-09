from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pytest

from milknado.domains.common import SessionEvent
from milknado.domains.coordinator.journal import (
    append_control_event,
    control_history,
    snapshot_control_history,
)
from milknado.domains.coordinator.model import ControlEvent
from milknado.domains.coordinator.turn_context import TurnContext
from milknado.domains.coordinator.turns import record_turn_event
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph


def _session(graph: MikadoGraph) -> str:
    return CoordinatorWorkflow(graph, graph.group_connection).start_goal("Deliver", "codex").id


def _assert_storage_compaction(
    conn: sqlite3.Connection, session_id: str, expected: tuple[str, ...]
) -> None:
    stored = cast(
        tuple[int | None] | None,
        conn.execute(
            "SELECT SUM(length(CAST(text AS BLOB))) FROM coordinator_events "
            + "WHERE session_id = ?",
            (session_id,),
        ).fetchone(),
    )
    assert stored is not None and stored[0] is not None
    assert stored[0] < sum(len(value) for value in expected) // 3
    limits = cast(
        tuple[int | None, int | None] | None,
        conn.execute(
            "SELECT MAX(stream_depth), SUM(stream_ref IS NULL) FROM coordinator_events "
            + "WHERE session_id = ? AND stream_key IS NOT NULL",
            (session_id,),
        ).fetchone(),
    )
    assert limits == (32, 2)


def test_cumulative_stream_uses_less_storage_without_changing_history(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    graph = MikadoGraph(path)
    session_id = _session(graph)
    expected = tuple("x" * (80 * index) for index in range(1, 65))
    for value in expected:
        event = SessionEvent(kind="assistant", text=value, event_id="native-1", state="streaming")
        record_turn_event(
            TurnContext(graph.group_connection, session_id, "turn-1"), event, "provider-1"
        )
    with sqlite3.connect(path) as conn:
        _assert_storage_compaction(conn, session_id, expected)
        history = control_history(conn, session_id)
        assert tuple(event.text for event in history) == expected
        recent, recovery, cursor = snapshot_control_history(conn, session_id, history[39].seq)
        assert tuple(event.text for event in recent) == expected[40:]
        assert recovery == ()
        assert cursor == history[-1].seq
    graph.close()
    reopened = MikadoGraph(path)
    with sqlite3.connect(path) as conn:
        assert tuple(event.text for event in control_history(conn, session_id)) == expected
    reopened.close()


def test_snapshot_history_does_not_prune_or_commit_expired_events(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    session_id = _session(graph)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    seq = append_control_event(
        graph.group_connection,
        session_id,
        ControlEvent(kind="diagnostic", text="expired", diagnostic_retention_seconds=60),
        now=now,
    )
    conn = graph.group_connection
    _ = conn.execute("BEGIN")
    try:
        recent, recovery, cursor = snapshot_control_history(
            conn, session_id, 0, now=now + timedelta(minutes=2)
        )
        assert recent == recovery == ()
        assert cursor == 0
        assert conn.in_transaction
        assert conn.execute("SELECT seq FROM coordinator_events WHERE seq = ?", (seq,)).fetchone()
    finally:
        conn.rollback()
    assert control_history(conn, session_id, now=now + timedelta(minutes=2)) == ()
    remaining = conn.execute("SELECT seq FROM coordinator_events WHERE seq = ?", (seq,))
    assert remaining.fetchone() is None
    graph.close()


def test_interleaved_identity_and_terminal_updates_keep_independent_chains(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    session_id = _session(graph)
    updates = (
        ("turn-a", "provider-a", "assistant", "same", "A", "streaming"),
        ("turn-b", "provider-b", "assistant", "same", "B", "streaming"),
        ("turn-a", "provider-a", "assistant", "same", "AB", "streaming"),
        ("turn-b", "provider-b", "assistant", "same", "BC", "streaming"),
        ("turn-a", "provider-a", "assistant", "same", "AB", "complete"),
        ("turn-a", "provider-a", "assistant", "same", "reset", "streaming"),
        ("turn-a", "provider-a", "error", "same", "oops", "streaming"),
        ("turn-a", "provider-a", "error", "same", "oops again", "streaming"),
    )
    for turn, provider, kind, native_id, value, state in updates:
        event = SessionEvent(
            kind=cast(Literal["assistant", "error"], kind),
            text=value,
            event_id=native_id,
            state=state,
        )
        record_turn_event(TurnContext(graph.group_connection, session_id, turn), event, provider)
    with sqlite3.connect(graph.db_path) as conn:
        rows = cast(
            list[tuple[str, str | None, int | None, int | None]],
            conn.execute(
                "SELECT text, stream_key, stream_ref, stream_depth FROM coordinator_events "
                + "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall(),
        )
        assert len({row[1] for row in rows}) == 3
        assert rows[2][0] == "B" and rows[2][2] is not None
        assert rows[3][0] == "C" and rows[3][2] is not None
        assert rows[4][0] == "AB" and rows[4][2:] == (None, 0)
        assert rows[5][0] == "reset" and rows[5][2:] == (None, 0)
        assert rows[7][0] == " again"
        assert tuple(item.text for item in control_history(conn, session_id)) == tuple(
            item[4] for item in updates
        )
    graph.close()


def test_nonstream_events_and_redaction_never_store_raw_payloads(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    session_id = _session(graph)
    events = (
        SessionEvent(kind="user", text="approve", event_id="input", state="delivered"),
        SessionEvent(kind="tool", text="secret=raw-tool", event_id="tool", state="complete"),
        SessionEvent(kind="permission", text="Approve", event_id="permission", state="requested"),
        SessionEvent(kind="assistant", text="token=alpha", event_id="message", state="streaming"),
        SessionEvent(
            kind="assistant", text="token=alpha more", event_id="message", state="streaming"
        ),
    )
    for event in events:
        record_turn_event(
            TurnContext(graph.group_connection, session_id, "turn"), event, "provider"
        )
    with sqlite3.connect(graph.db_path) as conn:
        rows = cast(
            list[tuple[str, str | None, int | None]],
            conn.execute(
                "SELECT text, stream_key, stream_ref FROM coordinator_events "
                + "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall(),
        )
        assert [row[0] for row in rows[:3]] == ["approve", "[tool payload elided]", "Approve"]
        assert all(row[1:] == (None, None) for row in rows[:3])
        assert rows[3][0] == "token=[REDACTED]"
        assert rows[4][0] == " more" and rows[4][2] is not None
        assert tuple(item.text for item in control_history(conn, session_id))[-2:] == (
            "token=[REDACTED]",
            "token=[REDACTED] more",
        )
        assert "raw-tool" not in repr(rows) and "alpha" not in repr(rows)
    graph.close()


def test_rollover_and_replacement_use_full_checkpoints(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    session_id = _session(graph)
    marker = "[Earlier text omitted]\n"
    values = ("a" * 8192, marker + "a" * (8192 - len(marker)), "replacement")
    for value in values:
        event = SessionEvent(kind="assistant", text=value, event_id="message", state="streaming")
        record_turn_event(
            TurnContext(graph.group_connection, session_id, "turn"), event, "provider"
        )
    with sqlite3.connect(graph.db_path) as conn:
        rows = cast(
            list[tuple[int | None, int | None]],
            conn.execute(
                "SELECT stream_ref, stream_depth FROM coordinator_events "
                + "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall(),
        )
        assert rows == [(None, 0)] * len(values)
        assert tuple(item.text for item in control_history(conn, session_id)) == values
    graph.close()


def test_cross_owner_stream_reference_fails_closed(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    session_id = _session(graph)
    foreign = CoordinatorWorkflow(graph, graph.group_connection).start_goal("Other", "codex")
    event = SessionEvent(kind="assistant", text="foreign", event_id="message", state="streaming")
    record_turn_event(TurnContext(graph.group_connection, foreign.id, "turn"), event, "provider")
    event = SessionEvent(kind="assistant", text="owned", event_id="message", state="streaming")
    record_turn_event(TurnContext(graph.group_connection, session_id, "turn"), event, "provider")
    with sqlite3.connect(graph.db_path) as conn:
        ids = cast(
            list[tuple[int, str]],
            conn.execute(
                "SELECT seq, session_id FROM coordinator_events WHERE stream_key IS NOT NULL "
                + "ORDER BY seq"
            ).fetchall(),
        )
        assert len(ids) == 2 and ids[0][1] == foreign.id and ids[1][1] == session_id
        _ = conn.execute(
            "UPDATE coordinator_events SET stream_ref = ?, stream_depth = 1 WHERE seq = ?",
            (ids[0][0], ids[1][0]),
        )
        with pytest.raises(ValueError, match="owner or identity"):
            _ = control_history(conn, session_id)
    graph.close()
