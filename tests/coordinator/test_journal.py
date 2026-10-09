from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.coordinator import ControlEvent
from milknado.domains.coordinator.journal import append_control_event, control_history
from milknado.domains.coordinator.persistence import start_coordinator
from milknado.domains.graph._persistence import create_tables, migrate


def _session(path: str) -> tuple[sqlite3.Connection, str]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA foreign_keys = ON")
    create_tables(conn)
    migrate(conn)
    _ = conn.execute(
        "INSERT INTO nodes (id, description, kind, created_at) VALUES (1, 'goal', 'goal', 'now')"
    )
    conn.commit()
    return conn, start_coordinator(conn, 1, "codex").id


def test_mixed_control_history_is_durable_and_ordered(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    conn, session_id = _session(path)
    with closing(conn):
        for kind in ("command", "approval", "graph_revision", "run_transition", "provider_event"):
            _ = append_control_event(conn, session_id, ControlEvent(kind=kind, text=kind))
    with closing(sqlite3.connect(path)) as reopened:
        entries = control_history(reopened, session_id)
        assert [entry.kind for entry in entries] == [
            "command",
            "approval",
            "graph_revision",
            "run_transition",
            "provider_event",
        ]
        assert [entry.seq for entry in entries] == [1, 2, 3, 4, 5]


def test_tool_payloads_are_elided_and_secrets_are_filtered(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    conn, session_id = _session(path)
    with closing(conn):
        _ = append_control_event(
            conn,
            session_id,
            ControlEvent(
                kind="tool",
                text="raw result secret",
                tool_name="Bash",
                status="done",
                tool_arguments="password=tool-argument-secret",
                tool_result="sk-test-result-secret",
            ),
        )
        _ = append_control_event(
            conn, session_id, ControlEvent(kind="assistant", text="Bearer bearer-secret")
        )
        _ = append_control_event(
            conn, session_id, ControlEvent(kind="assistant", text='Bearer "quoted bearer secret"')
        )
        _ = append_control_event(
            conn,
            session_id,
            ControlEvent(
                kind="assistant",
                text='{"api_key": "first secret", "password": "last secret"}',
            ),
        )
        entry = control_history(conn, session_id)[0]
        assert entry.tool_name == "Bash"
        assert entry.status == "done"
        assert entry.text == "[tool payload elided]"
    with closing(sqlite3.connect(path)) as reopened:
        rows = cast(
            list[tuple[str]],
            reopened.execute("SELECT text FROM coordinator_events ORDER BY seq").fetchall(),
        )
        raw = "\n".join(row[0] for row in rows)
        assert "tool-argument-secret" not in raw
        assert "test-result-secret" not in raw
        assert "bearer-secret" not in raw
        assert "quoted" not in raw
        assert "bearer secret" not in raw
        assert "first secret" not in raw
        assert "last secret" not in raw
        assert "[REDACTED]" in raw


def test_diagnostics_expire_and_reject_unbounded_retention(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    conn, session_id = _session(path)
    with closing(conn):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        _ = append_control_event(
            conn,
            session_id,
            ControlEvent(
                kind="diagnostic", text="api_key=private-token", diagnostic_retention_seconds=1
            ),
            now=now,
        )
        assert "[REDACTED]" in control_history(conn, session_id, now=now)[0].text
        with pytest.raises(ValueError, match="retention"):
            _ = append_control_event(
                conn,
                session_id,
                ControlEvent(
                    kind="diagnostic", text="bad", diagnostic_retention_seconds=31 * 86400
                ),
                now=now,
            )
    with closing(sqlite3.connect(path)) as reopened:
        assert control_history(reopened, session_id, now=now + timedelta(seconds=2)) == ()
        count = cast(
            tuple[int] | None,
            reopened.execute("SELECT COUNT(*) FROM coordinator_events").fetchone(),
        )
        assert count is not None and count[0] == 0


def test_control_event_rejects_invalid_size_and_timing(tmp_path: Path) -> None:
    conn, session_id = _session(str(tmp_path / "graph.db"))
    with closing(conn):
        with pytest.raises(ValueError, match="event kind"):
            _ = append_control_event(conn, session_id, ControlEvent(kind=""))
        with pytest.raises(ValueError, match="duration_ms"):
            _ = append_control_event(conn, session_id, ControlEvent(kind="tool", duration_ms=-1))
        with pytest.raises(ValueError, match="64 KiB"):
            _ = append_control_event(
                conn, session_id, ControlEvent(kind="assistant", text="x" * 65537)
            )
        assert control_history(conn, session_id) == ()


def test_diagnostics_use_utc_and_expiry_index(tmp_path: Path) -> None:
    conn, session_id = _session(str(tmp_path / "graph.db"))
    with closing(conn):
        local_time = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
        _ = append_control_event(
            conn,
            session_id,
            ControlEvent(kind="diagnostic", text="detail", diagnostic_retention_seconds=60),
            now=local_time,
        )
        entry = control_history(conn, session_id, now=datetime(2026, 1, 1, 10, tzinfo=UTC))[0]
        assert entry.created_at == "2026-01-01T10:00:00+00:00"
        assert control_history(conn, session_id, now=datetime(2026, 1, 1, 10, 2, tzinfo=UTC)) == ()
        with pytest.raises(ValueError, match="timezone-aware"):
            _ = append_control_event(
                conn, session_id, ControlEvent(kind="command"), now=datetime(2026, 1, 1)
            )
        with pytest.raises(ValueError, match="timezone-aware"):
            _ = control_history(conn, session_id, now=datetime(2026, 1, 1))
        index = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'idx_coordinator_events_expiry'"
            ).fetchone(),
        )
        assert index is not None and "WHERE expires_at IS NOT NULL" in index[0]


def test_authorization_header_value_is_fully_redacted_after_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.db")
    conn, session_id = _session(path)
    with closing(conn):
        for scheme in ("Bearer", "Basic"):
            _ = append_control_event(
                conn,
                session_id,
                ControlEvent(
                    kind="assistant",
                    text=f"Authorization: {scheme} credential-{scheme}\nnext: ok",
                ),
            )
    with closing(sqlite3.connect(path)) as reopened:
        assert [entry.text for entry in control_history(reopened, session_id)] == [
            "Authorization: [REDACTED]\nnext: ok",
            "Authorization: [REDACTED]\nnext: ok",
        ]


def test_unterminated_quoted_secret_has_bounded_redaction_time() -> None:
    payload = 'token="' + "\\" * 257
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from milknado.domains.coordinator.journal import redact_control_text; "
            + "import sys; print(redact_control_text(sys.argv[1]))",
            payload,
        ],
        capture_output=True,
        text=True,
        timeout=2,
        check=True,
    )
    assert result.stdout.strip() == 'token="[REDACTED]'
