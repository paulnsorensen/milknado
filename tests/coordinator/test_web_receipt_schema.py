from pathlib import Path
from typing import cast

from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.graph import MikadoGraph

EXPECTED_COLUMNS = [
    ("command_id", "TEXT", 0, 1),
    ("session_id", "TEXT", 1, 0),
    ("command_hash", "TEXT", 1, 0),
    ("status", "TEXT", 1, 0),
    ("result_json", "TEXT", 1, 0),
]


def _receipt_schema(graph: MikadoGraph) -> list[tuple[str, str, int, int]]:
    rows = cast(
        list[tuple[int, str, str, int, str | None, int]],
        graph.group_connection.execute("PRAGMA table_info(coordinator_web_receipts)").fetchall(),
    )
    return [(row[1], row[2], row[3], row[5]) for row in rows]


def test_receipt_schema_exists_on_fresh_open_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "graph.db"
    foreign_key_query = "PRAGMA foreign_key_list(coordinator_web_receipts)"
    graph = MikadoGraph(path)
    assert _receipt_schema(graph) == EXPECTED_COLUMNS
    foreign_keys = graph.group_connection.execute(foreign_key_query)
    assert list(foreign_keys) == []
    graph.close()

    reopened = MikadoGraph(path)
    assert _receipt_schema(reopened) == EXPECTED_COLUMNS
    foreign_keys = reopened.group_connection.execute(foreign_key_query)
    assert list(foreign_keys) == []
    reopened.close()


def test_command_reservation_uses_existing_receipt_schema(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    statements: list[str] = []
    graph.group_connection.set_trace_callback(statements.append)
    try:
        control = CoordinatorControl(graph, tmp_path)
        command = StartGoal("start-1", "Deliver", "codex")
        first = control.send_coordinator_command("", command)
        assert first.status == "accepted"
        assert control.send_coordinator_command("", command) == first
    finally:
        graph.group_connection.set_trace_callback(None)
        graph.close()
    assert any("INSERT OR IGNORE INTO coordinator_web_receipts" in sql for sql in statements)
    assert not any(
        "CREATE TABLE" in sql.upper() and "COORDINATOR_WEB_RECEIPTS" in sql.upper()
        for sql in statements
    )
