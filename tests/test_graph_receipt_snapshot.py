from __future__ import annotations

from pathlib import Path

from milknado.domains.common import NodeSpec, SessionContext
from milknado.domains.graph import GraphCommand, MikadoGraph
from milknado.domains.graph.commands import CommandReceipt, CommandStatus
from milknado.domains.graph.snapshot import connect_readonly, read_node_detail_connection

_NOW = "2026-09-12T00:00:00+00:00"
_FUTURE = "2026-09-13T00:00:00+00:00"
_EXPIRY = "2026-09-12T01:00:00+00:00"


def _command(command_id: str, node_id: int, run_id: str) -> GraphCommand:
    owner = "old-owner" if command_id == "rejected" else "owner-1"
    expires_at = _EXPIRY if command_id == "expired" else _FUTURE
    return GraphCommand(
        command_id=command_id,
        node_id=node_id,
        run_id=run_id,
        invocation_id="invoke-1",
        owner_incarnation=owner,
        action="steer",
        text=f"text-{command_id}",
        expires_at=expires_at,
    )


def _receipt_fixture(graph: MikadoGraph, tmp_path: Path, node_id: int) -> tuple[str, ...]:
    run_id = "run-receipts"
    _ = graph.runs.start(run_id, node_id, str(tmp_path / "run.log"), _NOW, 60)
    _ = graph.commands.publish_capabilities(
        run_id, node_id, "invoke-1", "owner-1", ("steer",), published_at=_NOW
    )
    expired_run = "run-expired"
    _ = graph.runs.start(expired_run, node_id, str(tmp_path / "expired.log"), _NOW, 60)
    _ = graph.commands.publish_capabilities(
        expired_run, node_id, "invoke-1", "owner-1", ("steer",), published_at=_NOW
    )
    _ = graph.sessions.start(expired_run, SessionContext(family="codex", cwd=str(tmp_path)))
    command_ids = ("queued", "submitted", "delivered", "rejected", "expired", "unconfirmed")
    queued = _command(command_ids[0], node_id, run_id)
    submitted = _command(command_ids[1], node_id, run_id)
    delivered = _command(command_ids[2], node_id, run_id)
    rejected = _command(command_ids[3], node_id, run_id)
    expired = _command(command_ids[4], node_id, expired_run)
    unconfirmed = _command(command_ids[5], node_id, run_id)
    _ = graph.commands.admit(queued, now=_NOW)
    _ = graph.commands.admit(submitted, now=_NOW)
    _ = graph.commands.submit(submitted, now=_NOW)
    _ = graph.commands.admit(delivered, now=_NOW)
    _ = graph.commands.submit(delivered, now=_NOW)
    _ = graph.commands.deliver(delivered, now=_NOW)
    _ = graph.commands.admit(rejected, now=_NOW)
    _ = graph.commands.admit(expired, now=_NOW)
    assert graph.commands.claim_pending(expired_run, "owner-1", now=_EXPIRY) == ()
    _ = graph.commands.admit(unconfirmed, now=_NOW)
    _ = graph.commands.unconfirm(unconfirmed, now=_NOW, detail="no delivery frame")
    return command_ids


def _expected_receipts(node_id: int) -> tuple[CommandReceipt, ...]:
    states: tuple[tuple[str, CommandStatus], ...] = (
        ("queued", "queued"),
        ("submitted", "queued"),
        ("submitted", "submitted"),
        ("delivered", "queued"),
        ("delivered", "submitted"),
        ("delivered", "delivered"),
        ("rejected", "rejected"),
        ("expired", "queued"),
        ("expired", "expired"),
        ("unconfirmed", "queued"),
        ("unconfirmed", "unconfirmed"),
    )
    details = {
        "rejected": "owner incarnation or invocation fence does not match",
        "expired": "command expired",
        "unconfirmed": "no delivery frame",
    }
    return tuple(
        CommandReceipt(
            command_id=command_id,
            status=status,
            node_id=node_id,
            run_id="run-expired" if command_id == "expired" else "run-receipts",
            invocation_id="invoke-1",
            owner_incarnation="old-owner" if command_id == "rejected" else "owner-1",
            action="steer",
            text=f"text-{command_id}",
            permission_id=None,
            expires_at=_EXPIRY if command_id == "expired" else _FUTURE,
            admitted_at=_NOW,
            recorded_at=_EXPIRY if status == "expired" else _NOW,
            detail=details.get(status),
        )
        for command_id, status in states
    )


def test_readonly_node_detail_discloses_ordered_receipts_and_pages(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    node = graph.add_node("node", spec=NodeSpec())
    command_ids = _receipt_fixture(graph, tmp_path, node.id)
    other = graph.add_node("other", spec=NodeSpec())
    other_run = "run-other"
    _ = graph.runs.start(other_run, other.id, str(tmp_path / "other.log"), _NOW, 60)
    _ = graph.commands.publish_capabilities(
        other_run, other.id, "invoke-1", "owner-1", ("steer",), published_at=_NOW
    )
    _ = graph.commands.admit(_command("other", other.id, other_run), now=_NOW)
    histories = tuple(graph.commands.history(command_id) for command_id in command_ids)
    expected = _expected_receipts(node.id)
    reader = connect_readonly(db_path)
    try:
        assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
        for page_number in range(5):
            detail = read_node_detail_connection(reader, node.id, page=page_number, limit=3).detail
            assert detail is not None
            page = detail.receipts
            offset = page_number * 3
            assert page.items == expected[offset : offset + 3]
            assert (page.offset, page.limit, page.total, page.state) == (offset, 3, 11, "loaded")
            assert page.has_more is (page_number < 3)
        assert graph.sessions.view("run-expired").events == ()
        assert tuple(item.status for item in graph.commands.history("expired")) == (
            "queued",
            "expired",
        )
        assert tuple(graph.commands.history(command_id) for command_id in command_ids) == histories
        assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        reader.close()
        graph.close()


def test_missing_session_differs_from_loaded_empty_session(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("node")
    _ = graph.runs.start("missing", node.id, str(tmp_path / "missing.log"), _NOW, 60)
    _ = graph.runs.start("empty", node.id, str(tmp_path / "empty.log"), _NOW, 60)
    _ = graph.sessions.start("empty", SessionContext(family="codex", cwd=str(tmp_path)))
    detail = graph.get_node_detail_snapshot(node.id, limit=10).detail
    assert detail is not None and detail.sessions.items is not None
    states = {session.run_id: session for session in detail.sessions.items}
    assert states["missing"].state == "missing"
    assert states["missing"].event_history.state == "missing"
    assert states["empty"].state == "loaded"
    assert states["empty"].event_history.items == ()
    assert states["empty"].event_history.state == "loaded"
    graph.close()
