from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from milknado.domains.common import SessionContext, SessionInput
from milknado.domains.graph import GraphCommand, MikadoGraph
from milknado.loop.sessions import SessionChannel

_NOW = datetime.now(UTC)
_LATER = (_NOW + timedelta(minutes=5)).isoformat()


def test_claimed_command_without_provider_receipt_is_unconfirmed(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "commands.db")
    try:
        node = graph.add_node("steerable")
        assert graph.claim_node(node.id, "run-1", now=_NOW.isoformat())
        graph.runs.start("run-1", node.id, "run.log", _NOW.isoformat(), 60)
        _ = graph.commands.publish_capabilities(
            "run-1", node.id, "invoke-1", "owner-1", ("steer",), (), published_at=_NOW.isoformat()
        )
        command = GraphCommand(
            command_id="command-1",
            node_id=node.id,
            run_id="run-1",
            invocation_id="invoke-1",
            owner_incarnation="owner-1",
            action="steer",
            text="redirect",
            permission_id=None,
            expires_at=_LATER,
        )
        assert graph.commands.admit(command, now=_NOW.isoformat()).status == "queued"

        def durable_drain() -> tuple[SessionInput, ...]:
            claimed = graph.commands.claim_pending("run-1", "owner-1", now=_NOW.isoformat())
            return tuple(
                SessionInput(
                    action=item.action,
                    text=item.text,
                    command_id=item.command_id,
                    request_id=item.permission_id or "",
                )
                for item in claimed
            )

        def record_state(input_command: SessionInput, state: str) -> None:
            assert input_command.command_id == "command-1"
            assert state == "unconfirmed"
            stored = graph.commands.command(input_command.command_id)
            assert stored is not None
            _ = graph.commands.unconfirm(stored, now=_NOW.isoformat())

        channel = SessionChannel(command_state_sink=record_state, durable_drain=durable_drain)
        channel.start(
            SessionContext(family="claude", cwd="/repo"), ("steer",), invocation_id="invoke-1"
        )
        assert channel.drain()
        channel.close()

        assert [receipt.status for receipt in graph.commands.history("command-1")] == [
            "queued",
            "submitted",
            "unconfirmed",
        ]
    finally:
        graph.close()
