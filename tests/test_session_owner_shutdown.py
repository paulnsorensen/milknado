from pathlib import Path
from typing import cast

import pytest

from milknado.adapters import LoopAdapter
from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.domains.graph import MikadoGraph, admit_session_command
from milknado.loop.manager import ManagedRun
from tests.graph_command_fixtures import owned_graph


def _failed_sink(_event: SessionEvent) -> None:
    raise RuntimeError("event sink unavailable")


@pytest.mark.parametrize("failed_sink", [False, True])
def test_close_rejects_command_admitted_after_final_drain(
    graph: MikadoGraph, tmp_path: Path, failed_sink: bool
) -> None:
    _ = owned_graph(graph)
    adapter = LoopAdapter(graph=graph)
    managed = cast(
        ManagedRun,
        adapter.create_run("claude", tmp_path, tmp_path / "task.md", (), run_id="run-1"),
    )
    context = SessionContext(family="claude", cwd=str(tmp_path))
    graph.sessions.start("run-1", context)
    channel = managed.state.session
    assert channel is not None
    channel.start(context, ("steer",), invocation_id="invoke-1")
    assert channel.drain() == ()
    command = SessionInput(action="steer", text="too late", command_id="late-command")
    assert admit_session_command(graph, "run-1", command) == command
    if failed_sink:
        channel.set_sink(_failed_sink)
        with pytest.raises(RuntimeError, match="event sink unavailable"):
            channel.close()
    else:
        channel.close()
    assert [receipt.status for receipt in graph.commands.history("late-command")] == [
        "queued",
        "rejected",
    ]
    capabilities = graph.commands.capabilities("run-1")
    assert capabilities is not None
    assert capabilities.actions == ()
    assert capabilities.permission_ids == ()
    assert (
        admit_session_command(
            graph, "run-1", SessionInput(action="steer", text="closed", command_id="after-close")
        )
        is None
    )
