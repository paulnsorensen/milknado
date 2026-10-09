import asyncio
from pathlib import Path
from typing import cast

import msgspec
import pytest
from starlette.requests import Request

from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_models import StartGoal
from milknado.domains.graph import MikadoGraph
from milknado.web.routes import coordinator as route
from milknado.web.routes.coordinator import _events  # pyright: ignore[reportPrivateUsage]


def test_stream_emits_graph_only_change_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    control = CoordinatorControl(graph, tmp_path)
    receipt = control.send_coordinator_command("", StartGoal("start", "Deliver", "codex"))
    result = cast(dict[str, object], receipt.result)
    session_id = cast(str, result["id"])
    goal_id = cast(int, result["goal_id"])
    cursor = control.read_coordinator_snapshot(session_id, 0).cursor
    initial = control.read_coordinator_snapshot(session_id, cursor)
    polls = 0

    class Connected:
        async def is_disconnected(self) -> bool:
            return polls >= 3

    async def advance(_: float) -> None:
        nonlocal polls
        polls += 1
        if polls == 1:
            graph.update_node(goal_id, description="Changed")

    monkeypatch.setattr(route, "sleep", advance)

    async def receive() -> tuple[dict[str, object], bool]:
        events = _events(
            cast(Request, cast(object, Connected())), control, initial, (session_id, cursor)
        )
        first = await asyncio.wait_for(anext(events), 2)
        ended = False
        try:
            _ = await asyncio.wait_for(anext(events), 2)
        except StopAsyncIteration:
            ended = True
        return cast(dict[str, object], msgspec.json.decode(first["data"].encode())), ended

    snapshot, ended = asyncio.run(receive())
    assert snapshot["cursor"] == cursor
    assert snapshot["events"] == []
    assert cast(dict[str, object], snapshot["goal"])["description"] == "Changed"
    assert ended
    assert polls == 3
    graph.close()
