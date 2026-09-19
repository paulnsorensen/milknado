from __future__ import annotations

import asyncio

from starlette.requests import Request

from milknado.web import LaunchToken, WebCommands, create_app
from milknado.web.fanout import SnapshotFanout
from milknado.web.routes.stream import stream_route
from tests.web.support import client_with_source


def test_stream_publishes_snapshot_event_and_unsubscribes() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    async def consume() -> dict[str, str]:
        async for event in fanout.events():
            return event
        raise AssertionError("stream ended without an event")

    async def exercise() -> dict[str, str]:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.to_thread(source.publish, source.snapshot())
        return await asyncio.wait_for(task, 2)

    event = asyncio.run(exercise())

    assert event["event"] == "snapshot"
    assert '"goal":"fixture goal"' in event["data"]


def test_stream_route_registers_fanout() -> None:
    _, _, source = client_with_source()
    app = create_app(source, WebCommands(), LaunchToken("test-token"))
    request = Request({"type": "http", "app": app})

    response = stream_route(request)

    assert response.media_type == "text/event-stream"
