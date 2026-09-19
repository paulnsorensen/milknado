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


def test_stream_handles_synchronous_replay_without_deadlock() -> None:
    _, _, source = client_with_source()
    subscribe = source.subscribe

    def replay(listener):
        unsubscribe = subscribe(listener)
        listener(source.snapshot())
        return unsubscribe

    source.subscribe = replay  # type: ignore[method-assign]
    fanout = SnapshotFanout(source)

    async def exercise() -> dict[str, str]:
        stream = fanout.events()
        return await asyncio.wait_for(stream.__anext__(), 2)

    event = asyncio.run(exercise())

    assert event["event"] == "snapshot"
    assert '"goal":"fixture goal"' in event["data"]


def test_stream_publishes_exact_frames_to_two_clients_and_unsubscribes() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    async def consume() -> dict[str, str]:
        stream = fanout.events()
        try:
            return await asyncio.wait_for(stream.__anext__(), 2)
        finally:
            await stream.aclose()

    async def exercise() -> tuple[dict[str, str], dict[str, str]]:
        first_task = asyncio.create_task(consume())
        second_task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.to_thread(source.publish, source.snapshot())
        return await asyncio.gather(first_task, second_task)

    first, second = asyncio.run(exercise())

    expected = {
        "event": "snapshot",
        "data": '{"goal":"fixture goal","active_runs":[],"terminal_runs":[],"completed":0,'
        '"failed":0,"stopped":0,"available":1,"event_lines":[],"listener_errors":'
        '["fixture listener error"],"graph":null,"node":null}',
    }
    assert first == expected
    assert second == expected
    assert len(source._listeners) == 0
