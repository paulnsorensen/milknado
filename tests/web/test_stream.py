from __future__ import annotations

import asyncio
from collections.abc import Callable

from starlette.requests import Request
from starlette.types import Message, Scope

from milknado.app.run_source import ExecutionSnapshot
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

    def replay(listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
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


def test_stream_replays_synchronous_setup_snapshot_to_joining_client() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)
    second_queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue()

    async def exercise() -> tuple[dict[str, str], ExecutionSnapshot]:
        loop = asyncio.get_running_loop()
        subscribe = source.subscribe

        def replay_then_join(
            listener: Callable[[ExecutionSnapshot], None],
        ) -> Callable[[], None]:
            unsubscribe = subscribe(listener)
            listener(source.snapshot())
            fanout._add(second_queue, loop)  # pyright: ignore[reportPrivateUsage]
            return unsubscribe

        source.subscribe = replay_then_join  # type: ignore[method-assign]
        first_stream = fanout.events()
        try:
            first_event = await asyncio.wait_for(first_stream.__anext__(), 2)
            second_snapshot = await asyncio.wait_for(second_queue.get(), 2)
            assert second_snapshot is not None
            return first_event, second_snapshot
        finally:
            await first_stream.aclose()
            fanout._remove(second_queue)  # pyright: ignore[reportPrivateUsage]

    event, snapshot = asyncio.run(exercise())

    assert event["event"] == "snapshot"
    assert snapshot == source.snapshot()


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
        + '"failed":0,"stopped":0,"available":1,"event_lines":[],"listener_errors":'
        + '["fixture listener error"],"graph":null,"node":null}',
    }
    assert first == expected
    assert second == expected
    assert len(source._listeners) == 0  # pyright: ignore[reportPrivateUsage]


def test_stream_replays_latest_snapshot_to_staggered_client() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    async def exercise() -> tuple[dict[str, str], dict[str, str]]:
        first_stream = fanout.events()
        first_task = asyncio.create_task(first_stream.__anext__())
        await asyncio.sleep(0)
        await asyncio.to_thread(source.publish, source.snapshot())
        first_event = await asyncio.wait_for(first_task, 2)

        second_stream = fanout.events()
        try:
            second_event = await asyncio.wait_for(second_stream.__anext__(), 2)
            return first_event, second_event
        finally:
            await first_stream.aclose()
            await second_stream.aclose()

    first, second = asyncio.run(exercise())

    assert second == first


def test_stream_asgi_emits_exact_frames_for_two_authenticated_clients() -> None:
    _, login, source = client_with_source()
    app = create_app(source, WebCommands(), login)
    expected = (
        b"event: snapshot\r\n"
        + b'data: {"goal":"fixture goal","active_runs":[],"terminal_runs":[],'
        + b'"completed":0,'
        + b'"failed":0,"stopped":0,"available":1,"event_lines":[],"listener_errors":'
        + b'["fixture listener error"],"graph":null,"node":null}\r\n\r\n'
    )

    async def request(disconnect: asyncio.Event) -> list[Message]:
        sent: list[Message] = []
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/stream",
            "raw_path": b"/api/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1"),
                (b"cookie", f"{login.cookie_name}={login.value}".encode()),
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
            "root_path": "",
        }

        async def receive() -> Message:
            _ = await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        await app(scope, receive, send)
        return sent

    async def exercise() -> tuple[list[Message], list[Message]]:
        first_disconnect = asyncio.Event()
        second_disconnect = asyncio.Event()
        first = asyncio.create_task(request(first_disconnect))
        second = asyncio.create_task(request(second_disconnect))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(source._listeners) == 1:  # pyright: ignore[reportPrivateUsage]
                break
        assert len(source._listeners) == 1  # pyright: ignore[reportPrivateUsage]
        await asyncio.to_thread(source.publish, source.snapshot())
        await asyncio.sleep(0)
        _ = first_disconnect.set()
        _ = second_disconnect.set()
        return await asyncio.gather(first, second)

    first, second = asyncio.run(exercise())
    for messages in (first, second):
        bodies = [
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        ]
        assert bodies == [expected]
    assert len(source._listeners) == 0  # pyright: ignore[reportPrivateUsage]


def test_stream_coalesces_threadsafe_callbacks_and_overflow_disconnects() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    async def exercise() -> None:
        stream = fanout.events()
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        queue = next(iter(fanout._clients))  # pyright: ignore[reportPrivateUsage]
        loop = asyncio.get_running_loop()
        original = loop.call_soon_threadsafe
        calls = 0

        def counted(callback: Callable[..., object], *args: object) -> object:
            nonlocal calls
            calls += 1
            return original(callback, *args)

        loop.call_soon_threadsafe = counted  # pyright: ignore[reportAttributeAccessIssue]
        try:
            snapshot = source.snapshot()
            for _ in range(1000):
                source.publish(snapshot)
            assert calls == 1
        finally:
            loop.call_soon_threadsafe = original
            await pending
            await stream.aclose()
        assert len(source._listeners) == 0  # pyright: ignore[reportPrivateUsage]

        stream = fanout.events()
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        queue = next(iter(fanout._clients))  # pyright: ignore[reportPrivateUsage]
        for _ in range(33):
            fanout._enqueue(queue, source.snapshot())  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
        assert len(source._listeners) == 0  # pyright: ignore[reportPrivateUsage]
        await pending
        await stream.aclose()

    asyncio.run(exercise())
