from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

import pytest
from starlette.types import Message, Scope

from milknado.app.run_source import ExecutionSnapshot
from milknado.web import WebCommands, create_app
from milknado.web.fanout import SnapshotFanout
from tests.web.support import client_with_source


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


def test_stream_queues_every_frame_and_disconnects_on_overflow() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    async def exercise() -> None:
        stream = fanout.events()
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
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
            for _ in range(3):
                source.publish(snapshot)
            assert calls == 1
            await pending
            for _ in range(2):
                _ = await stream.__anext__()
        finally:
            loop.call_soon_threadsafe = original
            await stream.aclose()

        queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue(32)
        fanout._add(queue, loop)  # pyright: ignore[reportPrivateUsage]
        for _ in range(33):
            source.publish(source.snapshot())
        await asyncio.sleep(0)
        assert len(source._listeners) == 0  # pyright: ignore[reportPrivateUsage]
        assert await queue.get() is None

    asyncio.run(exercise())


def test_stream_reconnect_waits_for_old_unsubscribe_before_replacement() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)
    active: list[Callable[[ExecutionSnapshot], None]] = []
    unsubscribe_started = threading.Event()
    release_unsubscribe = threading.Event()
    subscribe_count = 0

    def subscribe(listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        nonlocal subscribe_count
        subscribe_count += 1
        active.append(listener)

        def unsubscribe() -> None:
            unsubscribe_started.set()
            _ = release_unsubscribe.wait()
            active.clear()

        return unsubscribe

    def publish(snapshot: ExecutionSnapshot) -> None:
        for listener in tuple(active):
            listener(snapshot)

    source.subscribe = subscribe  # type: ignore[method-assign]
    source.publish = publish  # type: ignore[method-assign]

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        first_queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue()
        second_queue: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue()
        fanout._add(first_queue, loop)  # pyright: ignore[reportPrivateUsage]
        remove_task = asyncio.create_task(
            asyncio.to_thread(fanout._remove, first_queue)  # pyright: ignore[reportPrivateUsage]
        )
        _ = await asyncio.to_thread(unsubscribe_started.wait)
        add_task = asyncio.create_task(
            asyncio.to_thread(fanout._add, second_queue, loop)  # pyright: ignore[reportPrivateUsage]
        )
        await asyncio.sleep(0)
        assert subscribe_count == 1
        release_unsubscribe.set()
        await remove_task
        await add_task
        source.publish(source.snapshot())
        assert await asyncio.wait_for(second_queue.get(), 2) == source.snapshot()

    asyncio.run(exercise())
    assert subscribe_count == 2


def test_stream_terminates_clients_when_subscription_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)

    def fail(listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        _ = listener
        raise RuntimeError("subscription failed")

    source.subscribe = fail  # type: ignore[method-assign]

    async def exercise() -> None:
        stream = fanout.events()
        with pytest.raises(StopAsyncIteration):
            _ = await stream.__anext__()
        await stream.aclose()

    with caplog.at_level(logging.ERROR, logger="milknado.web.fanout"):
        asyncio.run(exercise())

    assert not fanout._clients  # pyright: ignore[reportPrivateUsage]
    assert caplog.records[0].getMessage() == "snapshot subscription failed"
    assert caplog.records[0].exc_info is not None
    assert not any("backpressure" in record.getMessage() for record in caplog.records)


def test_stream_ignores_callback_interleaved_with_generation_invalidation() -> None:
    _, _, source = client_with_source()
    fanout = SnapshotFanout(source)
    listeners: list[Callable[[ExecutionSnapshot], None]] = []

    def subscribe(listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        listeners.append(listener)
        return lambda: None

    source.subscribe = subscribe  # type: ignore[method-assign]

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        first: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue()
        second: asyncio.Queue[ExecutionSnapshot | None] = asyncio.Queue()
        fanout._add(first, loop)  # pyright: ignore[reportPrivateUsage]
        old_listener = listeners[0]
        started = threading.Event()

        def invoke_old_listener() -> None:
            started.set()
            old_listener(source.snapshot())

        with fanout._lock:  # pyright: ignore[reportPrivateUsage]
            thread = threading.Thread(target=invoke_old_listener)
            thread.start()
            assert started.wait(2)
            fanout._remove(first)  # pyright: ignore[reportPrivateUsage]
            fanout._add(second, loop)  # pyright: ignore[reportPrivateUsage]
        thread.join(2)
        assert not thread.is_alive()
        await asyncio.sleep(0)
        assert second.empty()
        fanout._remove(second)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(exercise())
