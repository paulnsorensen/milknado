import asyncio
from typing import cast

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from milknado.web.guards import RequestGuards
from milknado.web.login import LaunchToken
from tests.web.support import client


def test_foreign_host_is_rejected() -> None:
    response = cast(
        httpx.Response,
        client()[0].get("/api/snapshot", headers={"host": "evil.example"}),  # pyright: ignore[reportUnknownMemberType]
    )
    assert response.status_code == 400


def test_missing_origin_is_rejected_on_write() -> None:
    response = cast(
        httpx.Response,
        client()[0].post("/api/unknown", headers={"host": "127.0.0.1"}),  # pyright: ignore[reportUnknownMemberType]
    )
    assert response.status_code == 403


def test_matching_authority_with_port_is_allowed() -> None:
    response = cast(
        httpx.Response,
        client()[0].get(  # pyright: ignore[reportUnknownMemberType]
            "/api/snapshot",
            headers={"host": "127.0.0.1:8000", "origin": "http://127.0.0.1:8000"},
        ),
    )
    assert response.status_code != 403


async def _recording_write(request: Request) -> PlainTextResponse:
    calls = cast(list[str], request.app.state.calls)  # pyright: ignore[reportAny]
    calls.append("write")
    return PlainTextResponse("ok")


def _guarded_writer() -> tuple[TestClient, list[str]]:
    calls: list[str] = []
    app = Starlette(routes=[Route("/write", _recording_write, methods=["POST"])])
    app.state.calls = calls
    app.add_middleware(RequestGuards, login=LaunchToken("test-token"))
    return TestClient(app, base_url="http://127.0.0.1"), calls


def test_origin_guard_blocks_foreign_and_missing_without_handler_call() -> None:
    test_client, calls = _guarded_writer()
    foreign = cast(
        httpx.Response,
        test_client.post("/write", headers={"host": "127.0.0.1", "origin": "http://evil.example"}),  # pyright: ignore[reportUnknownMemberType]
    )
    missing = cast(httpx.Response, test_client.post("/write", headers={"host": "127.0.0.1"}))  # pyright: ignore[reportUnknownMemberType]
    assert foreign.status_code == 403
    assert missing.status_code == 403
    assert calls == []


def test_origin_guard_allows_matching_authority_and_calls_handler() -> None:
    test_client, calls = _guarded_writer()
    response = cast(
        httpx.Response,
        test_client.post(  # pyright: ignore[reportUnknownMemberType]
            "/write",
            headers={"host": "127.0.0.1:8000", "origin": "http://127.0.0.1:8000"},
        ),
    )
    assert response.status_code == 200
    assert calls == ["write"]


def test_non_http_scope_passes_through() -> None:
    calls: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        _ = receive
        _ = send
        calls.append(cast(str, scope["type"]))

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        _ = message

    async def exercise() -> None:
        guarded = RequestGuards(app, LaunchToken("test-token"))
        await guarded(
            {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}},
            receive,
            send,
        )

    asyncio.run(exercise())
    assert calls == ["lifespan"]
