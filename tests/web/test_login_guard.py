from typing import cast

import httpx
import pytest

from tests.web.support import client_with_source, headers, recording_commands


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/snapshot"),
        ("get", "/api/stream"),
        ("post", "/api/runs/run-1/session-input"),
        ("post", "/api/runs/run-1/cancel"),
        ("patch", "/api/nodes/1"),
    ],
)
def test_api_requires_login_cookie_and_calls_nothing(method: str, path: str) -> None:
    commands, recording = recording_commands()
    test_client, _, fixture = client_with_source(commands)
    test_client.cookies.clear()  # pyright: ignore[reportUnknownMemberType]
    response = cast(httpx.Response, getattr(test_client, method)(path, headers=headers()))
    assert response.status_code == 401
    assert fixture.snapshot_calls == 0
    assert recording.cancel_calls == []
    assert recording.force_stop_calls == []
    assert recording.stop_scheduling_calls == 0
