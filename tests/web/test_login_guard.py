from tests.web.support import client_with_source, headers, recording_commands


def test_api_requires_login_cookie_and_calls_nothing() -> None:
    commands, recording = recording_commands()
    test_client, _, fixture = client_with_source(commands)
    test_client.cookies.clear()  # pyright: ignore[reportUnknownMemberType]
    response = test_client.get("/api/snapshot", headers=headers())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert response.status_code == 401  # pyright: ignore[reportUnknownMemberType]
    assert fixture.snapshot_calls == 0
    assert recording.cancel_calls == []
    assert recording.force_stop_calls == []
    assert recording.stop_scheduling_calls == 0
