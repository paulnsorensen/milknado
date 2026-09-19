from milknado.web import WebCommands
from tests.web.support import client, headers


def test_owner_capabilities_are_available() -> None:
    commands = WebCommands(
        force_stop=lambda run_id: {"run_id": run_id},
        stop_scheduling=lambda: None,
    )
    payload = client(commands)[0].get("/api/snapshot", headers=headers()).json()["capabilities"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert payload["force_stop"]["available"] is True
    assert payload["stop_scheduling"]["available"] is True
