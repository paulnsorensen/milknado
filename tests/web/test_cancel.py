# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from milknado.web import WebCommands
from tests.web.support import client, headers, recording_commands


def test_cancel_calls_capability() -> None:
    commands, recording = recording_commands()
    response = client(commands)[0].post("/api/runs/run-1/cancel", headers=headers())
    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-1",
        "status": "cancelled",
        "terminal": True,
        "terminal_reason": "cancelled by request",
    }
    assert recording.cancel_calls == ["run-1"]


def test_cancel_unknown_run_returns_not_found() -> None:
    def cancel(run_id: str) -> dict[str, object]:
        raise ValueError(f"run {run_id!r} not found")

    response = client(commands=WebCommands(cancel=cancel))[0].post(
        "/api/runs/missing/cancel", headers=headers()
    )
    assert response.status_code == 404
    assert "not found" in response.json()["reason"]
