# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from milknado.web import WebCommands
from tests.web.support import client, headers


def test_observer_cannot_force_stop_or_stop_scheduling() -> None:
    test_client, _ = client(WebCommands())
    response = test_client.post("/api/runs/run-1/force-stop", headers=headers())
    assert response.status_code == 409
    assert response.json()["reason"] == "Force stop is unavailable."
    response = test_client.post("/api/scheduling/stop", headers=headers())
    assert response.status_code == 409
    assert response.json()["reason"] == "Stop scheduling is unavailable."
