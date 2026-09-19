# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from tests.web.support import client, headers


def test_snapshot_returns_goal_and_capabilities() -> None:
    response = client()[0].get("/api/snapshot", headers=headers())
    assert response.status_code == 200
    payload = response.json()
    assert payload["goal"] == "fixture goal"
    assert payload["listener_errors"] == ["fixture listener error"]
    assert response.json()["capabilities"]["force_stop"]["available"] is False
