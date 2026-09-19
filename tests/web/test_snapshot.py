# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from tests.web.support import client, headers


def test_snapshot_returns_goal_and_capabilities() -> None:
    response = client()[0].get("/api/snapshot", headers=headers())
    assert response.status_code == 200
    assert response.json()["goal"] == "fixture goal"
    assert response.json()["capabilities"]["force_stop"]["available"] is False
