# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from tests.web.support import client, headers


def test_api_requires_login_cookie() -> None:
    test_client, _ = client()
    test_client.cookies.clear()
    response = test_client.get("/api/snapshot", headers=headers())
    assert response.status_code == 401
