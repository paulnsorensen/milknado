# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from tests.web.support import client


def test_auth_exchanges_token_for_cookie() -> None:
    test_client, login = client()
    response = test_client.get(f"/auth?token={login.value}", follow_redirects=False)
    assert response.status_code == 303
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert response.headers["referrer-policy"] == "no-referrer"


def test_auth_rejects_wrong_token() -> None:
    response = client()[0].get("/auth?token=wrong", follow_redirects=False)
    assert response.status_code == 403
    assert "set-cookie" not in response.headers
