from tests.web.support import client


def test_foreign_host_is_rejected() -> None:
    response = client()[0].get("/api/snapshot", headers={"host": "evil.example"})
    assert response.status_code == 400


def test_missing_origin_is_rejected_on_write() -> None:
    response = client()[0].post("/api/unknown", headers={"host": "127.0.0.1"})
    assert response.status_code == 403
