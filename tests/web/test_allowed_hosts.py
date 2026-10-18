# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The Host allow-list widens past loopback only for operator-named host names."""

import pytest
from starlette.testclient import TestClient

from milknado.web import WebCommands, create_app
from milknado.web.guards import allowed_host
from milknado.web.login import LaunchToken
from tests.web.support import FixtureSnapshotSource

TAILNET = "crabbot.tail0a0bee.ts.net"


def _extra_host_client(extra_hosts: tuple[str, ...]) -> TestClient:
    login = LaunchToken("test-token")
    app = create_app(FixtureSnapshotSource(), WebCommands(), login, extra_hosts)
    test_client = TestClient(app, base_url=f"https://{TAILNET}:8443")
    test_client.cookies.set(login.cookie_name, login.value)
    return test_client


def test_extra_host_is_served_and_accepts_its_https_origin() -> None:
    test_client = _extra_host_client((TAILNET,))
    authority = f"{TAILNET.upper()}:8443"

    read = test_client.get("/api/snapshot", headers={"host": authority})
    write = test_client.post(
        "/api/scheduling/pause", headers={"host": authority, "origin": f"https://{authority}"}
    )
    foreign = test_client.post(
        "/api/scheduling/pause", headers={"host": authority, "origin": "https://evil.example"}
    )

    assert read.status_code == 200
    assert write.status_code == 409
    assert foreign.status_code == 403


def test_extra_host_is_rejected_unless_listed() -> None:
    response = _extra_host_client(()).get("/api/snapshot", headers={"host": f"{TAILNET}:8443"})

    assert response.status_code == 400


def test_create_app_normalizes_extra_hosts_from_direct_callers() -> None:
    test_client = _extra_host_client((f"{TAILNET.upper()}.",))

    response = test_client.get("/api/snapshot", headers={"host": f"{TAILNET}:8443"})

    assert response.status_code == 200


def test_trailing_dot_host_and_origin_case_match_the_same_host() -> None:
    test_client = _extra_host_client((TAILNET,))
    dotted = f"{TAILNET}.:8443"

    read = test_client.get("/api/snapshot", headers={"host": dotted})
    dotted_write = test_client.post(
        "/api/scheduling/pause", headers={"host": dotted, "origin": f"https://{dotted}"}
    )
    mixed_case_write = test_client.post(
        "/api/scheduling/pause",
        headers={"host": f"{TAILNET.upper()}:8443", "origin": f"https://{TAILNET}:8443"},
    )
    other_port = test_client.post(
        "/api/scheduling/pause",
        headers={"host": f"{TAILNET}:8443", "origin": f"https://{TAILNET}:9443"},
    )

    assert read.status_code == 200
    assert dotted_write.status_code == 409
    assert mixed_case_write.status_code == 409
    assert other_port.status_code == 403


def test_create_app_rejects_an_extra_host_that_is_not_a_host_name() -> None:
    with pytest.raises(ValueError, match="is not a host name"):
        _ = _extra_host_client(("host:80",))


@pytest.mark.parametrize("value", ["host:80", "http://host", "host/path", "", "-host", "a..b"])
def test_allowed_host_rejects_anything_but_a_host_name(value: str) -> None:
    with pytest.raises(ValueError, match="is not a host name"):
        _ = allowed_host(value)


def test_allowed_host_normalizes_case_and_trailing_dot() -> None:
    assert allowed_host(f" {TAILNET.upper()}. ") == TAILNET
