# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from starlette.routing import Route

from milknado.web.routes import discover_routes


def test_routes_are_discovered() -> None:
    paths = {route.path for route in discover_routes() if isinstance(route, Route)}
    assert "/auth" in paths
    assert "/api/snapshot" in paths
