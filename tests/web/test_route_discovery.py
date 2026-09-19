from milknado.web.routes import discover_routes


def test_routes_are_discovered() -> None:
    paths = {route.path for route in discover_routes()}
    assert "/auth" in paths
    assert "/api/snapshot" in paths
