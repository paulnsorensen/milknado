# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from starlette.routing import Route

from milknado.web.routes import discover_routes


def test_routes_are_discovered() -> None:
    routes = [route for route in discover_routes() if isinstance(route, Route)]
    assert {route.path for route in routes} == {
        "/",
        "/auth",
        "/login.js",
        "/assets/{path:path}",
        "/api/snapshot",
        "/api/stream",
        "/api/nodes",
        "/api/nodes/{node_id:int}",
        "/api/nodes/{node_id:int}/move",
        "/api/nodes/{node_id:int}/archive",
        "/api/reviews",
        "/api/reviews/{review_id:int}/decision",
        "/api/runs/{run_id}/session-input",
        "/api/runs/{run_id}/cancel",
        "/api/runs/{run_id}/force-stop",
        "/api/runs/{run_id}/changes",
        "/api/runs/{run_id}/diff",
        "/api/scheduling/stop",
    }
    writes = {
        (route.path, method)
        for route in routes
        for method in (route.methods or ())
        if method in {"POST", "PATCH"}
    }
    assert writes == {
        ("/api/nodes", "POST"),
        ("/api/nodes/{node_id:int}", "PATCH"),
        ("/api/nodes/{node_id:int}/move", "POST"),
        ("/api/nodes/{node_id:int}/archive", "POST"),
        ("/api/reviews/{review_id:int}/decision", "POST"),
        ("/api/runs/{run_id}/session-input", "POST"),
        ("/api/runs/{run_id}/cancel", "POST"),
        ("/api/runs/{run_id}/force-stop", "POST"),
        ("/api/scheduling/stop", "POST"),
    }
