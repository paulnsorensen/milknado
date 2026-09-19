# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""Authentication exchange endpoint."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.routing import Route

from milknado.web.app import WebContext
from milknado.web.login import LaunchToken


def auth_route(request: Request) -> RedirectResponse | PlainTextResponse:
    context = cast(WebContext, request.app.state.web)
    login = context.login
    token = request.query_params.get("token")
    if not isinstance(login, LaunchToken) or not login.verify(token):
        return PlainTextResponse("Invalid launch token.", status_code=403)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(login.cookie_name, login.value, httponly=True, samesite="strict")
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


ROUTES = (Route("/auth", auth_route, methods=["GET"]),)
