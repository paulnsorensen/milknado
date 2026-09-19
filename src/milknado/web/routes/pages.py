"""Frontend pages and authenticated static assets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from starlette.routing import Route

from milknado.web.app import WebContext

_STATIC_DIR = Path(__file__).parent.parent / "static"
_ASSETS_DIR = (_STATIC_DIR / "assets").resolve()
_LOGIN_PAGE = """<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>Milknado login</title></head>
  <body>
    <main>
      <h1>Milknado login</h1>
      <form action="/auth" method="get">
        <label for="launch-url">Paste the launch URL</label>
        <input id="launch-url" type="url" autocomplete="off" required>
        <input id="token" name="token" type="hidden">
        <button type="submit">Log in</button>
      </form>
    </main>
    <script src="/login.js"></script>
  </body>
</html>
"""


def index_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    if not context.login.verify(request.cookies.get(context.login.cookie_name)):
        return HTMLResponse(_LOGIN_PAGE)
    return Response((_STATIC_DIR / "index.html").read_bytes(), media_type="text/html")


def login_script_route(_request: Request) -> Response:
    return FileResponse(_STATIC_DIR / "login.js")


def assets_route(request: Request) -> Response:
    context = cast(WebContext, request.app.state.web)  # pyright: ignore[reportAny]
    if not context.login.verify(request.cookies.get(context.login.cookie_name)):
        return PlainTextResponse("Authentication required.", status_code=401)
    requested = cast(str, request.path_params.get("path", ""))
    asset = (_ASSETS_DIR / requested).resolve()
    if _ASSETS_DIR not in asset.parents or not asset.is_file():
        return PlainTextResponse("Not found.", status_code=404)
    return FileResponse(asset)


ROUTES = (
    Route("/", index_route, methods=["GET"]),
    Route("/login.js", login_script_route, methods=["GET"]),
    Route("/assets/{path:path}", assets_route, methods=["GET"]),
)
