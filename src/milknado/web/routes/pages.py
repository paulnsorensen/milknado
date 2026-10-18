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
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Milknado login</title>
    <style>
      :root {
        color-scheme: dark;
        --surface: #161826; --raised: #232532; --line: #2c2f40; --line-strong: #6f738a;
        --ink: #e9e9ed; --muted: #a3a7ba; --accent: #9184d9; --accent-text: #d2cefd;
      }
      @media (prefers-color-scheme: light) {
        :root {
          color-scheme: light;
          --surface: #e6e8ef; --raised: #f4f5f9; --line: #d0d3de; --line-strong: #767a8d;
          --ink: #1b1d29; --muted: #565a6b; --accent: #5d5294; --accent-text: #5d5294;
        }
      }
      * { box-sizing: border-box; }
      body {
        margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px;
        background: var(--surface); color: var(--ink);
        font: 400 13px/19px Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
      }
      main {
        width: min(440px, 100%); padding: 24px; background: var(--raised);
        border: 1px solid var(--line); border-radius: 10px;
      }
      .wordmark {
        margin: 0 0 8px; font-weight: 500; font-size: 14px; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--accent-text);
      }
      h1 { margin: 0 0 8px; font: 600 22px/28px "Source Serif 4", Georgia, serif; }
      p { margin: 0 0 16px; color: var(--muted); }
      form { display: flex; flex-direction: column; gap: 8px; }
      label { font-size: 11.5px; color: var(--muted); }
      input[type=url] {
        width: 100%; min-height: 36px; padding: 6px 10px; font: inherit; color: var(--ink);
        background: var(--surface); border: 1px solid var(--line-strong); border-radius: 3px;
      }
      input[type=url]::placeholder { color: var(--muted); }
      button {
        align-self: flex-start; min-height: 36px; padding: 6px 16px; font: inherit;
        font-weight: 500; color: var(--accent-text); background: transparent;
        border: 1px solid var(--accent); border-radius: 6px; cursor: pointer;
      }
      button:hover { background: color-mix(in srgb, var(--accent) 12%, transparent); }
      input:focus-visible, button:focus-visible {
        outline: 2px solid var(--accent); outline-offset: 2px;
      }
    </style>
  </head>
  <body>
    <main>
      <div class="wordmark">Milknado</div>
      <h1>Milknado login</h1>
      <p>Paste the launch URL printed in the terminal where you started Milknado.</p>
      <form action="/auth" method="get">
        <label for="launch-url">Paste the launch URL</label>
        <input id="launch-url" type="url" autocomplete="off" required
          placeholder="http://127.0.0.1:8765/auth?token=...">
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
    try:
        asset = (_ASSETS_DIR / requested).resolve()
    except (OSError, RuntimeError, ValueError):
        return PlainTextResponse("Not found.", status_code=404)
    if _ASSETS_DIR not in asset.parents or not asset.is_file():
        return PlainTextResponse("Not found.", status_code=404)
    return FileResponse(asset)


ROUTES = (
    Route("/", index_route, methods=["GET"]),
    Route("/login.js", login_script_route, methods=["GET"]),
    Route("/assets/{path:path}", assets_route, methods=["GET"]),
)
