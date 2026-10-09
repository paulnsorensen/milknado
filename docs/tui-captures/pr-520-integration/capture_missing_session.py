"""Capture matched missing-session browser states through the real local web app."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Browser, ConsoleMessage, Page, Response, expect, sync_playwright

import milknado
from milknado.app.watch import WatchSnapshotSource
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.graph import MikadoGraph
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from scripts._browser_server import BROWSER_TOKEN, BrowserServer

PROJECT_ROOT = Path("/tmp/milknado-pr520-offline-project")
SESSION_ID = "missing-session"
SNAPSHOT_PATH = f"/api/coordinators/{SESSION_ID}/snapshot"
MISSING_BODY = {"error": "Coordinator session does not exist."}
MISSING_NOTICE = "Coordinator session was not found."
SIZES = ((1440, 900), (390, 844))
RESOURCE_ERROR = (
    "Failed to load resource: the server responded with a status of 404 (Not Found)"
)


@dataclass
class BrowserEvidence:
    failures: list[str] = field(default_factory=list)
    missing_responses: list[Response] = field(default_factory=list)
    missing_console: list[str] = field(default_factory=list)

    def on_response(self, response: Response) -> None:
        if response.status >= 500:
            self.failures.append(f"HTTP {response.status}: {response.url}")
        if urlparse(response.url).path == SNAPSHOT_PATH:
            if response.status == 404:
                self.missing_responses.append(response)
            else:
                self.failures.append(f"snapshot HTTP {response.status}: {response.url}")

    def on_console(self, message: ConsoleMessage) -> None:
        if message.type != "error":
            return
        location = message.location.get("url", "")
        if message.text == RESOURCE_ERROR and urlparse(location).path == SNAPSHOT_PATH:
            self.missing_console.append(location)
        else:
            self.failures.append(f"console: {message.text} at {location}")

    def assert_expected(self, side: str) -> None:
        assert self.missing_responses, "browser did not request missing snapshot"
        urls = [response.url for response in self.missing_responses]
        assert self.missing_console == urls, (self.missing_console, urls)
        if side == "before":
            assert len(urls) >= 2, urls
        else:
            assert len(urls) == 1, urls
        assert not self.failures, "; ".join(self.failures)


def capture_page(page: Page, server: BrowserServer, side: str, output: Path) -> None:
    evidence = BrowserEvidence()
    page.on("pageerror", lambda error: evidence.failures.append(f"pageerror: {error}"))
    page.on("response", evidence.on_response)
    page.on("console", evidence.on_console)
    page.add_init_script(
        f"""localStorage.setItem('milknado.theme', 'dark');
localStorage.setItem('milknado.coordinator.session', '{SESSION_ID}');"""
    )
    response = page.goto(server.login_url)
    assert response is not None and response.status < 500
    expect(page.get_by_role("heading", name=str(PROJECT_ROOT))).to_be_visible()
    expect(page.get_by_role("complementary", name="Coordinator cockpit")).to_be_visible()
    expect(page.get_by_role("alert")).to_have_count(1)
    if side == "before":
        expect(page.get_by_role("button", name="New goal")).to_be_visible()
        expect(page.get_by_role("button", name="Start goal")).to_have_count(0)
    else:
        expect(page.get_by_role("button", name="New goal")).to_have_count(0)
        expect(page.get_by_role("button", name="Start goal")).to_be_visible()
        expect(page.get_by_role("alert")).to_contain_text(MISSING_NOTICE)
        assert page.evaluate("localStorage.getItem('milknado.coordinator.session')") is None
    reply = page.request.get(f"{server.base_url}{SNAPSHOT_PATH}")
    assert reply.status == 404 and reply.json() == MISSING_BODY
    page.wait_for_timeout(2300)
    evidence.assert_expected(side)
    print(
        f"EXPECTED 404 {side}: browser responses={len(evidence.missing_responses)}; "
        f"body={MISSING_BODY}"
    )
    if side == "after":
        expect(page.get_by_role("alert")).to_have_count(1)
    _ = page.evaluate("document.fonts.ready")
    viewport = page.viewport_size
    assert viewport is not None
    target = output / f"{side}-missing-session-{viewport['width']}x{viewport['height']}.png"
    page.screenshot(path=str(target), animations="disabled")
    print(f"CAPTURE {target} sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")


def capture_one(browser: Browser, side: str, size: tuple[int, int], output: Path) -> None:
    width, height = size
    with tempfile.TemporaryDirectory(prefix="milknado-pr520-missing-") as directory:
        db_path = Path(directory) / "graph.db"
        graph = MikadoGraph(db_path)
        watch = WatchSnapshotSource(PROJECT_ROOT, db_path)
        source = PolledSnapshotSource(watch, interval=0.05)
        control = CoordinatorControl(graph, PROJECT_ROOT)
        login = LaunchToken(BROWSER_TOKEN)
        commands = WebCommands(
            coordinator=control,
            graph_edits=GraphEditCommands(graph, frozenset({"implement"}), PROJECT_ROOT),
        )
        server = BrowserServer(create_app(source, commands, login), login)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            color_scheme="dark",
            device_scale_factor=1,
            locale="en-US",
            timezone_id="UTC",
        )
        try:
            assert graph.get_all_nodes() == []
            source.start()
            server.start()
            capture_page(context.new_page(), server, side, output)
        finally:
            context.close()
            server.stop()
            source.close()
            graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--side", choices=("before", "after"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    assert Path(milknado.__file__).resolve().is_relative_to(checkout), (
        "PYTHONPATH must import milknado from --checkout"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"OFFLINE {args.side}: real HTTP/SQLite UI; missing {SESSION_ID}; "
        "empty graph; dark theme; no live worker or control authority.",
        flush=True,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for size in SIZES:
                capture_one(browser, args.side, size, output)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
