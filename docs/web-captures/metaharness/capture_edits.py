"""Capture the metaharness graph-edit states from a running `milknado web` server.

Usage: python capture_edits.py <server-log> <out-dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright


def shot(page: Page, out: Path, name: str) -> None:
    page.wait_for_timeout(400)
    page.screenshot(path=str(out / f"{name}.png"))
    print("wrote", out / f"{name}.png")


def card(page: Page, text: str) -> Locator:
    return page.locator('[data-mh-node][data-mode="compact"]', has_text=text).first


def main(url: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(url)
        page.evaluate(
            "localStorage.setItem('milknado.view', 'metaharness');"
            " localStorage.setItem('milknado.theme', 'dark')"
        )
        page.reload()
        page.wait_for_selector(".mh-node")
        card(page, "Session stats strip").click()
        page.keyboard.press("Escape")
        card(page, "Layout issues inbox").hover()
        shot(page, out, "e1-hover-plus")
        box = card(page, "Session stats strip").bounding_box()
        target = card(page, "Layout issues inbox").bounding_box()
        assert box and target
        page.mouse.move(box["x"] + 20, box["y"] + 20)
        page.mouse.down()
        page.mouse.move(target["x"] + 30, target["y"] + 25, steps=8)
        shot(page, out, "e2-drag-over")
        page.mouse.up()
        page.wait_for_timeout(800)
        shot(page, out, "e3-after-drop")
        card(page, "Layout issues inbox").hover()
        page.get_by_role("button", name=re.compile("^Add a prerequisite to Layout")).click()
        shot(page, out, "e4-add-dialog")
        page.keyboard.press("Escape")
        page.get_by_role("button", name="Details").click()
        shot(page, out, "e5-details-actions")
        page.locator(".mh-crumbs").get_by_role("button", name="Projects").click()
        shot(page, out, "e6-projects-new-goal")
        page.locator(".mh-new-goal").click()
        shot(page, out, "e7-new-goal-dialog")
        page.keyboard.press("Escape")
        browser.close()
    print("errors:", errors or "none")


if __name__ == "__main__":
    log = Path(sys.argv[1]).read_text()
    match = re.search(r"http://127\.0\.0\.1:\d+/auth\?token=[A-Za-z0-9_-]+", log)
    if match is None:
        raise SystemExit("no login URL")
    main(match.group(0), Path(sys.argv[2]))
