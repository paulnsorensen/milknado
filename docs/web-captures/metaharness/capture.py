"""Capture metaharness screenshots from a running `milknado web` server.

Usage: python capture.py <token-url> <out-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

W, H = 1440, 900


def shot(page: Page, out: Path, name: str) -> None:
    page.wait_for_timeout(500)
    page.screenshot(path=str(out / f"{name}.png"))
    print("wrote", out / f"{name}.png")


def main(url: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H}, color_scheme="dark")
        page.on(
            "console",
            lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(url)
        page.evaluate(
            "localStorage.setItem('milknado.view', 'metaharness'); localStorage.setItem('milknado.theme', 'dark')"
        )
        page.reload()
        page.wait_for_selector('[data-view="metaharness"]')
        page.wait_for_selector(".mh-node")
        page.mouse.move(700, 450)
        shot(page, out, "01-thread")
        # Popover on the node that needs a decision.
        page.locator(".mh-node.is-need").first.click()
        shot(page, out, "02-popover")
        page.keyboard.press("Escape")
        # Diff takeover for the annotation node.
        page.locator("[data-mh-node]", has_text="Annotation").first.click()
        page.keyboard.press("Escape")
        diff = page.locator(".mh-chat-actions button", has_text="Diff")
        diff.first.wait_for(timeout=10000)
        diff.first.click()
        shot(page, out, "03-diff")
        page.keyboard.press("Escape")
        page.get_by_role("button", name="Details").click()
        shot(page, out, "04-details")
        page.locator("[data-mh-node]", has_text="Session stats strip").first.click()
        shot(page, out, "05-failed")
        page.keyboard.press("Escape")
        page.keyboard.press("g")
        shot(page, out, "06-expanded")
        page.keyboard.press("Escape")
        page.mouse.move(700, 450)
        page.mouse.move(2, 400)
        page.wait_for_selector(".mh-rail")
        shot(page, out, "07-rail")
        page.locator(".mh-rail-item", has_text="Knowledge").click()
        page.mouse.move(700, 450)
        shot(page, out, "08-knowledge")
        page.mouse.move(700, 450)
        page.mouse.move(2, 400)
        page.locator(".mh-rail-item", has_text="Stats").click()
        page.mouse.move(700, 450)
        shot(page, out, "09-stats")
        page.mouse.move(700, 450)
        page.mouse.move(2, 400)
        page.locator(".mh-rail-item", has_text="Issues").click()
        page.mouse.move(700, 450)
        shot(page, out, "10-issues")
        page.locator(".mh-crumbs button", has_text="Projects").click()
        shot(page, out, "11-projects")
        browser.close()
    print("console errors:", errors or "none")


def token_url(log: Path) -> str:
    import re

    match = re.search(r"http://127\.0\.0\.1:\d+/auth\?token=[A-Za-z0-9_-]+", log.read_text())
    if match is None:
        raise SystemExit(f"no login URL in {log}")
    return match.group(0)


if __name__ == "__main__":
    main(token_url(Path(sys.argv[1])), Path(sys.argv[2]))
