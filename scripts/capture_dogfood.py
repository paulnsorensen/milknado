"""Capture the seeded dashboard states used by the web dogfood report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from playwright.sync_api import expect, sync_playwright

DEFAULT_OUTPUT = Path("docs/web-ui")


def capture_dashboard(url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        _ = page.goto(url)
        expect(page.locator(".mk-shell")).to_be_visible()
        _ = page.screenshot(path=output_dir / "dogfood-round2-live.png", full_page=True)

        root_switcher = page.get_by_role("combobox", name="Root goal")
        _ = root_switcher.select_option("43")
        expect(page.get_by_role("heading", name="Parked roadmap 43")).to_be_visible()
        _ = page.screenshot(path=output_dir / "dogfood-round2-root-switcher.png", full_page=True)
        _ = root_switcher.select_option("87")
        _ = page.locator("button.mk-node", has_text="Live worker task").click()
        expect(page.get_by_text("none", exact=True)).to_be_visible()
        _ = page.screenshot(path=output_dir / "dogfood-round2-active-run.png", full_page=True)
        _ = page.locator("button.mk-node", has_text="Failed worker 1").click()

        expect(page.get_by_role("alert").filter(has_text="worker session gone")).to_be_visible()
        _ = page.get_by_role("button", name="Changes", exact=True).click()
        expect(page.get_by_text("No changes")).to_be_visible()
        _ = page.screenshot(
            path=output_dir / "dogfood-round2-failed-no-changes.png", full_page=True
        )

        _ = page.locator("button.mk-node", has_text="This long description verifies").click()
        expect(page.get_by_role("button", name="Expand description")).to_be_visible()
        for tab_name in ("Session", "Changes", "Details"):
            expect(page.get_by_role("button", name=tab_name, exact=True)).to_be_visible()
        _ = page.screenshot(
            path=output_dir / "dogfood-round2-long-description.png", full_page=True
        )
        _ = page.get_by_role("button", name="Expand description").click()
        _ = page.screenshot(
            path=output_dir / "dogfood-round2-long-description-expanded.png", full_page=True
        )
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("url", help="the authenticated dashboard launch URL")
    _ = parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    capture_dashboard(cast(str, args.url), cast(Path, args.output_dir))


if __name__ == "__main__":
    main()
