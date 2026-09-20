"""AC-14: the theme switch sets the palette and persists it across a reload."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.conftest import BrowserServer

pytestmark = pytest.mark.browser

DARK_SURFACE = "rgb(22, 24, 38)"
LIGHT_SURFACE = "rgb(230, 232, 239)"


def _shell_background(page: Page) -> str:
    background = page.eval_on_selector(  # pyright: ignore[reportAny]
        ".mk-shell", "el => getComputedStyle(el).backgroundColor"
    )
    return str(background)  # pyright: ignore[reportAny]


@pytest.mark.parametrize(
    "case",
    [
        ("Light", "light", LIGHT_SURFACE),
        ("Dark", "dark", DARK_SURFACE),
    ],
)
def test_theme_switch_sets_and_persists_the_palette(
    page: Page,
    browser_server: BrowserServer,
    case: tuple[str, str, str],
) -> None:
    button_name, theme, background = case
    _ = page.goto(browser_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name=button_name, exact=True).click()

    expect(page.locator("html")).to_have_attribute("data-theme", theme)
    assert _shell_background(page) == background

    _ = page.reload()
    page.wait_for_load_state("networkidle")

    expect(page.locator("html")).to_have_attribute("data-theme", theme)
    assert _shell_background(page) == background
