"""T3 web spike checks. The gate's browser lane runs only tests/browser."""

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser

DEMO = Path(__file__).resolve().parents[2] / "scripts/spikes/t3_web/index.html"


def test_completion_does_not_unlock_dependent_goal(page: Page) -> None:
    _ = page.goto(DEMO.as_uri())
    status = page.locator("#b-status")
    page.get_by_role("button", name="Complete A runtime").click()
    expect(page.locator("#runtime")).to_have_text("Completed")
    expect(page.locator("#review")).to_have_text("Evidence ready · awaiting review")
    expect(status).to_have_text("Blocked: A is not accepted")

    page.get_by_role("button", name="Accept A evidence").click()
    expect(page.locator("#review")).to_have_text("Evidence accepted")
    expect(status).to_have_text("Ready: A is accepted")


def test_runtime_uncertainty_and_stale_result(page: Page) -> None:
    _ = page.goto(DEMO.as_uri())
    page.get_by_role("button", name="Request cancellation").click()
    expect(page.locator("#runtime")).to_have_text("Cancellation requested · stop unconfirmed")
    page.get_by_role("button", name="Disconnect").click()
    expect(page.locator("#runtime")).to_have_text("Unknown · disconnected")
    expect(page.get_by_role("button", name="Complete A runtime")).to_be_disabled()

    page.get_by_role("button", name="Load replacement fixture").click()
    page.get_by_role("button", name="Replay old completion").click()
    expect(page.locator("#notice")).to_have_text("Ignored completion from attempt A-01.")
    expect(page.locator("#attempt")).to_have_text("A-02 · T3")
    expect(page.locator("#runtime")).to_have_text("Running")
    expect(page.locator("#review")).to_have_text("No evidence")


def test_late_completion_arrives_while_cancel_is_unconfirmed(page: Page) -> None:
    _ = page.goto(DEMO.as_uri())
    page.get_by_role("button", name="Request cancellation").click()
    expect(page.locator("#runtime")).to_have_text("Cancellation requested · stop unconfirmed")
    page.get_by_role("button", name="Complete A runtime").click()
    expect(page.locator("#runtime")).to_have_text("Completed")
    expect(page.locator("#review")).to_have_text("Evidence ready · awaiting review")
    expect(page.locator("#review-task")).to_be_visible()
    expect(page.get_by_role("button", name="Complete A runtime")).to_be_disabled()


def test_question_answer_does_not_approve_permission(page: Page) -> None:
    _ = page.goto(DEMO.as_uri())
    page.get_by_role("button", name="Answer fixture question").click()
    expect(page.locator("#question")).to_have_text("User question · answered in fixture")
    expect(page.get_by_role("button", name="Approve in T3")).to_be_disabled()
    expect(page.locator("#permission")).to_contain_text("Provider permission · pending")
