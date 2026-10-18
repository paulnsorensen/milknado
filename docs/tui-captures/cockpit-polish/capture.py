"""Capture the login page and coordinator cockpit.

Usage: capture.py <launch_log> <out_dir> [--proposal]

`--proposal` also starts one goal and captures its pending plan proposal.

The launch log comes from `milknado launch --no-open --port N --project-root DIR`.
Run with `python -I` from outside the repository.
"""

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

match = re.search(r"http://127\.0\.0\.1:\d+/auth\?token=\S+", Path(sys.argv[1]).read_text())
if match is None:
    sys.exit("no token URL in launch log")
token_url, out = match.group(0), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
base = token_url.split("/auth")[0]

PROPOSAL_VIEWS = [
    ("proposal-1440x900-dark", 1440, 900, "dark"),
    ("proposal-narrow-390x844-dark", 390, 844, "dark"),
]


def start_pending_proposal(request) -> str:
    """Start one Claude goal and request a plan; the fixture planner makes it pending."""
    started = request.post(f"{base}/api/coordinators/commands", data={
        "kind": "start_goal", "command_id": "capture-start",
        "description": "Ship the billing dashboard", "provider": "claude",
    }).json()
    session_id = started["result"]["id"]
    planned = request.post(f"{base}/api/coordinators/{session_id}/commands", data={
        "kind": "plan_goal", "command_id": "capture-plan",
    }, timeout=120_000).json()
    if planned["status"] != "accepted":
        sys.exit(f"plan_goal failed: {planned}")
    return session_id


with sync_playwright() as p:
    browser = p.chromium.launch()
    fresh = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    page = fresh.new_page()
    page.goto(base + "/")
    page.wait_for_timeout(1000)
    page.screenshot(path=out / "login-1440x900-dark.png")
    fresh.close()
    for name, w, h, scheme in [
        ("cockpit-1440x900-dark", 1440, 900, "dark"),
        ("cockpit-1280x720-light", 1280, 720, "light"),
        ("cockpit-narrow-390x844-dark", 390, 844, "dark"),
    ]:
        ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
        page = ctx.new_page()
        page.goto(token_url)
        page.wait_for_timeout(3000)
        page.screenshot(path=out / f"{name}.png")
        ctx.close()
    if "--proposal" in sys.argv:
        api = browser.new_context(extra_http_headers={"Origin": base})
        api.new_page().goto(token_url)
        session_id = start_pending_proposal(api.request)
        api.close()
        for name, w, h, scheme in PROPOSAL_VIEWS:
            ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
            page = ctx.new_page()
            page.goto(token_url)
            page.evaluate("id => localStorage.setItem('milknado.coordinator.session', id)", session_id)
            page.reload()
            page.wait_for_timeout(4000)
            page.screenshot(path=out / f"{name}.png")
            ctx.close()
    browser.close()
