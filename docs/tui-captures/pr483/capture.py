"""Matched PR 483 review frames from real Textual apps and a fixed snapshot fixture."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from playwright.async_api import async_playwright
from rich.text import Text
from textual.widgets import Input, Select, TabbedContent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-steering"))

from fixture import Source

from milknado.app.run_source import ExecutionSnapshot, NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch import AttachedWatchSource
from milknado.app.watch_tui import WatchApp

MATRIX = {
    "run": (
        "main",
        "help",
        "confirmation",
        "session",
        "permission",
        "error",
        "owner-unavailable",
        "review",
        "stop-confirmation",
        "submitted",
        "no-worktree",
    ),
    "watch": ("main", "help", "error", "owner-unavailable", "review", "no-worktree"),
    "attached-watch": (
        "main",
        "help",
        "session",
        "permission",
        "error",
        "owner-unavailable",
        "review",
        "submitted",
        "no-worktree",
    ),
}
SIZES = ((120, 40), (80, 24))


class ReviewSource(Source):
    """Populate the durable review field only in revisions that expose it."""

    def session_input(self, _run_id: str, _command: object) -> bool:
        """Admit the synthetic command without contacting a worker."""
        return True

    def snapshot(self, request: NodeSnapshotRequest | None = None) -> ExecutionSnapshot:
        snapshot = super().snapshot(request)
        if self.state != "review" or "pending_goal_reviews" not in {
            field.name for field in fields(snapshot)
        }:
            return snapshot
        from milknado.domains.graph import GoalReviewDecision, GoalReviewRecord

        review = GoalReviewRecord(
            review_id=4,
            goal_id=12,
            goal_revision="sha256:fixture-goal",
            evidence="The worker requests a change to the top-level goal.",
            proposed_change="Limit steering to the current execution.",
            decision=GoalReviewDecision.PENDING,
            affected_node_ids=None,
            reviewer="fixture-worker",
            assessed_at="2026-09-12T12:03:00+00:00",
            decided_at=None,
            decided_by=None,
        )
        return replace(snapshot, pending_goal_reviews=(review,))


def make_app(surface: str, state: str) -> ExecutionApp | WatchApp:
    source = ReviewSource(state, read_only=surface == "watch")
    if surface == "run":
        return ExecutionApp(source)
    if surface == "attached-watch":
        return WatchApp(AttachedWatchSource(source, source.session_input), read_only=False)
    return WatchApp(source)


async def capture(output: Path, surface: str, state: str, size: tuple[int, int]) -> dict:
    app = make_app(surface, state)
    app.theme = "textual-dark"
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if state in {"help", "confirmation", "stop-confirmation"}:
            await pilot.press({"help": "f1", "confirmation": "f", "stop-confirmation": "s"}[state])
        elif state == "no-worktree":
            app.action_open_detail()
            await pilot.pause()
            app.query_one("#run-tabs", TabbedContent).active = "changes"
        elif state in {"session", "permission", "submitted"}:
            await pilot.press("i")
            await pilot.pause()
            if state == "permission":
                app.query_one("#session-action", Select).value = "approve"
                app.query_one("#session-permission", Select).value = "perm-1"
            else:
                app.query_one("#session-input", Input).value = "Keep the change bounded."
                await pilot.pause()
                if state == "submitted":
                    await pilot.press("enter")
                    await asyncio.wait_for(app.workers.wait_for_complete(), timeout=10)
        await pilot.pause()
        stem = f"{surface}-{state}-{size[0]}x{size[1]}"
        svg = output / f"{stem}.svg"
        svg.write_text(app.export_screenshot(title="Milknado PR 483"), encoding="utf-8")
        focus = app.screen.focused
        return {
            "stem": stem,
            "surface": surface,
            "state": state,
            "size": size,
            "focus": focus.id if focus else None,
            "screen": type(app.screen).__name__,
            "pending_goal_reviews": len(getattr(app.snapshot, "pending_goal_reviews", ())),
            "submitted_draft": app.query_one("#session-input", Input).value
            if state == "submitted"
            else None,
        }


async def render_pngs(output: Path, records: list[dict]) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(device_scale_factor=1)
        for record in records:
            await page.goto((output / f"{record['stem']}.svg").as_uri())
            svg = page.locator("svg")
            await svg.screenshot(path=str(output / f"{record['stem']}.png"))
        await browser.close()


async def main(args: argparse.Namespace) -> None:
    import milknado

    root = args.source_root.resolve()
    if root not in Path(milknado.__file__).resolve().parents:
        raise ValueError(f"Wrong runtime: {milknado.__file__}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output / "manifest.json").read_text()) if args.resume else None
    if previous and previous["source_revision"] != revision:
        raise ValueError("Cannot resume captures from another revision.")
    records = previous["records"] if previous else []
    completed = {record["stem"] for record in records}
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for size in SIZES:
            for surface, states in MATRIX.items():
                for state in states:
                    if f"{surface}-{state}-{size[0]}x{size[1]}" not in completed:
                        records.append(await capture(output, surface, state, size))
    await render_pngs(output, [record for record in records if record["stem"] not in completed])
    manifest = {
        "source_revision": revision,
        "theme": "textual-dark",
        "clock": "12:00:00",
        "fixture": "../agent-steering/fixture.py from 7bae8d62bbd5af997727fda0f35ea60cf1ea79ce",
        "limits": [
            "Synthetic snapshots, not a live worker or CLI process.",
            "Attached watch uses an in-process synthetic command admission callback.",
            "Submitted input uses synthetic admission; "
            + "no provider transport or database persistence.",
            "Before runtime has no durable review field; "
            + "its existing review event stays identical.",
            "Browser rendering proves Textual layout, not emulator palette fidelity.",
        ],
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{revision}: {len(records)} SVG and PNG pairs in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Keep existing inspected frames.")
    asyncio.run(main(parser.parse_args()))
