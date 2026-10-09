"""Capture real HTTP/SQLite browser views on one checkout at fixed viewport sizes.

Run once per checkout with matching PYTHONPATH. Baseline proposal states show the
ordinary graph for the same goal; the baseline has no proposal controls.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID

from capture_browser_states import (
    GOAL,
    PLAN_STATES,
    PROPOSAL_FOCUS_STATES,
    BrowserErrors,
    after_state,
    expect_goal_node,
)
from playwright.sync_api import Browser, BrowserContext, Page, expect, sync_playwright

import milknado
from milknado.app.watch import WatchSnapshotSource
from milknado.domains.batching import BatchPlan
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices, StartGoal
from milknado.domains.graph import MikadoGraph
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app
from milknado.web.commands import GraphEditCommands
from scripts._browser_server import BROWSER_TOKEN, BrowserServer

PROJECT_ROOT = Path("/tmp/milknado-pr520-offline-project")
STATES = (
    "intake",
    "pending",
    "accepted",
    "rejected",
    "planner-unavailable",
    "stale",
    "applying",
    "recovery-unavailable",
    "coordinator-unavailable",
)
SIZES = ((1440, 900), (390, 844))
UUID_SEED = UUID(int=1)
FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class CaptureDateTime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return FIXED_NOW.astimezone(tz) if tz else FIXED_NOW.replace(tzinfo=None)


@contextmanager
def fixed_source_values() -> Iterator[None]:
    prefix = "milknado.domains.coordinator"
    with ExitStack() as stack:
        stack.enter_context(patch(f"{prefix}.persistence.uuid4", return_value=UUID_SEED))
        for module in ("persistence", "journal", "commands", "recovery"):
            stack.enter_context(patch(f"{prefix}.{module}.datetime", CaptureDateTime))
        yield


@dataclass(frozen=True)
class CaptureCase:
    revision: str
    state: str
    size: tuple[int, int]
    output: Path


RANDOM_UUID_SCRIPT = """Object.defineProperty(Crypto.prototype, 'randomUUID', {
  value: (() => { let n = 0; return () =>
    '00000000-0000-4000-8000-' + (++n).toString(16).padStart(12, '0'); })()
});"""


def planner_for(graph: MikadoGraph, root: Path) -> object:
    from milknado.domains.planning import PlanProposal, PlanResult, decode_manifest

    class PlannerStub:
        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            assert (goal, project_root) == (GOAL, root)
            assert target_goal_id > 0
            manifest = decode_manifest(
                {
                    "manifest_version": "milknado.plan.v2",
                    "goal": goal,
                    "goal_summary": goal,
                    "changes": [
                        {"id": "task-1", "path": "src/task-1.py", "description": "Task 1"}
                    ],
                }
            )
            return PlanProposal(manifest, root / "context.md")

        def prepare_proposal(self, proposal: PlanProposal, project_root: Path) -> BatchPlan:
            _ = (proposal, project_root)
            return BatchPlan((), (), "OPTIMAL")

        def apply_proposal(
            self, proposal: PlanProposal, *, target_goal_id: int, prepared_plan: BatchPlan
        ) -> PlanResult:
            _ = prepared_plan
            _ = graph.add_node(proposal.manifest.changes[0].description, target_goal_id)
            return PlanResult(True, 0, proposal.context_path, nodes_created=1)

    return PlannerStub()


def goal_id(graph: MikadoGraph) -> int:
    row = graph.group_connection.execute("SELECT goal_id FROM coordinator_sessions").fetchone()
    assert row is not None
    return int(row[0])


def seed_baseline(graph: MikadoGraph, root: Path, state: str) -> None:
    control = CoordinatorControl(graph, root)
    receipt = control.send_coordinator_command("", StartGoal("seed-goal", GOAL, "claude"))
    assert receipt.status == "accepted"
    if state == "accepted":
        _ = graph.add_node("Task 1", goal_id(graph))
    if state == "stale":
        _ = graph.add_node("Concurrent task", goal_id(graph))


def assert_graph(graph: MikadoGraph, state: str) -> None:
    if state in {"intake", "coordinator-unavailable"}:
        assert graph.get_all_nodes() == []
        return
    root = graph.get_node(goal_id(graph))
    assert root is not None and root.description == GOAL
    children = [node.description for node in graph.get_children(root.id)]
    expected = {"accepted": ["Task 1"], "stale": ["Concurrent task"]}.get(state, [])
    assert children == expected


def prepare_page(
    page: Page, server: BrowserServer, graph: MikadoGraph, case: CaptureCase
) -> BrowserErrors:
    revision, state = case.revision, case.state
    errors = BrowserErrors(revision == "after" and state == "coordinator-unavailable")
    page.on("pageerror", lambda error: errors.failures.append(f"pageerror: {error}"))
    page.on("console", errors.on_console)
    page.on("response", errors.on_response)
    page.add_init_script(RANDOM_UUID_SCRIPT)
    page.add_init_script("localStorage.setItem('milknado.theme', 'dark')")
    response = page.goto(server.login_url)
    assert response is not None and response.status < 500
    seeded = state not in {"intake", "coordinator-unavailable"}
    heading = GOAL if revision == "before" and seeded else str(PROJECT_ROOT)
    expect(page.get_by_role("heading", name=heading)).to_be_visible()
    if revision == "after":
        after_state(page, graph, state)
    elif seeded:
        expect_goal_node(page)
    _ = page.evaluate("document.fonts.ready")
    errors.assert_expected()
    return errors


def save_capture(page: Page, case: CaptureCase, graph: MikadoGraph, errors: BrowserErrors) -> None:
    revision, state = case.revision, case.state
    width, height = case.size
    proposal_focus = revision == "after" and state in PROPOSAL_FOCUS_STATES
    if proposal_focus:
        page.get_by_role("region", name="Plan proposals").scroll_into_view_if_needed()
    scroll = page.evaluate("""() => {
      const cockpit = document.querySelector('.mk-coordinator-cockpit');
      return {windowY: Math.round(scrollY), cockpitY: cockpit?.scrollTop ?? null,
        parentY: cockpit?.parentElement?.scrollTop ?? null};
    }""")
    print(f"SCROLL {revision}-{state}-{width}x{height} {scroll}", flush=True)
    assert_graph(graph, state)
    target = case.output / f"{revision}-{state}-{width}x{height}.png"
    page.screenshot(path=str(target), animations="disabled")
    full_target = case.output / f"{revision}-{state}-{width}x{height}-full.png"
    if proposal_focus:
        _ = page.locator(".mk-coordinator-cockpit").evaluate(
            "element => { element.scrollTop = 0; }"
        )
        page.screenshot(path=str(full_target), full_page=True, animations="disabled")
    errors.assert_expected()
    if errors.expect_coordinator_409:
        print(f"EXPECTED 409 {errors.coordinator_responses[0]}", flush=True)
    print(f"CAPTURE {target} sha256={hashlib.sha256(target.read_bytes()).hexdigest()}", flush=True)
    if proposal_focus:
        full_digest = hashlib.sha256(full_target.read_bytes()).hexdigest()
        print(f"SUPPLEMENT {full_target} sha256={full_digest}", flush=True)


def capture_context(browser: Browser, size: tuple[int, int]) -> BrowserContext:
    width, height = size
    return browser.new_context(
        viewport={"width": width, "height": height},
        color_scheme="dark",
        device_scale_factor=1,
        locale="en-US",
        timezone_id="UTC",
    )


def capture_one(browser: Browser, case: CaptureCase) -> None:
    revision, state = case.revision, case.state
    with tempfile.TemporaryDirectory(prefix="milknado-pr520-browser-") as directory:
        db_path = Path(directory) / "graph.db"
        graph = MikadoGraph(db_path)
        watch = WatchSnapshotSource(PROJECT_ROOT, db_path)
        source = PolledSnapshotSource(watch, interval=0.05)
        planner = (
            planner_for(graph, PROJECT_ROOT)
            if revision == "after" and state in PLAN_STATES
            else None
        )
        services = CoordinatorServices(planner=cast(object, planner))
        control = CoordinatorControl(graph, PROJECT_ROOT, services)
        login = LaunchToken(BROWSER_TOKEN)
        commands = WebCommands(
            coordinator=None if state == "coordinator-unavailable" else control,
            graph_edits=GraphEditCommands(graph, frozenset({"implement"}), PROJECT_ROOT),
        )
        server = BrowserServer(create_app(source, commands, login), login)
        context = capture_context(browser, case.size)
        try:
            with fixed_source_values():
                if revision == "before" and state not in {"intake", "coordinator-unavailable"}:
                    seed_baseline(graph, PROJECT_ROOT, state)
                source.start()
                server.start()
                page = context.new_page()
                errors = prepare_page(page, server, graph, case)
                save_capture(page, case, graph, errors)
        finally:
            context.close()
            server.stop()
            source.close()
            graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=("before", "after"), required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", choices=STATES, action="append")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    assert Path(milknado.__file__).resolve().is_relative_to(checkout), (
        "PYTHONPATH must import from --checkout"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    states = tuple(args.state) if args.state else STATES
    print(
        f"OFFLINE {args.revision}: real HTTP/SQLite UI; fixed {GOAL!r}, claude, dark theme, "
        f"source time {FIXED_NOW.isoformat()}, seeded UUID and browser command IDs. "
        "No live worker or control authority. "
        "Baseline proposals are N/A: ordinary graph only. "
        "After-only coordinator-unavailable has no visible cockpit; expected HTTP 409 "
        "and its console error are asserted. Stale advances graph revision; applying "
        "seeds a pending-to-applying record without a worker crash.",
        flush=True,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for state in states:
                for size in SIZES:
                    capture_one(browser, CaptureCase(args.revision, state, size, output))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
