"""Deterministic review frames for run quit and graceful stop controls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from playwright.async_api import async_playwright
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-steering"))

from fixture import Source

from milknado.app.run_tui import ExecutionApp, run_execution_tui
from milknado.app.watch_tui import WatchApp

SIZES = ((120, 40), (80, 24))
STATES = {"run": ("main", "quit-confirmation", "stop-confirmation"), "watch": ("main",)}


class WarningTranscriptApp(App[None]):
    CSS = "#warning { padding: 1 2; }"

    def __init__(self, warning: str) -> None:
        super().__init__()
        self.warning = warning

    def compose(self) -> ComposeResult:
        yield Static(
            f"Synthetic post-exit stderr transcript (wrapper only)\n\n{self.warning}",
            id="warning",
            markup=False,
        )


class CaptureSource(Source):
    def __init__(self, read_only: bool) -> None:
        super().__init__()
        self.read_only = read_only
        self.stop_calls = 0

    def stop_scheduling(self) -> None:
        self.stop_calls += 1

    def force_stop(self, _run_id: str) -> bool:
        self.stop_calls += 1
        return True

    def force_stop_all(self) -> bool:
        self.stop_calls += 1
        return False


async def capture(output: Path, surface: str, state: str, size: tuple[int, int]) -> dict:
    source = CaptureSource(read_only=surface == "watch")
    app = ExecutionApp(source) if surface == "run" else WatchApp(source)
    app.theme = "textual-dark"
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if state == "quit-confirmation":
            await pilot.press("q")
        elif state == "stop-confirmation":
            await pilot.press("s")
        await pilot.pause()
        stem = f"{surface}-{state}-{size[0]}x{size[1]}"
        (output / f"{stem}.svg").write_text(
            app.export_screenshot(title="Milknado deep-module reliability"), encoding="utf-8"
        )
        focus = app.screen.focused
        record = {
            "stem": stem,
            "surface": surface,
            "state": state,
            "size": size,
            "focus": focus.id if focus else None,
            "screen": type(app.screen).__name__,
            "active_runs": len(app.snapshot.active_runs),
        }
        if surface == "watch":
            await pilot.press("q")
            await pilot.pause()
    if source.stop_calls:
        raise AssertionError(f"{surface} {state}: unexpected worker stop call")
    return record


async def render_pngs(output: Path, records: list[dict]) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(device_scale_factor=1)
        for record in records:
            stem = record["stem"]
            await page.goto((output / f"{stem}.svg").as_uri())
            await page.locator("svg").screenshot(path=str(output / f"{stem}.png"))
        await browser.close()


async def capture_warning(output: Path) -> list[dict]:
    from milknado.app._shutdown import ShutdownIntent

    records = []
    for width, height in SIZES:
        source = CaptureSource(read_only=False)
        source.shutdown_intent = ShutdownIntent()
        stderr = StringIO()

        def run_with_failed_cleanup(app: ExecutionApp) -> None:
            app._cleanup_confirmed = app.controller.force_stop_all()

        with patch.object(ExecutionApp, "run", run_with_failed_cleanup):
            with redirect_stderr(stderr):
                run_execution_tui(source, feature_branch="fixture")
        warning = stderr.getvalue()
        expected = "Warning: force-stop cleanup did not finish; worker ownership remains.\n"
        if warning != expected or source.stop_calls != 1:
            raise AssertionError(f"Unexpected {width}x{height} warning transcript: {warning!r}")
        stem = f"run-unconfirmed-stop-warning-{width}x{height}"
        (output / f"{stem}.txt").write_text(warning, encoding="utf-8")
        app = WarningTranscriptApp(warning)
        app.theme = "textual-dark"
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            (output / f"{stem}.svg").write_text(
                app.export_screenshot(title="Synthetic post-exit stderr"), encoding="utf-8"
            )
        records.append({"stem": stem, "size": [width, height], "force_stop_all_calls": 1})
    return records


def presentation_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/milknado/app/run_tui.py",
        "src/milknado/app/run_view.py",
        "src/milknado/app/run_view_app.py",
        "src/milknado/app/watch_tui.py",
        "src/milknado/app/run_overlays.py",
        "src/milknado/app/run_layout.py",
        "docs/tui-captures/agent-steering/fixture.py",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def publish_pngs(output: Path, destination: Path, root: Path, revision: str) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_revision"] != revision:
        raise ValueError("Capture revision does not match the selected source checkout")
    if hashes := manifest.get("presentation_sha256"):
        if hashes != presentation_hashes(root):
            raise ValueError("Presentation files changed after capture")
    destination.mkdir(parents=True, exist_ok=True)
    for record in manifest["records"]:
        name = f"{record['stem']}.png"
        shutil.copyfile(output / name, destination / name)
    for record in manifest.get("warning_records", []):
        for suffix in (".png", ".txt"):
            name = f"{record['stem']}{suffix}"
            shutil.copyfile(output / name, destination / name)
    shutil.copyfile(manifest_path, destination / "manifest.json")


async def main(args: argparse.Namespace) -> None:
    import milknado

    root = args.source_root.resolve()
    if root not in Path(milknado.__file__).resolve().parents:
        raise ValueError(f"Wrong runtime: {milknado.__file__}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = args.output.resolve()
    if args.publish_to is not None:
        publish_pngs(output, args.publish_to.resolve(), root, revision)
        return
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for size in SIZES:
            for surface, states in STATES.items():
                for state in states:
                    records.append(await capture(output, surface, state, size))
    warning_records = await capture_warning(output) if args.warning else []
    await render_pngs(output, [*records, *warning_records])
    manifest = {
        "source_revision": revision,
        "presentation_sha256": presentation_hashes(root),
        "theme": "textual-dark",
        "clock": "12:00:00",
        "fixture": "docs/tui-captures/agent-steering/fixture.py",
        "watch_quit_local": "q exited each watch app with zero fixture stop calls",
        "limits": [
            "Synthetic snapshots; no live worker, controller, CLI, provider, or database.",
            "Fixture call evidence does not prove live-worker cleanup.",
            "Chromium renders Textual SVG; this does not prove terminal palette fidelity.",
        ],
        "records": records,
        "warning_records": warning_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{revision}: {len(records)} SVG and PNG pairs in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warning", action="store_true")
    parser.add_argument("--publish-to", type=Path)
    asyncio.run(main(parser.parse_args()))
