"""Capture the stale graph-highlight selection on run and watch views."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from dataclasses import replace
from html import unescape
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from playwright.async_api import async_playwright
from rich.text import Text
from textual.widgets import Tree

from tests.graph_navigation_fixtures import run_app, source

SIZES = ((80, 24), (120, 40))
SURFACES = ("run", "watch")
EXPECTED = {"before": (1, None), "after": (2, "run-2")}


async def capture(
    surface: str, size: tuple[int, int], expected: tuple[int, str | None]
) -> tuple[dict[str, object], str]:
    source_value = source()
    app = run_app(source_value, surface)
    app.theme = "textual-dark"
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        tree = app.query_one("#graph-tree", Tree)
        stale_node = tree.cursor_node
        assert stale_node is not None and getattr(stale_node.data, "node_id", None) == 1

        replacement = replace(source_value.current.active_runs[0], run_id="run-2", node_id=2)
        app.show_snapshot(replace(source_value.current, graph=None, active_runs=(replacement,)))
        app.select_tree_node(Tree.NodeHighlighted(stale_node))
        hint = (
            "i Session input" if surface == "run" and expected[1]
            else "s Stop scheduling" if surface == "run"
            else "q Quit"
        )
        for _ in range(20):
            await pilot.pause()
            svg = app.export_screenshot(title="Stale graph highlight")
            rendered = unescape(svg).replace("\xa0", " ")
            if "q Quit" in rendered and hint in rendered:
                break
        else:
            raise AssertionError(f"Footer hints did not render: {surface} {size}")

        selection = (app.selected_node_id, app.selected_run_id)
        assert selection == expected, (surface, size, selection, expected)
        stem = f"{surface}-{size[0]}x{size[1]}"
        focus = app.screen.focused
        record: dict[str, object] = {
            "image": f"{stem}.png",
            "surface": surface,
            "size": list(size),
            "selection": list(selection),
            "focus": focus.id if focus else None,
            "visible_status": "run-2 running",
        }
        if surface == "watch":
            await pilot.press("q")
    return record, svg


async def render(output: Path, captures: list[tuple[dict[str, object], str]]) -> None:
    with TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page(device_scale_factor=1)
            for record, svg in captures:
                image = str(record["image"])
                svg_path = temporary_root / f"{Path(image).stem}.svg"
                svg_path.write_text(svg, encoding="utf-8")
                await page.goto(svg_path.as_uri())
                await page.locator("svg").screenshot(path=str(output / image))
            await browser.close()


async def main(args: argparse.Namespace) -> None:
    import milknado

    root = args.source_root.resolve()
    if not Path(milknado.__file__).resolve().is_relative_to(root):
        raise ValueError(f"Wrong runtime: {milknado.__file__}")
    if not Path(source.__code__.co_filename).resolve().is_relative_to(root):
        raise ValueError(f"Wrong fixture: {source.__code__.co_filename}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    captures: list[tuple[dict[str, object], str]] = []
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for surface in SURFACES:
            for size in SIZES:
                captures.append(await capture(surface, size, EXPECTED[args.phase]))
    await render(output, captures)
    manifest = {
        "phase": args.phase,
        "source_revision": revision,
        "navigation_sha256": hashlib.sha256(
            (root / "src/milknado/app/session_navigation.py").read_bytes()
        ).hexdigest(),
        "theme": "textual-dark",
        "clock": "12:00:00",
        "fixture": "tests/graph_navigation_fixtures.py:source",
        "transition": "graph to graph=None with active run-2 on node 2, then stale node-1 highlight",
        "limits": [
            "Synthetic snapshots; no live worker, controller, CLI, provider, or database.",
            "Direct event delivery models a queued stale Tree.NodeHighlighted event.",
            "Chromium renders Textual SVG; terminal palette fidelity is not tested.",
        ],
        "records": [record for record, _ in captures],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.phase}: {len(captures)} PNGs in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=EXPECTED, required=True)
    asyncio.run(main(parser.parse_args()))
