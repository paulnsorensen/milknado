from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

from rich.text import Text
from textual.widgets import Static

sys.path.insert(0, str(Path(__file__).parent))

from fixture import PROVIDER_ACTIONS, Source  # noqa: E402
from milknado.app.run import ExecutionController  # noqa: E402
from milknado.app.run_tui import ExecutionApp  # noqa: E402
from milknado.app.watch_tui import WatchApp  # noqa: E402

SIZES = ((120, 40), (80, 24), (40, 15))
STATES = ("main", "session", "permission", "owner-unavailable", "error", "review")
FIXTURE_CONTRACT = {
    "graph": "GraphSnapshot with five nodes and two diamond dependency edges",
    "detail": "NodeDetailSnapshot with every MikadoNode field and populated related pages",
    "provider": "claude",
    "provider_actions": PROVIDER_ACTIONS["claude"],
    "mode": "synthetic fake-vendor fixture; no live worker",
}


def _svg_dimensions(svg: str) -> tuple[str, str]:
    match = re.search(r"viewBox=['\"]0 0 ([^ ]+) ([^'\"]+)['\"]", svg)
    if match is None:
        raise ValueError("screenshot has no SVG viewBox")
    return match.group(1), match.group(2)


async def _settle(pilot: object) -> None:
    for _ in range(3):
        await cast("object", pilot).pause()


async def capture(output: Path, surface: str, state: str, size: tuple[int, int]) -> dict[str, object]:
    source = Source(state, read_only=surface == "watch")
    app = (
        ExecutionApp(cast(ExecutionController, cast(object, source)))
        if surface == "run"
        else WatchApp(source)
    )
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        if state in {"session", "permission"} and surface == "run":
            await pilot.press("i")
            await _settle(pilot)
        overlay_visible = False
        if state == "confirmation" and surface == "run":
            app._set_confirmation("force", "run-12")  # type: ignore[attr-defined]
            await _settle(pilot)
            overlay = app.screen.query_one("#confirmation-overlay", Static)
            overlay_visible = overlay.region.area > 0
        path = output / f"{surface}-{state}-{size[0]}x{size[1]}.svg"
        svg = app.export_screenshot()
        width, height = _svg_dimensions(svg)
        record = {
            "file": path.name,
            "surface": surface,
            "state": state,
            "columns": size[0],
            "rows": size[1],
            "svg_width": width,
            "svg_height": height,
            "confirmation_visible": overlay_visible,
            "fixture": FIXTURE_CONTRACT["mode"],
        }
        metadata = html.escape(json.dumps(record, separators=(",", ":")))
        path.write_text(
            app.export_screenshot().replace(">", f"><metadata>{metadata}</metadata>", 1),
            encoding="utf-8",
        )
        return record


async def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for size in SIZES:
            for surface in ("run", "watch"):
                for state in STATES:
                    records.append(await capture(output, surface, state, size))
                if surface == "run":
                    records.append(await capture(output, surface, "confirmation", size))
    for record in records:
        svg = (output / str(record["file"])).read_text(encoding="utf-8")
        embedded = re.search(r"<metadata>(.+)</metadata>", svg)
        if embedded is None or json.loads(html.unescape(embedded.group(1))) != record:
            raise ValueError(f"metadata mismatch for {record['file']}")
    manifest = {
        "command": "uv run python docs/tui-captures/agent-steering/capture.py "
        + "--output docs/tui-captures/agent-steering/captures-after",
        "theme": "Textual default theme with fixed 12:00:00 clock",
        "fixture_contract": FIXTURE_CONTRACT,
        "states": records,
        "absent": [
            {"surface": "watch", "state": "confirmation", "reason": "view-only UI has no confirmation control"},
            {"surface": "run/watch", "state": "attached-watch", "reason": "requires a live owner process"},
        ],
        "provider_evidence": [
            {"provider": provider, "actions": actions, "mode": "fake-vendor protocol tests"}
            for provider, actions in PROVIDER_ACTIONS.items()
        ],
        "fixture_limits": [
            "synthetic snapshot",
            "fake-vendor session semantics",
            "no live worker",
            "no database",
            "no agent process",
            "no live compatibility claim",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
asyncio.run(main(args.output))
