"""Capture coordinator status in watch and attached-watch layouts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

from rich.text import Text

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-revision", required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / "docs/tui-captures/agent-steering"))

from fixture import Source  # noqa: E402
from milknado.app.watch import AttachedWatchSource  # noqa: E402
from milknado.app.watch_tui import WatchApp  # noqa: E402


class CoordinatorSource(Source):
    def coordinator_status(self) -> str:
        return "Goal 12: running\nRecovery: execution_group group-1: unavailable"


async def capture() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for width, height in ((80, 24), (40, 15)):
            for surface in ("watch", "attached-watch"):
                source = CoordinatorSource("main", read_only=surface == "watch")
                attached = AttachedWatchSource(source, lambda _run_id, _command: True)
                app = WatchApp(source) if surface == "watch" else WatchApp(attached, read_only=False)
                async with app.run_test(size=(width, height)) as pilot:
                    for _ in range(3):
                        await pilot.pause()
                    await pilot.press("c")
                    for _ in range(3):
                        await pilot.pause()
                    file = f"{surface}-coordinator-{width}x{height}.svg"
                    (args.output / file).write_text(app.export_screenshot(), encoding="utf-8")
                    records.append({"file": file, "surface": surface, "columns": width, "rows": height})
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "source_revision": args.source_revision,
                "source_root": str(args.source_root),
                "fixture": "synthetic coordinator and recovery status; no live provider",
                "interaction": "press c from main",
                "records": records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


asyncio.run(capture())
