"""PR 485 review-fix evidence: run help while a terminal run is selected."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from rich.text import Text

_PR483 = Path(__file__).resolve().parents[1] / "pr483" / "capture.py"
_SPEC = importlib.util.spec_from_file_location("pr483_capture", _PR483)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

TERMINAL_RUN = "run-13"


async def capture(output: Path, size: tuple[int, int]) -> dict:
    app = base.make_app("run", "main")
    app.theme = "textual-dark"
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.selected_run_id = TERMINAL_RUN
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        stem = f"run-help-terminal-{size[0]}x{size[1]}"
        svg = output / f"{stem}.svg"
        svg.write_text(app.export_screenshot(title="Milknado PR 485"), encoding="utf-8")
        focus = app.screen.focused
        return {
            "stem": stem,
            "surface": "run",
            "state": "help-terminal",
            "size": size,
            "selected_run": app.selected_run_id,
            "focus": focus.id if focus else None,
            "screen": type(app.screen).__name__,
        }


async def main(args: argparse.Namespace) -> None:
    import milknado

    root = args.source_root.resolve()
    if root not in Path(milknado.__file__).resolve().parents:
        raise ValueError(f"Wrong runtime: {milknado.__file__}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        records = [await capture(output, size) for size in base.SIZES]
    await base.render_pngs(output, records)
    manifest = {
        "source_revision": revision,
        "theme": "textual-dark",
        "clock": "12:00:00",
        "fixture": "../agent-steering/fixture.py through ../pr483/capture.py",
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{revision}: {len(records)} SVG and PNG captures in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
