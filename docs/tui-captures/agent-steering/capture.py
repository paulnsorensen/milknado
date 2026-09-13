from __future__ import annotations

import argparse
import asyncio
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

from rich.text import Text
from textual.widgets import Static

sys.path.insert(0, str(Path(__file__).parent))

PROVIDER_ACTIONS: dict[str, tuple[str, ...]] = {}
Source: object
ExecutionController: object
ExecutionApp: object
WatchApp: object


def _load_runtime() -> None:
    global ExecutionApp, ExecutionController, PROVIDER_ACTIONS, Source, WatchApp
    from fixture import PROVIDER_ACTIONS as actions, Source as source
    from milknado.app.run import ExecutionController as controller
    from milknado.app.run_tui import ExecutionApp as run_app
    from milknado.app.watch_tui import WatchApp as watch_app

    PROVIDER_ACTIONS = actions
    Source = source
    ExecutionController = controller
    ExecutionApp = run_app
    WatchApp = watch_app

SIZES = ((120, 40), (80, 24), (40, 15))
ATTACHED_SIZES = ((120, 40), (80, 24))
BASELINE_STATES = ("main", "session", "permission", "error")
FINAL_STATES = ("main", "session", "permission", "owner-unavailable", "error", "review")
ATTACHED_STATES = ("session", "permission", "owner-unavailable", "error")
BASELINE_SOURCE_REVISION = "a6b2dee711f84ebad07427e01adb7175be750fad"
CAPTURE_SCRIPT = "docs/tui-captures/agent-steering/capture.py"
FIXTURE_CONTRACT = {
    "graph": "GraphSnapshot with five nodes and two diamond dependency edges",
    "detail": "NodeDetailSnapshot with every MikadoNode field and populated related pages",
    "provider": "claude",
    "provider_actions": (),
    "mode": "synthetic fake-vendor fixture; no live worker",
}


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _archive_source(revision: str, output: Path, capture_set: str) -> None:
    repo = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
    resolved = _git("rev-parse", "--verify", f"{revision}^{{commit}}", cwd=repo)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", resolved],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="milknado-capture-") as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        target = root / CAPTURE_SCRIPT
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), target)
        shutil.copy2(Path(__file__).with_name("fixture.py"), target.with_name("fixture.py"))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(root / "src"), env.get("PYTHONPATH", "")))
        )
        command = [
            sys.executable,
            str(target),
            "--output",
            str(output.resolve()),
            "--capture-set",
            capture_set,
            "--source-revision",
            revision,
            "--resolved-source-revision",
            resolved,
            "--source-root",
            str(root),
        ]
        subprocess.run(command, cwd=root, env=env, check=True)


def _source_revision(args: argparse.Namespace) -> str:
    if args.resolved_source_revision:
        return args.resolved_source_revision
    repo = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
    return _git("rev-parse", "--verify", f"{args.source_revision}^{{commit}}", cwd=repo)


def _validate_source(capture_set: str, revision: str, source_root: Path | None) -> None:
    if capture_set == "baseline" and revision != BASELINE_SOURCE_REVISION:
        raise ValueError(
            f"baseline capture set requires {BASELINE_SOURCE_REVISION}, got {revision}"
        )
    if source_root is not None:
        package = Path(cast(str, sys.modules["milknado"].__file__)).resolve()
        if source_root.resolve() not in package.parents:
            raise RuntimeError(f"source revision mismatch: imported {package} outside {source_root}")


def _svg_dimensions(svg: str) -> tuple[str, str]:
    match = re.search(r"viewBox=['\"]0 0 ([^ ]+) ([^'\"]+)['\"]", svg)
    if match is None:
        raise ValueError("screenshot has no SVG viewBox")
    return match.group(1), match.group(2)


async def _settle(pilot: object) -> None:
    for _ in range(3):
        await cast("object", pilot).pause()


def _admit_synthetic_owner(_run_id: str, _command: object) -> bool:
    return True


async def capture(
    output: Path,
    surface: str,
    state: str,
    size: tuple[int, int],
    source_revision: str,
) -> dict[str, object]:
    source = Source(state, read_only=surface == "watch")
    owner_synthetic = surface == "attached-watch"
    if owner_synthetic:
        from milknado.app.watch import AttachedWatchSource

        attached = AttachedWatchSource(source, _admit_synthetic_owner)
        app = WatchApp(attached, read_only=False)
    elif surface == "run":
        app = ExecutionApp(cast(ExecutionController, cast(object, source)))
    else:
        app = WatchApp(source)
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        if state in {"session", "permission"} and surface in {"run", "attached-watch"}:
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
            "owner_mode": "synthetic-attached-owner" if owner_synthetic else None,
            "owner_synthetic": owner_synthetic,
            "source_revision": source_revision,
        }
        metadata = html.escape(json.dumps(record, separators=(",", ":")))
        path.write_text(svg.replace(">", f"><metadata>{metadata}</metadata>", 1), encoding="utf-8")
        return record


def _manifest_command(capture_set: str, source_revision: str) -> str:
    output = (
        "docs/tui-captures/agent-steering/captures"
        if capture_set == "baseline"
        else "docs/tui-captures/agent-steering/captures-after"
    )
    return (
        f"uv run python {CAPTURE_SCRIPT} --output {output} "
        f"--capture-set {capture_set} --source-revision {source_revision}"
    )


def _absent(capture_set: str) -> list[dict[str, str]]:
    absent = [{"surface": "watch", "state": "confirmation", "reason": "view-only UI has no confirmation control"}]
    if capture_set == "baseline":
        absent.extend(
            [
                {"surface": "run/watch", "state": "attached-watch", "reason": "not implemented in baseline source"},
                {"surface": "run/watch", "state": "goal-review", "reason": "not implemented in baseline source"},
            ]
        )
    return absent


async def main(args: argparse.Namespace) -> None:
    if args.source_root is None:
        _archive_source(args.source_revision, args.output.resolve(), args.capture_set)
        return
    _load_runtime()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_revision = _source_revision(args)
    _validate_source(args.capture_set, source_revision, args.source_root)
    states = BASELINE_STATES if args.capture_set == "baseline" else FINAL_STATES
    records: list[dict[str, object]] = []
    with patch("textual.widgets._header.HeaderClock.render", return_value=Text("12:00:00")):
        for size in SIZES:
            for surface in ("run", "watch"):
                for state in states:
                    records.append(await capture(output, surface, state, size, source_revision))
                if surface == "run":
                    records.append(await capture(output, surface, "confirmation", size, source_revision))
        if args.capture_set == "final":
            for size in ATTACHED_SIZES:
                for state in ATTACHED_STATES:
                    records.append(
                        await capture(output, "attached-watch", state, size, source_revision)
                    )
    for record in records:
        svg = (output / str(record["file"])).read_text(encoding="utf-8")
        embedded = re.search(r"<metadata>(.+)</metadata>", svg)
        if embedded is None:
            raise ValueError(f"metadata missing for {record['file']}")
        parsed = json.loads(html.unescape(embedded.group(1)))
        if parsed != record:
            raise ValueError(f"metadata mismatch for {record['file']}")
        if parsed["source_revision"] != source_revision:
            raise ValueError(f"source revision mismatch for {record['file']}")
    manifest = {
        "command": _manifest_command(args.capture_set, args.source_revision),
        "capture_set": args.capture_set,
        "source_revision": source_revision,
        "source_revision_argument": args.source_revision,
        "theme": "Textual default theme with fixed 12:00:00 clock",
        "fixture_contract": FIXTURE_CONTRACT,
        "states": records,
        "absent": _absent(args.capture_set),
        "process_boundary_evidence": {
            "status": "missing",
            "owner_process": "not recorded",
            "watch_process": "not recorded",
            "boundary": (
                "No committed capture starts a worker owner and a separately launched "
                "attached-watch process or records their live session handoff."
            ),
            "synthetic_surface": (
                "attached-watch records use an in-process synthetic owner adapter; "
                "they do not prove a live process boundary."
            ),
        },
        "provider_evidence": [
            {"provider": provider, "actions": actions, "mode": "fake-vendor protocol tests"}
            for provider, actions in PROVIDER_ACTIONS.items()
        ],
        "provider_evidence_scope": (
            "One-shot fake-vendor protocol probes do not establish live mid-session compatibility."
        ),
        "fixture_limits": [
            "synthetic snapshot",
            "fake-vendor session semantics",
            "no live worker",
            "no database",
            "no agent process",
            "no live attached-watch owner",
            "no live mid-session compatibility claim",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--capture-set", choices=("baseline", "final"), required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--resolved-source-revision")
parser.add_argument("--source-root", type=Path)
args = parser.parse_args()
asyncio.run(main(args))
