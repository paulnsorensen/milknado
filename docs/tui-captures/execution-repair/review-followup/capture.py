"""Reproduce review captures with real widgets and synthetic worker state."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Protocol, cast, final

root, mode, state, proof_path = sys.argv[1:]
sys.path[:0] = [str(Path(root) / "src"), root]

from textual.binding import Binding, BindingType
from textual.pilot import Pilot
from textual.widgets import Select, Static

from milknado.app.run import ExecutionSnapshot
from milknado.app.run_overlays import FooterHint, RunFooter
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.domains.common import SessionEvent


class CaptureSource(Protocol):
    current: ExecutionSnapshot


@final
class BindingOwner(Static):
    can_focus: bool = True
    BINDINGS: ClassVar[list[BindingType]] = [Binding("f6", "proof", "Owner proof")]
    DEFAULT_CSS: ClassVar[str] = "BindingOwner { dock: top; height: 1; }"
    reached: bool = False

    def action_proof(self) -> None:
        self.reached = True
        self.update("Owned action reached")


spec = importlib.util.spec_from_file_location(
    "affinage_fixture", Path(root) / "docs/tui-captures/execution-repair/fixture.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Capture fixture is unavailable")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
make_app = cast(
    Callable[[str, str, str], tuple[ExecutionSnapshotApp, CaptureSource]],
    cast(object, fixture.make_app),
)
fixture_state = {
    "permission": "session",
    "confirmation": "main",
    "footer-owner": "main",
}.get(state, state)
app, source = make_app(mode, fixture_state, "omp")


async def prepare_owner(pilot: Pilot[object]) -> dict[str, bool]:
    owner = BindingOwner("Owned action not reached")
    footer = app.query_one(RunFooter)
    await app.screen.mount(owner, before=footer)
    _ = owner.focus()
    await pilot.pause()
    footer.update_hints()
    await pilot.pause()
    hint = next(hint for hint in app.query(FooterHint) if "Owner proof" in str(hint.render()))
    clicked = await pilot.click(hint, offset=(2, 0))
    await pilot.pause()
    owner.update(f"Owned action {'reached' if owner.reached else 'not reached'} — READY")
    await pilot.pause()
    return {"footer_click_handled": clicked, "binding_owner_reached": owner.reached}


async def prepare_permission(pilot: Pilot[object]) -> dict[str, str | bool]:
    run = app.snapshot.active_runs[0]
    assert run.session is not None
    permissions = tuple(
        SessionEvent(kind="permission", event_id=name, text=label, state="requested")
        for name, label in (("first", "Approve alpha"), ("second", "Approve beta"))
    )
    current = replace(
        app.snapshot,
        active_runs=(replace(run, session=replace(run.session, permissions=permissions)),),
    )
    source.current = current
    app.show_snapshot(current)
    await pilot.press("i")
    await pilot.pause()
    action = cast(Select[str], app.query_one("#session-action", Select))
    action.value = "approve"
    await pilot.pause()
    selector = cast(Select[str], app.query_one("#session-permission", Select))
    selector.value = "first"
    await pilot.pause()
    _ = selector.focus()
    await pilot.pause()
    with app.prevent(Select.Changed):
        selector.value = "second"
        app.show_snapshot(current)
    await pilot.pause()
    assert isinstance(selector.value, str)
    return {"permission_value": selector.value, "permission_focused": selector.has_focus}


async def prepare(pilot: Pilot[object]) -> None:
    await pilot.pause()
    facts: dict[str, object] = {}
    if state == "permission":
        facts.update(await prepare_permission(pilot))
    if state == "footer-owner":
        facts.update(await prepare_owner(pilot))
    if state == "confirmation":
        await pilot.press("f")
        await pilot.pause()
        cancel = next(hint for hint in app.query(FooterHint) if "Cancel" in str(hint.render()))
        facts["cancel_footer_click_handled"] = await pilot.click(cancel, offset=(2, 0))
        await pilot.pause()
    if state == "empty-graph":
        await pilot.press("j")
        await pilot.pause()
    _ = Path(proof_path).write_text(
        json.dumps(
            {
                "mode": mode,
                "state": state,
                "selected_run_id": app.selected_run_id,
                "active_run_ids": [run.run_id for run in app.snapshot.active_runs],
                "terminal_run_ids": [run.run_id for run in app.snapshot.terminal_runs],
                "screen": type(app.screen).__name__,
                **facts,
            },
            indent=2,
        )
    )
    source.current = replace(source.current, goal=f"Affinage {state} READY")
    app.show_snapshot(source.current)


_ = app.run(auto_pilot=prepare)
