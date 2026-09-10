from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import Protocol, cast, final

import pytest
from rich.console import RenderableType
from rich.text import Text
from textual.widgets import Button, DataTable, Input, Select, Static

from milknado.adapters import ChangedFile, GitAdapter
from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_tui import ExecutionApp
from milknado.app.watch_tui import WatchApp
from milknado.domains.common import (
    SessionAction,
    SessionContext,
    SessionEvent,
    SessionInput,
    SessionView,
)
from milknado.loop.sessions import SessionChannel
from tests.test_execution_tui import FakeController, snapshot


class _WorkerManager(Protocol):
    async def wait_for_complete(self) -> None: ...


def _wait_for_workers(app: ExecutionApp) -> _WorkerManager:
    return cast(_WorkerManager, app.workers)


@dataclass(kw_only=True)
class SnapshotController(FakeController):
    submissions: list[tuple[str, SessionInput]] = field(default_factory=list)
    accepted: bool = True

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        self.submissions.append((run_id, command))
        return self.accepted


def plain(app: ExecutionApp | WatchApp, selector: str) -> str:
    renderable = app.query_one(selector, Static).render()
    return renderable.plain if isinstance(renderable, Text) else str(renderable)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def changed_context(root: Path, filename: str, after: str) -> SessionContext:
    root.mkdir()
    _ = _git(root, "init", "-q")
    _ = _git(root, "config", "user.email", "test@example.com")
    _ = _git(root, "config", "user.name", "Test")
    _ = (root / filename).write_text("before\n")
    _ = _git(root, "add", filename)
    _ = _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD").strip()
    _ = (root / filename).write_text(after)
    return SessionContext(family="omp", cwd=str(root), base_oid=base)


def _session_view(context: SessionContext, request_id: str, prompt: str) -> SessionView:
    return SessionView(
        context=context,
        actions=("steer", "follow_up", "approve", "deny"),
        active=True,
        permissions=(
            SessionEvent(kind="permission", event_id=request_id, text=prompt, state="requested"),
        ),
    )


def two_run_snapshot(first: SessionContext, second: SessionContext) -> ExecutionSnapshot:
    current = snapshot(second=True)
    return replace(
        current,
        active_runs=(
            replace(
                current.active_runs[0],
                session=_session_view(first, "first-permission", "Approve first"),
            ),
            replace(
                current.active_runs[1],
                session=_session_view(second, "second-permission", "Approve second"),
            ),
        ),
    )


@final
@dataclass(kw_only=True)
class _SessionController(FakeController):
    channel: SessionChannel

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        if run_id != self.snapshot().active_runs[0].run_id:
            return False
        accepted = self.channel.submit(command)
        self.show_session()
        return accepted

    def show_session(self) -> None:
        current = self.snapshot()
        selected = replace(current.active_runs[0], session=self.channel.view())
        self.initial_snapshot = replace(current, active_runs=(selected,))
        self.publish(self.initial_snapshot)


@pytest.fixture
def controller() -> _SessionController:
    channel = SessionChannel()
    channel.start(
        SessionContext(family="omp", cwd="/repo", base_oid="base"), ("steer", "approve", "deny")
    )
    for request_id, text in (("first", "Approve file edit"), ("second", "Choose alpha or beta")):
        channel.publish(
            SessionEvent(kind="permission", event_id=request_id, text=text, state="requested")
        )
    current = snapshot()
    selected = replace(current.active_runs[0], session=channel.view())
    return _SessionController(
        channel=channel, initial_snapshot=replace(current, active_runs=(selected,))
    )


@pytest.mark.asyncio
async def test_palette_quit_requires_confirmation_and_preserves_session_draft(
    controller: _SessionController,
) -> None:
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("i", *"keep this input", "escape", "h")
        await pilot.pause()
        assert app.screen.is_modal
        await pilot.press("escape", "ctrl+p", *"quit")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.is_running
        assert controller.stop_requests == 0
        await pilot.press("escape", "i")
        await pilot.pause()
        assert controller.stop_requests == 0
        assert app.query_one("#session-input", Input).value == "keep this input"
        await pilot.press("enter")
        await _wait_for_workers(app).wait_for_complete()
        inputs = [event for event in controller.channel.view().events if event.kind == "user"]
        assert [(event.text, event.action, event.state) for event in inputs] == [
            ("keep this input", "steer", "queued"),
        ]


@pytest.mark.asyncio
async def test_native_input_events_enqueue_exact_text_and_clear_the_editor(
    controller: _SessionController,
) -> None:
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("i")
        await pilot.press(*"  preserve whitespace  ", "enter")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        inputs = [event for event in controller.channel.view().events if event.kind == "user"]
        assert [(event.text, event.action, event.state) for event in inputs] == [
            ("  preserve whitespace  ", "steer", "queued"),
        ]
        assert app.query_one("#session-input", Input).value == ""


@pytest.mark.asyncio
async def test_permission_choice_survives_refresh_and_preserves_reply_text(
    controller: _SessionController,
) -> None:
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("i")
        request_id = controller.channel.view().permissions[1].event_id
        app.query_one("#session-action", Select).value = "approve"
        app.query_one("#session-permission", Select).value = request_id
        await pilot.pause()
        controller.show_session()
        await pilot.pause()
        assert cast(Select[str], app.query_one("#session-action", Select)).value == "approve"
        assert cast(Select[str], app.query_one("#session-permission", Select)).value == request_id
        await pilot.press(*"beta", "enter")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        inputs = [event for event in controller.channel.view().events if event.kind == "user"]
        assert [(event.event_id, event.text, event.action, event.state) for event in inputs] == [
            (request_id, "beta", "approve", "queued"),
        ]


@pytest.mark.asyncio
async def test_session_controls_and_drafts_are_scoped_to_selected_run(
    tmp_path: Path,
) -> None:
    first = SessionContext(family="omp", cwd=str(tmp_path / "first"), base_oid="first")
    second = SessionContext(family="omp", cwd=str(tmp_path / "second"), base_oid="second")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(first, second), replay_subscription=False
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("i")
        action = cast(Select[SessionAction], app.query_one("#session-action", Select))
        permission = cast(Select[str], app.query_one("#session-permission", Select))
        action.value = "approve"
        permission.value = "first-permission"
        await pilot.pause()
        await pilot.press(*"first")
        controller.publish(controller.snapshot())
        await pilot.pause()
        await pilot.press(*" draft")
        await pilot.pause()
        assert app.query_one("#session-input", Input).value == "first draft"
        assert app.query_one("#session-input", Input).has_focus

        await pilot.press("escape", "j")
        await pilot.pause()
        assert app.selected_run_id == "run-2"

        action.value = "follow_up"
        permission.value = "second-permission"
        await pilot.pause()
        await pilot.press("i", *"second draft")
        await pilot.pause()
        controller.publish(controller.snapshot())
        await pilot.pause()

        await pilot.press("escape", "k")
        await pilot.pause()
        assert app.selected_run_id == "run-1"
        assert app.query_one("#session-input", Input).value == "first draft"
        assert action.value == "approve"
        assert permission.value == "first-permission"

        await pilot.press("escape", "j")
        await pilot.pause()
        assert app.selected_run_id == "run-2"
        assert app.query_one("#session-input", Input).value == "second draft"
        assert action.value == "follow_up"
        assert permission.value == "second-permission"


@pytest.mark.asyncio
async def test_rejected_session_input_retains_exact_draft() -> None:
    current = snapshot()
    session = SessionView(actions=("steer",), active=True)
    selected = replace(current.active_runs[0], session=session)
    controller = SnapshotController(
        initial_snapshot=replace(current, active_runs=(selected,)),
        accepted=False,
        replay_subscription=False,
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("i")
        await pilot.pause()
        await pilot.press(*"keep this draft", "enter")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()

        assert controller.submissions == [
            ("run-1", SessionInput(action="steer", text="keep this draft"))
        ]
        assert app.query_one("#session-input", Input).value == "keep this draft"


@pytest.mark.asyncio
async def test_failed_git_inspection_is_visible_in_changes_panel(tmp_path: Path) -> None:
    missing = tmp_path / "missing-worktree"
    context = SessionContext(family="omp", cwd=str(missing), base_oid="base")
    current = snapshot()
    selected = replace(
        current.active_runs[0],
        session=SessionView(context=context, active=True),
    )
    controller = SnapshotController(
        initial_snapshot=replace(current, active_runs=(selected,)),
        replay_subscription=False,
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("x")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()

        expected = f"git session changes failed: worktree is unavailable: {missing}"
        assert plain(app, "#changes-state") == expected
        assert plain(app, "#diff-text") == expected


@pytest.mark.asyncio
async def test_failed_git_diff_is_visible_after_changed_files_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = changed_context(tmp_path / "diff-failure", "changed.txt", "after\n")
    current = snapshot()
    selected = replace(
        current.active_runs[0],
        session=SessionView(context=context, active=True),
    )
    controller = SnapshotController(
        initial_snapshot=replace(current, active_runs=(selected,)),
        replay_subscription=False,
    )
    original_changes = GitAdapter.session_changes

    def remove_worktree_after_changes(
        adapter: GitAdapter, loaded_context: SessionContext
    ) -> tuple[ChangedFile, ...]:
        result = original_changes(adapter, loaded_context)
        shutil.rmtree(loaded_context.cwd)
        return result

    monkeypatch.setattr(GitAdapter, "session_changes", remove_worktree_after_changes)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("x")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()

        expected = f"git session changes failed: worktree is unavailable: {context.cwd}"
        assert plain(app, "#changes-state") == "1 changed file"
        assert plain(app, "#diff-text") == expected


@pytest.mark.asyncio
async def test_stale_changes_response_cannot_replace_newly_selected_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = changed_context(tmp_path / "first", "first.txt", "first after\n")
    second = changed_context(tmp_path / "second", "second.txt", "second after\n")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(first, second), replay_subscription=False
    )
    first_started = Event()
    first_release = Event()
    second_finished = Event()
    original_changes = GitAdapter.session_changes

    def delayed_changes(adapter: GitAdapter, context: SessionContext) -> tuple[ChangedFile, ...]:
        if context.cwd == first.cwd:
            first_started.set()
            if not first_release.wait(timeout=5):
                raise AssertionError("first Git response did not get released")
        result = original_changes(adapter, context)
        if context.cwd == second.cwd:
            second_finished.set()
        return result

    monkeypatch.setattr(GitAdapter, "session_changes", delayed_changes)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("x")
        assert await asyncio.to_thread(first_started.wait, 5)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("j")
        await pilot.pause()
        assert app.selected_run_id == "run-2"
        controller.publish(controller.snapshot())
        first_release.set()
        assert await asyncio.to_thread(second_finished.wait, 5)
        await pilot.pause()

        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        table = cast(DataTable[RenderableType], app.query_one("#changes-files", DataTable))
        assert table.get_row_at(0)[1] == "second.txt"
        assert plain(app, "#changes-state") == "1 changed file"
        diff = plain(app, "#diff-text")
        assert "+second after" in diff
        assert "+first after" not in diff


@pytest.mark.asyncio
async def test_stale_diff_response_cannot_replace_newly_selected_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = changed_context(tmp_path / "first", "first.txt", "first after\n")
    second = changed_context(tmp_path / "second", "second.txt", "second after\n")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(first, second), replay_subscription=False
    )
    first_started = Event()
    first_release = Event()
    second_finished = Event()
    original_diff = GitAdapter.session_diff

    def delayed_diff(adapter: GitAdapter, context: SessionContext, path: str) -> str:
        if context.cwd == first.cwd:
            first_started.set()
            if not first_release.wait(timeout=5):
                raise AssertionError("first diff response did not get released")
        result = original_diff(adapter, context, path)
        if context.cwd == second.cwd:
            second_finished.set()
        return result

    monkeypatch.setattr(GitAdapter, "session_diff", delayed_diff)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("x")
        assert await asyncio.to_thread(first_started.wait, 5)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("j")
        await pilot.pause()
        assert app.selected_run_id == "run-2"
        controller.publish(controller.snapshot())
        first_release.set()
        assert await asyncio.to_thread(second_finished.wait, 5)
        await pilot.pause()

        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        table = cast(DataTable[RenderableType], app.query_one("#changes-files", DataTable))
        assert table.get_row_at(0)[1] == "second.txt"
        assert plain(app, "#changes-state") == "1 changed file"
        diff = plain(app, "#diff-text")
        assert "+second after" in diff
        assert "+first after" not in diff


@pytest.mark.asyncio
async def test_watch_session_controls_are_read_only() -> None:
    current = snapshot()
    selected = replace(
        current.active_runs[0],
        session=SessionView(actions=("steer",), active=True),
    )
    source = FakeController(initial_snapshot=replace(current, active_runs=(selected,)))
    app = WatchApp(source, poll_interval=60.0)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        await pilot.pause()

        message_input = app.query_one("#session-input", Input)
        submit = app.query_one("#session-submit", Button)
        assert app.query_one("#runs", DataTable).has_focus
        assert not message_input.has_focus
        assert message_input.disabled
        assert submit.disabled
        assert not app.query_one("#session-input-row").display
        assert not app.query_one("#structured-controls").display
        assert not app.query_one("#actions", Static).display
