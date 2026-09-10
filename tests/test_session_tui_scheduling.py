from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Protocol, cast

import pytest
from textual.widgets import DataTable, Input
from typing_extensions import override

from milknado.adapters import ChangedFile, GitAdapter
from milknado.app.run import ExecutionController
from milknado.app.run_tui import ExecutionApp
from milknado.domains.common import SessionContext, SessionInput
from tests.test_session_tui import (
    SnapshotController,
    changed_context,
    plain,
    two_run_snapshot,
)


class _WorkerManager(Protocol):
    async def wait_for_complete(self) -> None: ...


def _wait_for_workers(app: ExecutionApp) -> _WorkerManager:
    return cast(_WorkerManager, app.workers)


@dataclass(kw_only=True)
class _OrderedController(SnapshotController):
    first_started: Event = field(default_factory=Event)
    release_first: Event = field(default_factory=Event)
    second_finished: Event = field(default_factory=Event)
    reject_first: bool = False

    @override
    def session_input(self, run_id: str, command: SessionInput) -> bool:
        if command.text == "first":
            self.first_started.set()
            assert self.release_first.wait(5), "First input was not released"
        self.submissions.append((run_id, command))
        if command.text == "second":
            self.second_finished.set()
        if command.text == "first" and self.reject_first:
            raise ValueError("First input rejected")
        return True


@dataclass(kw_only=True)
class _DuplicateController(SnapshotController):
    first_started: Event = field(default_factory=Event)
    release_first: Event = field(default_factory=Event)
    second_started: Event = field(default_factory=Event)
    release_second: Event = field(default_factory=Event)

    @override
    def session_input(self, run_id: str, command: SessionInput) -> bool:
        submission_number = len(self.submissions)
        if submission_number == 0:
            self.first_started.set()
            assert self.release_first.wait(5), "First input was not released"
        elif submission_number == 1:
            self.second_started.set()
            assert self.release_second.wait(5), "Second input was not released"
        self.submissions.append((run_id, command))
        if submission_number == 1:
            raise ValueError("Second input rejected")
        return True


@dataclass(kw_only=True)
class _RevisionController(SnapshotController):
    first_started: Event = field(default_factory=Event)
    release_first: Event = field(default_factory=Event)

    @override
    def session_input(self, run_id: str, command: SessionInput) -> bool:
        self.first_started.set()
        assert self.release_first.wait(5), "First input was not released"
        self.submissions.append((run_id, command))
        return True


@pytest.mark.asyncio
async def test_identical_queued_submissions_retain_draft_after_rejection(
    tmp_path: Path,
) -> None:
    context = changed_context(tmp_path / "repo", "work.txt", "after\n")
    controller = _DuplicateController(
        initial_snapshot=two_run_snapshot(context, context), replay_subscription=False
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("i", *"same draft", "enter")
        assert await asyncio.to_thread(controller.first_started.wait, 3)
        try:
            await pilot.press("enter")
            controller.release_first.set()
            assert await asyncio.to_thread(controller.second_started.wait, 3)
            await pilot.pause()
            assert app.query_one("#session-input", Input).value == "same draft"
        finally:
            controller.release_first.set()
            controller.release_second.set()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert [(run_id, command.text) for run_id, command in controller.submissions] == [
            ("run-1", "same draft"),
            ("run-1", "same draft"),
        ]
        assert app.query_one("#session-input", Input).value == "same draft"


@pytest.mark.asyncio
async def test_accepted_response_does_not_clear_newer_identical_draft_revision(
    tmp_path: Path,
) -> None:
    context = changed_context(tmp_path / "repo", "work.txt", "after\n")
    controller = _RevisionController(
        initial_snapshot=two_run_snapshot(context, context), replay_subscription=False
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("i", *"same draft", "enter")
        assert await asyncio.to_thread(controller.first_started.wait, 3)
        try:
            await pilot.press("end", "ctrl+u", *"other draft")
            await pilot.pause()
            await pilot.press("end", "ctrl+u", *"same draft")
            await pilot.pause()
            assert app.query_one("#session-input", Input).value == "same draft"
        finally:
            controller.release_first.set()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert controller.submissions == [
            ("run-1", SessionInput(action="steer", text="same draft"))
        ]
        assert app.query_one("#session-input", Input).value == "same draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_first", [False, True])
async def test_rapid_inputs_preserve_order_after_a_slow_controller_reply(
    tmp_path: Path, reject_first: bool
) -> None:
    context = changed_context(tmp_path / "repo", "work.txt", "after\n")
    controller = _OrderedController(
        initial_snapshot=two_run_snapshot(context, context),
        replay_subscription=False,
        reject_first=reject_first,
    )
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("i", *"first", "enter")
        assert await asyncio.to_thread(controller.first_started.wait, 3)
        try:
            await pilot.press("ctrl+a", "ctrl+k", *"second", "enter")
            _ = await asyncio.to_thread(controller.second_finished.wait, 0.5)
        finally:
            controller.release_first.set()
        assert await asyncio.to_thread(controller.second_finished.wait, 3)
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert [(run_id, command.text) for run_id, command in controller.submissions] == [
            ("run-1", "first"),
            ("run-1", "second"),
        ]
        assert app.query_one("#session-input", Input).value == ""


@pytest.mark.asyncio
async def test_context_change_clears_old_diff_and_bounds_streaming_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = changed_context(tmp_path / "first", "first.txt", "first-only\n")
    second = changed_context(tmp_path / "second", "second.txt", "second-only\n")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(first, first), replay_subscription=False
    )
    entered, release, lock = Event(), Event(), Lock()
    active = peak = 0
    original = GitAdapter.session_changes

    def delayed(adapter: GitAdapter, context: SessionContext) -> tuple[ChangedFile, ...]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if context == second:
                entered.set()
                assert release.wait(5), "Second worktree inspection was not released"
            return original(adapter, context)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(GitAdapter, "session_changes", delayed)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert "+first-only" in plain(app, "#diff-text")
        controller.initial_snapshot = two_run_snapshot(second, first)
        controller.publish(controller.initial_snapshot)
        assert await asyncio.to_thread(entered.wait, 3)
        try:
            for _ in range(20):
                controller.publish(controller.initial_snapshot)
            await pilot.pause()
            assert "first-only" not in plain(app, "#diff-text")
            assert app.query_one("#changes-files", DataTable).row_count == 0
        finally:
            release.set()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert "+second-only" in plain(app, "#diff-text")
        assert peak == 1


@pytest.mark.asyncio
async def test_file_change_clears_previous_diff_until_selected_file_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = changed_context(tmp_path / "repo", "first.txt", "first-only\n")
    _ = (Path(context.cwd) / "second.txt").write_text("second-only\n")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(context, context), replay_subscription=False
    )
    entered, release = Event(), Event()
    original = GitAdapter.session_diff

    def delayed(adapter: GitAdapter, selected: SessionContext, path: str) -> str:
        if path == "second.txt":
            entered.set()
            assert release.wait(5), "Selected diff was not released"
        return original(adapter, selected, path)

    monkeypatch.setattr(GitAdapter, "session_diff", delayed)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert "+first-only" in plain(app, "#diff-text")
        await pilot.press("x", "down", "enter")
        assert await asyncio.to_thread(entered.wait, 3)
        try:
            assert "first-only" not in plain(app, "#diff-text")
        finally:
            release.set()
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert "+second-only" in plain(app, "#diff-text")


@pytest.mark.asyncio
async def test_periodic_refresh_preserves_keyboard_file_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = changed_context(tmp_path / "repo", "first.txt", "first-only\n")
    _ = (Path(context.cwd) / "second.txt").write_text("second-only\n")
    controller = SnapshotController(
        initial_snapshot=two_run_snapshot(context, context), replay_subscription=False
    )
    ready, refreshed = Event(), Event()
    original = GitAdapter.session_changes

    def inspected(adapter: GitAdapter, selected: SessionContext) -> tuple[ChangedFile, ...]:
        result = original(adapter, selected)
        if ready.is_set():
            refreshed.set()
        return result

    monkeypatch.setattr(GitAdapter, "session_changes", inspected)
    app = ExecutionApp(cast(ExecutionController, cast(object, controller)))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        await pilot.press("x", "down")
        ready.set()
        assert await asyncio.to_thread(refreshed.wait, 3)
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_workers(app).wait_for_complete()
        await pilot.pause()
        assert "+second-only" in plain(app, "#diff-text")
