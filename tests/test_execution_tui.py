from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Protocol, cast

import pytest
from rich.console import RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import DataTable, Input, Static

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_panels import RunDetailPanel
from milknado.app.run_tui import ExecutionApp


class _WorkerManager(Protocol):
    async def wait_for_complete(self) -> None: ...


def _as_execution_controller(controller: object) -> ExecutionController:
    return cast(ExecutionController, controller)


def _execution_app(controller: object) -> ExecutionApp:
    return ExecutionApp(_as_execution_controller(controller))


def _input(app: ExecutionApp, widget_id: str) -> Input:
    return app.query_one(widget_id, Input)


def _static(app: ExecutionApp, widget_id: str) -> Static:
    return app.query_one(widget_id, Static)


def _plain(app: ExecutionApp, widget_id: str) -> str:
    return cast(Text, _static(app, widget_id).render()).plain


def _output(app: ExecutionApp) -> VerticalScroll:
    return app.query_one("#output", VerticalScroll)


def _runs(app: ExecutionApp) -> DataTable[RenderableType]:
    return cast(DataTable[RenderableType], app.query_one("#runs", DataTable))


def _wait_for_workers(app: ExecutionApp) -> _WorkerManager:
    return cast(_WorkerManager, app.workers)


def _worker_call(app: ExecutionApp, name: str, *args: object) -> None:
    method = cast(Callable[..., object], getattr(app, name))
    _ = method(*args)


@dataclass
class FakeController:
    guidance: list[tuple[str, str]] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)
    force_stops: list[str] = field(default_factory=list)
    listener: Callable[[ExecutionSnapshot], None] | None = None
    stop_requests: int = 0
    run_result: object = "run-result"
    run_calls: list[dict[str, object]] = field(default_factory=list)
    rejected_guidance: bool = False
    initial_snapshot: ExecutionSnapshot | None = None
    replay_subscription: bool = True
    guidance_error: RuntimeError | None = None
    control_error: RuntimeError | None = None

    def snapshot(self) -> ExecutionSnapshot:
        return self.initial_snapshot or snapshot()

    def run(self, **kwargs: object) -> object:
        self.run_calls.append(kwargs)
        return self.run_result

    def stop_scheduling(self) -> None:
        self.stop_requests += 1
        if self.control_error is not None:
            raise self.control_error

    unsubscribed: bool = False

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self.listener = listener
        if self.replay_subscription:
            listener(self.snapshot())

        def unsubscribe() -> None:
            self.listener = None
            self.unsubscribed = True

        return unsubscribe

    def publish(self, replacement: ExecutionSnapshot) -> None:
        assert self.listener is not None
        self.listener(replacement)

    def queue_guidance(self, run_id: str, text: str) -> bool:
        self.guidance.append((run_id, text))
        if self.guidance_error is not None:
            raise self.guidance_error
        return not self.rejected_guidance

    def cancel(self, run_id: str) -> None:
        self.cancellations.append(run_id)
        if self.control_error is not None:
            raise self.control_error

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        _ = timeout
        self.force_stops.append(run_id)
        if self.control_error is not None:
            raise self.control_error
        return True


def snapshot(
    *,
    second: bool = False,
    output: tuple[str, ...] = ("first output", "latest output"),
    event_lines: tuple[str, ...] = ("run-1 started",),
) -> ExecutionSnapshot:
    runs = [
        ActiveRunSnapshot(
            run_id="run-1",
            node_id=1,
            description="First task",
            status=ExecutionRunStatus.RUNNING,
            progress="iteration 2",
            stop_requested=False,
            actions=RunActionAvailability(),
            output=output,
            pending_guidance=(),
            elapsed_seconds=12.0,
            progress_pct=50.0,
            eta_seconds=8.0,
            attempt=1,
            max_attempts=3,
            stalled=False,
        )
    ]
    if second:
        runs.append(
            ActiveRunSnapshot(
                run_id="run-2",
                node_id=2,
                description="Second task",
                status=ExecutionRunStatus.RUNNING,
                progress=None,
                stop_requested=False,
                actions=RunActionAvailability(force_stop_reason="No child process is active."),
                output=(),
                pending_guidance=("check tests",),
                elapsed_seconds=30.0,
                progress_pct=None,
                eta_seconds=None,
                attempt=2,
                max_attempts=3,
                stalled=True,
            )
        )
    return ExecutionSnapshot(
        goal="Responsive run",
        active_runs=tuple(runs),
        terminal_runs=(),
        completed=1,
        failed=0,
        stopped=0,
        available=2,
        event_lines=event_lines,
    )


def stopped_snapshot() -> ExecutionSnapshot:
    return ExecutionSnapshot(
        goal="Responsive run",
        active_runs=(),
        terminal_runs=(
            TerminalRunSnapshot(
                run_id="run-1",
                node_id=1,
                description="First task",
                status=ExecutionRunStatus.STOPPED,
                output=("[bold]literal worker output[/bold]",),
                pending_guidance=("not delivered",),
                duration_seconds=64.0,
            ),
        ),
        completed=0,
        failed=0,
        stopped=1,
        available=0,
        event_lines=("run-1 stopped",),
    )


@pytest.mark.asyncio
async def test_wide_view_queues_guidance_and_confirms_force_stop() -> None:

    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        assert app.compact is False
        _input(app, "#guidance").value = "check tests"
        _ = await pilot.click("#guidance")
        await pilot.press("enter")
        await _wait_for_workers(app).wait_for_complete()
        assert controller.guidance == [("run-1", "check tests")]
        assert _input(app, "#guidance").value == ""
        app.action_force()
        assert "Force stop run-1?" in _plain(app, "#confirmation")
        assert controller.force_stops == []
        await pilot.press("y")
        await _wait_for_workers(app).wait_for_complete()
        assert controller.force_stops == ["run-1"]


@pytest.mark.asyncio
async def test_compact_drilldown_preserves_composer_and_exposes_action_reason() -> None:

    controller = FakeController(initial_snapshot=snapshot(second=True))
    app = _execution_app(controller)
    async with app.run_test(size=(70, 30)) as pilot:
        assert app.compact is True
        assert app.route == "list"
        await pilot.press("down", "enter")
        assert app.route == "detail"
        assert app.selected_run_id == "run-2"
        assert "No child process is active." in _plain(app, "#actions")
        _input(app, "#guidance").value = "hold this"

        await pilot.resize_terminal(width=120, height=36)
        assert app.compact is False
        assert app.selected_run_id == "run-2"
        assert _input(app, "#guidance").value == "hold this"
        await pilot.resize_terminal(width=70, height=30)
        assert app.compact is True
        assert app.route == "detail"
        await pilot.press("escape")
        assert app.route == "list"
        await pilot.press("enter")
        assert _input(app, "#guidance").value == "hold this"


@pytest.mark.asyncio
async def test_help_and_quit_confirmation_only_show_current_actions() -> None:

    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("h")
        assert "g queue guidance" in _plain(app, "#help")
        assert "f force stop" in _plain(app, "#help")
        await pilot.press("escape", "q")
        assert "Stop scheduling and gracefully stop 1 active run?" in (
            _plain(app, "#confirmation")
        )
        await pilot.press("y")
        await _wait_for_workers(app).wait_for_complete()
        assert controller.stop_requests == 1


@pytest.mark.asyncio
async def test_unmount_releases_controller_snapshot_subscription() -> None:

    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)):
        assert controller.listener is not None

    assert controller.unsubscribed is True


@pytest.mark.asyncio
async def test_synchronous_subscription_replay_updates_on_the_app_thread() -> None:

    controller = FakeController(replay_subscription=True, initial_snapshot=snapshot(second=True))
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)):
        assert app.selected_run_id == "run-1"
        assert _runs(app).row_count == 2


@pytest.mark.asyncio
async def test_rejected_guidance_remains_available_for_correction() -> None:

    controller = FakeController(rejected_guidance=True)
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        guidance = _input(app, "#guidance")
        guidance.value = "retry after review"
        _ = await pilot.click("#guidance")
        await pilot.press("enter")
        await _wait_for_workers(app).wait_for_complete()

        assert controller.guidance == [("run-1", "retry after review")]
        assert guidance.value == "retry after review"


@pytest.mark.asyncio
async def test_guidance_rejection_during_shutdown_preserves_input() -> None:

    controller = FakeController(guidance_error=RuntimeError("execution has finished"))
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        guidance = _input(app, "#guidance")
        guidance.value = "keep this guidance"
        _ = await pilot.click("#guidance")
        await pilot.press("enter")
        await _wait_for_workers(app).wait_for_complete()

        assert controller.guidance == [("run-1", "keep this guidance")]
        assert guidance.value == "keep this guidance"


@pytest.mark.asyncio
async def test_terminal_control_rejections_do_not_fail_tui_workers() -> None:

    controller = FakeController(control_error=RuntimeError("execution has finished"))
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)):
        _worker_call(app, "_cancel", "run-1")
        _worker_call(app, "_force_stop", "run-1")
        _worker_call(app, "_stop_scheduling")
        await _wait_for_workers(app).wait_for_complete()

    assert controller.cancellations == ["run-1"]
    assert controller.force_stops == ["run-1"]
    assert controller.stop_requests == 1


def test_execution_and_shutdown_workers_use_distinct_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    app = _execution_app(FakeController())
    groups: list[str] = []

    def capture_worker(_work: object, **kwargs: object) -> None:
        group = kwargs["group"]
        assert isinstance(group, str)
        groups.append(group)

    monkeypatch.setattr(app, "run_worker", capture_worker)

    _worker_call(app, "_run_execution")
    _worker_call(app, "_stop_scheduling")

    assert groups == ["execution", "controls"]


@pytest.mark.asyncio
async def test_compact_layout_follows_focused_pane() -> None:

    app = _execution_app(FakeController(initial_snapshot=snapshot(second=True)))

    async with app.run_test(size=(70, 30)) as pilot:
        _ = _input(app, "#guidance").focus()
        await pilot.pause()
        await pilot.resize_terminal(width=120, height=36)
        await pilot.resize_terminal(width=70, height=30)

        assert app.route == "detail"
        assert app.has_class("detail")


@pytest.mark.asyncio
async def test_compact_layout_returns_to_the_focused_run_table() -> None:

    app = _execution_app(FakeController(initial_snapshot=snapshot(second=True)))

    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.press("enter")
        assert app.route == "detail"
        await pilot.resize_terminal(width=120, height=36)
        _ = _input(app, "#guidance").focus()
        await pilot.pause()
        _ = _runs(app).focus()
        await pilot.pause()
        await pilot.resize_terminal(width=70, height=30)

        assert app.route == "list"
        assert app.has_class("list")


@pytest.mark.asyncio
async def test_pointer_scroll_pauses_auto_follow() -> None:

    app = _execution_app(FakeController())

    async with app.run_test(size=(120, 36)):
        app.query_one("#detail", RunDetailPanel).on_mouse_scroll_down(
            cast(MouseScrollDown, cast(object, None))
        )

        assert app.auto_follow is False
        app.query_one("#detail", RunDetailPanel).on_mouse_scroll_up(
            cast(MouseScrollUp, cast(object, None))
        )
        assert _output(app).border_title == "Output (paused; press r to resume)"
        rendered = app.export_screenshot().replace("&#160;", " ")
        assert "Output (paused; press r to resume)" in rendered


@pytest.mark.asyncio
async def test_events_height_budget_yields_to_workspace_at_small_terminal() -> None:
    events = tuple(f"run-1 event {index}" for index in range(30))
    app = _execution_app(FakeController(initial_snapshot=snapshot(event_lines=events)))

    async with app.run_test(size=(120, 14)):
        workspace_height = app.query_one("#workspace").region.height
        events_height = app.query_one("#events").region.height

        assert workspace_height > events_height


@pytest.mark.asyncio
async def test_quit_stops_future_scheduling_before_waiting_for_run_result() -> None:

    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("q", "y")
        await _wait_for_workers(app).wait_for_complete()

        assert controller.stop_requests == 1


@pytest.mark.asyncio
async def test_quit_waits_for_an_in_flight_execution_without_active_runs() -> None:

    controller = FakeController(
        initial_snapshot=ExecutionSnapshot(
            goal="Responsive run",
            active_runs=(),
            terminal_runs=(),
            completed=0,
            failed=0,
            stopped=0,
            available=0,
            event_lines=(),
        )
    )
    app = _execution_app(controller)

    class InFlightWorker:
        is_finished: bool = False

    async with app.run_test(size=(120, 36)) as pilot:
        worker_attr = "_execution_worker"
        setattr(app, worker_attr, InFlightWorker())
        await pilot.press("q")

        assert "Stop scheduling and gracefully stop 0 active runs?" in (
            _plain(app, "#confirmation")
        )
        await pilot.press("y")
        await _wait_for_workers(app).wait_for_complete()

        assert controller.stop_requests == 1


@pytest.mark.asyncio
async def test_cancel_binding_targets_the_selected_run() -> None:
    controller = FakeController(initial_snapshot=snapshot(second=True))
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        app.action_next_run()
        await pilot.press("c")
        await _wait_for_workers(app).wait_for_complete()

        assert app.selected_run_id == "run-2"
        assert controller.cancellations == ["run-2"]


@pytest.mark.asyncio
async def test_worker_thread_snapshot_is_applied_on_the_app_thread() -> None:
    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        publisher = Thread(target=controller.publish, args=(snapshot(second=True),))
        publisher.start()
        await pilot.pause()
        publisher.join(timeout=1)

        assert not publisher.is_alive()
        assert _runs(app).row_count == 2
        assert app.snapshot.active_runs[1].run_id == "run-2"


@pytest.mark.asyncio
async def test_stopped_run_retains_literal_output_and_undelivered_guidance() -> None:
    app = _execution_app(FakeController(initial_snapshot=stopped_snapshot()))

    async with app.run_test(size=(120, 36)):
        assert _runs(app).row_count == 1
        assert "[bold]literal worker output[/bold]" in (_plain(app, "#output-text"))
        assert "Undelivered guidance: not delivered" in _plain(app, "#summary")
        assert "output retained for inspection" in _plain(app, "#actions")
        assert _input(app, "#guidance").disabled is True


@pytest.mark.asyncio
async def test_paused_scroll_offset_survives_compact_and_wide_layouts() -> None:
    lines = tuple(f"worker output {index}" for index in range(100))
    app = _execution_app(FakeController(initial_snapshot=snapshot(output=lines)))

    async with app.run_test(size=(120, 24)) as pilot:
        app.pause_auto_follow()
        output = _output(app)
        output.scroll_to(y=8, animate=False)
        await pilot.pause()
        paused_offset = output.scroll_offset.y
        assert paused_offset > 0

        await pilot.resize_terminal(width=70, height=24)
        await pilot.resize_terminal(width=120, height=24)

        assert app.auto_follow is False
        assert output.scroll_offset.y == paused_offset


@pytest.mark.asyncio
async def test_paused_output_position_survives_a_compact_route_round_trip() -> None:
    lines = tuple(f"worker output {index}" for index in range(100))
    app = _execution_app(FakeController(initial_snapshot=snapshot(output=lines)))

    async with app.run_test(size=(120, 24)) as pilot:
        app.pause_auto_follow()
        output = _output(app)
        output.scroll_to(y=8, animate=False)
        await pilot.pause()

        await pilot.resize_terminal(width=80, height=24)
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.resize_terminal(width=120, height=24)

        assert output.scroll_offset.y == 8


@pytest.mark.asyncio
async def test_guidance_composer_stays_on_screen_on_a_short_terminal() -> None:
    lines = tuple(f"worker output {index}" for index in range(100))
    events = tuple(f"run-1 event {index}" for index in range(30))
    app = _execution_app(
        FakeController(initial_snapshot=snapshot(output=lines, event_lines=events))
    )

    async with app.run_test(size=(120, 24)):
        guidance = _input(app, "#guidance").region

        assert guidance.height == 3
        assert app.screen.region.contains_region(guidance)
        assert _output(app).container_size.height >= 3


@pytest.mark.asyncio
async def test_quit_graceful_stop_keeps_force_stop_escalation_available() -> None:
    controller = FakeController()
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("q", "y")
        await _wait_for_workers(app).wait_for_complete()
        assert controller.stop_requests == 1

        app.action_force()
        await pilot.press("y")
        await _wait_for_workers(app).wait_for_complete()
        assert controller.force_stops == ["run-1"]


@pytest.mark.asyncio
async def test_navigation_actions_and_resume_auto_follow() -> None:
    app = _execution_app(FakeController(initial_snapshot=snapshot(second=True)))

    async with app.run_test(size=(70, 30)) as pilot:
        app.action_previous_run()
        assert app.selected_run_id == "run-2"
        app.action_next_run()
        assert app.selected_run_id == "run-1"

        app.action_open_detail()
        assert app.route == "detail"
        _ = app.action_back()
        assert app.route == "list"
        app.action_open_detail()

        app.action_focus_guidance()
        await pilot.pause()
        assert _input(app, "#guidance").has_focus
        app.pause_auto_follow()
        app.action_resume_output()
        assert app.auto_follow is True


@pytest.mark.asyncio
async def test_quit_without_runs_exits_immediately() -> None:
    controller = FakeController(
        initial_snapshot=ExecutionSnapshot(
            goal="Responsive run",
            active_runs=(),
            terminal_runs=(),
            completed=0,
            failed=0,
            stopped=0,
            available=0,
            event_lines=(),
        )
    )
    app = _execution_app(controller)

    async with app.run_test(size=(120, 36)):
        app.action_previous_run()
        app.action_next_run()
        app.action_open_detail()
        app.action_force()
        assert app.route == "list"
        await app.action_quit()

    assert controller.stop_requests == 0


@pytest.mark.asyncio
async def test_execution_worker_forwards_options_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = FakeController()
    app = ExecutionApp(
        _as_execution_controller(controller),
        feature_branch="feature",
        strict=True,
        spec_text="spec",
        spec_path=Path("spec.md"),
    )
    exits: list[object] = []
    monkeypatch.setattr(app, "exit", exits.append)

    async with app.run_test(size=(120, 36)):
        await _wait_for_workers(app).wait_for_complete()

    assert exits == ["run-result"]
    assert controller.run_calls == [
        {
            "feature_branch": "feature",
            "strict": True,
            "spec_text": "spec",
            "spec_path": Path("spec.md"),
        }
    ]


def test_tui_entry_returns_the_execution_result(monkeypatch: pytest.MonkeyPatch) -> None:
    import milknado.app.run_tui as run_tui

    expected = object()

    class FakeApp:
        def __init__(self, controller: FakeController, **kwargs: object) -> None:
            assert controller is expected_controller
            assert kwargs["feature_branch"] == "feature"

        def run(self) -> object:
            return expected

    expected_controller = FakeController()
    monkeypatch.setattr(run_tui, "ExecutionApp", FakeApp)

    assert (
        run_tui.run_execution_tui(
            cast(ExecutionController, cast(object, expected_controller)), feature_branch="feature"
        )
        is expected
    )
