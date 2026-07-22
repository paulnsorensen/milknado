from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from milknado.domains.execution import ActiveRunSnapshot, ExecutionSnapshot
from milknado.loop import RunStatus


@dataclass
class FakeController:
    guidance: list[tuple[str, str]] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)
    force_stops: list[str] = field(default_factory=list)
    listener: Callable[[ExecutionSnapshot], None] | None = None
    stop_requests: int = 0
    run_result: object = "run-result"
    rejected_guidance: bool = False
    initial_snapshot: ExecutionSnapshot | None = None
    replay_subscription: bool = False
    guidance_error: RuntimeError | None = None
    control_error: RuntimeError | None = None
    
    def snapshot(self) -> ExecutionSnapshot:
        return self.initial_snapshot or snapshot()

    def run(self, **_kwargs: object) -> object:
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
        self.force_stops.append(run_id)
        if self.control_error is not None:
            raise self.control_error
        return True


def snapshot(*, second: bool = False) -> ExecutionSnapshot:
    runs = [
        ActiveRunSnapshot(
            run_id="run-1",
            node_id=1,
            description="First task",
            status=RunStatus.RUNNING,
            progress="iteration 2",
            stop_requested=False,
            force_stop_available=True,
            unavailable_action_reasons=(),
            output=("first output", "latest output"),
            pending_guidance=(),
        )
    ]
    if second:
        runs.append(
            ActiveRunSnapshot(
                run_id="run-2",
                node_id=2,
                description="Second task",
                unavailable_action_reasons=(("force_stop", "No child process is active."),),
                status=RunStatus.RUNNING,
                progress=None,
                stop_requested=False,
                force_stop_available=False,
                output=(),
                pending_guidance=("check tests",),
            )
        )
    return ExecutionSnapshot(
        goal="Responsive run",
        active_runs=tuple(runs),
        completed=1,
        failed=0,
        available=2,
        event_lines=("run-1 started",),
    )


@pytest.mark.asyncio
async def test_wide_view_queues_guidance_and_confirms_force_stop() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController()
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        assert app.compact is False
        app.query_one("#guidance").value = "check tests"
        await pilot.click("#guidance")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert controller.guidance == [("run-1", "check tests")]
        assert app.query_one("#guidance").value == ""
        app.action_force()
        assert "Force stop run-1?" in app.query_one("#confirmation").render().plain
        assert controller.force_stops == []
        await pilot.press("y")
        await app.workers.wait_for_complete()
        assert controller.force_stops == ["run-1"]


@pytest.mark.asyncio
async def test_compact_drilldown_preserves_composer_and_exposes_action_reason() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(initial_snapshot=snapshot(second=True))
    app = ExecutionApp(controller)
    async with app.run_test(size=(70, 30)) as pilot:
        assert app.compact is True
        assert app.route == "list"
        await pilot.press("down", "enter")
        assert app.route == "detail"
        assert app.selected_run_id == "run-2"
        assert "No child process is active." in app.query_one("#actions").render().plain
        app.query_one("#guidance").value = "hold this"

        await pilot.resize_terminal(width=120, height=36)
        assert app.compact is False
        assert app.selected_run_id == "run-2"
        assert app.query_one("#guidance").value == "hold this"
        await pilot.resize_terminal(width=70, height=30)
        assert app.compact is True
        assert app.route == "detail"
        await pilot.press("escape")
        assert app.route == "list"
        await pilot.press("enter")
        assert app.query_one("#guidance").value == "hold this"


@pytest.mark.asyncio
async def test_help_and_quit_confirmation_only_show_current_actions() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController()
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("h")
        assert "g queue guidance" in app.query_one("#help").render().plain
        assert "f force stop" in app.query_one("#help").render().plain
        await pilot.press("escape", "q")
        assert "Quit and cancel 1 active run?" in app.query_one("#confirmation").render().plain
        await pilot.press("y")
        await app.workers.wait_for_complete()
        assert controller.stop_requests == 1


@pytest.mark.asyncio
async def test_unmount_releases_controller_snapshot_subscription() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController()
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)):
        assert controller.listener is not None

    assert controller.unsubscribed is True


@pytest.mark.asyncio
async def test_synchronous_subscription_replay_updates_on_the_app_thread() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(replay_subscription=True, initial_snapshot=snapshot(second=True))
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)):
        assert app.selected_run_id == "run-1"
        assert app.query_one("#runs").row_count == 2


@pytest.mark.asyncio
async def test_rejected_guidance_remains_available_for_correction() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(rejected_guidance=True)
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        guidance = app.query_one("#guidance")
        guidance.value = "retry after review"
        await pilot.click("#guidance")
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert controller.guidance == [("run-1", "retry after review")]
        assert guidance.value == "retry after review"




@pytest.mark.asyncio
async def test_guidance_rejection_during_shutdown_preserves_input() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(guidance_error=RuntimeError("execution has finished"))
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        guidance = app.query_one("#guidance")
        guidance.value = "keep this guidance"
        await pilot.click("#guidance")
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert controller.guidance == [("run-1", "keep this guidance")]
        assert guidance.value == "keep this guidance"


@pytest.mark.asyncio
async def test_terminal_control_rejections_do_not_fail_tui_workers() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(control_error=RuntimeError("execution has finished"))
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)):
        app._cancel("run-1")
        app._force_stop("run-1")
        app._stop_scheduling()
        await app.workers.wait_for_complete()

    assert controller.cancellations == ["run-1"]
    assert controller.force_stops == ["run-1"]
    assert controller.stop_requests == 1


def test_execution_and_shutdown_workers_use_distinct_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from milknado.app.run_tui import ExecutionApp

    app = ExecutionApp(FakeController())
    groups: list[str] = []
    monkeypatch.setattr(
        app,
        "run_worker",
        lambda _work, **kwargs: groups.append(kwargs["group"]),
    )

    app._run_execution()
    app._stop_scheduling()

    assert groups == ["execution", "controls"]
@pytest.mark.asyncio
async def test_compact_layout_follows_focused_pane() -> None:
    from milknado.app.run_tui import ExecutionApp

    app = ExecutionApp(FakeController(initial_snapshot=snapshot(second=True)))

    async with app.run_test(size=(70, 30)) as pilot:
        app.query_one("#guidance").focus()
        await pilot.pause()
        await pilot.resize_terminal(width=120, height=36)
        await pilot.resize_terminal(width=70, height=30)

        assert app.route == "detail"
        assert app.has_class("detail")



@pytest.mark.asyncio
async def test_compact_layout_returns_to_the_focused_run_table() -> None:
    from milknado.app.run_tui import ExecutionApp

    app = ExecutionApp(FakeController(initial_snapshot=snapshot(second=True)))

    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.press("enter")
        assert app.route == "detail"
        await pilot.resize_terminal(width=120, height=36)
        app.query_one("#guidance").focus()
        await pilot.pause()
        app.query_one("#runs").focus()
        await pilot.pause()
        await pilot.resize_terminal(width=70, height=30)

        assert app.route == "list"
        assert app.has_class("list")

@pytest.mark.asyncio
async def test_pointer_scroll_pauses_auto_follow() -> None:
    from milknado.app.run_tui import ExecutionApp

    app = ExecutionApp(FakeController())

    async with app.run_test(size=(120, 36)):
        app.query_one("#detail").on_mouse_scroll_down(None)  # type: ignore[arg-type]

        assert app.auto_follow is False
        assert "paused; press r to resume" in app.query_one("#output").render().plain


@pytest.mark.asyncio
async def test_quit_stops_future_scheduling_before_waiting_for_run_result() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController()
    app = ExecutionApp(controller)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("q", "y")
        await app.workers.wait_for_complete()

        assert controller.stop_requests == 1



@pytest.mark.asyncio
async def test_quit_waits_for_an_in_flight_execution_without_active_runs() -> None:
    from milknado.app.run_tui import ExecutionApp

    controller = FakeController(initial_snapshot=ExecutionSnapshot(
        goal="Responsive run", active_runs=(), completed=0, failed=0, available=0, event_lines=()
    ))
    app = ExecutionApp(controller)

    class InFlightWorker:
        is_finished = False

    async with app.run_test(size=(120, 36)) as pilot:
        app._execution_worker = InFlightWorker()  # type: ignore[assignment]
        await pilot.press("q")

        assert "Quit and cancel 0 active runs?" in app.query_one("#confirmation").render().plain
        await pilot.press("y")
        await app.workers.wait_for_complete()

        assert controller.stop_requests == 1

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

    assert run_tui.run_execution_tui(expected_controller, feature_branch="feature") is expected
