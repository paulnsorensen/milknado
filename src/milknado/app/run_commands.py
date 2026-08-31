"""Controller-command workers shared by the execution TUI app."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar, cast

from textual import work
from textual.widgets import Input

from milknado.app.run import ExecutionController

_T = TypeVar("_T")
_WidgetT = TypeVar("_WidgetT")


class _ExecutionAppLike(Protocol):
    """Host surface supplied by the concrete Textual execution app."""

    controller: ExecutionController
    feature_branch: str | None
    strict: bool
    spec_text: str | None
    spec_path: Path | None

    def call_from_thread(
        self, callback: Callable[..., object], *args: object, **kwargs: object
    ) -> object: ...

    def exit(self, result: object | None = None) -> None: ...

    def notify(self, message: str, **_kwargs: object) -> object: ...

    def query_one(self, _selector: str, _expect_type: type[_WidgetT]) -> _WidgetT: ...


class ExecutionCommandsMixin:
    """Thread-backed controller commands; mixed into `ExecutionApp`."""

    @work(thread=True, group="execution", exclusive=True)
    def _run_execution(self) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        result = app.controller.run(
            feature_branch=app.feature_branch or "",
            strict=app.strict,
            spec_text=app.spec_text,
            spec_path=app.spec_path,
        )
        _ = app.call_from_thread(app.exit, result)

    @work(thread=True, exclusive=False)
    def _queue_guidance(self, run_id: str, text: str) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        try:
            accepted = app.controller.queue_guidance(run_id, text)
        except RuntimeError:
            accepted = False
        if accepted:
            _ = app.call_from_thread(self._clear_guidance_if_unchanged, text)
        else:
            _ = app.call_from_thread(
                app.notify, "This run no longer accepts guidance.", severity="warning"
            )

    def _clear_guidance_if_unchanged(self, submitted: str) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        guidance = app.query_one("#guidance", Input)
        if guidance.value.strip() == submitted:
            guidance.value = ""

    def _run_control(self, action: Callable[[], _T]) -> _T | None:
        try:
            return action()
        except RuntimeError as error:
            if str(error) != "execution has finished":
                raise
            return None

    @work(thread=True, group="controls", exclusive=False)
    def _cancel(self, run_id: str) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        _ = self._run_control(lambda: app.controller.cancel(run_id))

    @work(thread=True, group="controls", exclusive=False)
    def _force_stop(self, run_id: str) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        completed = self._run_control(lambda: app.controller.force_stop(run_id))
        if completed is False:
            _ = app.call_from_thread(
                app.notify, "Force-stop cleanup did not finish in time.", severity="warning"
            )

    @work(thread=True, group="controls", exclusive=False)
    def _stop_scheduling(self) -> None:
        app = cast(_ExecutionAppLike, cast(object, self))
        _ = self._run_control(app.controller.stop_scheduling)
