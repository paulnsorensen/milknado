"""Controller-command workers shared by the execution TUI app."""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.widgets import Input

from milknado.app.run import ExecutionController


class ExecutionCommandsMixin:
    """Thread-backed controller commands; mixed into `ExecutionApp`."""

    controller: ExecutionController
    feature_branch: str | None
    strict: bool
    spec_text: str | None
    spec_path: Path | None

    @work(thread=True, group="execution", exclusive=True)
    def _run_execution(self) -> None:
        result = self.controller.run(
            feature_branch=self.feature_branch or "",
            strict=self.strict,
            spec_text=self.spec_text,
            spec_path=self.spec_path,
        )
        self.call_from_thread(self.exit, result)

    @work(thread=True, exclusive=False)
    def _queue_guidance(self, run_id: str, text: str) -> None:
        try:
            accepted = self.controller.queue_guidance(run_id, text)
        except RuntimeError:
            accepted = False
        if accepted:
            self.call_from_thread(self._clear_guidance_if_unchanged, text)
        else:
            self.call_from_thread(
                self.notify, "This run no longer accepts guidance.", severity="warning"
            )

    def _clear_guidance_if_unchanged(self, submitted: str) -> None:
        guidance = self.query_one("#guidance", Input)
        if guidance.value.strip() == submitted:
            guidance.value = ""

    def _run_control(self, action: object) -> object:
        try:
            return action()
        except RuntimeError as error:
            if str(error) != "execution has finished":
                raise
            return None

    @work(thread=True, group="controls", exclusive=False)
    def _cancel(self, run_id: str) -> None:
        self._run_control(lambda: self.controller.cancel(run_id))

    @work(thread=True, group="controls", exclusive=False)
    def _force_stop(self, run_id: str) -> None:
        completed = self._run_control(lambda: self.controller.force_stop(run_id))
        if completed is False:
            self.call_from_thread(
                self.notify, "Force-stop cleanup did not finish in time.", severity="warning"
            )

    @work(thread=True, group="controls", exclusive=False)
    def _stop_scheduling(self) -> None:
        self._run_control(self.controller.stop_scheduling)
