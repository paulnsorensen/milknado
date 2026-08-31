"""Tests for the run engine."""

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from milknado.loop._agent import AgentResult  # pyright: ignore[reportMissingTypeStubs]
from milknado.loop._events import (  # pyright: ignore[reportMissingTypeStubs]
    BoundEmitter,
    EventType,
    NullEmitter,
    QueueEmitter,
)
from milknado.loop._run_types import (  # pyright: ignore[reportMissingTypeStubs]
    Command,
    CompletionVerdict,
    RunConfig,
    RunState,
    RunStatus,
)
from milknado.loop._runner import RunResult  # pyright: ignore[reportMissingTypeStubs]
from milknado.loop.adapters import select_adapter  # pyright: ignore[reportMissingTypeStubs]
from milknado.loop.engine import (  # pyright: ignore[reportMissingTypeStubs]
    _assemble_prompt,  # pyright: ignore[reportPrivateUsage]
    _delay_if_needed,  # pyright: ignore[reportPrivateUsage]
    _handle_control_signals,  # pyright: ignore[reportPrivateUsage]
    _run_commands,  # pyright: ignore[reportPrivateUsage]
    run_loop,
)
from tests.loop.helpers import (
    MOCK_RUN_COMMAND,
    MOCK_SUBPROCESS,
    drain_events,  # pyright: ignore[reportUnknownVariableType]
    event_types,  # pyright: ignore[reportUnknownVariableType]
    events_of_type,  # pyright: ignore[reportUnknownVariableType]
    fail_proc,
    make_config,
    make_state,
    ok_proc,
    ok_run_result,
    timeout_proc,
)


class TestRunLoop:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_single_iteration(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.completed == 1
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert mock_run.call_count == 1

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_multiple_iterations(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=3)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert state.completed == 3
        assert mock_run.call_count == 3

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_ordinary_loop_ignores_unrelated_completion_probe(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=2,
            stop_on_completion_signal=True,
            completion_probe=lambda: False,
        )
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert mock_run.call_count == 2
        assert state.status is RunStatus.COMPLETED

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_failed_iterations_counted(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=2)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert state.completed == 0
        assert state.failed == 2

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_stop_on_error(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=5, stop_on_error=True)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert mock_run.call_count == 1
        assert state.failed == 1

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_stop_on_error_sets_failed_status(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        """When stop_on_error triggers, status should be FAILED, not COMPLETED."""
        config = make_config(tmp_path, max_iterations=5, stop_on_error=True)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.status == RunStatus.FAILED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "error"  # pyright: ignore[reportUnknownMemberType]

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_max_consecutive_failures_stops_run(self, mock_run: MagicMock, tmp_path: Path):
        """The cap turns a would-be-infinite crash loop into a bounded FAILED run."""
        config = make_config(tmp_path, max_iterations=10, max_consecutive_failures=3)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert mock_run.call_count == 3
        assert state.failed == 3
        assert state.status == RunStatus.FAILED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "error"  # pyright: ignore[reportUnknownMemberType]

    @patch(MOCK_SUBPROCESS)
    def test_consecutive_failure_count_resets_on_success(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        """Sporadic failures below the cap never end the run."""
        mock_run.side_effect = [
            fail_proc(),
            fail_proc(),
            ok_proc(),
            fail_proc(),
            fail_proc(),
            ok_proc(),
        ]
        config = make_config(tmp_path, max_iterations=6, max_consecutive_failures=3)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert mock_run.call_count == 6
        assert state.completed == 2
        assert state.failed == 4
        assert state.status == RunStatus.COMPLETED

    @patch(MOCK_SUBPROCESS)
    def test_last_captured_stderr_stored_for_failure_reporting(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        """Captured stderr survives on state so coordinators can log the failure cause."""
        mock_run.side_effect = lambda *a, **k: fail_proc(  # pyright: ignore[reportUnknownLambdaType]
            stderr_text="Error: unknown flag: --mcp-config\n"
        )
        config = make_config(tmp_path, max_iterations=1, log_dir=tmp_path / "logs")
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert "unknown flag: --mcp-config" in (state.last_captured_stderr or "")

    @patch(MOCK_SUBPROCESS, side_effect=timeout_proc)
    def test_timeout_counted(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=1, timeout=5)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert state.timed_out_count == 1
        assert state.failed == 1

    @patch("milknado.loop.engine.execute_agent")
    def test_force_stop_remains_stopped_with_stop_on_error(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(tmp_path, max_iterations=2, stop_on_error=True)
        state = make_state()
        mock_execute_agent.return_value = AgentResult(returncode=-9, force_stopped=True)

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 1
        assert state.status is RunStatus.STOPPED
        assert state.failed == 0

    @patch("milknado.loop.engine._delay_if_needed")
    @patch("milknado.loop.engine.execute_agent")
    def test_retriable_timeout_keeps_guidance_open(
        self, mock_execute_agent: MagicMock, mock_delay: MagicMock, tmp_path: Path
    ):
        config = make_config(tmp_path, max_iterations=2)
        state = make_state()
        mock_execute_agent.side_effect = [
            AgentResult(returncode=None, timed_out=True),
            AgentResult(returncode=0),
        ]

        def admit_guidance(_config: RunConfig, observed_state: RunState, _emit: BoundEmitter):
            assert observed_state.timed_out_count == 1
            assert observed_state.queue_guidance("retry with the failing test") is True

        mock_delay.side_effect = admit_guidance

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 2
        assert state.status is RunStatus.COMPLETED

    @patch(MOCK_SUBPROCESS)
    def test_prompt_read_from_ralph_file(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc()
        config = make_config(tmp_path, "my prompt text", max_iterations=1)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert mock_run.call_args.args[0][-1] == "my prompt text"

    @patch(MOCK_SUBPROCESS)
    def test_log_dir_creates_files(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc(stdout_text="output\n")
        log_dir = tmp_path / "logs"
        config = make_config(tmp_path, max_iterations=2, log_dir=log_dir)
        state = make_state()

        run_loop(config, state, NullEmitter())

        log_files = sorted(log_dir.iterdir())
        assert len(log_files) == 2
        assert log_files[0].name.startswith("001_")
        assert log_files[1].name.startswith("002_")


class TestPromiseCompletionSignals:
    @patch("milknado.loop.engine.execute_agent")
    def test_tagged_promise_does_not_stop_by_default(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """Without ``stop_on_completion_signal`` the loop must run all iterations.

        A non-streaming adapter (here ``echo`` → GenericAdapter) only sees
        promise completion when the engine elects to capture stdout, which
        in turn requires either logging or the explicit opt-in.  Skipping
        the buffer is the whole point of the gating, so
        ``state.promise_completed`` legitimately stays False here.
        """
        config = make_config(tmp_path, max_iterations=3)
        state = make_state()
        emitter = NullEmitter()
        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout=None,
        )

        run_loop(config, state, emitter)

        assert mock_execute_agent.call_count == 3
        assert state.completed == 3
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is False
        assert [call.args[0].on_output_line for call in mock_execute_agent.call_args_list] == [  # pyright: ignore[reportAny]
            None,
            None,
            None,
        ]
        assert [
            call.args[0].capture_result_text  # pyright: ignore[reportAny]
            for call in mock_execute_agent.call_args_list
        ] == [True, True, True]
        # Generic adapter requires full stdout, but the user didn't opt in
        # and no log dir is set — capture should stay off to avoid
        # buffering verbose agent output.
        assert [call.args[0].capture_stdout for call in mock_execute_agent.call_args_list] == [  # pyright: ignore[reportAny]
            False,
            False,
            False,
        ]

    @patch("milknado.loop.engine.execute_agent")
    def test_capture_stdout_off_for_streaming_adapter_without_logging(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """Claude exposes ``result_text`` directly; the engine must not
        force-buffer the full stdout transcript even when the user opts
        into ``stop_on_completion_signal``."""
        config = make_config(
            tmp_path,
            agent="claude",
            max_iterations=1,
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            result_text="<promise>RALPH_PROMISE_COMPLETE</promise>",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_args.args[0].capture_stdout is False  # pyright: ignore[reportAny]
        assert state.promise_completed is True

    @patch("milknado.loop.engine.execute_agent")
    def test_capture_stdout_on_for_blocking_adapter_with_opt_in(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """Generic / Copilot adapters need the full stdout buffer to
        scan for the promise tag — engine must opt in when the user
        opts into completion signalling."""
        config = make_config(
            tmp_path,
            max_iterations=1,
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="<promise>RALPH_PROMISE_COMPLETE</promise>\n",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_args.args[0].capture_stdout is True  # pyright: ignore[reportAny]
        assert state.promise_completed is True

    @patch("milknado.loop.engine.execute_agent")
    def test_capture_stdout_on_when_log_dir_set(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """Logging always needs the buffer regardless of completion signal."""
        log_dir = tmp_path / "logs"
        config = make_config(tmp_path, max_iterations=1, log_dir=log_dir)
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="anything\n",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_args.args[0].capture_stdout is True  # pyright: ignore[reportAny]

    @patch("milknado.loop.engine.execute_agent")
    def test_tagged_promise_stops_early_when_enabled(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=5,
            stop_on_completion_signal=True,
        )
        state = make_state()
        emitter = QueueEmitter()
        emitter.wants_agent_output_lines = lambda: True
        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="<promise>RALPH_PROMISE_COMPLETE</promise>\n",
        )

        run_loop(config, state, emitter)

        mock_execute_agent.assert_called_once()
        assert mock_execute_agent.call_args.args[0].capture_result_text is True  # pyright: ignore[reportAny]
        assert state.iteration == 1
        assert state.completed == 1
        assert state.failed == 0
        assert state.total == 1
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is True

        events = drain_events(emitter)  # pyright: ignore[reportUnknownVariableType]
        completed_events = events_of_type(events, EventType.ITERATION_COMPLETED)  # pyright: ignore[reportUnknownVariableType]
        assert len(completed_events) == 1  # pyright: ignore[reportUnknownArgumentType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "completed"  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["total"] == 1  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["completed"] == 1  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["failed"] == 0  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["timed_out_count"] == 0  # pyright: ignore[reportUnknownMemberType]

    @patch("milknado.loop.engine.execute_agent")
    def test_custom_promise_text_matches_inner_tag_text(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=4,
            completion_signal="CUSTOM_DONE",
            stop_on_completion_signal=True,
        )
        state = make_state()
        emitter = QueueEmitter()
        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="<promise>CUSTOM_DONE</promise>\n",
        )

        run_loop(config, state, emitter)

        mock_execute_agent.assert_called_once()
        assert state.iteration == 1
        assert state.completed == 1
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is True

        events = drain_events(emitter)  # pyright: ignore[reportUnknownVariableType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "completed"  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["total"] == 1  # pyright: ignore[reportUnknownMemberType]
        assert stop_event.data["completed"] == 1  # pyright: ignore[reportUnknownMemberType]

    @patch("milknado.loop.engine.execute_agent")
    def test_promise_tag_normalizes_inner_whitespace_before_matching(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=4,
            completion_signal="CUSTOM DONE",
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="<promise>\n  CUSTOM\tDONE  \n</promise>\n",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 1
        assert state.iteration == 1
        assert state.completed == 1
        assert state.failed == 0
        assert state.total == 1
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is True

    @patch("milknado.loop.engine.execute_agent")
    def test_untagged_raw_text_does_not_match_completion_signal(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=3,
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            result_text="done RALPH_PROMISE_COMPLETE without promise tags",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 3
        assert state.completed == 3
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is False

    @patch("milknado.loop.engine.execute_agent")
    def test_different_tagged_promise_text_does_not_match(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(
            tmp_path,
            max_iterations=3,
            completion_signal="CUSTOM_DONE",
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            result_text="<promise>CUSTOM_DONE_NOW</promise>",
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 3
        assert state.completed == 3
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is False

    @patch("milknado.loop.engine.execute_agent")
    def test_structured_agents_ignore_raw_stdout_for_promise_detection(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """ClaudeAdapter only looks at ``result`` events — embedded
        promise tags inside ``status`` or ``assistant`` JSON messages
        must not trigger early completion."""
        config = make_config(
            tmp_path,
            agent="claude",
            max_iterations=2,
            stop_on_completion_signal=True,
        )
        state = make_state()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            result_text="done without promise tag",
            captured_stdout='{"type":"status","message":"<promise>RALPH_PROMISE_COMPLETE</promise>"}\n',
        )

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 2
        assert state.completed == 2
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED
        assert state.promise_completed is False

    @patch("milknado.loop.engine.execute_agent")
    def test_blocking_captured_stdout_is_echoed_when_peek_is_off(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        emitter = QueueEmitter()

        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="plain blocking output\n",
        )

        run_loop(config, state, emitter)

        completed_event = events_of_type(drain_events(emitter), EventType.ITERATION_COMPLETED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert completed_event.data["echo_stdout"] == "plain blocking output\n"  # pyright: ignore[reportUnknownMemberType]


class TestCompletionVerifier:
    """Tier-2 completion verifier gating the exit-0 + promise stop path.

    Three branches: absent verifier (back-compat accept), verifier passes
    (accept), verifier rejects (inject feedback + keep looping, fail loud at
    the iteration budget).
    """

    @staticmethod
    def _promise_result() -> AgentResult:
        return AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="<promise>RALPH_PROMISE_COMPLETE</promise>\n",
        )

    @patch("milknado.loop.engine.execute_agent")
    def test_absent_verifier_accepts_promise(self, mock_execute_agent: MagicMock, tmp_path: Path):
        """``completion_verifier=None`` keeps the legacy promise-only accept."""
        config = make_config(tmp_path, max_iterations=5, stop_on_completion_signal=True)
        assert config.completion_verifier is None
        state = make_state()
        mock_execute_agent.return_value = self._promise_result()

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 1
        assert state.iteration == 1
        assert state.status == RunStatus.COMPLETED

    @patch("milknado.loop.engine.execute_agent")
    def test_verifier_pass_completes_run(self, mock_execute_agent: MagicMock, tmp_path: Path):
        """``ok=True`` accepts the promise and stops the run."""
        calls: list[int] = []

        def verifier() -> CompletionVerdict:
            calls.append(1)
            return CompletionVerdict(ok=True, feedback="")

        config = make_config(
            tmp_path,
            max_iterations=5,
            stop_on_completion_signal=True,
            completion_verifier=verifier,
        )
        state = make_state()
        mock_execute_agent.return_value = self._promise_result()

        run_loop(config, state, NullEmitter())

        assert len(calls) == 1
        assert mock_execute_agent.call_count == 1
        assert state.iteration == 1
        assert state.status == RunStatus.COMPLETED

    @patch("milknado.loop.engine.execute_agent")
    def test_verifier_reject_then_retry_injects_feedback(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """``ok=False`` keeps looping; the feedback lands in the next prompt.

        First promise is rejected with a concrete message; the run must not
        stop, the rejection feedback must appear verbatim in the *second*
        iteration's assembled prompt, and a subsequent ``ok=True`` then
        completes the run.
        """
        feedback = "gate failed: just check-llm reported 2 lint errors"
        verdicts = iter(
            [
                CompletionVerdict(ok=False, feedback=feedback),
                CompletionVerdict(ok=True, feedback=""),
            ]
        )

        config = make_config(
            tmp_path,
            "do the work",
            max_iterations=5,
            stop_on_completion_signal=True,
            completion_verifier=lambda: next(verdicts),
        )
        state = make_state()
        mock_execute_agent.return_value = self._promise_result()

        run_loop(config, state, NullEmitter())

        assert mock_execute_agent.call_count == 2
        first_prompt = mock_execute_agent.call_args_list[0].args[0].prompt  # pyright: ignore[reportAny]
        second_prompt = mock_execute_agent.call_args_list[1].args[0].prompt  # pyright: ignore[reportAny]
        assert feedback not in first_prompt
        assert feedback in second_prompt
        assert state.iteration == 2
        assert state.status == RunStatus.COMPLETED

    @patch("milknado.loop.engine.execute_agent")
    def test_feedback_cleared_after_one_retry(self, mock_execute_agent: MagicMock, tmp_path: Path):
        """A single rejection must not leak feedback into every later prompt.

        Reject once, then accept on iteration 2: iteration 2 carries the
        feedback, but had a third iteration run it would not — proven here by
        asserting the run stopped at 2 and only the second prompt carried it.
        """
        feedback = "fix the empty diff"
        verdicts = iter(
            [
                CompletionVerdict(ok=False, feedback=feedback),
                CompletionVerdict(ok=True, feedback=""),
            ]
        )
        config = make_config(
            tmp_path,
            "do the work",
            max_iterations=5,
            stop_on_completion_signal=True,
            completion_verifier=lambda: next(verdicts),
        )
        state = make_state()
        mock_execute_agent.return_value = self._promise_result()

        run_loop(config, state, NullEmitter())

        prompts = [c.args[0].prompt for c in mock_execute_agent.call_args_list]  # pyright: ignore[reportAny]
        assert [feedback in p for p in prompts] == [False, True]  # pyright: ignore[reportAny]

    @patch("milknado.loop.engine.execute_agent")
    def test_verifier_reject_exhausts_budget_fails_loud(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """Persistent rejection to the iteration budget fails the run loudly.

        Every promise is rejected, so the run never accepts.  At the
        ``max_iterations`` budget the run must end FAILED (not COMPLETED) and
        emit an error message — not silently report success.
        """
        config = make_config(
            tmp_path,
            "do the work",
            max_iterations=3,
            stop_on_completion_signal=True,
            completion_verifier=lambda: CompletionVerdict(ok=False, feedback="still failing"),
        )
        state = make_state()
        emitter = QueueEmitter()
        mock_execute_agent.return_value = self._promise_result()

        run_loop(config, state, emitter)

        assert mock_execute_agent.call_count == 3
        assert state.iteration == 3
        assert state.status == RunStatus.FAILED

        events = drain_events(emitter)  # pyright: ignore[reportUnknownVariableType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "error"  # pyright: ignore[reportUnknownMemberType]
        error_logs = [  # pyright: ignore[reportUnknownVariableType]
            e
            for e in events_of_type(events, EventType.LOG_MESSAGE)  # pyright: ignore[reportUnknownVariableType]
            if "verifier never accepted" in e.data["message"]  # pyright: ignore[reportUnknownMemberType]
        ]
        assert len(error_logs) == 1  # pyright: ignore[reportUnknownArgumentType]

    @patch("milknado.loop.engine.execute_agent")
    def test_verifier_not_consulted_without_promise(
        self, mock_execute_agent: MagicMock, tmp_path: Path
    ):
        """No promise → no completion to gate → verifier is never called."""
        calls: list[int] = []
        config = make_config(
            tmp_path,
            max_iterations=2,
            stop_on_completion_signal=True,
            completion_verifier=lambda: calls.append(1) or CompletionVerdict(ok=True, feedback=""),
        )
        state = make_state()
        mock_execute_agent.return_value = AgentResult(
            returncode=0,
            elapsed=0.01,
            captured_stdout="no promise here\n",
        )

        run_loop(config, state, NullEmitter())

        assert calls == []
        assert mock_execute_agent.call_count == 2
        assert state.status == RunStatus.COMPLETED

    def test_assemble_prompt_appends_verifier_feedback(self, tmp_path: Path):
        """``_assemble_prompt`` appends feedback verbatim when supplied."""
        config = make_config(tmp_path, "base prompt", max_iterations=1)
        state = make_state()
        state.iteration = 2

        result = _assemble_prompt(config, state, {}, "address: empty diff")

        assert result.startswith("base prompt")
        assert "address: empty diff" in result

    def test_assemble_prompt_omits_feedback_when_none(self, tmp_path: Path):
        """No feedback → prompt is unchanged from the no-feedback path."""
        config = make_config(tmp_path, "base prompt", max_iterations=1)
        state = make_state()
        state.iteration = 1

        assert _assemble_prompt(config, state, {}, None) == "base prompt"


class TestRunLoopDefaults:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_runs_without_emitter(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        """run_loop works when called without an explicit emitter (uses NullEmitter)."""
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()

        run_loop(config, state)

        assert state.completed == 1
        assert state.status == RunStatus.COMPLETED


class TestRunLoopEvents:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_events_emitted_in_order(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        types = event_types(events)
        assert types[0] == EventType.RUN_STARTED
        assert types[-1] == EventType.RUN_STOPPED
        # Verify lifecycle ordering within the iteration
        assert types.index(EventType.ITERATION_STARTED) < types.index(EventType.PROMPT_ASSEMBLED)
        assert types.index(EventType.PROMPT_ASSEMBLED) < types.index(EventType.ITERATION_COMPLETED)
        assert types.index(EventType.ITERATION_COMPLETED) < types.index(EventType.RUN_STOPPED)

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_failure_event_emitted(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert EventType.ITERATION_FAILED in event_types(events)

    @patch(MOCK_SUBPROCESS, side_effect=timeout_proc)
    def test_timeout_event_emitted(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=1, timeout=5)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert EventType.ITERATION_TIMED_OUT in event_types(events)

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_all_events_have_run_id(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        for event in drain_events(q):  # pyright: ignore[reportUnknownVariableType]
            assert event.run_id == "test-run-001"


class TestRunStateControls:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_stop_request(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=100)
        state = make_state()

        def stop_after_first(*_args: object, **_kwargs: object):
            state.request_stop()
            return ok_proc()

        mock_run.side_effect = stop_after_first

        run_loop(config, state, NullEmitter())

        assert mock_run.call_count == 1
        assert state.status == RunStatus.STOPPED

    @patch(MOCK_SUBPROCESS)
    def test_keyboard_interrupt_sets_stopped(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.side_effect = KeyboardInterrupt
        config = make_config(tmp_path, max_iterations=5)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.status == RunStatus.STOPPED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "user_requested"  # pyright: ignore[reportUnknownMemberType]

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_pause_and_resume(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=3)
        state = make_state()

        call_count = 0

        def track_calls(*_args: object, **_kwargs: object):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                state.request_pause()

                def resume_later():
                    time.sleep(0.1)
                    state.request_resume()

                threading.Thread(target=resume_later, daemon=True).start()
            return ok_proc()

        mock_run.side_effect = track_calls

        run_loop(config, state, NullEmitter())

        assert state.completed == 3
        assert state.status == RunStatus.COMPLETED

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_stop_while_paused(self, mock_run: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=100)
        state = make_state()

        def pause_then_stop(*_args: object, **_kwargs: object):
            state.request_pause()

            def stop_later():
                time.sleep(0.1)
                state.request_stop()

            threading.Thread(target=stop_later, daemon=True).start()
            return ok_proc()

        mock_run.side_effect = pause_then_stop

        run_loop(config, state, NullEmitter())

        assert state.status == RunStatus.STOPPED
        assert mock_run.call_count == 1


class TestRalphArgs:
    @patch(MOCK_SUBPROCESS)
    def test_args_resolved_in_prompt(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc()
        config = make_config(
            tmp_path,
            "---\nargs:\n  - dir\n  - focus\n---\nResearch {{ args.dir }} focus: {{ args.focus }}",
            max_iterations=1,
            args={"dir": "./src", "focus": "perf"},
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        prompt_arg = mock_run.call_args.args[0][-1]  # pyright: ignore[reportAny]
        assert prompt_arg == "Research ./src focus: perf"

    @patch(MOCK_SUBPROCESS)
    def test_empty_args_clears_placeholders(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc()
        config = make_config(
            tmp_path,
            "Before {{ args.opt }} after",
            max_iterations=1,
            args={},
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        prompt_arg = mock_run.call_args.args[0][-1]  # pyright: ignore[reportAny]
        assert prompt_arg == "Before  after"


class TestCommandExecution:
    @patch(MOCK_SUBPROCESS)
    @patch(MOCK_RUN_COMMAND)
    def test_commands_output_injected(
        self, mock_run_cmd: MagicMock, mock_agent: MagicMock, tmp_path: Path
    ):
        mock_run_cmd.return_value = ok_run_result(output="test output\n")
        mock_agent.return_value = ok_proc()

        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: tests\n    run: uv run pytest\n---\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "Results:\n\n{{ commands.tests }}",
            max_iterations=1,
            commands=[Command(name="tests", run="uv run pytest")],
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        prompt_arg = mock_agent.call_args.args[0][-1]  # pyright: ignore[reportAny]
        assert "test output" in prompt_arg
        assert "{{ commands.tests }}" not in prompt_arg

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    @patch(MOCK_RUN_COMMAND)
    def test_multiple_commands_all_executed(
        self,
        mock_run_cmd: MagicMock,
        mock_agent: MagicMock,  # pyright: ignore[reportUnusedParameter]
        tmp_path: Path,
    ):
        """All commands in the list are executed and their outputs collected."""

        call_count = 0

        def per_command(**kwargs: object):
            nonlocal call_count
            call_count += 1
            name = kwargs.get("command", "")
            return ok_run_result(output=f"output-{name}")

        mock_run_cmd.side_effect = per_command

        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: tests\n    run: pytest\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "  - name: lint\n    run: ruff check\n---\n"
            "{{ commands.tests }}\n{{ commands.lint }}",
            max_iterations=1,
            commands=[
                Command(name="tests", run="pytest"),
                Command(name="lint", run="ruff check"),
            ],
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        assert mock_run_cmd.call_count == 2

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    @patch(MOCK_RUN_COMMAND)
    def test_dotslash_command_uses_ralph_dir_as_cwd(
        self,
        mock_run_cmd: MagicMock,
        mock_agent: MagicMock,  # pyright: ignore[reportUnusedParameter]
        tmp_path: Path,
    ):
        """Commands starting with ./ run relative to the ralph directory."""
        mock_run_cmd.return_value = ok_run_result(output="ok")

        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: local\n    run: ./check.sh\n---\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "{{ commands.local }}",
            max_iterations=1,
            commands=[Command(name="local", run="./check.sh")],
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        passed_cwd = mock_run_cmd.call_args.kwargs["cwd"]  # pyright: ignore[reportAny]
        assert passed_cwd == config.ralph_dir

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    @patch(MOCK_RUN_COMMAND)
    def test_regular_command_uses_project_root_as_cwd(
        self,
        mock_run_cmd: MagicMock,
        mock_agent: MagicMock,  # pyright: ignore[reportUnusedParameter]
        tmp_path: Path,
    ):
        """Commands without ./ prefix run from the project root."""
        mock_run_cmd.return_value = ok_run_result(output="ok")

        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: tests\n    run: uv run pytest\n---\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "{{ commands.tests }}",
            max_iterations=1,
            commands=[Command(name="tests", run="uv run pytest")],
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        passed_cwd = mock_run_cmd.call_args.kwargs["cwd"]  # pyright: ignore[reportAny]
        assert passed_cwd == config.project_root

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_agent_spawned_in_project_root(self, mock_run: MagicMock, tmp_path: Path):
        """The agent subprocess runs in the run's project root, not the orchestrator cwd."""
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        run_loop(config, state, NullEmitter())

        assert mock_run.call_args.kwargs["cwd"] == config.project_root

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    @patch(MOCK_RUN_COMMAND)
    def test_command_timeout_passed_through(
        self,
        mock_run_cmd: MagicMock,
        mock_agent: MagicMock,  # pyright: ignore[reportUnusedParameter]
        tmp_path: Path,
    ):
        """Command timeout from frontmatter is forwarded to run_command."""
        mock_run_cmd.return_value = ok_run_result(output="ok")

        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "  - name: slow\n    run: sleep 1\n    timeout: 300\n---\n"
            "{{ commands.slow }}",
            max_iterations=1,
            commands=[Command(name="slow", run="sleep 1", timeout=300)],
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        passed_timeout = mock_run_cmd.call_args.kwargs["timeout"]  # pyright: ignore[reportAny]
        assert passed_timeout == 300


class TestAgentCommandParsing:
    """Tests for malformed agent command handling in the engine."""

    def test_malformed_agent_command_raises_value_error(self, tmp_path: Path):
        """shlex.split on an agent with unmatched quotes produces a clear error."""
        config = make_config(tmp_path, max_iterations=1, agent="omp 'unterminated")
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.status == RunStatus.FAILED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        log_events = events_of_type(events, EventType.LOG_MESSAGE)  # pyright: ignore[reportUnknownVariableType]
        assert any("Invalid agent command syntax" in e.data["message"] for e in log_events)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @patch(MOCK_SUBPROCESS)
    def test_agent_path_spoof_fails_before_popen(self, mock_popen: MagicMock, tmp_path: Path):
        config = make_config(tmp_path, max_iterations=1, agent="/tmp/claude")
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.status == RunStatus.FAILED
        mock_popen.assert_not_called()

    @patch(MOCK_SUBPROCESS)
    def test_agent_not_found_raises_file_not_found_error(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        """FileNotFoundError from the agent subprocess is re-raised with a helpful message."""
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'nonexistent'")
        config = make_config(tmp_path, max_iterations=1, agent="omp")
        q = QueueEmitter()
        state = make_state()

        run_loop(config, state, q)

        assert state.status == RunStatus.FAILED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        log_events = events_of_type(events, EventType.LOG_MESSAGE)  # pyright: ignore[reportUnknownVariableType]
        assert any("Agent command not found" in e.data["message"] for e in log_events)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


class TestRunLoopCrashHandling:
    """Tests for the broad exception handler in run_loop (engine.py lines 269-276)."""

    @patch(MOCK_SUBPROCESS)
    def test_unexpected_exception_sets_failed_status(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.side_effect = RuntimeError("disk full")
        config = make_config(tmp_path, max_iterations=5)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert state.status == RunStatus.FAILED

    @patch(MOCK_SUBPROCESS)
    def test_unexpected_exception_emits_log_and_stop_events(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        mock_run.side_effect = RuntimeError("disk full")
        config = make_config(tmp_path, max_iterations=5)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        types = event_types(events)
        assert EventType.LOG_MESSAGE in types
        assert EventType.RUN_STOPPED in types

        log_event = events_of_type(events, EventType.LOG_MESSAGE)[0]  # pyright: ignore[reportUnknownVariableType]
        assert log_event.data["level"] == "error"  # pyright: ignore[reportUnknownMemberType]
        assert "disk full" in log_event.data["message"]  # pyright: ignore[reportUnknownMemberType]
        assert "traceback" in log_event.data  # pyright: ignore[reportUnknownMemberType]

        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "error"  # pyright: ignore[reportUnknownMemberType]

    @patch(MOCK_SUBPROCESS)
    def test_unexpected_exception_only_runs_one_iteration(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        mock_run.side_effect = RuntimeError("boom")
        config = make_config(tmp_path, max_iterations=10)
        state = make_state()

        run_loop(config, state, NullEmitter())

        assert mock_run.call_count == 1
        assert state.iteration == 1

    @patch("milknado.loop.engine.parse_frontmatter")
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_crash_in_prompt_assembly_handled(
        self,
        mock_run: MagicMock,  # pyright: ignore[reportUnusedParameter]
        mock_parse: MagicMock,
        tmp_path: Path,
    ):
        mock_parse.side_effect = ValueError("corrupt YAML")
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()
        q = QueueEmitter()

        run_loop(config, state, q)

        assert state.status == RunStatus.FAILED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        log_events = events_of_type(events, EventType.LOG_MESSAGE)  # pyright: ignore[reportUnknownVariableType]
        assert any("corrupt YAML" in e.data["message"] for e in log_events)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


class TestDelayIfNeeded:
    def test_no_delay_when_zero(self, tmp_path: Path):
        config = make_config(tmp_path, delay=0, max_iterations=5)
        state = make_state()
        state.iteration = 1
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        start = time.monotonic()
        _delay_if_needed(config, state, emit)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1
        assert drain_events(q) == []

    def test_delay_sleeps_between_iterations(self, tmp_path: Path):
        config = make_config(tmp_path, delay=0.15, max_iterations=5)
        state = make_state()
        state.iteration = 1
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        start = time.monotonic()
        _delay_if_needed(config, state, emit)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.1
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].type == EventType.LOG_MESSAGE
        assert "Waiting" in events[0].data["message"]  # pyright: ignore[reportUnknownMemberType]

    def test_delay_message_uses_format_duration(self, tmp_path: Path):
        """Delay log message should use format_duration for consistency with
        the rest of the UI — e.g. '2m 0s' instead of raw '120s'."""
        config = make_config(tmp_path, delay=120, max_iterations=5)
        state = make_state()
        state.iteration = 1
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        # Request stop immediately so we don't actually wait 120s
        state.request_stop()
        _delay_if_needed(config, state, emit)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        msg = events[0].data["message"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert "2m 0s" in msg, f"Expected formatted duration '2m 0s' in message, got: {msg!r}"

    def test_no_delay_on_last_iteration(self, tmp_path: Path):
        config = make_config(tmp_path, delay=0.5, max_iterations=3)
        state = make_state()
        state.iteration = 3  # last iteration
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        start = time.monotonic()
        _delay_if_needed(config, state, emit)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1
        assert drain_events(q) == []

    def test_delay_with_unlimited_iterations(self, tmp_path: Path):
        config = make_config(tmp_path, delay=0.15, max_iterations=None)
        state = make_state()
        state.iteration = 100
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        start = time.monotonic()
        _delay_if_needed(config, state, emit)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.1

    def test_delay_exits_early_on_stop_request(self, tmp_path: Path):
        """Stop requests during delay should be respected, not blocked
        for the full delay duration."""
        config = make_config(tmp_path, delay=5.0, max_iterations=None)
        state = make_state()
        state.iteration = 1
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        # Request stop from another thread after a short interval
        def stop_soon():
            time.sleep(0.1)
            state.request_stop()

        threading.Thread(target=stop_soon, daemon=True).start()

        start = time.monotonic()
        _delay_if_needed(config, state, emit)
        elapsed = time.monotonic() - start

        # Should exit well before the full 5s delay
        assert elapsed < 2.0


class TestHandleControlSignals:
    """Unit tests for _handle_control_signals — stop and pause logic."""

    def test_returns_true_when_no_signals(self):
        state = make_state()
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        result = _handle_control_signals(state, emit)

        assert result is True
        assert state.status == RunStatus.PENDING

    def test_returns_false_when_stop_requested(self):
        state = make_state()
        state.request_stop()
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        result = _handle_control_signals(state, emit)

        assert result is False
        assert state.status == RunStatus.STOPPED

    def test_paused_then_resumed_returns_true(self):
        state = make_state()
        state.request_pause()
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        def resume_soon():
            time.sleep(0.05)
            state.request_resume()

        threading.Thread(target=resume_soon, daemon=True).start()

        result = _handle_control_signals(state, emit)

        assert result is True
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        types = event_types(events)
        assert EventType.RUN_PAUSED in types
        assert EventType.RUN_RESUMED in types

    def test_paused_then_stop_returns_false(self):
        state = make_state()
        state.request_pause()
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        def stop_soon():
            time.sleep(0.05)
            state.request_stop()

        threading.Thread(target=stop_soon, daemon=True).start()

        result = _handle_control_signals(state, emit)

        assert result is False
        assert state.status == RunStatus.STOPPED
        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        types = event_types(events)
        assert EventType.RUN_PAUSED in types
        assert EventType.RUN_RESUMED not in types

    def test_stop_detected_during_pause_poll_interval(self):
        """Stop flag set without resume event exercises the polling-loop guard.

        Normally request_stop() sets both the flag and the resume event, so
        wait_for_unpause unblocks immediately.  This test sets only the flag
        to cover the guard inside the polling loop (engine.py _wait_for_resume)
        that checks stop_requested after each poll timeout.
        """
        state = make_state()
        state.request_pause()
        q = QueueEmitter()
        emit = BoundEmitter(q, state.run_id)

        def set_stop_flag():
            time.sleep(0.01)
            state._stop_event.set()  # pyright: ignore[reportPrivateUsage]

        threading.Thread(target=set_stop_flag, daemon=True).start()

        result = _handle_control_signals(state, emit)

        assert result is False
        assert state.status == RunStatus.STOPPED


class TestRunCommands:
    """Unit tests for _run_commands — command execution and cwd resolution."""

    @patch(MOCK_RUN_COMMAND)
    def test_returns_name_to_output_mapping(self, mock_run_cmd: MagicMock, tmp_path: Path):
        mock_run_cmd.return_value = ok_run_result(output="test output")
        commands = [Command(name="tests", run="pytest")]

        result = _run_commands(
            commands, ralph_dir=tmp_path / "ralph", project_root=tmp_path, user_args={}
        )

        assert result == {"tests": "test output"}

    @patch(MOCK_RUN_COMMAND)
    def test_multiple_commands(self, mock_run_cmd: MagicMock, tmp_path: Path):
        call_count = 0

        def per_command(**kwargs: object):  # pyright: ignore[reportUnusedParameter]
            nonlocal call_count
            call_count += 1
            return ok_run_result(output=f"out-{call_count}")

        mock_run_cmd.side_effect = per_command
        commands = [
            Command(name="a", run="cmd-a"),
            Command(name="b", run="cmd-b"),
        ]

        result = _run_commands(
            commands, ralph_dir=tmp_path / "ralph", project_root=tmp_path, user_args={}
        )

        assert len(result) == 2
        assert result["a"] == "out-1"
        assert result["b"] == "out-2"

    @patch(MOCK_RUN_COMMAND)
    def test_dotslash_uses_ralph_dir(self, mock_run_cmd: MagicMock, tmp_path: Path):
        mock_run_cmd.return_value = ok_run_result(output="ok")
        ralph_dir = tmp_path / "my-ralph"
        commands = [Command(name="local", run="./check.sh")]

        _ = _run_commands(commands, ralph_dir=ralph_dir, project_root=tmp_path, user_args={})

        assert mock_run_cmd.call_args.kwargs["cwd"] == ralph_dir

    @patch(MOCK_RUN_COMMAND)
    def test_regular_command_uses_project_root(self, mock_run_cmd: MagicMock, tmp_path: Path):
        mock_run_cmd.return_value = ok_run_result(output="ok")
        ralph_dir = tmp_path / "my-ralph"
        commands = [Command(name="tests", run="pytest")]

        _ = _run_commands(commands, ralph_dir=ralph_dir, project_root=tmp_path, user_args={})

        assert mock_run_cmd.call_args.kwargs["cwd"] == tmp_path

    @patch(MOCK_RUN_COMMAND)
    def test_timeout_passed_to_run_command(self, mock_run_cmd: MagicMock, tmp_path: Path):
        mock_run_cmd.return_value = ok_run_result(output="ok")
        commands = [Command(name="slow", run="sleep 1", timeout=300)]

        _ = _run_commands(commands, ralph_dir=tmp_path, project_root=tmp_path, user_args={})

        assert mock_run_cmd.call_args.kwargs["timeout"] == 300

    def test_empty_commands_returns_empty_dict(self, tmp_path: Path):
        result = _run_commands([], ralph_dir=tmp_path, project_root=tmp_path, user_args={})

        assert result == {}

    @patch(MOCK_RUN_COMMAND)
    def test_resolves_args_in_command_run_string(self, mock_run_cmd: MagicMock, tmp_path: Path):
        mock_run_cmd.return_value = ok_run_result(output="issue content")
        commands = [Command(name="issue", run="gh issue view {{ args.issue }} --json title")]

        _ = _run_commands(
            commands,
            ralph_dir=tmp_path,
            project_root=tmp_path,
            user_args={"issue": "42"},
        )

        assert mock_run_cmd.call_args.kwargs["command"] == "gh issue view 42 --json title"

    @patch(MOCK_RUN_COMMAND)
    def test_dotslash_detection_after_args_resolution(
        self, mock_run_cmd: MagicMock, tmp_path: Path
    ):
        mock_run_cmd.return_value = ok_run_result(output="ok")
        ralph_dir = tmp_path / "my-ralph"
        commands = [Command(name="check", run="./{{ args.script }}")]

        _ = _run_commands(
            commands,
            ralph_dir=ralph_dir,
            project_root=tmp_path,
            user_args={"script": "check.sh"},
        )

        assert mock_run_cmd.call_args.kwargs["cwd"] == ralph_dir
        assert mock_run_cmd.call_args.kwargs["command"] == "./check.sh"

    @patch(MOCK_RUN_COMMAND)
    def test_arg_values_with_spaces_are_shell_quoted_in_commands(
        self, mock_run_cmd: MagicMock, tmp_path: Path
    ):
        """Arg values containing spaces must be shell-quoted when substituted
        into command run strings so shlex.split treats them as single tokens."""
        mock_run_cmd.return_value = ok_run_result(output="found it")
        commands = [Command(name="search", run="grep {{ args.pattern }} src/")]

        _ = _run_commands(
            commands,
            ralph_dir=tmp_path,
            project_root=tmp_path,
            user_args={"pattern": "hello world"},
        )

        resolved_cmd = mock_run_cmd.call_args.kwargs["command"]  # pyright: ignore[reportAny]
        # The value must be quoted so shlex.split produces a single token
        import shlex

        tokens = shlex.split(resolved_cmd)  # pyright: ignore[reportAny]
        assert tokens == ["grep", "hello world", "src/"]

    @patch(MOCK_RUN_COMMAND)
    def test_dotslash_detected_after_leading_whitespace_from_empty_arg(
        self, mock_run_cmd: MagicMock, tmp_path: Path
    ):
        """When an optional arg placeholder before ./ resolves to empty,
        the leading whitespace must not prevent ./  detection for cwd."""
        mock_run_cmd.return_value = ok_run_result(output="ok")
        ralph_dir = tmp_path / "my-ralph"
        commands = [Command(name="check", run="{{ args.flag }} ./check.sh")]

        _ = _run_commands(commands, ralph_dir=ralph_dir, project_root=tmp_path, user_args={})

        assert mock_run_cmd.call_args.kwargs["cwd"] == ralph_dir

    @patch(MOCK_RUN_COMMAND)
    def test_timed_out_command_output_includes_notice(
        self, mock_run_cmd: MagicMock, tmp_path: Path
    ):
        """When a command times out, the output injected into the prompt
        should include a notice so the agent knows the data is incomplete."""
        mock_run_cmd.return_value = RunResult(
            returncode=None,
            output="partial output",
            timed_out=True,
        )
        commands = [Command(name="slow", run="sleep 100", timeout=5)]

        result = _run_commands(commands, ralph_dir=tmp_path, project_root=tmp_path, user_args={})

        assert "partial output" in result["slow"]
        assert "timed out" in result["slow"].lower()

    @patch(MOCK_RUN_COMMAND)
    def test_timed_out_command_uses_formatted_duration(
        self, mock_run_cmd: MagicMock, tmp_path: Path
    ):
        """The timeout notice injected into command output should use
        format_duration() for consistency with the rest of the UI — e.g.
        '2m 0s' instead of raw '120s'."""
        mock_run_cmd.return_value = RunResult(
            returncode=None,
            output="partial",
            timed_out=True,
        )
        commands = [Command(name="slow", run="sleep 200", timeout=120)]

        result = _run_commands(commands, ralph_dir=tmp_path, project_root=tmp_path, user_args={})

        assert "2m 0s" in result["slow"]

    @patch(MOCK_RUN_COMMAND, side_effect=FileNotFoundError("no-such-binary"))
    def test_command_not_found_raises_with_context(self, mock_run_cmd: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        commands = [Command(name="missing", run="no-such-binary --flag")]

        with pytest.raises(FileNotFoundError, match="Command 'missing' binary not found"):
            _ = _run_commands(commands, ralph_dir=tmp_path, project_root=tmp_path, user_args={})

    @patch(MOCK_RUN_COMMAND, side_effect=ValueError("No closing quotation"))
    def test_command_invalid_syntax_raises_with_context(
        self,
        mock_run_cmd: MagicMock,  # pyright: ignore[reportUnusedParameter]
        tmp_path: Path,
    ):
        """ValueError from run_command (e.g. malformed quotes) is re-raised
        with the command name so the user knows which command failed."""
        commands = [Command(name="broken", run="echo 'unterminated")]

        with pytest.raises(ValueError, match="Command 'broken' has invalid syntax"):
            _ = _run_commands(commands, ralph_dir=tmp_path, project_root=tmp_path, user_args={})


class TestAssemblePrompt:
    """Unit tests for _assemble_prompt — reading and resolving the prompt template."""

    def test_reads_prompt_from_ralph_file(self, tmp_path: Path):
        config = make_config(tmp_path, "simple prompt", max_iterations=1)
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "simple prompt"

    def test_resolves_command_placeholders(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: tests\n    run: pytest\n---\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "Results: {{ commands.tests }}",
            max_iterations=1,
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {"tests": "all passed"})

        assert result == "Results: all passed"

    def test_resolves_args_placeholders(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "---\nagent: echo\nargs:\n  - dir\n---\nSearch {{ args.dir }}",
            max_iterations=1,
            args={"dir": "./src"},
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "Search ./src"

    def test_clears_unresolved_placeholders(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "Before {{ args.missing }} after",
            max_iterations=1,
            args={},
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "Before  after"

    def test_strips_html_comments(self, tmp_path: Path):
        config = make_config(tmp_path, "Before <!-- hidden --> after", max_iterations=1)
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "Before  after"

    def test_commit_footer_appended_to_prompt(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "simple prompt",
            max_iterations=1,
            commit_footer="Co-authored-by: Team <team@example.com>",
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result.startswith("simple prompt")
        assert "Co-authored-by: Team <team@example.com>" in result

    def test_no_footer_when_unset(self, tmp_path: Path):
        config = make_config(tmp_path, "simple prompt", max_iterations=1)
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "simple prompt"

    def test_arg_values_not_resolved_as_command_placeholders(self, tmp_path: Path):
        """Arg values containing {{ commands.X }} must appear literally,
        not be re-processed as command placeholders."""
        config = make_config(
            tmp_path,
            "---\nagent: echo\ncommands:\n  - name: tests\n    run: pytest\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "args:\n  - filter\n---\n"
            "Filter: {{ args.filter }}\nTests: {{ commands.tests }}",
            max_iterations=1,
            args={"filter": "{{ commands.tests }}"},
            commands=[Command(name="tests", run="pytest")],
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {"tests": "5 passed"})

        assert "Filter: {{ commands.tests }}" in result
        assert "Tests: 5 passed" in result

    def test_resolves_ralph_placeholders(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "Name: {{ ralph.name }}, Iter: {{ ralph.iteration }}, Max: {{ ralph.max_iterations }}",
            max_iterations=5,
        )
        state = make_state()
        state.iteration = 3

        result = _assemble_prompt(config, state, {})

        assert result == "Name: my-ralph, Iter: 3, Max: 5"

    def test_ralph_max_iterations_empty_when_unlimited(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "Max: {{ ralph.max_iterations }}",
            max_iterations=None,
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "Max: "

    def test_ralph_name_is_ralph_dir_name(self, tmp_path: Path):
        config = make_config(
            tmp_path,
            "Name: {{ ralph.name }}",
            max_iterations=1,
        )
        state = make_state()
        state.iteration = 1

        result = _assemble_prompt(config, state, {})

        assert result == "Name: my-ralph"


class TestInMemoryPrompt:
    """Curd 3 — RunConfig(prompt=...) runs the body without a file read."""

    def test_assemble_uses_prompt_body_without_reading_file(self, tmp_path: Path):
        config = RunConfig(
            agent="omp",
            ralph_dir=tmp_path,
            prompt="Search {{ args.dir }} now",
            args={"dir": "./src"},
            max_iterations=1,
        )
        state = make_state()
        state.iteration = 1

        with patch("pathlib.Path.read_text") as mock_read:
            result = _assemble_prompt(config, state, {})

        mock_read.assert_not_called()
        assert result == "Search ./src now"

    def test_prompt_body_is_not_frontmatter_parsed(self, tmp_path: Path):
        # A leading '---' block stays verbatim — it is the body, not frontmatter.
        body = "---\nnot: parsed\n---\nreal prompt"
        config = RunConfig(
            agent="omp",
            ralph_dir=tmp_path,
            prompt=body,
            max_iterations=1,
        )
        state = make_state()
        state.iteration = 1

        assert _assemble_prompt(config, state, {}) == body

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_run_loop_with_in_memory_prompt(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        config = RunConfig(
            agent="omp",
            ralph_dir=tmp_path,
            prompt="do work",
            max_iterations=1,
        )
        state = make_state()
        q = QueueEmitter()

        with patch("pathlib.Path.read_text") as mock_read:
            run_loop(config, state, q)

        mock_read.assert_not_called()
        assert state.status == RunStatus.COMPLETED
        assert state.completed == 1


class TestCommitFooterInLoop:
    @patch(MOCK_SUBPROCESS)
    def test_commit_footer_instruction_in_agent_input(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc()
        config = make_config(
            tmp_path,
            "do work",
            max_iterations=1,
            commit_footer="Co-authored-by: Team <team@example.com>",
        )
        state = make_state()
        run_loop(config, state, NullEmitter())

        prompt_arg = mock_run.call_args.args[0][-1]  # pyright: ignore[reportAny]
        assert "Co-authored-by: Team <team@example.com>" in prompt_arg

    @patch(MOCK_SUBPROCESS)
    def test_no_footer_unset_no_trailer_in_agent_input(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = ok_proc()
        config = make_config(tmp_path, "do work", max_iterations=1)
        state = make_state()
        run_loop(config, state, NullEmitter())

        prompt_arg = mock_run.call_args.args[0][-1]  # pyright: ignore[reportAny]
        assert "Co-authored-by" not in prompt_arg


class TestAgentOutputLineFiltering:
    """Tests for AGENT_OUTPUT_LINE event filtering (medium-01)."""

    @patch(MOCK_SUBPROCESS)
    def test_agent_output_line_not_emitted_when_peek_off(
        self, mock_run: MagicMock, tmp_path: Path
    ):
        """When peek is off, no AGENT_OUTPUT_LINE events are emitted."""
        mock_run.return_value = ok_proc(stdout_text="line1\nline2\nline3\n")
        q = QueueEmitter()
        config = make_config(tmp_path, max_iterations=1)
        state = make_state()

        run_loop(config, state, q)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        output_events = events_of_type(events, EventType.AGENT_OUTPUT_LINE)  # pyright: ignore[reportUnknownVariableType]
        assert output_events == []


def _write_opencode_stub(tmp_path: Path) -> Path:
    """Write an executable ``opencode`` stub emitting opencode-shaped JSON.

    The stub ignores stdin entirely (arg-delivery), prints a tool_use and a
    step_finish event followed by a promise completion tag, then exits 0.
    Named ``opencode`` so ``select_adapter`` dispatches to OpenCodeAdapter.
    """
    script = tmp_path / "opencode"
    _ = script.write_text(
        "#!/usr/bin/env python3\n"  # pyright: ignore[reportImplicitStringConcatenation]
        "import sys\n"
        'print(\'{"type": "step_start", "part": {}}\', flush=True)\n'
        'print(\'{"type": "tool_use", "part": {"name": "Edit"}}\', flush=True)\n'
        'print(\'{"type": "step_finish", "part": {"tokens": 10}}\', flush=True)\n'
        "print('<promise>RALPH_PROMISE_COMPLETE</promise>', flush=True)\n"
    )
    script.chmod(0o755)
    return script


class TestOpenCodeEndToEnd:
    """End-to-end run_loop with a real arg-delivery opencode stub (FR-9)."""

    def test_opencode_stub_runs_and_completes_on_promise(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        # The autouse _disable_streaming fixture forces blocking mode on every
        # adapter; re-enable streaming on the registered opencode instance so
        # this test exercises the real JSON-parsing activity path.
        opencode_adapter = select_adapter(["opencode", "run"])
        monkeypatch.setattr(opencode_adapter, "supports_streaming", True)

        _ = _write_opencode_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
        config = make_config(
            tmp_path,
            agent="opencode run",
            max_iterations=3,
            stop_on_completion_signal=True,
        )
        state = make_state()
        emitter = QueueEmitter()

        run_loop(config, state, emitter)

        # Promise tag in stdout stops the loop after the first iteration.
        assert state.promise_completed is True
        assert state.iteration == 1
        assert state.completed == 1
        assert state.failed == 0
        assert state.status == RunStatus.COMPLETED

        events = drain_events(emitter)  # pyright: ignore[reportUnknownVariableType]
        activity = events_of_type(events, EventType.AGENT_ACTIVITY)  # pyright: ignore[reportUnknownVariableType]
        kinds = [e.data["raw"].get("type") for e in activity]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert "tool_use" in kinds
        assert "step_finish" in kinds

        stop_event = events_of_type(events, EventType.RUN_STOPPED)[0]  # pyright: ignore[reportUnknownVariableType]
        assert stop_event.data["reason"] == "completed"  # pyright: ignore[reportUnknownMemberType]
